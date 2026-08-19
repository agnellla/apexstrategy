"use client";

import { useState } from "react";
import type { WindowPoint } from "@/lib/types";
import { signed } from "@/lib/data";

/**
 * The pit window: time gained or lost against the current plan, for every lap you could
 * box on.
 *
 * This replaced a twelve-row ranked table, and the reason is worth recording. At Hungary
 * lap 20 those twelve rows were four adjacent laps and one compound choice wearing twelve
 * labels, and four of them were inside Monte Carlo noise of each other. A ranked list
 * presents that as an ordering. The same numbers on one axis show what is actually true:
 * a floor two or three laps wide, and a cliff after it.
 *
 * The floor is the window. Its width is the answer to "how long can I wait", which is the
 * question being asked on the lap, and the slope past it is what waiting costs.
 */

const H = 210;
const PAD_L = 52;
const PAD_R = 18;
const PAD_T = 16;

export function PitWindow({
  points,
  bestLap,
  windowLaps,
  executedLap,
  currentLap,
}: {
  points: WindowPoint[];
  bestLap: number;
  windowLaps: number[];
  executedLap: number | null;
  currentLap: number;
}) {
  const [hover, setHover] = useState<WindowPoint | null>(null);
  if (points.length < 2) return null;

  const W = 760;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - 34;

  const laps = points.map((p) => p.lap);
  const loLap = Math.min(...laps);
  const hiLap = Math.max(...laps);
  const deltas = points.map((p) => p.delta_s);
  const lo = Math.min(...deltas, 0);
  const hi = Math.max(...deltas, 0);
  const pad = (hi - lo) * 0.12 || 1;

  const x = (lap: number) => PAD_L + ((lap - loLap) / (hiLap - loLap || 1)) * plotW;
  const y = (v: number) => PAD_T + ((hi + pad - v) / (hi - lo + 2 * pad)) * plotH;

  const path = points.map((p, i) => `${i ? "L" : "M"}${x(p.lap)} ${y(p.delta_s)}`).join(" ");
  const winLo = Math.min(...windowLaps);
  const winHi = Math.max(...windowLaps);

  const step = niceStep(hi - lo + 2 * pad);
  const ticks: number[] = [];
  for (let t = Math.ceil((lo - pad) / step) * step; t <= hi + pad; t += step) ticks.push(+t.toFixed(4));

  return (
    <div style={{ position: "relative" }}>
      <div className="table-scroll">
        <svg
          className="chart"
          viewBox={`0 0 ${W} ${H}`}
          style={{ minWidth: 560 }}
          role="img"
          aria-label={`Time gained or lost for each possible pit lap from ${loLap} to ${hiLap}. The window runs from lap ${winLo} to ${winHi}.`}
          onMouseLeave={() => setHover(null)}
        >
          {/* The window itself: every lap the simulation cannot separate from the best. */}
          <rect
            x={x(winLo) - 4}
            y={PAD_T}
            width={x(winHi) - x(winLo) + 8}
            height={plotH}
            fill="var(--accent)"
            opacity="0.12"
          />

          {ticks.map((t) => (
            <g key={t}>
              <line className={t === 0 ? "axis-line" : "grid-line"} x1={PAD_L} y1={y(t)} x2={W - PAD_R} y2={y(t)} />
              <text x={PAD_L - 8} y={y(t) + 3} textAnchor="end" className="tick-text">
                {t === 0 ? "0" : signed(t, Math.abs(step) < 1 ? 1 : 0)}
              </text>
            </g>
          ))}

          <path d={path} fill="none" stroke="var(--simulated)" strokeWidth="2" strokeDasharray="5 3" />

          {points.map((p) => {
            const isBest = p.lap === bestLap;
            return (
              <g key={p.lap} onMouseEnter={() => setHover(p)}>
                <rect x={x(p.lap) - 9} y={PAD_T} width={18} height={plotH} fill="transparent" />
                <circle
                  cx={x(p.lap)}
                  cy={y(p.delta_s)}
                  r={isBest ? 6 : 4}
                  fill={isBest ? "var(--accent)" : "var(--simulated)"}
                  stroke="var(--surface)"
                  strokeWidth="2"
                />
              </g>
            );
          })}

          {executedLap !== null && executedLap >= loLap && executedLap <= hiLap && (
            <g>
              <line
                x1={x(executedLap)}
                y1={PAD_T}
                x2={x(executedLap)}
                y2={PAD_T + plotH}
                stroke="var(--observed)"
                strokeWidth="1.5"
              />
              <text x={x(executedLap) + 6} y={PAD_T + 11} className="tick-text" style={{ fill: "var(--ink-2)" }}>
                boxed here
              </text>
            </g>
          )}

          <line className="axis-line" x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} />
          {points
            .filter((_, i) => i % Math.ceil(points.length / 9) === 0)
            .map((p) => (
              <text key={`x${p.lap}`} x={x(p.lap)} y={PAD_T + plotH + 16} textAnchor="middle" className="tick-text">
                {p.lap}
              </text>
            ))}
          <text
            x={PAD_L + plotW / 2}
            y={H - 4}
            textAnchor="middle"
            style={{ fontSize: 11.5, fill: "var(--ink-3)" }}
          >
            Lap you box on. Seconds gained against the current plan, so lower is better.
          </text>
        </svg>
      </div>

      <div
        aria-live="polite"
        style={{
          marginTop: 8,
          fontSize: 13,
          color: "var(--ink-2)",
          fontFamily: "var(--font-mono)",
          minHeight: "1.4em",
        }}
      >
        {hover
          ? `Lap ${hover.lap} · ${signed(hover.delta_s)}s · ${hover.compound} · ${hover.n_stops}-stop`
          : `Window laps ${winLo} to ${winHi} · currently on lap ${currentLap}`}
      </div>
    </div>
  );
}

function niceStep(range: number) {
  const raw = range / 4;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  return (n >= 5 ? 5 : n >= 2 ? 2 : 1) * mag;
}

export function WindowTable({ points, bestLap }: { points: WindowPoint[]; bestLap: number }) {
  return (
    <div className="table-scroll">
      <table>
        <caption style={{ captionSide: "top", textAlign: "left", fontSize: 12, color: "var(--ink-3)", paddingBottom: 8 }}>
          Best plan available for each pit lap, as numbers.
        </caption>
        <thead>
          <tr>
            <th scope="col">Box on lap</th>
            <th scope="col">Gain</th>
            <th scope="col">Next tyre</th>
            <th scope="col">Stops left</th>
          </tr>
        </thead>
        <tbody>
          {points.map((p) => (
            <tr key={p.lap} style={p.lap === bestLap ? { background: "var(--surface-sunk)" } : undefined}>
              <th scope="row" className="mono">
                {p.lap}
              </th>
              <td className="num" style={{ color: p.delta_s < 0 ? "var(--gain)" : undefined }}>
                {signed(p.delta_s)}
              </td>
              <td>{p.compound}</td>
              <td className="num">{p.n_stops}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
