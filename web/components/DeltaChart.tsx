"use client";

import { useState } from "react";
import type { ScenarioRow } from "@/lib/types";
import { planLabel, secs, signed } from "@/lib/data";

/**
 * Candidate plans as a dot-and-interval chart against the baseline.
 *
 * The measure is a paired difference, so zero is the baseline and the axis is signed.
 * Each row carries two marks with different meanings, which is the whole reason this is
 * not a bar chart: the dot is the expected difference and the bar behind it runs out to
 * the downside, the mean of the worst tenth. A plan can be faster in expectation and
 * carry more downside than the one below it, and that has to be visible in one glance.
 *
 * Everything drawn here is Counterfactual apart from the baseline row: these are plans
 * nobody ran, and no outcome exists to check them against.
 */

const ROW = 26;
const LABEL_W = 168;

export function DeltaChart({ rows, baselineLabel }: { rows: ScenarioRow[]; baselineLabel: string }) {
  const [hover, setHover] = useState<number | null>(null);

  const width = 760;
  const plotW = width - LABEL_W - 56;
  const height = rows.length * ROW + 52;

  const values = rows.flatMap((r) => [r.delta_s, r.cvar10_delta_s, 0]);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = (hi - lo) * 0.08 || 1;
  const x = (v: number) => LABEL_W + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * plotW;

  const step = niceStep(hi - lo + 2 * pad);
  const ticks: number[] = [];
  for (let t = Math.ceil((lo - pad) / step) * step; t <= hi + pad; t += step) ticks.push(+t.toFixed(6));

  return (
    <div style={{ position: "relative" }}>
      <div className="table-scroll">
        <svg
          className="chart"
          viewBox={`0 0 ${width} ${height}`}
          style={{ minWidth: 600 }}
          role="img"
          aria-label="Expected time difference and downside for each candidate plan"
        >
          {ticks.map((t) => (
            <line key={t} className="grid-line" x1={x(t)} y1={0} x2={x(t)} y2={rows.length * ROW} />
          ))}

          {/* Zero is the baseline, so it gets the only emphasised rule on the chart. */}
          <line
            x1={x(0)}
            y1={0}
            x2={x(0)}
            y2={rows.length * ROW}
            stroke="var(--ink)"
            strokeWidth="1.5"
          />

          {rows.map((r, i) => {
            const y = i * ROW + ROW / 2;
            const isBase = r.is_baseline;
            const color = isBase ? "var(--ink)" : r.delta_s <= 0 ? "var(--gain)" : "var(--loss)";
            return (
              <g
                key={r.label}
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
              >
                <rect
                  x={LABEL_W - 8}
                  y={i * ROW}
                  width={width - LABEL_W + 8}
                  height={ROW}
                  fill={hover === i ? "var(--surface-sunk)" : "transparent"}
                />
                <text
                  x={LABEL_W - 16}
                  y={y + 4}
                  textAnchor="end"
                  className="num"
                  style={{ fontSize: 11, fill: isBase ? "var(--ink)" : "var(--ink-2)" }}
                >
                  {planLabel(r.label)}
                </text>

                {/* Downside reach: expected difference out to the worst tenth. */}
                <line
                  x1={x(r.delta_s)}
                  y1={y}
                  x2={x(r.cvar10_delta_s)}
                  y2={y}
                  stroke={color}
                  strokeWidth="2"
                  strokeDasharray={isBase ? undefined : "3 2"}
                  opacity="0.5"
                />
                <line
                  x1={x(r.cvar10_delta_s)}
                  y1={y - 4}
                  x2={x(r.cvar10_delta_s)}
                  y2={y + 4}
                  stroke={color}
                  strokeWidth="1.5"
                  opacity="0.5"
                />
                <circle cx={x(r.delta_s)} cy={y} r="4.5" fill={color} stroke="var(--surface)" strokeWidth="2" />
                <title>
                  {`${r.label}: ${signed(r.delta_s)}s expected, downside ${signed(r.cvar10_delta_s)}s`}
                </title>
              </g>
            );
          })}

          <line className="axis-line" x1={LABEL_W} y1={rows.length * ROW} x2={width - 40} y2={rows.length * ROW} />
          {ticks.map((t) => (
            <text key={`t${t}`} x={x(t)} y={rows.length * ROW + 16} textAnchor="middle" className="tick-text">
              {t === 0 ? "0" : signed(t, Math.abs(step) < 1 ? 1 : 0)}
            </text>
          ))}
          <text
            x={LABEL_W + plotW / 2}
            y={rows.length * ROW + 40}
            textAnchor="middle"
            style={{ fontSize: 11.5, fill: "var(--ink-2)" }}
          >
            Seconds against {baselineLabel.toLowerCase()}. Left is faster.
          </text>
        </svg>
      </div>

      <div style={{ display: "flex", gap: 20, marginTop: 10, flexWrap: "wrap" }}>
        <LegendItem>
          <svg width="16" height="10" aria-hidden>
            <circle cx="8" cy="5" r="4" fill="var(--ink-2)" />
          </svg>
          Expected difference
        </LegendItem>
        <LegendItem>
          <svg width="22" height="10" aria-hidden>
            <line x1="0" y1="5" x2="22" y2="5" stroke="var(--ink-2)" strokeWidth="2" strokeDasharray="3 2" opacity="0.5" />
          </svg>
          Reach to the worst tenth
        </LegendItem>
        <LegendItem>
          <svg width="16" height="10" aria-hidden>
            <line x1="8" y1="0" x2="8" y2="10" stroke="var(--ink)" strokeWidth="1.5" />
          </svg>
          {baselineLabel}
        </LegendItem>
      </div>
    </div>
  );
}

function LegendItem({ children }: { children: React.ReactNode }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 7,
        fontSize: 12,
        color: "var(--ink-2)",
      }}
    >
      {children}
    </span>
  );
}

function niceStep(range: number) {
  const raw = range / 5;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  return (n >= 5 ? 5 : n >= 2 ? 2 : 1) * mag;
}

export function DeltaTable({ rows }: { rows: ScenarioRow[] }) {
  return (
    <div className="table-scroll">
      <table>
        <caption
          style={{
            captionSide: "top",
            textAlign: "left",
            fontSize: 12,
            color: "var(--ink-3)",
            paddingBottom: 8,
          }}
        >
          Same data as the chart above, as numbers.
        </caption>
        <thead>
          <tr>
            <th scope="col">Plan</th>
            <th scope="col">Stops</th>
            <th scope="col">Expected</th>
            <th scope="col">MC error</th>
            <th scope="col">P(faster)</th>
            <th scope="col">Downside</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} style={r.is_baseline ? { background: "var(--surface-sunk)" } : undefined}>
              <th scope="row" className="mono">
                {planLabel(r.label)}
                {r.is_baseline && (
                  <span style={{ color: "var(--ink-3)", marginLeft: 8, fontSize: 11 }}>as executed</span>
                )}
              </th>
              <td className="num">{r.n_stops}</td>
              <td className="num" style={{ color: r.delta_s < 0 ? "var(--gain)" : undefined }}>
                {signed(r.delta_s)}
              </td>
              <td className="num" style={{ color: "var(--ink-3)" }}>±{secs(r.delta_se_s)}</td>
              <td className="num">{(r.p_faster * 100).toFixed(0)}%</td>
              <td className="num">{signed(r.cvar10_delta_s)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
