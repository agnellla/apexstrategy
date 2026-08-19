"use client";

import type { Explanation } from "@/lib/types";
import { signed } from "@/lib/data";

/**
 * The decomposition, as a waterfall.
 *
 * A waterfall is the right form here for a reason that is not aesthetic: the components
 * are guaranteed to sum to the total, so a chart whose bars physically stack to the net
 * is telling the literal truth about the arithmetic. If a segment ever failed to land on
 * the next one, the Python side would have raised before this shipped.
 *
 * Diverging colour, because the measure has polarity: time gained against time lost, with
 * the net drawn in neutral ink so it reads as a sum rather than as a third category.
 */

const ROW = 34;
const LABEL_W = 150;
const VALUE_W = 78;

interface Segment {
  label: string;
  value: number;
  note?: string;
}

export function Decomposition({ explanation }: { explanation: Explanation }) {
  const c = explanation.components;
  const segments: Segment[] = [
    { label: "Fresher rubber", value: c.tyre_age_s, note: "running a newer set, compound held fixed" },
    { label: "Compound choice", value: c.compound_s, note: "a different compound, tyre age held fixed" },
    { label: "Pit-lane time", value: c.pit_lane_s, note: "stops one plan makes and the other does not" },
    { label: "Sampling residue", value: c.sampling_s, note: "mean zero; its size is how settled this is" },
  ];

  const width = 720;
  const plotW = width - LABEL_W - VALUE_W - 16;
  const height = (segments.length + 1) * ROW + 12;

  let running = 0;
  const spans = segments.map((s) => {
    const from = running;
    running += s.value;
    return { ...s, from, to: running };
  });

  const all = [0, ...spans.map((s) => s.to), explanation.total_s];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = (hi - lo) * 0.1 || 1;
  const x = (v: number) => LABEL_W + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * plotW;

  return (
    <div>
      <svg
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Waterfall of the components that sum to the total time difference"
      >
        <line x1={x(0)} y1={0} x2={x(0)} y2={(segments.length + 1) * ROW} stroke="var(--rule-strong)" strokeWidth="1" />

        {spans.map((s, i) => {
          const y = i * ROW + 7;
          const x0 = x(Math.min(s.from, s.to));
          const x1 = x(Math.max(s.from, s.to));
          const gain = s.value < 0;
          return (
            <g key={s.label}>
              <text x={LABEL_W - 14} y={y + 13} textAnchor="end" style={{ fontSize: 12.5, fill: "var(--ink-2)" }}>
                {s.label}
              </text>
              <rect
                x={x0}
                y={y}
                width={Math.max(1.5, x1 - x0 - 2)}
                height={ROW - 14}
                fill={gain ? "var(--gain)" : "var(--loss)"}
              />
              {/* Connector to where the next segment starts. */}
              {i < spans.length - 1 && (
                <line
                  x1={x(s.to)}
                  y1={y}
                  x2={x(s.to)}
                  y2={y + ROW}
                  stroke="var(--rule-strong)"
                  strokeWidth="1"
                />
              )}
              <text
                x={width - VALUE_W + 8}
                y={y + 13}
                className="num"
                style={{ fontSize: 12.5, fill: "var(--ink)" }}
              >
                {signed(s.value)}
              </text>
              <title>{`${s.label}: ${signed(s.value)}s (${s.note})`}</title>
            </g>
          );
        })}

        {/* The net, drawn from zero so its length is comparable to the segments. */}
        <g>
          <line
            x1={LABEL_W - 8}
            y1={segments.length * ROW + 2}
            x2={width - 8}
            y2={segments.length * ROW + 2}
            className="axis-line"
          />
          <text
            x={LABEL_W - 14}
            y={segments.length * ROW + 24}
            textAnchor="end"
            style={{ fontSize: 13, fill: "var(--ink)", fontWeight: 600 }}
          >
            Net
          </text>
          <rect
            x={x(Math.min(0, explanation.total_s))}
            y={segments.length * ROW + 11}
            width={Math.max(1.5, Math.abs(x(explanation.total_s) - x(0)) - 2)}
            height={ROW - 14}
            fill="var(--neutral)"
          />
          <text
            x={width - VALUE_W + 8}
            y={segments.length * ROW + 24}
            className="num"
            style={{ fontSize: 13, fill: "var(--ink)", fontWeight: 500 }}
          >
            {signed(explanation.total_s)}
          </text>
        </g>
      </svg>

      <div style={{ display: "flex", gap: 20, marginTop: 4, flexWrap: "wrap" }}>
        <Key color="var(--gain)">Time gained</Key>
        <Key color="var(--loss)">Time lost</Key>
        <Key color="var(--neutral)">Net</Key>
      </div>
    </div>
  );
}

function Key({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 7, fontSize: 12, color: "var(--ink-2)" }}>
      <span aria-hidden style={{ width: 11, height: 11, background: color, border: "1px solid var(--rule-strong)" }} />
      {children}
    </span>
  );
}

export function DecompositionTable({ explanation }: { explanation: Explanation }) {
  const c = explanation.components;
  const rows: [string, number][] = [
    ["Fresher rubber", c.tyre_age_s],
    ["Compound choice", c.compound_s],
    ["Pit-lane time", c.pit_lane_s],
    ["Sampling residue", c.sampling_s],
  ];
  return (
    <table>
      <caption style={{ captionSide: "top", textAlign: "left", fontSize: 12, color: "var(--ink-3)", paddingBottom: 8 }}>
        Same components as the chart, as numbers.
      </caption>
      <thead>
        <tr>
          <th scope="col">Component</th>
          <th scope="col">Seconds</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, v]) => (
          <tr key={label}>
            <th scope="row">
              {label}
            </th>
            <td className="num">{signed(v)}</td>
          </tr>
        ))}
        <tr>
          <th scope="row" className="strong">
            Net
          </th>
          <td className="num" style={{ fontWeight: 500, borderBottom: 0 }}>{signed(explanation.total_s)}</td>
        </tr>
      </tbody>
    </table>
  );
}
