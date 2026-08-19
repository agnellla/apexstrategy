"use client";

import { useState } from "react";
import type { RacePayload } from "@/lib/types";
import { CompoundChip, compoundColor } from "./Primitives";

/**
 * Stint timeline: one row per driver, one bar per set of tyres.
 *
 * Every mark here is Observed. Adjacent stints are separated by a surface gap rather than
 * a border, so a stop reads as a break in the run and not as a mark of its own.
 *
 * The gap is 4px rather than the 2px a stacked bar would use, because here it is doing
 * real work: a driver who takes two stints on the same compound (Hamilton and Perez both
 * did at this race) has nothing but that gap to show a stop happened, and at 2px the stop
 * that the whole product exists to reason about was almost invisible.
 *
 * A stint that ran to the flag carries no mark on the bar at all, which took two tries to
 * arrive at. Fading it produced a 55% amber and a 55% blue that read as two more
 * compounds; hatching it at this bar height produced a tint rather than a texture and read
 * the same way. Both were spending the encoding on information the reader already has,
 * since the last stint of anyone who finished ran to the flag by definition. It is in the
 * tooltip and marked in the table, and the palette stays three colours meaning three
 * compounds.
 */

const ROW = 22;
const GAP = 4;
const LABEL_W = 52;

export function StintTimeline({ race }: { race: RacePayload }) {
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);
  const [table, setTable] = useState(false);

  const drivers = race.drivers.map((d) => d.code);
  const byDriver = new Map(drivers.map((d) => [d, [] as RacePayload["stints"]]));
  for (const s of race.stints) byDriver.get(s.driver)?.push(s);

  const width = 900;
  const plotW = width - LABEL_W - 8;
  const height = drivers.length * ROW + 26;
  const x = (lap: number) => LABEL_W + (lap / race.total_laps) * plotW;

  const ticks = [1, ...Array.from({ length: Math.floor(race.total_laps / 10) }, (_, i) => (i + 1) * 10)];

  return (
    <div style={{ position: "relative" }}>
      <div className="table-scroll">
        <svg
          className="chart"
          viewBox={`0 0 ${width} ${height}`}
          style={{ minWidth: 640 }}
          role="img"
          aria-label={`Stint timeline for ${drivers.length} drivers over ${race.total_laps} laps`}
        >
          {race.caution_laps.length > 0 &&
            race.caution_laps.map((lap) => (
              <rect
                key={`c${lap}`}
                x={x(lap - 1)}
                y={0}
                width={Math.max(1, x(lap) - x(lap - 1))}
                height={drivers.length * ROW}
                fill="var(--ink)"
                opacity="0.07"
              />
            ))}

          {ticks.map((t) => (
            <line key={t} className="grid-line" x1={x(t)} y1={0} x2={x(t)} y2={drivers.length * ROW} />
          ))}

          {drivers.map((code, i) => {
            const stints = byDriver.get(code) ?? [];
            return (
              <g key={code}>
                <text
                  x={LABEL_W - 10}
                  y={i * ROW + ROW / 2 + 4}
                  textAnchor="end"
                  className="num"
                  style={{ fontSize: 11, fill: "var(--ink-2)" }}
                >
                  {code}
                </text>
                {stints.map((s) => {
                  const x0 = x(s.start_lap - 1);
                  const x1 = x(s.end_lap);
                  const label = `${code} · ${s.compound} · laps ${s.start_lap} to ${s.end_lap} (${s.laps})${
                    s.censored ? " · ran to the flag" : ""
                  }`;
                  return (
                    <rect
                      key={s.stint}
                      x={x0}
                      y={i * ROW + 3}
                      width={Math.max(1, x1 - x0 - GAP)}
                      height={ROW - 7}
                      fill={compoundColor(s.compound)}
                      onMouseEnter={(e) =>
                        setHover({
                          x: e.nativeEvent.offsetX,
                          y: i * ROW,
                          text: label,
                        })
                      }
                      onMouseLeave={() => setHover(null)}
                    >
                      <title>{label}</title>
                    </rect>
                  );
                })}
              </g>
            );
          })}

          <line
            className="axis-line"
            x1={LABEL_W}
            y1={drivers.length * ROW}
            x2={width - 8}
            y2={drivers.length * ROW}
          />
          {ticks.map((t) => (
            <text
              key={`l${t}`}
              x={x(t)}
              y={drivers.length * ROW + 15}
              textAnchor="middle"
              className="tick-text"
            >
              {t}
            </text>
          ))}
        </svg>
      </div>

      <div
        style={{
          display: "flex",
          gap: 20,
          flexWrap: "wrap",
          marginTop: 12,
          alignItems: "center",
        }}
      >
        {(["SOFT", "MEDIUM", "HARD"] as const).map((c) => (
          <CompoundChip key={c} compound={c} />
        ))}
        <span style={{ display: "inline-flex", alignItems: "center", gap: 7, fontSize: 12, color: "var(--ink-2)" }}>
          <svg width="11" height="11" aria-hidden>
            <rect width="11" height="11" fill="var(--ink)" opacity="0.07" />
            <rect width="11" height="11" fill="none" stroke="var(--rule-strong)" />
          </svg>
          Caution
        </span>
        <button
          className="control"
          aria-pressed={table}
          onClick={() => setTable((v) => !v)}
          style={{ marginLeft: "auto" }}
        >
          {table ? "Hide table" : "Show table"}
        </button>
      </div>

      {table && (
        <div className="table-scroll" style={{ marginTop: 14, maxHeight: 340, overflowY: "auto" }}>
          <table>
            <caption style={{ captionSide: "top", textAlign: "left", fontSize: 12, color: "var(--ink-3)", paddingBottom: 8 }}>
              Every stint as numbers. The compound colours sit below the 3:1 contrast line
              against paper, so the names are always available in text. A + on the lap count
              marks a stint that ran to the flag, where the length is a lower bound on what
              the set could do rather than a choice.
            </caption>
            <thead>
              <tr>
                <th scope="col">Driver</th>
                <th scope="col">Stint</th>
                <th scope="col">Compound</th>
                <th scope="col">From</th>
                <th scope="col">To</th>
                <th scope="col">Laps</th>
              </tr>
            </thead>
            <tbody>
              {race.stints.map((s) => (
                <tr key={`${s.driver}-${s.stint}`}>
                  <th scope="row" className="num" style={{ textTransform: "none", letterSpacing: 0, fontSize: 12, color: "var(--ink)", fontWeight: 400, borderBottom: "1px solid var(--rule)" }}>
                    {s.driver}
                  </th>
                  <td className="num">{s.stint}</td>
                  <td style={{ textAlign: "right" }}>
                    <span style={{ display: "inline-flex", justifyContent: "flex-end", width: "100%" }}>
                      <CompoundChip compound={s.compound} />
                    </span>
                  </td>
                  <td className="num">{s.start_lap}</td>
                  <td className="num">{s.end_lap}</td>
                  <td className="num">
                    {s.laps}
                    {s.censored && <span style={{ color: "var(--ink-3)" }}> +</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {hover && (
        <div
          role="status"
          style={{
            position: "absolute",
            left: Math.min(hover.x, 520),
            top: hover.y + 26,
            background: "var(--ink)",
            color: "var(--paper)",
            padding: "5px 8px",
            fontSize: 12,
            pointerEvents: "none",
            zIndex: 5,
            whiteSpace: "nowrap",
          }}
        >
          {hover.text}
        </div>
      )}
    </div>
  );
}
