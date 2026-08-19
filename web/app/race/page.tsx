"use client";

import { useState } from "react";
import Link from "next/link";
import { useCircuit, useIndex, circuitName, pct } from "@/lib/data";
import { Callout, Panel, ProvenanceKey, Skeleton, SkeletonRows } from "@/components/Primitives";
import { RacePicker } from "@/components/RacePicker";
import { StintTimeline } from "@/components/StintTimeline";

export default function RaceOverview() {
  const index = useIndex();
  const [circuit, setCircuit] = useState("hungary");
  const file = useCircuit(circuit);

  return (
    <>
      <section className="shell" style={{ padding: "44px 24px 26px" }}>
        <div className="eyebrow" style={{ marginBottom: 12 }}>Race</div>
        <h2 style={{ maxWidth: "20ch" }}>What actually happened</h2>
        <p className="note" style={{ marginTop: 14 }}>
          Observed only. Every mark on this page is measurement, which is why the yield
          decomposition sits beside the timeline rather than on a diagnostics page nobody
          opens: it is the honest reading of how much of this race the models were allowed
          to learn from.
        </p>
      </section>

      <div className="shell" style={{ display: "grid", gap: 28, paddingBottom: 16 }}>
        {index.loading && <Skeleton height={40} label="Loading races" />}
        {index.data && (
          <RacePicker index={index.data} value={circuit} onChange={setCircuit} />
        )}

        {file.loading && <Skeleton height={430} label={`Loading ${circuitName(circuit)}`} />}
        {file.error && <Callout tone="caution">Could not load this race: {file.error}</Callout>}

        {file.data && (
          <>
            {file.data.method.excluded && (
              <Callout tone="caution">
                <strong>Stored, and excluded from model fitting.</strong>{" "}
                {file.data.method.reason} No strategy recommendation is offered for this
                race; showing one anyway would be the exact thing the exclusion exists to
                prevent.
              </Callout>
            )}

            <Panel
              title="Stints"
              provenance="observed"
              meta={`${file.data.race.total_laps} laps · ${file.data.race.drivers.length} starters`}
            >
              <StintTimeline race={file.data.race} />
              <p className="note" style={{ marginTop: 14, marginBottom: 0 }}>
                Shaded columns are laps run under a safety car or virtual safety car. One
                colour means one compound and nothing else; a stint that ran to the flag is
                marked in the tooltip and in the table, because its length is a lower bound
                on what the set could do rather than a choice anybody made.
              </p>
            </Panel>

            <div style={{ display: "grid", gap: 28, gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))" }}>
              <Panel
                title="Where the laps went"
                provenance="observed"
                meta={`${pct(file.data.race.data_quality.clean_yield, 1)} clean`}
              >
                <p className="note" style={{ marginTop: 0 }}>
                  Laps are flagged, never dropped. A model that quietly trains on 41% of a
                  race while reporting on all of it is the failure this panel exists to
                  make impossible.
                </p>
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Excluded for</th>
                      <th scope="col">Laps</th>
                      <th scope="col">Share</th>
                    </tr>
                  </thead>
                  <tbody>
                    {file.data.race.data_quality.exclusions.map((e) => (
                      <tr key={e.reason}>
                        <th scope="row" className="mono"
                        >
                          {e.reason.replace(/_/g, " ")}
                        </th>
                        <td className="num">{e.n_laps}</td>
                        <td className="num" style={{ color: "var(--ink-3)" }}>
                          {pct(e.n_laps / file.data!.race.data_quality.total_laps_recorded, 1)}
                        </td>
                      </tr>
                    ))}
                    <tr>
                      <th scope="row" className="strong">
                        Clean, and usable
                      </th>
                      <td className="num" style={{ borderBottom: 0, fontWeight: 500 }}>
                        {file.data.race.data_quality.clean_laps}
                      </td>
                      <td className="num" style={{ borderBottom: 0 }}>
                        {pct(file.data.race.data_quality.clean_yield, 1)}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </Panel>

              <Panel title="Reading the marks" meta="applies site-wide">
                <p className="note" style={{ marginTop: 0 }}>
                  Four classes, applied consistently on every chart. The difference between
                  an honest tool and a misleading one is whether a projection can be
                  mistaken for a measurement.
                </p>
                {index.data ? (
                  <ProvenanceKey
                    classes={["observed", "predicted", "simulated", "counterfactual"]}
                    descriptions={index.data.provenance}
                  />
                ) : (
                  <SkeletonRows rows={4} />
                )}
              </Panel>
            </div>

            {!file.data.method.excluded && (
              <Panel title="Decisions computed for this race" meta={`${file.data.decisions.length} states`}>
                <p className="note" style={{ marginTop: 0 }}>
                  The grid is frozen at four decision laps for the six leading runners at
                  each. Pick one on the{" "}
                  <Link href="/simulator/">simulator</Link>, or go straight to the{" "}
                  <Link href="/decision/">recommendation and its arithmetic</Link>.
                </p>
              </Panel>
            )}
          </>
        )}
      </div>
    </>
  );
}
