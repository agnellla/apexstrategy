"use client";

import { useState } from "react";
import Link from "next/link";
import { useCircuit, useIndex, circuitName, planLabel, signed } from "@/lib/data";
import { Callout, Panel, Skeleton } from "@/components/Primitives";
import { RacePicker } from "@/components/RacePicker";
import { DecisionPicker, pickDecision } from "@/components/DecisionPicker";
import { Decomposition, DecompositionTable } from "@/components/Decomposition";
import type { Explanation } from "@/lib/types";

/**
 * Does the arithmetic on screen actually add up?
 *
 * The Python side already guarantees this to a millionth of a second and raises if it
 * ever fails, but the payload is rounded to three decimals for the wire, so the true
 * residual arrives as a meaningless 0. What is worth checking here is what a reader can
 * check: whether the four numbers printed on the page sum to the number printed under
 * them. Anything else would be reporting a proof the reader cannot follow.
 */
function reconciles(why: Explanation): boolean {
  const c = why.components;
  const shown = c.tyre_age_s + c.compound_s + c.pit_lane_s + c.sampling_s;
  return Math.abs(shown - why.total_s) < 0.005;
}

export default function DecisionPanel() {
  const index = useIndex();
  const [circuit, setCircuit] = useState("hungary");
  const [lap, setLap] = useState(20);
  const [driver, setDriver] = useState("VER");
  const [showTable, setShowTable] = useState(false);
  const file = useCircuit(circuit);

  const decision = file.data ? pickDecision(file.data, lap, driver) : null;
  const why = decision?.explanation ?? null;

  return (
    <div className="shell" style={{ padding: "56px 24px 0", display: "grid", gap: 28 }}>
      <header>
        <div className="eyebrow" style={{ marginBottom: 14 }}>Decision panel</div>
        <h2 style={{ maxWidth: "20ch" }}>An explanation you can add up</h2>
        <p className="note" style={{ marginTop: 16 }}>
          The recommendation comes out of the simulator, so it is explained by the
          simulator. The components below are the same seconds the simulator added,
          regrouped by cause; they reconcile to the headline to within a millionth of a
          second, and the code raises rather than reports if they ever fail to.
        </p>
      </header>

      {index.loading && <Skeleton height={40} label="Loading races" />}
      {index.data && <RacePicker index={index.data} value={circuit} onChange={setCircuit} />}

      {file.loading && <Skeleton height={460} label={`Loading ${circuitName(circuit)}`} />}
      {file.error && <Callout tone="caution">Could not load this race: {file.error}</Callout>}

      {file.data && file.data.decisions.length === 0 && (
        <Callout tone="caution">
          <strong>No recommendation is offered for this race.</strong>{" "}
          {file.data.method.reason}
        </Callout>
      )}

      {file.data && decision && (
        <>
          <DecisionPicker
            file={file.data}
            lap={lap}
            driver={decision.ego}
            onLap={setLap}
            onDriver={setDriver}
          />

          {!why && (
            <Callout>
              At this state the engine agrees with the plan that was actually run, so there
              is no difference to decompose.
            </Callout>
          )}

          {why && (
            <>
              <Panel title="The claim" provenance="counterfactual">
                <div style={{ display: "grid", gap: 6 }}>
                  <div className="eyebrow">Instead of {planLabel(why.baseline.label)}</div>
                  <p style={{ fontSize: 22, fontFamily: "var(--font-display)", margin: 0 }}>
                    {planLabel(why.plan.label)}
                  </p>
                  <p className="num" style={{ fontSize: 30, margin: "6px 0 0", color: why.total_s < 0 ? "var(--gain)" : "var(--loss)" }}>
                    {signed(why.total_s)}s
                  </p>
                  <p className="note" style={{ margin: "2px 0 0" }}>
                    over the {why.to_lap - why.from_lap} laps still to run, across{" "}
                    {why.n_sims.toLocaleString()} paired simulations.
                  </p>
                </div>
              </Panel>

              <Panel
                title="Where the difference comes from"
                provenance="counterfactual"
                meta={reconciles(why) ? "components sum to the net" : "does not reconcile"}
              >
                <Decomposition explanation={why} />

                <div style={{ display: "flex", gap: 8, marginTop: 18 }}>
                  <button className="control" aria-pressed={showTable} onClick={() => setShowTable((v) => !v)}>
                    {showTable ? "Hide table" : "Show table"}
                  </button>
                </div>
                {showTable && (
                  <div style={{ marginTop: 14 }}>
                    <DecompositionTable explanation={why} />
                  </div>
                )}

                {!why.is_settled && (
                  <div style={{ marginTop: 18 }}>
                    <Callout tone="caution">
                      The sampling row is {signed(why.components.sampling_s)}s against a{" "}
                      {signed(why.total_s)}s headline. That row is mean zero by
                      construction, so its size here says this comparison is not settled and
                      the honest answer is more draws, not a tighter story.
                    </Callout>
                  </div>
                )}
              </Panel>

              <Panel title="Two effects that cancel, and two that are missing">
                <p className="note" style={{ marginTop: 0 }}>
                  Reference pace and the in-race track trend are the two largest terms in
                  any absolute projection, and neither appears above. Both plans run the
                  same laps of the same race, so both enter both totals identically and
                  vanish in the subtraction. What survives is about the strategy.
                </p>
                <dl style={{ display: "grid", gap: 14, margin: "18px 0 0" }}>
                  {why.not_included.map((n) => (
                    <div key={n.term}>
                      <dt style={{ fontWeight: 600, fontSize: 13.5 }}>{n.term}</dt>
                      <dd className="note" style={{ margin: "3px 0 0" }}>
                        {n.why}
                        {n.measured_bias_s_per_dirty_air_lap !== undefined && (
                          <>
                            {" "}Measured at{" "}
                            <span className="num">
                              {signed(n.measured_bias_s_per_dirty_air_lap)}s
                            </span>{" "}
                            per lap spent in traffic, and one-sided: a plan that rejoins
                            behind someone is flattered by every number on this page.
                          </>
                        )}
                        {n.available_as && <> Available as: {n.available_as}.</>}
                      </dd>
                    </div>
                  ))}
                </dl>
              </Panel>

              <p className="note">
                The full list of what this build does not model is on the{" "}
                <Link href="/method/">method page</Link>.
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}
