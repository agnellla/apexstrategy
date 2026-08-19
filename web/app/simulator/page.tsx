"use client";

import { useState } from "react";
import Link from "next/link";
import { useCircuit, useIndex, circuitName, planLabel } from "@/lib/data";
import { Callout, Panel, Skeleton, SkeletonRows } from "@/components/Primitives";
import { RacePicker } from "@/components/RacePicker";
import { DecisionPicker, pickDecision } from "@/components/DecisionPicker";
import { StateStrip } from "@/components/StateStrip";
import { DeltaChart, DeltaTable } from "@/components/DeltaChart";

export default function Simulator() {
  const index = useIndex();
  const [circuit, setCircuit] = useState("hungary");
  const [lap, setLap] = useState(20);
  const [driver, setDriver] = useState("VER");
  const [showTable, setShowTable] = useState(false);
  const file = useCircuit(circuit);

  const decision = file.data ? pickDecision(file.data, lap, driver) : null;

  return (
    <div className="shell" style={{ padding: "56px 24px 0", display: "grid", gap: 28 }}>
      <header>
        <div className="eyebrow" style={{ marginBottom: 14 }}>Strategy simulator</div>
        <h2 style={{ maxWidth: "22ch" }}>Every legal plan, run against the same luck</h2>
        <p className="note" style={{ marginTop: 16 }}>
          Around ninety pit plans survive the two-compound rule and the stint-length
          filter at a typical state. All of them are simulated against identical random
          draws, so the difference between two plans is a difference in strategy rather
          than in weather. Without that pairing a nine-second gap reads as a coin flip.
        </p>
      </header>

      {index.loading && <Skeleton height={40} label="Loading races" />}
      {index.data && <RacePicker index={index.data} value={circuit} onChange={setCircuit} />}

      {file.loading && <Skeleton height={520} label={`Loading ${circuitName(circuit)}`} />}
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
            onLap={(l) => setLap(l)}
            onDriver={setDriver}
          />

          <Panel title="State at the decision lap" provenance="observed">
            <StateStrip decision={decision} />
          </Panel>

          <Panel
            title="Candidate plans"
            provenance="counterfactual"
            meta={`${decision.candidates_evaluated} evaluated · ${decision.candidates_shown} shown · ${decision.n_sims} draws each`}
          >
            {!decision.baseline_in_support && (
              <div style={{ marginBottom: 16 }}>
                <Callout tone="caution">
                  The plan this driver actually ran keeps a set of tyres past anything
                  observed for that compound, so the model is extrapolating to price it.
                  The differences below are measured against that extrapolation and should
                  not be read as time that was on the table.
                </Callout>
              </div>
            )}

            <DeltaChart rows={decision.candidates} baselineLabel="As executed" />

            <div style={{ display: "flex", gap: 8, marginTop: 18, marginBottom: 4 }}>
              <button
                className="control"
                aria-pressed={showTable}
                onClick={() => setShowTable((v) => !v)}
              >
                {showTable ? "Hide table" : "Show table"}
              </button>
            </div>
            {showTable && (
              <div style={{ marginTop: 12 }}>
                <DeltaTable rows={decision.candidates} />
              </div>
            )}

            <p className="note" style={{ marginTop: 18, marginBottom: 0 }}>
              {decision.indistinguishable_from_best.length > 1 ? (
                <>
                  <strong>
                    {decision.indistinguishable_from_best.length} of these plans are not
                    separated by this simulation.
                  </strong>{" "}
                  They sit within two Monte Carlo standard errors of each other, so ranking
                  one above another here is noise. Presenting the top row as the answer
                  would overstate what was measured.
                </>
              ) : (
                <>
                  The top plan is separated from the rest by more than two Monte Carlo
                  standard errors.
                </>
              )}{" "}
              MC error is not model error: the simulator's own accuracy is roughly 0.5% of
              the projected window, and more draws do not shrink it. See{" "}
              <Link href="/method/">method</Link>.
            </p>
          </Panel>

          <Panel title="Recommendation" provenance="counterfactual">
            <p style={{ fontSize: 20, fontFamily: "var(--font-display)", margin: "0 0 6px" }}>
              {planLabel(decision.recommendation)}
            </p>
            <p className="note" style={{ margin: 0 }}>
              Ranked on expected race time. Positions and championship points need
              overtaking, which this build does not model, so it does not report them.
            </p>
            <div style={{ marginTop: 18 }}>
              <Link className="control" href="/decision/" style={{ textDecoration: "none" }}>
                See why, as arithmetic
              </Link>
            </div>
          </Panel>
        </>
      )}

      {file.loading && <SkeletonRows rows={5} />}
    </div>
  );
}
