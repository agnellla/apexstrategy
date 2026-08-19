"use client";

import { useState } from "react";
import Link from "next/link";
import type { Decision } from "@/lib/types";
import { costOfWaiting, planLabel, signed, windowLaps } from "@/lib/data";
import { Callout, CompoundChip, ProvenanceTag } from "./Primitives";
import { PitWindow, WindowTable } from "./PitWindow";
import { Decomposition } from "./Decomposition";

/**
 * The call, and the six things that qualify it.
 *
 * Ordered by what gets read first on a pit wall, which is not the order the engine
 * computes them in. The verb and the lap are the answer. Then how wide the window is, then
 * what waiting costs, then whether the call is actually separated from the alternatives,
 * then what a safety car would do to it. Everything about how the number was produced is
 * below the fold, behind a control, because it is evidence rather than instruction.
 */
export function TheCall({ decision }: { decision: Decision }) {
  const [showWork, setShowWork] = useState(false);
  const [showTable, setShowTable] = useState(false);

  const s = decision.state;
  const best = decision.window.find((p) => p.label === decision.recommendation) ?? decision.window[0];
  const laps = windowLaps(decision.window);
  const winLo = laps.length ? Math.min(...laps) : (best?.lap ?? s.lap);
  const winHi = laps.length ? Math.max(...laps) : (best?.lap ?? s.lap);
  const executed = decision.options.find((o) => o.is_baseline);
  const executedLap = executed?.next_lap ?? null;

  const staysOut = !best;
  const live = !staysOut && winLo <= s.lap + 2;
  const verb = staysOut ? "STAY OUT" : live ? "BOX" : "HOLD";
  const gain = decision.options[0]?.delta_s ?? 0;
  const settled = decision.indistinguishable_from_best.length <= 1;
  const waiting = best ? costOfWaiting(decision.window, winHi) : null;

  return (
    <div style={{ display: "grid", gap: 26 }}>
      {/* --- the answer --- */}
      <section
        style={{
          border: "1px solid var(--rule-strong)",
          background: "var(--surface)",
          padding: "26px 26px 22px",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <div className="eyebrow">
            {decision.ego} · P{s.position} · lap {s.lap} of {s.total_laps}
          </div>
          <ProvenanceTag kind="counterfactual" />
        </div>

        <div style={{ display: "flex", alignItems: "baseline", gap: 20, flexWrap: "wrap", marginTop: 10 }}>
          <h1 className="verdict">{verb}</h1>
          {!staysOut && <div className="verdict-lap">L{best.lap}</div>}
        </div>

        {!staysOut && (
          <div style={{ marginTop: 4, fontSize: 15, color: "var(--ink-2)" }}>
            Window laps <span className="num">{winLo}</span> to <span className="num">{winHi}</span>
            {winHi > winLo && (
              <>
                {" "}
                <span style={{ color: "var(--ink-3)" }}>({winHi - winLo + 1} laps to play with)</span>
              </>
            )}
            {" · onto "}
            <span style={{ display: "inline-flex", verticalAlign: "middle" }}>
              <CompoundChip compound={best.compound} />
            </span>
          </div>
        )}

        <div className="stat-grid" style={{ marginTop: 22 }}>
          <div className="stat">
            <dt className="eyebrow">Worth</dt>
            <dd>
              <span className="headline-num" style={{ color: gain < 0 ? "var(--gain)" : "var(--loss)" }}>
                {signed(gain)}s
              </span>
            </dd>
          </div>
          <div className="stat">
            <dt className="eyebrow">Confidence</dt>
            <dd>
              <span className="num" style={{ fontSize: 17 }}>
                {(decision.options[0]?.p_faster * 100).toFixed(0)}%
              </span>
              <span style={{ color: "var(--ink-3)", fontSize: 12, marginLeft: 6 }}>
                it is faster
              </span>
            </dd>
          </div>
          <div className="stat">
            <dt className="eyebrow">Each lap you wait</dt>
            <dd>
              <span className="num" style={{ fontSize: 17 }}>
                {waiting === null ? "—" : `${signed(waiting)}s`}
              </span>
              <span style={{ color: "var(--ink-3)", fontSize: 12, marginLeft: 6 }}>past L{winHi}</span>
            </dd>
          </div>
          <div className="stat">
            <dt className="eyebrow">Tyre now</dt>
            <dd>
              <CompoundChip compound={s.compound} age={s.tyre_age} />
            </dd>
          </div>
        </div>

        {!settled && (
          <div style={{ marginTop: 16 }}>
            <Callout>
              <strong>This is close.</strong>{" "}
              {decision.indistinguishable_from_best.length} plans sit within Monte Carlo noise
              of each other, so the ordering above is not a real ranking. Pick on the things
              this model does not see: track position, who you are covering, how the tyre
              actually feels.
            </Callout>
          </div>
        )}
      </section>

      {/* --- the window --- */}
      {decision.window.length > 1 && (
        <section className="panel">
          <div className="panel-head">
            <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
              <h3>The window</h3>
              <ProvenanceTag kind="simulated" />
            </div>
            <div className="eyebrow">{decision.n_sims} draws per lap</div>
          </div>
          <div className="panel-body">
            <PitWindow
              points={decision.window}
              bestLap={best?.lap ?? s.lap}
              windowLaps={laps.length ? laps : [best?.lap ?? s.lap]}
              executedLap={executedLap}
              currentLap={s.lap}
            />
            <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
              <button className="control" aria-pressed={showTable} onClick={() => setShowTable((v) => !v)}>
                {showTable ? "Hide numbers" : "Show numbers"}
              </button>
            </div>
            {showTable && (
              <div style={{ marginTop: 14 }}>
                <WindowTable points={decision.window} bestLap={best?.lap ?? s.lap} />
              </div>
            )}
          </div>
        </section>
      )}

      {/* --- the real choices --- */}
      <section className="panel">
        <div className="panel-head">
          <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
            <h3>What else is on the table</h3>
            <ProvenanceTag kind="counterfactual" />
          </div>
          <div className="eyebrow">
            {decision.candidates_evaluated} plans, {decision.options.length} shapes
          </div>
        </div>
        <div className="panel-body">
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th scope="col">Plan</th>
                  <th scope="col">Box</th>
                  <th scope="col">Worth</th>
                  <th scope="col">Faster</th>
                </tr>
              </thead>
              <tbody>
                {decision.options.map((o) => (
                  <tr key={o.label} style={o.is_baseline ? { background: "var(--surface-sunk)" } : undefined}>
                    <th scope="row">
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 9 }}>
                        <span className="num" style={{ color: "var(--ink-3)", fontSize: 12 }}>
                          {o.n_stops}-stop
                        </span>
                        {o.next_compound !== "none" && <CompoundChip compound={o.next_compound} />}
                        {o.is_baseline && (
                          <span style={{ color: "var(--ink-3)", fontSize: 11.5 }}>as executed</span>
                        )}
                      </span>
                    </th>
                    <td className="num">{o.next_lap ? `L${o.next_lap}` : "—"}</td>
                    <td className="num" style={{ color: o.delta_s < 0 ? "var(--gain)" : undefined }}>
                      {signed(o.delta_s)}
                      <span style={{ color: "var(--ink-3)", fontSize: 11 }}> ±{o.delta_se_s.toFixed(2)}</span>
                    </td>
                    <td className="num">{(o.p_faster * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note" style={{ marginTop: 14, marginBottom: 0 }}>
            {decision.candidates_evaluated} legal plans were simulated and they collapse to
            these {decision.options.length}. The rest are the same shapes a lap or two apart,
            which is what the window above is for.
          </p>
        </div>
      </section>

      {/* --- what breaks it --- */}
      {decision.caution && (
        <section className="panel">
          <div className="panel-head">
            <h3>If a safety car falls in the next ten laps</h3>
            <div className="eyebrow">assumption, not forecast</div>
          </div>
          <div className="panel-body">
            <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
              <span
                className="headline-num"
                style={{ color: Math.abs(decision.caution.option_value_s) < 0.5 ? "var(--ink)" : "var(--loss)" }}
              >
                {signed(decision.caution.option_value_s)}s
              </span>
              <span className="note" style={{ margin: 0 }}>
                {Math.abs(decision.caution.option_value_s) < 0.5
                  ? "The call does not change. A caution over this window moves the plan's advantage by less than half a second."
                  : "A caution over this window materially changes the plan's advantage, so the call is exposed."}
              </span>
            </div>
            <div style={{ marginTop: 16 }}>
              <Callout tone="caution">
                Safety cars are never predicted here (there are 27 caution laps in the whole
                seven-race sample, 0.2%, which is nowhere near enough to fit a rate on). This
                is a contrast under an assumed caution, and it leans on the two thinnest
                constants in the project: the caution lap multiplier rests on a single virtual
                safety car, and the caution pit multiplier is assumed rather than fitted,
                because no green-to-caution stop pairs exist in the data at all. Trust the
                sign more than the size.
              </Callout>
            </div>
          </div>
        </section>
      )}

      {/* --- the machinery, out of the way --- */}
      <section className="panel">
        <div className="panel-head">
          <h3>Behind the number</h3>
          <button className="control" aria-pressed={showWork} onClick={() => setShowWork((v) => !v)}>
            {showWork ? "Hide" : "Show"}
          </button>
        </div>
        {showWork && (
          <div className="panel-body" style={{ display: "grid", gap: 22 }}>
            <div className="stat-grid">
              <div className="stat">
                <dt className="eyebrow">Reference pace</dt>
                <dd className="num">{s.reference_pace_s.toFixed(3)}s</dd>
              </div>
              <div className="stat">
                <dt className="eyebrow">Clean laps seen</dt>
                <dd className="num" style={{ color: s.clean_laps_observed < 5 ? "var(--loss)" : undefined }}>
                  {s.clean_laps_observed}
                </dd>
              </div>
              <div className="stat">
                <dt className="eyebrow">Track trend</dt>
                <dd className="num">{signed(s.race_progress_adjustment_s_per_lap, 3)}s/lap</dd>
              </div>
              <div className="stat">
                <dt className="eyebrow">Compounds used</dt>
                <dd style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  {s.compounds_used.map((c) => (
                    <CompoundChip key={c} compound={c} />
                  ))}
                </dd>
              </div>
            </div>

            {decision.explanation && (
              <div>
                <div className="eyebrow" style={{ marginBottom: 12 }}>
                  Where {planLabel(decision.recommendation)} gets its {signed(gain)}s
                </div>
                <Decomposition explanation={decision.explanation} />
              </div>
            )}

            {!decision.baseline_in_support && (
              <Callout tone="caution">
                The plan actually run keeps a set past anything observed for that compound, so
                the model extrapolates to price it. Read the comparison as indicative.
              </Callout>
            )}

            <p className="note" style={{ margin: 0 }}>
              Nothing here was fitted on this race. The full validation, and the list of what
              this build does not model, is on the <Link href="/method/">method page</Link>.
            </p>
          </div>
        )}
      </section>
    </div>
  );
}
