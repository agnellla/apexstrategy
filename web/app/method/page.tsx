import type { Metadata } from "next";
import Link from "next/link";
import { Panel } from "@/components/Primitives";

export const metadata: Metadata = {
  title: "Method · ApexStrategy",
  description:
    "How the simulator was validated, what it does not model, and which numbers should " +
    "not be trusted.",
};

export default function Method() {
  return (
    <div className="shell" style={{ padding: "56px 24px 0", display: "grid", gap: 28 }}>
      <header>
        <div className="eyebrow" style={{ marginBottom: 14 }}>Method</div>
        <h2 style={{ maxWidth: "24ch" }}>What was validated, and what was not</h2>
        <p className="note" style={{ marginTop: 16 }}>
          A strategy tool is only worth the gate it passed. This page is the gate, the
          numbers it produced, and the parts of the problem the build deliberately does not
          touch. Nothing here is softened for the demo.
        </p>
        <p className="note">
          <strong>The one to read first:</strong> there is no overtaking model, so nothing
          on this site knows about track position. It will tell you a stop is worth a second
          and it cannot tell you whether you come out ahead of the car you are racing. On a
          circuit where the answer is decided by who you rejoin behind, that gap is the
          whole question, and the tool is silent on it by construction rather than by
          oversight.
        </p>
      </header>

      <Panel title="The gate" meta="Phase 7">
        <p className="note" style={{ marginTop: 0 }}>
          Before the simulator was allowed to recommend anything, it had to reproduce races
          it had been handed the real strategies for. If it cannot rebuild a race whose
          answers it already has, no counterfactual downstream of it is worth reading.
        </p>
        <table>
          <thead>
            <tr>
              <th scope="col">Decision lap</th>
              <th scope="col">Clean-lap error</th>
              <th scope="col">Race-time error</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["5", "1.21 s", "0.95%"],
              ["15", "0.89 s", "0.52%"],
              ["25", "0.70 s", "0.42%"],
              ["35", "0.76 s", "0.52%"],
            ].map(([lap, mae, pct]) => (
              <tr key={lap}>
                <th scope="row" className="mono">
                  Lap {lap}
                </th>
                <td className="num">{mae}</td>
                <td className="num">{pct}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note" style={{ marginBottom: 0, marginTop: 14 }}>
          18 of 20 replays land inside 1% of the projected window, and 14 of 15 from
          decision lap 15 onward. The two failures are both at Suzuka, a circuit flagged as
          the weak one at every earlier stage. They are reported, not excluded.
        </p>
      </Panel>

      <Panel title="Nothing is fitted on the race it is shown against" meta="leave-one-race-out">
        <p className="note" style={{ margin: 0 }}>
          The pace model, the pit-loss distribution, the noise components and the
          compound stint caps are all refitted per race with that race held out. Circuit
          constants leak exactly like model weights, so they are fitted inside the fold
          rather than once up front. With one race per circuit there is no honest random
          split available, and none is used.
        </p>
      </Panel>

      <Panel title="How confident the intervals are" meta="Phase 8">
        <p className="note" style={{ marginTop: 0 }}>
          Uncertainty is sampled at the level each effect lives at: how the track trends
          today, how well a car's pace is pinned down, whether this set of tyres is a good
          one, how a single lap goes. Independent per-lap noise averages out over a stint
          and produces absurd confidence, which is what the first version did.
        </p>
        <table>
          <thead>
            <tr>
              <th scope="col">Interval</th>
              <th scope="col">Should cover</th>
              <th scope="col">Actually covers</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["50%", "0.50", "0.56"],
              ["80%", "0.80", "0.81"],
              ["95%", "0.95", "0.90"],
            ].map(([lvl, want, got]) => (
              <tr key={lvl}>
                <th scope="row" className="mono">
                  {lvl}
                </th>
                <td className="num">{want}</td>
                <td className="num">{got}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note" style={{ marginBottom: 0, marginTop: 14 }}>
          <strong>The headline 0.81 is partly two errors cancelling.</strong> Excluding
          Suzuka, coverage runs 0.65 / 0.91 / 0.96: four circuits are still too wide, and
          Suzuka's point-estimate bias pulls the pooled figure back onto nominal. Narrowing
          the intervals would fix four circuits and break the fifth, on a sample of five
          races, so it is reported rather than tuned away.
        </p>
      </Panel>

      <Panel title="What this build does not model">
        <dl style={{ display: "grid", gap: 18, margin: 0 }}>
          <Item term="Overtaking and track position">
            Cars pass freely on cumulative time. There is no wheel-to-wheel model, so the
            tool answers how much time a plan gains or loses, never what position it
            finishes in, and never how many championship points it is worth. An expected
            finishing position is an average over an ordinal scale and would be a made-up
            number here.
          </Item>
          <Item term="Traffic on rejoin">
            Projections are clean-air times. The replay measures the cost of this at about
            0.07 s per lap spent in traffic, roughly 11% of total replay error, and it is
            one-sided: a plan that rejoins behind someone is flattered.
          </Item>
          <Item term="Safety cars">
            Never predicted. Across all seven races the store holds 27 caution laps, 0.2% of
            the total, which is nowhere near enough to calibrate a rate on. Cautions are an
            assumption supplied to the simulator, and the grid here assumes the ones that
            actually happened.
          </Item>
          <Item term="Weather changes">
            Dry running only. One race in the set went to intermediates and is stored,
            excluded from fitting, and shown without a recommendation.
          </Item>
          <Item term="The counterfactual itself">
            This is the limit that does not go away with more data. A strategy nobody ran
            has no outcome to check against. Reproduction was validated and spread was
            validated; the recommendation rests on those two holding for plans that were
            never executed.
          </Item>
        </dl>
      </Panel>

      <Panel title="Scope">
        <p className="note" style={{ margin: 0 }}>
          Seven races from the 2024 season, of which five are used for fitting. Monaco is
          stored and excluded as a pathological outlier, Silverstone as a mixed-conditions
          race. Five circuits is a small sample and every conclusion on this site should be
          read against that. Full build log and reasoning in the{" "}
          <a href="https://github.com/agnellla/apexstrategy">repository</a>.
        </p>
      </Panel>

      <p className="note">
        <Link href="/">Back to the race overview</Link>
      </p>
    </div>
  );
}

function Item({ term, children }: { term: string; children: React.ReactNode }) {
  return (
    <div>
      <dt style={{ fontWeight: 600, fontSize: 14 }}>{term}</dt>
      <dd className="note" style={{ margin: "4px 0 0" }}>
        {children}
      </dd>
    </div>
  );
}
