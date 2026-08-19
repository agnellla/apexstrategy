import type { Decision } from "@/lib/types";
import { CompoundChip } from "./Primitives";
import { secs } from "@/lib/data";

/**
 * What the pit wall knows at lap L, and nothing it does not.
 *
 * `clean_laps_observed` is here rather than buried because it is the honest measure of how
 * much this state rests on. At lap 10 a driver stuck in traffic may have produced almost
 * no clean running, and every number downstream inherits that.
 */
export function StateStrip({ decision }: { decision: Decision }) {
  const s = decision.state;
  const thin = s.clean_laps_observed < 5;

  const items: [string, React.ReactNode][] = [
    ["Position", <span key="p" className="num">P{s.position}</span>],
    ["Lap", <span key="l" className="num">{s.lap} / {s.total_laps}</span>],
    ["On", <CompoundChip key="c" compound={s.compound} age={s.tyre_age} />],
    [
      "Used",
      <span key="u" style={{ display: "inline-flex", gap: 10, flexWrap: "wrap" }}>
        {s.compounds_used.map((c) => (
          <CompoundChip key={c} compound={c} />
        ))}
      </span>,
    ],
    ["Stops made", <span key="st" className="num">{s.stops_made}</span>],
    [
      "Clean laps seen",
      <span key="cl" className="num" style={{ color: thin ? "var(--loss)" : undefined }}>
        {s.clean_laps_observed}
      </span>,
    ],
    ["Reference pace", <span key="r" className="num">{secs(s.reference_pace_s, 3)}s</span>],
  ];

  return (
    <div>
      <dl
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(126px, 1fr))",
          gap: 0,
          margin: 0,
          border: "1px solid var(--rule)",
        }}
      >
        {items.map(([label, value], i) => (
          <div
            key={label}
            style={{
              padding: "11px 14px",
              borderRight: i < items.length - 1 ? "1px solid var(--rule)" : undefined,
            }}
          >
            <dt className="eyebrow" style={{ marginBottom: 5 }}>
              {label}
            </dt>
            <dd style={{ margin: 0, fontSize: 14 }}>{value}</dd>
          </div>
        ))}
      </dl>
      {thin && (
        <p className="note" style={{ marginTop: 10, marginBottom: 0 }}>
          This car has produced under five clean laps, so its pace estimate is thin and
          every projection built on it is correspondingly loose. The simulator widens its
          intervals for exactly this, and the number is shown rather than smoothed over.
        </p>
      )}
    </div>
  );
}
