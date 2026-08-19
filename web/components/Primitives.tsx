import type { Compound, ProvenanceClass } from "@/lib/types";

/** Compound colour is validated; the text label is not decoration.
 *  The light-mode amber sits below 3:1 against paper, so the relief rule applies and
 *  every compound mark ships with its name visible. */
export const COMPOUND_VAR: Record<string, string> = {
  SOFT: "var(--soft)",
  MEDIUM: "var(--medium)",
  HARD: "var(--hard)",
  INTERMEDIATE: "var(--neutral)",
  WET: "var(--neutral)",
  UNKNOWN: "var(--neutral)",
};

export const compoundColor = (c: string) => COMPOUND_VAR[c] ?? "var(--neutral)";

export function CompoundChip({ compound, age }: { compound: Compound; age?: number }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        fontSize: 12,
        whiteSpace: "nowrap",
      }}
    >
      <span
        aria-hidden
        style={{
          width: 9,
          height: 9,
          background: compoundColor(compound),
          border: "1px solid var(--rule-strong)",
          flex: "none",
        }}
      />
      <span style={{ letterSpacing: "0.04em" }}>{compound}</span>
      {age !== undefined && (
        <span className="num" style={{ color: "var(--ink-3)" }}>
          {age}L
        </span>
      )}
    </span>
  );
}

export function Eyebrow({ children }: { children: React.ReactNode }) {
  return <div className="eyebrow">{children}</div>;
}

export function Panel({
  title,
  meta,
  provenance,
  children,
}: {
  title: string;
  meta?: React.ReactNode;
  provenance?: ProvenanceClass;
  children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
          <h3>{title}</h3>
          {provenance && <ProvenanceTag kind={provenance} />}
        </div>
        {meta && <div className="eyebrow">{meta}</div>}
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}

const PROVENANCE_STYLE: Record<ProvenanceClass, { label: string; color: string; dash: string }> = {
  observed: { label: "Observed", color: "var(--observed)", dash: "" },
  predicted: { label: "Predicted", color: "var(--predicted)", dash: "" },
  simulated: { label: "Simulated", color: "var(--simulated)", dash: "4 3" },
  counterfactual: { label: "Counterfactual", color: "var(--counterfactual)", dash: "2 3" },
};

/** Provenance is carried by a drawn sample of the mark itself, not by a colour swatch:
 *  a dashed line means the same thing here as it does inside every chart on the site. */
export function ProvenanceTag({ kind }: { kind: ProvenanceClass }) {
  const s = PROVENANCE_STYLE[kind];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 7,
        fontFamily: "var(--font-mono)",
        fontSize: 10.5,
        letterSpacing: "0.1em",
        textTransform: "uppercase",
        color: "var(--ink-2)",
        border: "1px solid var(--rule)",
        padding: "2px 7px",
      }}
    >
      <svg width="20" height="8" aria-hidden focusable="false">
        {kind === "predicted" && (
          <rect x="0" y="1" width="20" height="6" fill={s.color} opacity="0.16" />
        )}
        {kind === "simulated" && (
          <path d="M0 6 L20 2 L20 6 Z" fill={s.color} opacity="0.16" />
        )}
        <line
          x1="0"
          y1="4"
          x2="20"
          y2="4"
          stroke={s.color}
          strokeWidth="2"
          strokeDasharray={s.dash || undefined}
        />
      </svg>
      {s.label}
    </span>
  );
}

export function ProvenanceKey({
  classes,
  descriptions,
}: {
  classes: ProvenanceClass[];
  descriptions: Record<string, { label: string; meaning: string }>;
}) {
  return (
    <dl
      style={{
        display: "grid",
        gridTemplateColumns: "auto 1fr",
        gap: "8px 16px",
        margin: 0,
        alignItems: "baseline",
      }}
    >
      {classes.map((c) => (
        <div key={c} style={{ display: "contents" }}>
          <dt>
            <ProvenanceTag kind={c} />
          </dt>
          <dd style={{ margin: 0, fontSize: 13, color: "var(--ink-2)" }}>
            {descriptions[c]?.meaning}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function Skeleton({ height = 220, label }: { height?: number; label?: string }) {
  return (
    <div
      className="skeleton"
      role="status"
      aria-live="polite"
      aria-busy="true"
      style={{ height, display: "grid", placeItems: "center" }}
    >
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "var(--ink-3)",
        }}
      >
        {label ?? "Loading"}
      </span>
    </div>
  );
}

export function SkeletonRows({ rows = 6 }: { rows?: number }) {
  return (
    <div role="status" aria-busy="true" aria-live="polite">
      {Array.from({ length: rows }, (_, i) => (
        <div
          key={i}
          className="skeleton-line"
          style={{ height: 33, opacity: 1 - i * 0.1 }}
        />
      ))}
    </div>
  );
}

export function Callout({
  tone = "note",
  children,
}: {
  tone?: "note" | "caution";
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        border: "1px solid var(--rule-strong)",
        borderTop: `2px solid ${tone === "caution" ? "var(--loss)" : "var(--rule-strong)"}`,
        background: "var(--surface-sunk)",
        padding: "12px 14px",
        fontSize: 13,
        lineHeight: 1.55,
        color: "var(--ink-2)",
      }}
    >
      {children}
    </div>
  );
}
