"use client";

import type { CircuitFile, Decision } from "@/lib/types";

export function DecisionPicker({
  file,
  lap,
  driver,
  onLap,
  onDriver,
}: {
  file: CircuitFile;
  lap: number;
  driver: string;
  onLap: (l: number) => void;
  onDriver: (d: string) => void;
}) {
  const laps = [...new Set(file.decisions.map((d) => d.from_lap))].sort((a, b) => a - b);
  const drivers = file.decisions.filter((d) => d.from_lap === lap).map((d) => d.ego);

  return (
    <div style={{ display: "flex", gap: 28, flexWrap: "wrap", alignItems: "flex-end" }}>
      <Field label="Decision lap">
        <div className="segmented" role="group" aria-label="Decision lap">
          {laps.map((l) => (
            <button key={l} className="control num" data-active={l === lap} onClick={() => onLap(l)}>
              {l}
            </button>
          ))}
        </div>
      </Field>
      <Field label="Driver">
        <div className="segmented" role="group" aria-label="Driver">
          {drivers.map((d) => (
            <button key={d} className="control num" data-active={d === driver} onClick={() => onDriver(d)}>
              {d}
            </button>
          ))}
        </div>
      </Field>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="eyebrow" style={{ marginBottom: 7 }}>
        {label}
      </div>
      {children}
    </div>
  );
}

export function pickDecision(file: CircuitFile, lap: number, driver: string): Decision | null {
  return (
    file.decisions.find((d) => d.from_lap === lap && d.ego === driver) ??
    file.decisions.find((d) => d.from_lap === lap) ??
    file.decisions[0] ??
    null
  );
}
