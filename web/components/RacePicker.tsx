"use client";

import type { IndexFile } from "@/lib/types";
import { circuitName } from "@/lib/data";

export function RacePicker({
  index,
  value,
  onChange,
}: {
  index: IndexFile;
  value: string;
  onChange: (c: string) => void;
}) {
  return (
    <div className="segmented" role="group" aria-label="Race">
      {index.races.map((r) => (
        <button
          key={r.circuit}
          className="control"
          data-active={r.circuit === value}
          onClick={() => onChange(r.circuit)}
          title={r.in_training_pool ? undefined : "Stored, and excluded from model fitting"}
        >
          {circuitName(r.circuit)}
          {!r.in_training_pool && (
            <span style={{ marginLeft: 6, opacity: 0.6 }} aria-label="excluded from fitting">
              (excl.)
            </span>
          )}
        </button>
      ))}
    </div>
  );
}
