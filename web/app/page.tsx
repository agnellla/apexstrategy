"use client";

import { useState } from "react";
import Link from "next/link";
import { useCircuit, useIndex, circuitName } from "@/lib/data";
import { Callout, Skeleton } from "@/components/Primitives";
import { RacePicker } from "@/components/RacePicker";
import { DecisionPicker, pickDecision } from "@/components/DecisionPicker";
import { TheCall } from "@/components/TheCall";

export default function Call() {
  const index = useIndex();
  const [circuit, setCircuit] = useState("hungary");
  const [lap, setLap] = useState(20);
  const [driver, setDriver] = useState("VER");
  const file = useCircuit(circuit);
  const decision = file.data ? pickDecision(file.data, lap, driver) : null;

  return (
    <div className="shell" style={{ padding: "34px 24px 0", display: "grid", gap: 24 }}>
      <div style={{ display: "grid", gap: 16 }}>
        {index.loading && <Skeleton height={38} label="Loading races" />}
        {index.data && <RacePicker index={index.data} value={circuit} onChange={setCircuit} />}
        {file.data && file.data.decisions.length > 0 && decision && (
          <DecisionPicker
            file={file.data}
            lap={lap}
            driver={decision.ego}
            onLap={setLap}
            onDriver={setDriver}
          />
        )}
      </div>

      {file.loading && <Skeleton height={420} label={`Loading ${circuitName(circuit)}`} />}
      {file.error && <Callout tone="caution">Could not load this race: {file.error}</Callout>}

      {file.data && file.data.decisions.length === 0 && (
        <Callout tone="caution">
          <strong>No call is offered for this race.</strong> {file.data.method.reason} The
          race itself is on the <Link href="/race/">race page</Link>.
        </Callout>
      )}

      {decision && <TheCall decision={decision} />}

      {decision && (
        <p className="note" style={{ marginTop: 4 }}>
          Ranked on expected race time. Track position, undercut and overcut are not
          modelled, so this never tells you where you finish, only how much time a plan is
          worth. <Link href="/method/">What else it does not model</Link>.
        </p>
      )}
    </div>
  );
}
