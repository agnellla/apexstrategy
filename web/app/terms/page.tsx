import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Terms · ApexStrategy",
  description: "Terms of use for the ApexStrategy showcase.",
};

export default function Terms() {
  return (
    <div className="shell" style={{ padding: "56px 24px 0", maxWidth: 760 }}>
      <div className="eyebrow" style={{ marginBottom: 14 }}>Terms of use</div>
      <h2>Terms</h2>
      <p className="note" style={{ marginTop: 18 }}>
        Last updated 19 August 2026.
      </p>

      <h3 style={{ marginTop: 32 }}>What this is</h3>
      <p className="note">
        ApexStrategy is a personal portfolio project demonstrating race-strategy modelling.
        It is not a commercial product, not a betting or wagering service, and not
        affiliated with, endorsed by, or connected to Formula 1, the FIA, Formula One
        Management, or any competing team. All trademarks belong to their owners.
      </p>

      <h3 style={{ marginTop: 32 }}>No warranty</h3>
      <p className="note">
        The site is provided as is, without warranty of any kind. The figures shown are the
        output of a statistical model fitted to a five-race sample, with documented
        limitations set out on the <Link href="/method/">method page</Link>. They are
        illustrative of a modelling approach and are not advice of any kind, sporting,
        financial, or otherwise. Do not rely on them for any decision that matters.
      </p>

      <h3 style={{ marginTop: 32 }}>Data</h3>
      <p className="note">
        Timing data is derived from the public FastF1 archive of the 2024 Formula 1 season
        and is used here for non-commercial research and demonstration. No official timing
        feed is redistributed; what this site serves is the derived output of the models in
        the repository.
      </p>

      <h3 style={{ marginTop: 32 }}>Your use of the site</h3>
      <p className="note">
        You may read, link to, and reference this site freely. The source code is published
        in the linked repository under its own licence, which governs reuse of the code.
        Please do not represent the outputs here as official Formula 1 data or as
        professional strategy analysis.
      </p>

      <h3 style={{ marginTop: 32 }}>Changes</h3>
      <p className="note">
        These terms may change as the project does. The date at the top reflects the last
        revision.
      </p>

      <p className="note" style={{ marginTop: 40 }}>
        <Link href="/">Back to the race overview</Link>
      </p>
    </div>
  );
}
