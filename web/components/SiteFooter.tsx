import Link from "next/link";

export function SiteFooter() {
  return (
    <footer
      style={{
        borderTop: "1px solid var(--rule)",
        marginTop: 72,
        padding: "28px 0 44px",
        background: "var(--surface)",
      }}
    >
      <div
        className="shell"
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 24,
          justifyContent: "space-between",
          alignItems: "flex-start",
        }}
      >
        <p className="note" style={{ maxWidth: "46ch", margin: 0 }}>
          A portfolio project, not a commercial product and not affiliated with Formula 1,
          the FIA, or any team. Timing data is derived from the public FastF1 archive of
          the 2024 season.
        </p>
        <nav style={{ display: "flex", gap: 18, fontSize: 13 }}>
          <Link href="/method/">Method</Link>
          <Link href="/terms/">Terms</Link>
          <Link href="/privacy/">Privacy</Link>
          <a href="https://github.com/agnellla/apexstrategy">Source</a>
        </nav>
      </div>
    </footer>
  );
}
