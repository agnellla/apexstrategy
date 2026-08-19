"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const PAGES = [
  { href: "/", label: "The call" },
  { href: "/race/", label: "Race" },
  { href: "/simulator/", label: "All plans" },
  { href: "/decision/", label: "Why" },
  { href: "/method/", label: "Method" },
];

export function SiteNav() {
  const path = usePathname();
  return (
    <header
      style={{
        borderBottom: "1px solid var(--rule)",
        background: "var(--surface)",
        position: "sticky",
        top: 0,
        zIndex: 20,
      }}
    >
      <div
        className="shell"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 24,
          height: 56,
        }}
      >
        <Link
          href="/"
          style={{
            textDecoration: "none",
            fontFamily: "var(--font-display)",
            fontSize: 19,
            letterSpacing: "-0.02em",
            whiteSpace: "nowrap",
          }}
        >
          ApexStrategy
        </Link>

        <nav style={{ display: "flex", gap: 2 }}>
          {PAGES.map((p) => {
            const active = path === p.href;
            return (
              <Link
                key={p.href}
                href={p.href}
                className="control"
                data-active={active}
                style={{ textDecoration: "none", display: "inline-block" }}
              >
                {p.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
