import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Newsreader } from "next/font/google";
import "./globals.css";
import { SiteNav } from "@/components/SiteNav";
import { SiteFooter } from "@/components/SiteFooter";

/*
  Type: a serif with real optical sizing for display, a grotesque with flared terminals
  for running text, and a mono with tabular figures for every number on the site. Chosen
  against the house defaults on purpose; a timing number that shifts width as it ticks is
  the tell that nobody looked at the numerals.
*/

const newsreader = Newsreader({
  subsets: ["latin"],
  variable: "--font-newsreader",
  display: "swap",
  axes: ["opsz"],
});

const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-plex-sans",
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "ApexStrategy",
  description:
    "Formula 1 race strategy under uncertainty: a validated race simulator, ranked pit " +
    "plans, and an explanation that is arithmetic rather than attribution.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // The font variables go on <html>, not <body>: globals.css composes them into
    // --font-ui and friends at :root, and a custom property cannot reference one that is
    // only defined on a descendant. Put them on <body> and every declaration silently
    // resolves to nothing, which renders the whole site in Times.
    <html lang="en" className={`${newsreader.variable} ${plexSans.variable} ${plexMono.variable}`}>
      <body>
        <SiteNav />
        <main>{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
