import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Privacy · ApexStrategy",
  description: "What this site collects, which is nothing.",
};

export default function Privacy() {
  return (
    <div className="shell" style={{ padding: "56px 24px 0", maxWidth: 760 }}>
      <div className="eyebrow" style={{ marginBottom: 14 }}>Privacy</div>
      <h2>Privacy</h2>
      <p className="note" style={{ marginTop: 18 }}>
        Last updated 19 August 2026.
      </p>

      <h3 style={{ marginTop: 32 }}>What is collected</h3>
      <p className="note">
        Nothing. This site is a set of static files and precomputed JSON. There is no
        account system, no login, no contact form, no comment system, and no database. It
        sets no cookies and uses no local storage. There is no analytics script, no tag
        manager, no session recording, and no advertising or tracking pixel of any kind.
      </p>

      <h3 style={{ marginTop: 32 }}>Third parties</h3>
      <p className="note">
        The site is hosted on Vercel, which as the hosting provider processes standard
        server request logs, including IP address and user agent, for delivery and abuse
        prevention. That processing is governed by Vercel&rsquo;s own privacy policy and is
        outside this project&rsquo;s control. Typefaces are self-hosted and served from the
        same origin, so viewing a page makes no request to a font provider or any other
        third party.
      </p>

      <h3 style={{ marginTop: 32 }}>Data on the page</h3>
      <p className="note">
        The Formula 1 timing data shown is public historical sporting data about
        competitors in a professional series. It contains no information about you.
        Selections you make, such as race, decision lap, and driver, are held in the
        browser tab for as long as it is open and are never transmitted anywhere.
      </p>

      <h3 style={{ marginTop: 32 }}>Contact</h3>
      <p className="note">
        Questions about this project can be raised as an issue in the{" "}
        <a href="https://github.com/agnellla/apexstrategy">source repository</a>.
      </p>

      <p className="note" style={{ marginTop: 40 }}>
        <Link href="/">Back to the race overview</Link>
      </p>
    </div>
  );
}
