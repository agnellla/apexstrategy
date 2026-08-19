# ApexStrategy showcase

The Next.js half of D9's two front ends. It carries the UI/UX story; the internal
diagnostics live in the Python CLIs (`replay`, `calibration`, `strategy --explain`), which
is why no Streamlit app was built.

## It has no backend, on purpose

Everything under `public/data/` is precomputed by the Python side:

```bash
uv run python -m apexstrategy.showcase            # all seven races, ~5 min
uv run python -m apexstrategy.showcase --circuits hungary --laps 20
```

That produces one JSON file per race plus an index, about 800 KB in total, and the app is
a static export (`output: "export"`). No cold starts, no hosting cost, and every figure on
screen was produced by the same code the Phase 7 replay gate ran against. The trade-off is
real and stated: a visitor explores grid points, not arbitrary states.

Re-run the export whenever the engine changes. `lib/types.ts` is hand-written against the
export's shape so that a contract change fails the type check rather than rendering
`undefined` into a table.

## Local

```bash
npm install
npm run dev          # http://localhost:3000
npm run build        # static export to out/
npm run typecheck
```

## Deploying

Point Vercel at this directory as the project root. The framework preset is Next.js and
needs no further configuration; `output: "export"` produces static files that Vercel serves
directly. `public/data/` is committed, so a deploy does not need Python or the Parquet
store.

## Design notes

Register is an engineering report rather than a product page: paper and ink, hairline
rules, square corners, tabular figures, no shadows and no motion.

Chart colour is validated rather than chosen. The three compound hues clear the lightness
band, chroma floor, all-pairs CVD separation and normal-vision floor against both the light
and dark surfaces. The light amber lands at 2.7:1 against paper, below the 3:1 line, so it
ships under the relief rule: every compound mark carries a visible text label and every
chart has a table view.

Provenance colouring is a hard requirement from the architecture doc and is applied on
every view. Class is carried by treatment first (solid, band, dash, fan) and hue second, so
a projection cannot be mistaken for a measurement.
