"""Precomputed scenario JSON for the static showcase front end.

D9 splits the front end in two and gives this half the UI/UX story: a Next.js app on
Vercel, fed by a discretised grid computed offline. There is no backend and no cold
start, and the cost of that is the only real trade-off: a visitor can explore grid points
rather than arbitrary states. In a showcase that limit is invisible, and in exchange every
number on the page was produced by the same code the replay gate ran against.

Nothing in here models anything. It calls the engine, walks a grid, and serialises.
"""
