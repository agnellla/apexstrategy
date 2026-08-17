"""Simulation layer: race state, scenarios, and the Monte Carlo engine.

This is the boundary between the offline half of the system (data, fitted models) and
the online half (projections, strategy). Everything upstream of ``RaceState`` is batch;
everything downstream must be fast enough to sit behind a UI.

``RaceState`` is where the temporal firewall lives: it is constructed from a frame
truncated at lap L, so it *cannot* contain information from later laps. That is
structural prevention, not vigilance (architecture-proposal.md, Part 13).
"""
