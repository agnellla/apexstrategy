"use client";

import { useEffect, useState } from "react";
import type { CircuitFile, IndexFile } from "./types";

/**
 * The grid is static JSON under /data, fetched at run time rather than bundled.
 *
 * Bundling it would inline about a megabyte into the first paint for a page that shows
 * one race. Fetching means there is a real loading state, which is why the skeletons in
 * this app are load-bearing rather than decorative.
 */

const cache = new Map<string, unknown>();

async function loadJSON<T>(path: string): Promise<T> {
  const hit = cache.get(path);
  if (hit) return hit as T;
  const res = await fetch(path);
  if (!res.ok) throw new Error(`could not load ${path} (${res.status})`);
  const data = (await res.json()) as T;
  cache.set(path, data);
  return data;
}

export interface Async<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

function useJSON<T>(path: string | null): Async<T> {
  const [state, setState] = useState<Async<T>>({
    data: null,
    error: null,
    loading: true,
  });

  useEffect(() => {
    if (!path) return;
    let live = true;
    setState({ data: null, error: null, loading: true });
    loadJSON<T>(path)
      .then((data) => live && setState({ data, error: null, loading: false }))
      .catch((e: Error) => live && setState({ data: null, error: e.message, loading: false }));
    return () => {
      live = false;
    };
  }, [path]);

  return state;
}

const base = () => (process.env.NODE_ENV === "production" ? "" : "");

export const useIndex = () => useJSON<IndexFile>(`${base()}/data/index.json`);
export const useCircuit = (circuit: string | null) =>
  useJSON<CircuitFile>(circuit ? `${base()}/data/${circuit}.json` : null);

export const CIRCUIT_NAMES: Record<string, string> = {
  bahrain: "Bahrain",
  suzuka: "Suzuka",
  monaco: "Monaco",
  barcelona: "Barcelona",
  austria: "Red Bull Ring",
  britain: "Silverstone",
  hungary: "Hungaroring",
};

export const circuitName = (key: string) => CIRCUIT_NAMES[key] ?? key;

/** Signed seconds, always with an explicit sign so a gain never reads as a level. */
export const signed = (v: number, digits = 2) =>
  `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;

export const secs = (v: number, digits = 2) => v.toFixed(digits);

export const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;

/** The engine writes plans as "L22 HARD -> L45 HARD". The arrow is a real arrow on screen. */
export const planLabel = (label: string) => label.replace(/\s*->\s*/g, " \u2192 ");

/** Laps whose best plan is within `n` Monte Carlo standard errors of the best overall.
 *
 * The window is not "the best lap plus or minus two". It is every lap the simulation
 * cannot tell apart from the best one, which is a claim about what was measured rather
 * than a rule of thumb, and it is usually asymmetric: boxing early costs little and
 * boxing late falls off a cliff. */
export function windowLaps(points: { lap: number; delta_s: number }[], tolerance = 0.25) {
  if (!points.length) return [];
  const best = Math.min(...points.map((p) => p.delta_s));
  return points.filter((p) => p.delta_s - best <= tolerance).map((p) => p.lap);
}

/** What one more lap of waiting costs, measured at the edge of the window. */
export function costOfWaiting(points: { lap: number; delta_s: number }[], fromLap: number) {
  const here = points.find((p) => p.lap === fromLap);
  const next = points.find((p) => p.lap === fromLap + 1);
  if (!here || !next) return null;
  return next.delta_s - here.delta_s;
}
