/**
 * Shapes emitted by `apexstrategy.showcase.export`. Hand-written rather than generated,
 * because the export is a deliberate contract: if the Python side changes what it emits,
 * this file should fail to compile rather than the page quietly rendering undefined.
 */

export type Compound = "SOFT" | "MEDIUM" | "HARD" | "INTERMEDIATE" | "WET" | "UNKNOWN";

export type ProvenanceClass =
  | "observed"
  | "predicted"
  | "simulated"
  | "counterfactual";

export interface PitInstruction {
  lap: number;
  compound: Compound;
}

export interface ScenarioRow {
  ego: string;
  plan: PitInstruction[];
  rival_policy: string;
  label: string;
  rank: number;
  n_stops: number;
  next_stop: string;
  median_s: number;
  mean_s: number;
  /** Mean paired difference against the baseline. Negative is faster. */
  delta_s: number;
  /** Monte Carlo standard error of that difference. Not model error. */
  delta_se_s: number;
  p_faster: number;
  /** Mean of the worst tenth of the paired difference. The risk column. */
  cvar10_delta_s: number;
  regret_s: number;
  cvar10_regret_s: number;
  cvar10_s: number;
  is_baseline: boolean;
}

export interface DecisionState {
  lap: number;
  total_laps: number;
  track_status: string;
  position: number;
  compound: Compound;
  tyre_age: number;
  stops_made: number;
  compounds_used: Compound[];
  gap_to_leader_s: number;
  /** How much clean running the reference pace rests on. Often tiny early. */
  clean_laps_observed: number;
  reference_pace_s: number;
  race_progress_adjustment_s_per_lap: number;
}

export interface Explanation {
  ego: string;
  from_lap: number;
  to_lap: number;
  plan: { label: string; plan: PitInstruction[] };
  baseline: { label: string; plan: PitInstruction[] };
  n_sims: number;
  total_s: number;
  components: {
    tyre_age_s: number;
    compound_s: number;
    pit_lane_s: number;
    sampling_s: number;
  };
  residual_s: number;
  is_settled: boolean;
  not_included: {
    term: string;
    why: string;
    measured_bias_s_per_dirty_air_lap?: number;
    direction?: string;
    available_as?: string;
  }[];
}

export interface Decision {
  ego: string;
  from_lap: number;
  to_lap: number;
  n_sims: number;
  seed: number;
  rival_policy: string;
  caution_laps: number[];
  objective: string;
  baseline: string;
  recommendation: string;
  candidates: ScenarioRow[];
  candidates_evaluated: number;
  candidates_shown: number;
  indistinguishable_from_best: string[];
  state: DecisionState;
  /** False when the executed plan runs a stint past anything observed for its compound,
   *  which makes every delta measured against it an extrapolation. */
  baseline_in_support: boolean;
  explanation: Explanation | null;
  wall_time_s: number;
}

export interface Stint {
  driver: string;
  stint: number;
  compound: Compound;
  start_lap: number;
  end_lap: number;
  laps: number;
  /** Ran to the flag, so its length is a lower bound on the set, not a choice. */
  censored: boolean;
}

export interface RacePayload {
  circuit: string;
  season: number;
  round: number;
  total_laps: number;
  in_training_pool: boolean;
  drivers: {
    code: string;
    team: string;
    grid: number | null;
    finish: number | null;
    classified: string;
    retired: boolean;
    laps_completed: number;
  }[];
  stints: Stint[];
  stops: {
    driver: string;
    lap: number;
    stop: number;
    compound_fitted: Compound | null;
    duration_s: number | null;
    under_caution: boolean;
  }[];
  caution_laps: number[];
  data_quality: {
    total_laps_recorded: number;
    clean_laps: number;
    clean_yield: number;
    exclusions: { reason: string; n_laps: number }[];
  };
  constants: {
    race_progress_s_per_lap?: number;
    pit_loss_median_s?: number;
    pit_loss_n?: number;
  };
}

export interface Method {
  held_out: string;
  in_training_pool: boolean;
  excluded?: boolean;
  reason?: string;
  note: string;
  n_sims?: number;
  objective?: string;
  rival_policy?: string;
  not_modelled?: string[];
  wall_time_s: number;
}

export interface CircuitFile {
  schema: number;
  race: RacePayload;
  decisions: Decision[];
  method: Method;
  provenance: Record<ProvenanceClass, { label: string; meaning: string }>;
}

export interface IndexFile {
  schema: number;
  demo_circuit: string;
  decision_laps: number[];
  races: {
    circuit: string;
    season: number;
    round: number;
    total_laps: number;
    in_training_pool: boolean;
    n_decisions: number;
    clean_yield: number;
    size_kb: number;
  }[];
  provenance: Record<ProvenanceClass, { label: string; meaning: string }>;
}
