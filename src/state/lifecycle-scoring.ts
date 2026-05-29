// Adaptive Knowledge Lifecycle - pure scoring functions (no I/O).
//
// Clean-room implementation of a FinMem-style importance/recency/maturity
// scoring model for ranking and garbage-collecting stored observations.
// Every function here is pure: it returns new values and never reads or
// writes the store. I/O (getLifecycle/setLifecycle/listLifecycle) lives in
// VectorIndex (src/state/vector-index.ts); callers wire decay + reinforce
// around those async methods. See lifecycle-gc.ts and hybrid-search.ts.
//
// LifecycleFields shape (from vector-index.ts):
//   importance  0-100   how valuable the observation is
//   recency     0-1     exp-decayed freshness since last update
//   accessCount int     number of read accesses
//   updateCount int     number of content updates
//   maturity    string  "draft" | "validated" | "core"
//   createdAt   epoch ms
//   updatedAt   epoch ms

import type { LifecycleFields } from "./vector-index.js";

// Tuning constants. Exported so tests and the GC can reference the exact
// values rather than re-deriving them.
export const DECAY = {
  // Recency uses an exponential half-life: recency = exp(-days / TAU).
  // RECENCY_TAU = 30 means recency falls to ~0.37 after 30 days and
  // ~0.13 after 60 days. (Spec: exp(-days/30).)
  RECENCY_TAU: 30,
  // Importance erodes geometrically: importance *= IMPORTANCE_DECAY^days.
  // 0.995^day loses ~0.5%/day, ~14% over 30 days, ~26% over 60 days.
  IMPORTANCE_DECAY: 0.995,
  // Reinforcement increments.
  ACCESS_BOOST: 3, // +importance per read access
  UPDATE_BOOST: 5, // +importance per content update
  IMPORTANCE_MAX: 100,
  IMPORTANCE_MIN: 0,
} as const;

export const MS_PER_DAY = 86_400_000;

// Maturity tiers as a discriminated set of string literals. Stored as a
// plain string in LifecycleFields, but used here for clarity.
export type Maturity = "draft" | "validated" | "core";

// Hysteresis thresholds for maturity transitions. The promote/demote gaps
// are deliberately non-adjacent so an observation hovering near a boundary
// does not oscillate between tiers on tiny importance jitter.
export const MATURITY = {
  PROMOTE_DRAFT_TO_VALIDATED: 65, // draft -> validated when importance >= 65
  PROMOTE_VALIDATED_TO_CORE: 85, // validated -> core when importance >= 85
  DEMOTE_CORE_TO_VALIDATED: 60, // core -> validated when importance < 60
  DEMOTE_VALIDATED_TO_DRAFT: 35, // validated -> draft when importance < 35
} as const;

// Compound-score blend weights (must sum to 1.0 for the additive base).
export const SCORE_WEIGHTS = {
  RRF: 0.6,
  IMPORTANCE: 0.2,
  RECENCY: 0.2,
} as const;

function clampImportance(v: number): number {
  if (v < DECAY.IMPORTANCE_MIN) return DECAY.IMPORTANCE_MIN;
  if (v > DECAY.IMPORTANCE_MAX) return DECAY.IMPORTANCE_MAX;
  return v;
}

// Fresh lifecycle record for a newly created observation.
export function defaultLifecycle(nowMs: number): LifecycleFields {
  return {
    importance: 50,
    recency: 1,
    accessCount: 0,
    updateCount: 0,
    maturity: "draft",
    createdAt: nowMs,
    updatedAt: nowMs,
  };
}

// Time-decay an existing record relative to its updatedAt. Pure: returns a
// new object and deliberately does NOT advance updatedAt (decay reflects
// elapsed time, it is not itself an update event). recency is recomputed
// from scratch each call so repeated decays do not compound; importance
// erodes geometrically and is clamped to >= 0.
export function applyDecay(f: LifecycleFields, nowMs: number): LifecycleFields {
  const days = Math.max(0, (nowMs - f.updatedAt) / MS_PER_DAY);
  const recency = Math.exp(-days / DECAY.RECENCY_TAU);
  const importance = clampImportance(
    f.importance * Math.pow(DECAY.IMPORTANCE_DECAY, days),
  );
  return {
    ...f,
    importance,
    recency,
  };
}

// Reinforce on a read access: bump importance, increment accessCount, reset
// recency to full, and stamp updatedAt to now (so subsequent decay measures
// from this access). Pure - returns a new object.
export function reinforceOnAccess(
  f: LifecycleFields,
  nowMs: number,
): LifecycleFields {
  return {
    ...f,
    importance: clampImportance(f.importance + DECAY.ACCESS_BOOST),
    accessCount: f.accessCount + 1,
    recency: 1,
    updatedAt: nowMs,
  };
}

// Reinforce on a content update: stronger importance bump than a read,
// increment updateCount, reset recency, stamp updatedAt. Pure.
export function reinforceOnUpdate(
  f: LifecycleFields,
  nowMs: number,
): LifecycleFields {
  return {
    ...f,
    importance: clampImportance(f.importance + DECAY.UPDATE_BOOST),
    updateCount: f.updateCount + 1,
    recency: 1,
    updatedAt: nowMs,
  };
}

// Multiplicative ranking boost by maturity tier. core knowledge is favored,
// draft is mildly penalized. Unknown/legacy values fall back to neutral.
export function tierBoost(maturity: string): number {
  switch (maturity) {
    case "core":
      return 1.15;
    case "validated":
      return 1.0;
    case "draft":
      return 0.85;
    default:
      return 1.0;
  }
}

// Decide the next maturity tier from current importance, with hysteresis.
// Promotion uses high thresholds; demotion uses lower ones; the gap between
// them is the dead band that prevents tier flapping. Anything outside the
// known tiers is returned unchanged. Caller should pass a record whose
// importance already reflects any decay/reinforcement it wants considered.
export function nextMaturity(f: LifecycleFields): string {
  const imp = f.importance;
  switch (f.maturity) {
    case "draft":
      return imp >= MATURITY.PROMOTE_DRAFT_TO_VALIDATED ? "validated" : "draft";
    case "validated":
      if (imp >= MATURITY.PROMOTE_VALIDATED_TO_CORE) return "core";
      if (imp < MATURITY.DEMOTE_VALIDATED_TO_DRAFT) return "draft";
      return "validated";
    case "core":
      return imp < MATURITY.DEMOTE_CORE_TO_VALIDATED ? "validated" : "core";
    default:
      return f.maturity;
  }
}

// Blend a normalized RRF score with lifecycle signal into a final ranking
// score. When no lifecycle record exists (undefined), return the RRF score
// untouched so backends without lifecycle data (e.g. the in-memory backend)
// rank exactly as before. Otherwise: additive base of normalized RRF +
// importance + recency, then scaled by the maturity tier boost.
//
// NOTE: this function does NOT decay `f` itself. Callers must pass an
// already-decayed record (e.g. applyDecay(f, now)) so importance/recency
// reflect elapsed time at scoring moment.
export function compoundScore(
  rrfNorm: number,
  f: LifecycleFields | undefined,
): number {
  if (!f) return rrfNorm;
  const base =
    SCORE_WEIGHTS.RRF * rrfNorm +
    SCORE_WEIGHTS.IMPORTANCE * (f.importance / 100) +
    SCORE_WEIGHTS.RECENCY * f.recency;
  return base * tierBoost(f.maturity);
}
