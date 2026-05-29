// Adaptive Knowledge Lifecycle - GC candidate finder.
//
// Surfaces stored observations whose lifecycle signal has decayed to the
// point they are candidates for review/eviction. This module ONLY reads and
// reports - it never deletes. Deletion (if any) is a separate, gated step so
// a human or higher-level policy decides. core-tier knowledge is never
// flagged regardless of decay.
//
// Degrades gracefully: with no vector backend, or a backend that does not
// track lifecycle (listLifecycle returns an empty Map), the result is [].

import type { VectorIndex } from "../state/vector-index.js";
import { applyDecay } from "../state/lifecycle-scoring.js";

const MS_PER_DAY = 86_400_000;

// Decayed-importance floor below which a non-core record is GC-eligible.
const IMPORTANCE_FLOOR = 35;

// Staleness ceilings per tier (days since last update). draft knowledge is
// reaped sooner than validated; core is never reaped here.
const STALE_DAYS = {
  draft: 60,
  validated: 120,
} as const;

export interface GcCandidate {
  obsId: string;
  reason: string;
  importance: number; // decayed importance at evaluation time
  maturity: string;
  ageDays: number; // days since updatedAt
}

// Walk every lifecycle record, decay it to `nowMs`, and collect non-core
// records that are either low-importance or stale-for-their-tier. Returns
// the candidates with a human-readable reason; does not mutate anything.
export async function findGcCandidates(
  vector: VectorIndex | null,
  nowMs: number,
): Promise<GcCandidate[]> {
  if (!vector) return [];

  const all = await vector.listLifecycle();
  if (all.size === 0) return [];

  const candidates: GcCandidate[] = [];

  for (const [obsId, raw] of all) {
    // core is protected - never a GC candidate.
    if (raw.maturity === "core") continue;

    const decayed = applyDecay(raw, nowMs);
    const ageDays = Math.max(0, (nowMs - raw.updatedAt) / MS_PER_DAY);

    const lowImportance = decayed.importance < IMPORTANCE_FLOOR;

    const staleCeiling =
      raw.maturity === "draft"
        ? STALE_DAYS.draft
        : raw.maturity === "validated"
          ? STALE_DAYS.validated
          : Infinity; // unknown tiers: only low-importance can flag them
    const stale = ageDays > staleCeiling;

    if (!lowImportance && !stale) continue;

    const reasons: string[] = [];
    if (lowImportance) {
      reasons.push(
        `decayed importance ${decayed.importance.toFixed(1)} < ${IMPORTANCE_FLOOR}`,
      );
    }
    if (stale) {
      reasons.push(
        `stale for tier "${raw.maturity}": ${ageDays.toFixed(0)}d > ${staleCeiling}d`,
      );
    }

    candidates.push({
      obsId,
      reason: reasons.join("; "),
      importance: decayed.importance,
      maturity: raw.maturity,
      ageDays,
    });
  }

  return candidates;
}
