// Adaptive Knowledge Lifecycle - periodic maintenance sweep (off the hot path).
//
// Two jobs, both no-ops on a backend without lifecycle data (the in-memory
// backend's listLifecycle returns an empty Map, so this returns immediately):
//
//   1. Maturity maintenance. PROMOTION happens eagerly at the reinforcement
//      sites (search-access in hybrid-search.ts and content-update in
//      remember.ts both recompute the tier the moment importance rises).
//      DEMOTION can only be judged against DECAYED importance, which no hot
//      path computes - so this sweep decays each record and persists any tier
//      that should drop. The decayed importance itself is NOT written back;
//      decay is a read-time projection, not a stored mutation, so only the
//      maturity field is updated (which does not reset the recency clock).
//
//   2. GC review. Surface non-core records whose decayed signal has fallen
//      below the floor or gone stale for their tier. Reports to the log only -
//      it never deletes. Deletion stays a separate, human-gated decision.
import type { ISdk } from "iii-sdk";
import { getVectorIndex } from "./search.js";
import { applyDecay, nextMaturity } from "../state/lifecycle-scoring.js";
import { findGcCandidates } from "./lifecycle-gc.js";
import { logger } from "../logger.js";

export function registerLifecycleSweepFunction(sdk: ISdk): void {
  sdk.registerFunction("mem::lifecycle-sweep", async () => {
    const vector = getVectorIndex();
    if (!vector) {
      return {
        success: true,
        skipped: "no vector index",
        maturityChanged: 0,
        gcCandidates: 0,
      };
    }

    const now = Date.now();
    const all = await vector.listLifecycle();
    if (all.size === 0) {
      return { success: true, maturityChanged: 0, gcCandidates: 0 };
    }

    // 1. Maturity maintenance against decayed importance.
    let maturityChanged = 0;
    for (const [obsId, raw] of all) {
      const target = nextMaturity(applyDecay(raw, now));
      if (target !== raw.maturity) {
        try {
          await vector.setLifecycle(obsId, { maturity: target });
          maturityChanged++;
        } catch {
          // best-effort: a single failed tier write must not abort the sweep
        }
      }
    }

    // 2. GC review (read-only). Reuse the `all` map already scanned above so
    // the sweep does ONE listLifecycle() table scan, not two.
    const candidates = await findGcCandidates(vector, now, all);

    logger.info("Lifecycle sweep complete", {
      records: all.size,
      maturityChanged,
      gcCandidates: candidates.length,
      candidates: candidates.map(
        (c) =>
          `${c.obsId} [${c.maturity}] imp=${c.importance.toFixed(1)} age=${c.ageDays.toFixed(0)}d: ${c.reason}`,
      ),
    });

    return { success: true, maturityChanged, gcCandidates: candidates.length };
  });
}
