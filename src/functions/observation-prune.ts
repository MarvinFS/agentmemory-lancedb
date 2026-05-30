import type { ISdk } from "iii-sdk";
import type {
  Session,
  CompressedObservation,
  ObservationType,
} from "../types.js";
import { KV } from "../state/schema.js";
import { StateKV } from "../state/kv.js";
import { flushIndexSave, vectorIndexOptimize } from "./search.js";
import { deleteObservationCascade } from "./_observation-cascade.js";
import { BREADCRUMB_TYPES } from "./compress-synthetic.js";
import { logger } from "../logger.js";

const MS_PER_DAY = 24 * 60 * 60 * 1000;

// Default prune target: every breadcrumb type EXCEPT `error`. Errors are the
// one breadcrumb worth keeping by default (postmortem value), so an explicit
// `types: ["error"]` is required to remove them. This is the backlog-clear +
// scheduled-prune mechanism that complements the importance/decay fix: it
// deletes the breadcrumbs the lifecycle GC has not reached yet, with the full
// search/vector/image-ref cascade so no orphan index entries survive.
const DEFAULT_PRUNE_TYPES: ObservationType[] = [...BREADCRUMB_TYPES].filter(
  (t) => t !== "error",
);

interface ObservationPruneInput {
  types?: ObservationType[];
  olderThanDays?: number;
  dryRun?: boolean;
}

interface ObservationPruneResult {
  dryRun: boolean;
  wouldDelete?: number;
  deleted?: number;
  byType: Record<string, number>;
}

export function registerObservationPruneFunction(
  sdk: ISdk,
  kv: StateKV,
): void {
  sdk.registerFunction(
    "mem::observations-prune",
    async (data: ObservationPruneInput): Promise<ObservationPruneResult> => {
      const dryRun = data?.dryRun ?? false;

      const typeSet = new Set<ObservationType>(
        data?.types && data.types.length > 0 ? data.types : DEFAULT_PRUNE_TYPES,
      );
      const olderThanMs =
        typeof data?.olderThanDays === "number" && data.olderThanDays > 0
          ? data.olderThanDays * MS_PER_DAY
          : undefined;

      const now = Date.now();
      const byType: Record<string, number> = {};
      let count = 0;

      const sessions = await kv.list<Session>(KV.sessions).catch(() => []);
      for (const session of sessions) {
        const observations = await kv
          .list<CompressedObservation>(KV.observations(session.id))
          .catch(() => []);

        for (const o of observations) {
          // Only compressed breadcrumbs carry a type; skip anything else.
          if (!o.type || !typeSet.has(o.type)) continue;
          if (olderThanMs !== undefined) {
            if (!o.timestamp) continue;
            const age = now - new Date(o.timestamp).getTime();
            // An unparseable timestamp yields NaN; treat it as "keep" rather
            // than deleting it, so a corrupt timestamp never causes an
            // age-bounded prune to remove a record it cannot date.
            if (Number.isNaN(age) || age <= olderThanMs) continue;
          }

          if (dryRun) {
            byType[o.type] = (byType[o.type] ?? 0) + 1;
            count++;
            continue;
          }

          // Delete + full cascade (search/vector/image-ref/audit) under one
          // guard: a failure on any single observation logs and skips that one
          // record instead of aborting the whole sweep (which would also skip
          // the final flushIndexSave and leave the remaining sessions
          // unprocessed). count/byType only advance on a fully-applied delete.
          try {
            await kv.delete(KV.observations(session.id), o.id);
            await deleteObservationCascade(kv, sdk, {
              obsId: o.id,
              imageData: o.imageData,
              imageRef: o.imageRef,
              removeFromIndex: true,
              audit: {
                operation: "delete",
                functionId: "mem::observations-prune",
                details: {
                  resource: "observation",
                  reason: "observation_prune",
                  type: o.type,
                  sessionId: session.id,
                  dryRun,
                },
              },
            });
          } catch (err) {
            logger.warn("Observation prune delete failed", {
              resource: "observation",
              id: o.id,
              sessionId: session.id,
              error: err instanceof Error ? err.message : String(err),
            });
            continue;
          }
          byType[o.type] = (byType[o.type] ?? 0) + 1;
          count++;
        }
      }

      if (!dryRun && count > 0) {
        await flushIndexSave();
        // Each per-id vectorIndexRemove above created a new Lance version;
        // compact them in one shot after the sweep, never per-id. Guarded +
        // best-effort: a compaction failure must not fail the prune. No-op on
        // non-LanceDB backends.
        try {
          await vectorIndexOptimize();
        } catch (err) {
          logger.warn("Vector index optimize after prune failed", {
            error: err instanceof Error ? err.message : String(err),
          });
        }
      }

      logger.info("Observation prune complete", {
        dryRun,
        count,
        byType,
        types: [...typeSet],
        olderThanDays: data?.olderThanDays,
      });

      return dryRun
        ? { dryRun, wouldDelete: count, byType }
        : { dryRun, deleted: count, byType };
    },
  );
}
