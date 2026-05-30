import type { ISdk } from "iii-sdk";
import type {
  Session,
  CompressedObservation,
  ObservationType,
} from "../types.js";
import { KV } from "../state/schema.js";
import { StateKV } from "../state/kv.js";
import { recordAudit } from "./audit.js";
import { getSearchIndex, vectorIndexRemove, flushIndexSave } from "./search.js";
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
      const { decrementImageRef } = await import("./image-refs.js");

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
            if (age <= olderThanMs) continue;
          }

          if (dryRun) {
            byType[o.type] = (byType[o.type] ?? 0) + 1;
            count++;
            continue;
          }

          try {
            await kv.delete(KV.observations(session.id), o.id);
          } catch (err) {
            logger.warn("Observation prune delete failed", {
              resource: "observation",
              id: o.id,
              sessionId: session.id,
              error: err instanceof Error ? err.message : String(err),
            });
            continue;
          }
          getSearchIndex().remove(o.id);
          await vectorIndexRemove(o.id);
          if (o.imageData) await decrementImageRef(kv, sdk, o.imageData);
          if (o.imageRef && o.imageRef !== o.imageData) {
            await decrementImageRef(kv, sdk, o.imageRef);
          }
          await recordAudit(kv, "delete", "mem::observations-prune", [o.id], {
            resource: "observation",
            reason: "observation_prune",
            type: o.type,
            sessionId: session.id,
            dryRun,
          });
          byType[o.type] = (byType[o.type] ?? 0) + 1;
          count++;
        }
      }

      if (!dryRun && count > 0) {
        await flushIndexSave();
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
