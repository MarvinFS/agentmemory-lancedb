import type { ISdk } from "iii-sdk";
import type { Memory, CompressedObservation, Session } from "../types.js";
import { KV } from "../state/schema.js";
import { StateKV } from "../state/kv.js";
import { recordAudit } from "./audit.js";
import { deleteAccessLog } from "./access-tracker.js";
import { getSearchIndex, vectorIndexRemove, flushIndexSave, vectorIndexOptimize } from "./search.js";
import { deleteObservationCascade } from "./_observation-cascade.js";
import { logger } from "../logger.js";

const MS_PER_DAY = 24 * 60 * 60 * 1000;
const CONTRADICTION_THRESHOLD = 0.9;

interface AutoForgetResult {
  ttlExpired: string[];
  contradictions: Array<{
    memoryA: string;
    memoryB: string;
    similarity: number;
  }>;
  lowValueObs: string[];
  dryRun: boolean;
}

export function registerAutoForgetFunction(sdk: ISdk, kv: StateKV): void {
  sdk.registerFunction("mem::auto-forget", 
    async (data: { dryRun?: boolean }): Promise<AutoForgetResult> => {
      const dryRun = data?.dryRun ?? false;
      const now = Date.now();
      const { decrementImageRef } = await import("./image-refs.js");

      const result: AutoForgetResult = {
        ttlExpired: [],
        contradictions: [],
        lowValueObs: [],
        dryRun,
      };

      const memories = await kv.list<Memory>(KV.memories);
      const deletedIds = new Set<string>();
      for (const mem of memories) {
        if (mem.forgetAfter) {
          const expiry = new Date(mem.forgetAfter).getTime();
          if (now > expiry) {
            result.ttlExpired.push(mem.id);
            deletedIds.add(mem.id);
            if (!dryRun) {
              if (mem.imageRef) {
                await decrementImageRef(kv, sdk, mem.imageRef);
              }
              await kv.delete(KV.memories, mem.id);
              await recordAudit(kv, "delete", "mem::auto-forget", [mem.id], {
                resource: "memory",
                reason: "auto-forget TTL",
                timestamp: mem.forgetAfter,
              });
              await deleteAccessLog(kv, mem.id);
              getSearchIndex().remove(mem.id);
              await vectorIndexRemove(mem.id);
            }
          }
        }
      }

      const latestMemories = memories
        .filter((m) => m.isLatest !== false && !deletedIds.has(m.id))
        .sort(
          (a, b) =>
            new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
        )
        .slice(0, 1000);

      const tokenCache = new Map<string, Set<string>>();
      for (const mem of latestMemories) {
        tokenCache.set(
          mem.id,
          new Set(
            mem.content
              .toLowerCase()
              .split(/\s+/)
              .filter((t) => t.length > 2),
          ),
        );
      }

      const memById = new Map(latestMemories.map((m) => [m.id, m]));
      const conceptIndex = new Map<string, string[]>();
      for (const mem of latestMemories) {
        const concepts = mem.concepts || [];
        for (const c of concepts) {
          const key = c.toLowerCase();
          if (!conceptIndex.has(key)) conceptIndex.set(key, []);
          conceptIndex.get(key)!.push(mem.id);
        }
      }

      const compared = new Set<string>();
      for (const [, memIds] of conceptIndex) {
        for (let i = 0; i < memIds.length; i++) {
          for (let j = i + 1; j < memIds.length; j++) {
            const key =
              memIds[i] < memIds[j]
                ? `${memIds[i]}|${memIds[j]}`
                : `${memIds[j]}|${memIds[i]}`;
            if (compared.has(key)) continue;
            compared.add(key);

            const setA = tokenCache.get(memIds[i])!;
            const setB = tokenCache.get(memIds[j])!;
            let intersection = 0;
            if (setA.size === 0 && setB.size === 0) continue;
            if (setA.size === 0 || setB.size === 0) continue;
            for (const word of setA) {
              if (setB.has(word)) intersection++;
            }
            const sim =
              intersection / (setA.size + setB.size - intersection);

            if (sim > CONTRADICTION_THRESHOLD) {
              const memA = memById.get(memIds[i])!;
              const memB = memById.get(memIds[j])!;
              result.contradictions.push({
                memoryA: memA.id,
                memoryB: memB.id,
                similarity: sim,
              });

              if (!dryRun) {
                const older =
                  new Date(memA.createdAt).getTime() <
                    new Date(memB.createdAt).getTime()
                    ? memA
                    : memB;
                older.isLatest = false;
                await kv.set(KV.memories, older.id, older);
                await recordAudit(kv, "forget", "mem::auto-forget", [older.id], {
                  resource: "memory",
                  reason: "auto-forget contradiction",
                  olderId: older.id,
                  similarity: sim,
                });
              }
            }
          }
        }
      }

      const sessions = await kv.list<Session>(KV.sessions);
      const obsPerSession: CompressedObservation[][] = [];
      for (let batch = 0; batch < sessions.length; batch += 10) {
        const chunk = sessions.slice(batch, batch + 10);
        const results = await Promise.all(
          chunk.map((s) =>
            kv
              .list<CompressedObservation>(KV.observations(s.id))
              .catch(() => [] as CompressedObservation[]),
          ),
        );
        obsPerSession.push(...results);
      }
      for (let i = 0; i < sessions.length; i++) {
        for (const obs of obsPerSession[i]) {
          if (!obs.timestamp) continue;
          const age = now - new Date(obs.timestamp).getTime();
          // The `?? 5` default means an observation written WITHOUT an importance field is never pruned here (5 > 2) - harmless today since synthetic breadcrumbs always set importance, but a latent trap for externally-written observations.
          if (age > 180 * MS_PER_DAY && (obs.importance ?? 5) <= 2) {
            result.lowValueObs.push(obs.id);
            if (!dryRun) {
              let deletedOk = false;
              try {
                await kv.delete(KV.observations(sessions[i].id), obs.id);
                deletedOk = true;
              } catch {
                deletedOk = false;
              }
              if (deletedOk) {
                await deleteObservationCascade(kv, sdk, {
                  obsId: obs.id,
                  imageData: obs.imageData,
                  imageRef: obs.imageRef,
                  removeFromIndex: true,
                  audit: {
                    operation: "delete",
                    functionId: "mem::auto-forget",
                    details: {
                      resource: "observation",
                      reason: "auto-forget low-value observation",
                      sessionId: sessions[i].id,
                      timestamp: obs.timestamp,
                    },
                  },
                });
              }
            }
          }
        }
      }

      if (!dryRun && (result.ttlExpired.length > 0 || result.lowValueObs.length > 0)) {
        await flushIndexSave();
        // Each per-id vectorIndexRemove above (TTL memories + low-value obs)
        // created a new Lance version; compact them once after the sweep,
        // never per-id. Guarded + best-effort: a compaction failure must not
        // fail auto-forget. No-op on non-LanceDB backends.
        try {
          await vectorIndexOptimize();
        } catch (err) {
          logger.warn("Vector index optimize after auto-forget failed", {
            error: err instanceof Error ? err.message : String(err),
          });
        }
      }

      logger.info("Auto-forget complete", {
        ttlExpired: result.ttlExpired.length,
        contradictions: result.contradictions.length,
        lowValueObs: result.lowValueObs.length,
        dryRun,
      });
      return result;
    },
  );
}
