import { describe, it, expect, vi } from "vitest";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

import { registerSmartSearchFunction } from "../src/functions/smart-search.js";
import { mockSdk } from "./helpers/mocks.js";
import type { CompressedObservation, Memory, Session } from "../src/types.js";

// A KV mock whose get() returns the RAW Map value -- i.e. `undefined` on a
// miss, NOT `?? null`. This mirrors the real engine (state::get) behaviour and
// is what exposes the findObservation bug; the shared mockKV coalesces misses
// to null and therefore MASKS it (existing expand tests pass against the bug).
function rawMissKV() {
  const store = new Map<string, Map<string, unknown>>();
  return {
    get: async <T>(scope: string, key: string): Promise<T | null> =>
      store.get(scope)?.get(key) as T, // undefined on miss (no ?? null)
    set: async <T>(scope: string, key: string, data: T): Promise<T> => {
      if (!store.has(scope)) store.set(scope, new Map());
      store.get(scope)!.set(key, data);
      return data;
    },
    delete: async (scope: string, key: string): Promise<void> => {
      store.get(scope)?.delete(key);
    },
    list: async <T>(scope: string): Promise<T[]> => {
      const entries = store.get(scope);
      return entries ? (Array.from(entries.values()) as T[]) : [];
    },
  };
}

function makeObs(over: Partial<CompressedObservation>): CompressedObservation {
  return {
    id: "obs_x",
    sessionId: "ses_real",
    timestamp: "2026-02-01T00:00:00Z",
    type: "command_run",
    title: "real observation",
    facts: ["f"],
    narrative: "n",
    concepts: [],
    files: [],
    importance: 5,
    ...over,
  };
}

describe("findObservation undefined-on-miss regression", () => {
  it("Bug A: bare obsId resolves when a partial/id-less session precedes the real one in the same batch", async () => {
    const sdk = mockSdk();
    const kv = rawMissKV();

    // Partial (id-less) record FIRST, real session SECOND -- both land in the
    // first 5-item scan batch. The partial yields kv.get("mem:obs:undefined",
    // obsId) === undefined; the old predicate `.find(r => r !== null)` selects
    // that undefined and stops, dropping the real hit later in the batch.
    await kv.set("mem:sessions", "partial", { status: "completed" } as never);
    const realSession: Session = {
      id: "ses_real", project: "p", cwd: "/tmp",
      startedAt: "2026-02-01T00:00:00Z", status: "active", observationCount: 1,
    };
    await kv.set("mem:sessions", "ses_real", realSession);
    await kv.set("mem:obs:ses_real", "obs_x", makeObs({ id: "obs_x", sessionId: "ses_real" }));

    registerSmartSearchFunction(sdk as never, kv as never, async () => []);
    const result = (await sdk.trigger("mem::smart-search", {
      expandIds: ["obs_x"],
    })) as { mode: string; results: Array<{ observation: CompressedObservation }> };

    expect(result.mode).toBe("expanded");
    expect(result.results.length).toBe(1); // FAILS against `!== null` (returns 0)
    expect(result.results[0].observation.id).toBe("obs_x");
  });

  it("Bug B: bare obsId for a saved memory resolves via KV.memories with agentId preserved", async () => {
    const sdk = mockSdk();
    const kv = rawMissKV();

    const memory: Memory = {
      id: "mem_x", createdAt: "2026-02-01T00:00:00Z", updatedAt: "2026-02-01T00:00:00Z",
      type: "architecture", title: "saved memory", content: "body",
      concepts: ["c"], files: [], sessionIds: ["memory"], strength: 7,
      version: 1, isLatest: true, agentId: "claude",
    };
    await kv.set("mem:memories", "mem_x", memory);

    registerSmartSearchFunction(sdk as never, kv as never, async () => []);
    const result = (await sdk.trigger("mem::smart-search", {
      expandIds: ["mem_x"],
    })) as { mode: string; results: Array<{ observation: CompressedObservation }> };

    expect(result.mode).toBe("expanded");
    expect(result.results.length).toBe(1);
    expect(result.results[0].observation.title).toBe("saved memory");
    expect(result.results[0].observation.agentId).toBe("claude");
  });
});
