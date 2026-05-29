import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { IndexPersistence } from "../src/state/index-persistence.js";
import type { IndexBlobStore } from "../src/state/index-blob-store.js";
import { SearchIndex } from "../src/state/search-index.js";
import { VectorIndex, MemoryVectorIndex } from "../src/state/vector-index.js";
import type { CompressedObservation } from "../src/types.js";

// Tiny in-memory IndexBlobStore: a Map keyed by the blob key ("data" /
// "vectors"). Replaces the old kv mock — IndexPersistence now writes/reads
// blobs through this abstraction rather than a scoped KV. The blob keys are
// exactly the ones save()/load() use, so assertions check store.get("data")
// and store.get("vectors") directly.
function makeBlobStore(): IndexBlobStore {
  const store = new Map<string, string>();
  return {
    set: async (key: string, value: string): Promise<void> => {
      store.set(key, value);
    },
    get: async (key: string): Promise<string | null> => {
      return store.has(key) ? store.get(key)! : null;
    },
  };
}

function makeObs(
  overrides: Partial<CompressedObservation> = {},
): CompressedObservation {
  return {
    id: "obs_1",
    sessionId: "ses_1",
    timestamp: new Date().toISOString(),
    type: "file_edit",
    title: "Edit auth middleware",
    subtitle: "JWT validation",
    facts: ["Added token check"],
    narrative: "Modified the auth middleware to validate JWT tokens",
    concepts: ["authentication", "jwt"],
    files: ["src/middleware/auth.ts"],
    importance: 7,
    ...overrides,
  };
}

describe("IndexPersistence", () => {
  let blobStore: IndexBlobStore;

  beforeEach(() => {
    vi.useFakeTimers();
    blobStore = makeBlobStore();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("saves and loads BM25 index round-trip", async () => {
    const bm25 = new SearchIndex();
    bm25.add(makeObs({ id: "obs_1", title: "auth handler" }));

    const persistence = new IndexPersistence(blobStore, bm25, null);
    await persistence.save();

    const loaded = await persistence.load();
    expect(loaded.bm25).not.toBeNull();
    expect(loaded.bm25!.size).toBe(1);
    const results = loaded.bm25!.search("auth");
    expect(results.length).toBe(1);
  });

  it("saves and loads vector index round-trip", async () => {
    const bm25 = new SearchIndex();
    const vector = new VectorIndex(new MemoryVectorIndex());
    await vector.add("obs_1", "ses_1", new Float32Array([0.1, 0.2, 0.3]));

    const persistence = new IndexPersistence(blobStore, bm25, vector);
    await persistence.save();

    // The in-memory backend is NOT self-persisting (persistsExternally
    // false), so save() writes the legacy "vectors" blob and load() returns
    // its JSON. Rehydrate it into a fresh index and confirm the round-trip.
    const loaded = await persistence.load();
    expect(loaded.vectorJson).not.toBeNull();

    const restored = new VectorIndex(new MemoryVectorIndex());
    await restored.restoreFrom(loaded.vectorJson!);
    expect(restored.size).toBe(1);
    const results = await restored.search(new Float32Array([0.1, 0.2, 0.3]), 1);
    expect(results[0]?.obsId).toBe("obs_1");
  });

  it("scheduleSave debounces multiple calls", async () => {
    const bm25 = new SearchIndex();
    const persistence = new IndexPersistence(blobStore, bm25, null);

    persistence.scheduleSave();
    persistence.scheduleSave();
    persistence.scheduleSave();

    await expect(blobStore.get("data")).resolves.toBeNull();

    vi.advanceTimersByTime(5000);
    await vi.runAllTimersAsync();

    const saved = await blobStore.get("data");
    expect(saved).not.toBeNull();
  });

  it("stop clears the pending timer", async () => {
    const bm25 = new SearchIndex();
    bm25.add(makeObs({ id: "obs_1", title: "auth handler" }));
    const persistence = new IndexPersistence(blobStore, bm25, null);

    persistence.scheduleSave();
    persistence.stop();

    vi.advanceTimersByTime(10000);
    const saved = await blobStore.get("data");
    expect(saved).toBeNull();
  });

  it("returns null indexes when nothing has been saved", async () => {
    const bm25 = new SearchIndex();
    const vector = new VectorIndex(new MemoryVectorIndex());
    const persistence = new IndexPersistence(blobStore, bm25, vector);

    const loaded = await persistence.load();
    expect(loaded.bm25).toBeNull();
    expect(loaded.vectorJson).toBeNull();
  });

  it("scheduled save swallows blobStore.set rejection without unhandledRejection (#204)", async () => {
    const set = vi.fn(async () => {
      const err = new Error(
        "TIMEOUT: invocation timed out after 30000ms",
      ) as Error & { code?: string; function_id?: string };
      err.code = "TIMEOUT";
      err.function_id = "state::set";
      throw err;
    });
    const failingBlobStore: IndexBlobStore = {
      set,
      get: async () => null,
    };
    const bm25 = new SearchIndex();
    bm25.add(makeObs({ id: "obs_1", title: "auth handler" }));
    const persistence = new IndexPersistence(failingBlobStore, bm25, null);

    let unhandled = false;
    const onUnhandled = () => {
      unhandled = true;
    };
    process.on("unhandledRejection", onUnhandled);

    try {
      persistence.scheduleSave();
      vi.advanceTimersByTime(5000);
      await vi.runAllTimersAsync();
      // give microtasks a chance to flush
      await Promise.resolve();
      expect(set).toHaveBeenCalled();
      expect(unhandled).toBe(false);
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  it("save() does not throw when blobStore.set rejects (#204)", async () => {
    const failingBlobStore: IndexBlobStore = {
      set: vi.fn(async () => {
        throw new Error("TIMEOUT");
      }),
      get: async () => null,
    };
    const bm25 = new SearchIndex();
    bm25.add(makeObs({ id: "obs_1", title: "auth handler" }));
    const persistence = new IndexPersistence(failingBlobStore, bm25, null);

    await expect(persistence.save()).resolves.toBeUndefined();
  });
});
