import { describe, it, expect, afterAll } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createLanceBackends } from "../src/state/stores/lancedb-store.js";
import type { PersistenceBackends } from "../src/state/vector-store.js";

// The LanceDB backend depends on the @lancedb/lancedb native module, an
// optionalDependency that may be absent (no platform binary on this CI
// runner). Probe it once at module load; describe.skipIf below makes the
// whole suite a clean skip rather than a failure when it is missing.
const lancedbAvailable: boolean = await import("@lancedb/lancedb")
  .then(() => true)
  .catch(() => false);

const DIMS = 8;

// Distinct unit vectors so cosine ranking is unambiguous. The query equals
// vector "a", so "a" must come back best-first.
function unitVec(slot: number): Float32Array {
  const v = new Float32Array(DIMS);
  v[slot % DIMS] = 1;
  return v;
}

describe.skipIf(!lancedbAvailable)("LanceDB vector backend (contract)", () => {
  // Each createLanceBackends opens ${dataDir}/lancedb; use a unique tmp dir
  // per suite and clean it up in afterAll.
  const dataDir = mkdtempSync(join(tmpdir(), "agentmemory-lance-"));

  afterAll(() => {
    rmSync(dataDir, { recursive: true, force: true });
  });

  async function open(dir: string): Promise<PersistenceBackends> {
    return createLanceBackends({
      kv: null as never,
      dataDir: dir,
      dimensions: DIMS,
      backend: "lancedb",
    });
  }

  it("adds vectors and searches them best-first", async () => {
    const { vector } = await open(dataDir);
    await vector.add("a", "ses_1", unitVec(0));
    await vector.add("b", "ses_1", unitVec(1));
    await vector.add("c", "ses_1", unitVec(2));
    expect(vector.size).toBe(3);

    const hits = await vector.search(unitVec(0), 3);
    expect(hits.length).toBe(3);
    expect(hits[0].obsId).toBe("a");
    expect(hits[0].sessionId).toBe("ses_1");
    // Best-first: the exact match outscores the orthogonal vectors.
    expect(hits[0].score).toBeGreaterThan(hits[1].score);
    expect(hits[1].score).toBeGreaterThanOrEqual(hits[2].score);
  });

  it("returns default lifecycle fields for a freshly-added vector", async () => {
    const { vector } = await open(dataDir);
    const map = await vector.getLifecycle(["a"]);
    const a = map.get("a");
    expect(a).toBeDefined();
    expect(a!.importance).toBe(50);
    expect(a!.maturity).toBe("draft");
    expect(a!.recency).toBe(1);
    expect(a!.accessCount).toBe(0);
    expect(a!.updateCount).toBe(0);
  });

  it("setLifecycle updates the stored fields", async () => {
    const { vector } = await open(dataDir);
    await vector.setLifecycle("a", { importance: 90, maturity: "core" });
    const map = await vector.getLifecycle(["a"]);
    const a = map.get("a");
    expect(a!.importance).toBe(90);
    expect(a!.maturity).toBe("core");
  });

  it("listLifecycle returns one entry per stored vector", async () => {
    const { vector } = await open(dataDir);
    const all = await vector.listLifecycle();
    expect(all.size).toBe(3);
    expect(all.has("a")).toBe(true);
    expect(all.has("b")).toBe(true);
    expect(all.has("c")).toBe(true);
  });

  it("validateDimensions reports no mismatch for the correct dimension", async () => {
    const { vector } = await open(dataDir);
    const report = await vector.validateDimensions(DIMS);
    expect(report.mismatches).toHaveLength(0);
    expect(Array.from(report.seenDimensions)).toEqual([DIMS]);
  });

  it("validateDimensions flags a wrong expected dimension", async () => {
    const { vector } = await open(dataDir);
    const report = await vector.validateDimensions(DIMS * 2);
    expect(report.mismatches.length).toBeGreaterThanOrEqual(1);
  });

  it("blobStore set/get round-trips a value", async () => {
    const { blobStore } = await open(dataDir);
    await blobStore.set("data", "bm25-serialized-payload");
    const got = await blobStore.get("data");
    expect(got).toBe("bm25-serialized-payload");
    expect(await blobStore.get("missing-key")).toBeNull();
  });

  it("re-adding an obsId preserves its lifecycle fields", async () => {
    const { vector } = await open(dataDir);
    // "a" was set to importance 90 / core above. Re-add with a new vector;
    // the LanceDB backend updates only vector/session/updated_at on a match,
    // leaving lifecycle columns untouched.
    await vector.setLifecycle("a", { importance: 77 });
    await vector.add("a", "ses_2", unitVec(3));
    const map = await vector.getLifecycle(["a"]);
    expect(map.get("a")!.importance).toBe(77);
    // Re-add of an existing id does not change the row count.
    expect(vector.size).toBe(3);
  });

  it("remove decrements size", async () => {
    const { vector } = await open(dataDir);
    expect(vector.size).toBe(3);
    await vector.remove("c");
    expect(vector.size).toBe(2);
    const hits = await vector.search(unitVec(2), 3);
    expect(hits.find((h) => h.obsId === "c")).toBeUndefined();
  });

  it("clear zeroes the size", async () => {
    const { vector } = await open(dataDir);
    expect(vector.size).toBeGreaterThan(0);
    await vector.clear();
    expect(vector.size).toBe(0);
    expect(await vector.search(unitVec(0), 3)).toEqual([]);
  });

  it("reopens persisted rows and blobs from disk", async () => {
    // Fresh dataDir so this is independent of the clear() above.
    const reopenDir = mkdtempSync(join(tmpdir(), "agentmemory-lance-reopen-"));
    try {
      const first = await open(reopenDir);
      await first.vector.add("p1", "ses_x", unitVec(0));
      await first.vector.add("p2", "ses_x", unitVec(1));
      await first.vector.setLifecycle("p1", { importance: 88 });
      await first.blobStore.set("data", "persisted-bm25");

      // Second open on the same dataDir: init() reads countRows() from disk
      // and the blob table is reopened.
      const second = await open(reopenDir);
      expect(second.vector.size).toBe(2);

      const hits = await second.vector.search(unitVec(0), 2);
      expect(hits[0].obsId).toBe("p1");

      const lc = await second.vector.getLifecycle(["p1"]);
      expect(lc.get("p1")!.importance).toBe(88);

      expect(await second.blobStore.get("data")).toBe("persisted-bm25");
    } finally {
      rmSync(reopenDir, { recursive: true, force: true });
    }
  });
});
