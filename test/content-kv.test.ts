import { describe, it, expect, afterAll } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ISdk } from "iii-sdk";
import { createLanceBackends } from "../src/state/stores/lancedb-store.js";
import type { ContentKvStore } from "../src/state/content-kv-router.js";
import {
  ScopeRoutingKV,
  ContentMigrationState,
  backfillContentIfIncomplete,
  isContentScope,
  contentKeyOf,
} from "../src/state/content-kv-router.js";
import { KV } from "../src/state/schema.js";

// Same native-module gate as lancedb-store.test.ts: skip the whole suite when
// @lancedb/lancedb has no platform binary on this runner.
const lancedbAvailable: boolean = await import("@lancedb/lancedb")
  .then(() => true)
  .catch(() => false);

const DIMS = 8;

async function openContent(dir: string): Promise<ContentKvStore> {
  const { contentKv } = await createLanceBackends({
    kv: null as never,
    dataDir: dir,
    dimensions: DIMS,
    backend: "lancedb",
  });
  if (!contentKv) throw new Error("contentKv missing from lancedb backends");
  return contentKv;
}

// Minimal in-memory iii engine: a Map-backed state::get/set/delete/list so the
// ScopeRoutingKV fallthrough (non-content scopes) and migration read-fallback
// can be exercised without a live engine.
function makeFakeSdk(): { sdk: ISdk; store: Map<string, Map<string, unknown>> } {
  const store = new Map<string, Map<string, unknown>>();
  const ensure = (s: string): Map<string, unknown> => {
    let m = store.get(s);
    if (!m) {
      m = new Map();
      store.set(s, m);
    }
    return m;
  };
  const sdk = {
    trigger(req: { function_id: string; payload: Record<string, unknown> }) {
      const { function_id, payload } = req;
      const scope = payload.scope as string;
      const key = payload.key as string;
      if (function_id === "state::get") {
        const m = store.get(scope);
        return Promise.resolve(m && m.has(key) ? m.get(key) : null);
      }
      if (function_id === "state::set") {
        ensure(scope).set(key, payload.value);
        return Promise.resolve(payload.value);
      }
      if (function_id === "state::delete") {
        store.get(scope)?.delete(key);
        return Promise.resolve(undefined);
      }
      if (function_id === "state::list") {
        const m = store.get(scope);
        return Promise.resolve(m ? Array.from(m.values()) : []);
      }
      return Promise.resolve(null);
    },
  } as unknown as ISdk;
  return { sdk, store };
}

describe("content scope membership (pure)", () => {
  it("classifies static + prefixed content scopes, excludes others", () => {
    expect(isContentScope(KV.memories)).toBe(true);
    expect(isContentScope(KV.sessions)).toBe(true);
    expect(isContentScope(KV.summaries)).toBe(true);
    expect(isContentScope(KV.observations("s1"))).toBe(true);
    expect(isContentScope(KV.enrichedChunks("s1"))).toBe(true);
    expect(isContentScope(KV.graphNodes)).toBe(false);
    expect(isContentScope(KV.config)).toBe(false);
    expect(isContentScope(KV.audit)).toBe(false);
  });

  it("derives the per-scope key (summaries on sessionId, else id)", () => {
    expect(contentKeyOf(KV.memories, { id: "mem_1" })).toBe("mem_1");
    expect(contentKeyOf(KV.observations("s1"), { id: "obs_1" })).toBe("obs_1");
    expect(contentKeyOf(KV.summaries, { sessionId: "s1", id: "ignored" })).toBe("s1");
    expect(contentKeyOf(KV.memories, { noId: true })).toBeNull();
  });
});

describe.skipIf(!lancedbAvailable)("LanceContentKvStore (contract)", () => {
  const dataDir = mkdtempSync(join(tmpdir(), "agentmemory-content-"));
  afterAll(() => rmSync(dataDir, { recursive: true, force: true }));

  it("set/get round-trips and returns null for misses", async () => {
    const c = await openContent(dataDir);
    const mem = { id: "mem_1", content: "hello" };
    await c.set(KV.memories, "mem_1", mem);
    expect(await c.get(KV.memories, "mem_1")).toEqual(mem);
    expect(await c.get(KV.memories, "missing")).toBeNull();
  });

  it("delete writes a tombstone: get null, getRaw deleted, list excludes", async () => {
    const c = await openContent(dataDir);
    await c.set(KV.memories, "mem_del", { id: "mem_del" });
    await c.delete(KV.memories, "mem_del");
    expect(await c.get(KV.memories, "mem_del")).toBeNull();
    const raw = await c.getRaw(KV.memories, "mem_del");
    expect(raw.found).toBe(true);
    expect(raw.deleted).toBe(true);
    const list = await c.list<{ id: string }>(KV.memories);
    expect(list.find((m) => m.id === "mem_del")).toBeUndefined();
  });

  it("composite pk keeps the same key independent across scopes", async () => {
    const c = await openContent(dataDir);
    await c.set(KV.observations("sA"), "shared", { id: "shared", from: "A" });
    await c.set(KV.observations("sB"), "shared", { id: "shared", from: "B" });
    expect((await c.get<{ from: string }>(KV.observations("sA"), "shared"))!.from).toBe("A");
    expect((await c.get<{ from: string }>(KV.observations("sB"), "shared"))!.from).toBe("B");
  });

  it("pk is collision-safe for adversarial scope/key splits", async () => {
    const c = await openContent(dataDir);
    // ["a:b","c"] vs ["a","b:c"] would collide under naive scope+":"+key.
    await c.set("mem:obs:a:b", "c", { id: "c", tag: "left" });
    await c.set("mem:obs:a", "b:c", { id: "b:c", tag: "right" });
    expect((await c.get<{ tag: string }>("mem:obs:a:b", "c"))!.tag).toBe("left");
    expect((await c.get<{ tag: string }>("mem:obs:a", "b:c"))!.tag).toBe("right");
  });

  it("findByKey resolves a bare key across obs scopes, skips tombstones", async () => {
    const c = await openContent(dataDir);
    await c.set(KV.observations("sX"), "obs_bare", { id: "obs_bare", v: 1 });
    const hit = await c.findByKey<{ v: number }>("obs_bare", [KV.memories], ["mem:obs:"]);
    expect(hit?.scope).toBe(KV.observations("sX"));
    expect(hit?.value.v).toBe(1);
    await c.delete(KV.observations("sX"), "obs_bare");
    expect(await c.findByKey("obs_bare", [KV.memories], ["mem:obs:"])).toBeNull();
  });

  it("bulkInsertIfAbsent inserts new rows but preserves existing + tombstones", async () => {
    const c = await openContent(dataDir);
    await c.set(KV.sessions, "ses_keep", { id: "ses_keep", v: "new-write" });
    await c.set(KV.sessions, "ses_tomb", { id: "ses_tomb" });
    await c.delete(KV.sessions, "ses_tomb");
    await c.bulkInsertIfAbsent([
      { scope: KV.sessions, key: "ses_keep", value: { id: "ses_keep", v: "stale" } },
      { scope: KV.sessions, key: "ses_tomb", value: { id: "ses_tomb", v: "resurrected?" } },
      { scope: KV.sessions, key: "ses_fresh", value: { id: "ses_fresh", v: "ok" } },
    ]);
    expect((await c.get<{ v: string }>(KV.sessions, "ses_keep"))!.v).toBe("new-write");
    expect(await c.get(KV.sessions, "ses_tomb")).toBeNull(); // tombstone preserved
    expect((await c.get<{ v: string }>(KV.sessions, "ses_fresh"))!.v).toBe("ok");
  });

  it("countScope counts non-tombstone rows", async () => {
    const dir = mkdtempSync(join(tmpdir(), "agentmemory-content-count-"));
    try {
      const c = await openContent(dir);
      await c.set(KV.summaries, "s1", { sessionId: "s1" });
      await c.set(KV.summaries, "s2", { sessionId: "s2" });
      await c.delete(KV.summaries, "s2");
      expect(await c.countScope(KV.summaries)).toBe(1);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("reopens persisted content rows from disk (durability)", async () => {
    const dir = mkdtempSync(join(tmpdir(), "agentmemory-content-reopen-"));
    try {
      const first = await openContent(dir);
      await first.set(KV.memories, "mem_p", { id: "mem_p", content: "persisted" });
      const second = await openContent(dir);
      expect((await second.get<{ content: string }>(KV.memories, "mem_p"))!.content).toBe(
        "persisted",
      );
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe.skipIf(!lancedbAvailable)("ScopeRoutingKV routing", () => {
  const dataDir = mkdtempSync(join(tmpdir(), "agentmemory-route-"));
  afterAll(() => rmSync(dataDir, { recursive: true, force: true }));

  it("routes content scopes to content_kv and non-content to iii", async () => {
    const content = await openContent(dataDir);
    const { sdk, store } = makeFakeSdk();
    const migration = new ContentMigrationState();
    migration.markAllComplete();
    const kv = new ScopeRoutingKV(sdk, { content, migration });

    await kv.set(KV.memories, "mem_r", { id: "mem_r", x: 1 });
    // content scope did NOT touch the iii store
    expect(store.get(KV.memories)).toBeUndefined();
    expect(await kv.get<{ x: number }>(KV.memories, "mem_r")).toEqual({ id: "mem_r", x: 1 });

    await kv.set(KV.config, "cfg", { v: 9 });
    // non-content scope landed in iii, not content_kv
    expect(store.get(KV.config)?.get("cfg")).toEqual({ v: 9 });
    expect(await kv.get<{ v: number }>(KV.config, "cfg")).toEqual({ v: 9 });
  });

  it("during migration reads fall back to iii; tombstones suppress fallback", async () => {
    const content = await openContent(dataDir);
    const { sdk, store } = makeFakeSdk();
    const migration = new ContentMigrationState(); // nothing complete -> migrating
    const kv = new ScopeRoutingKV(sdk, { content, migration });

    // A pre-existing iii row not yet backfilled is visible via fallback.
    store.set(KV.sessions, new Map([["ses_iii", { id: "ses_iii", src: "iii" }]]));
    expect((await kv.get<{ src: string }>(KV.sessions, "ses_iii"))!.src).toBe("iii");

    // A delete tombstones in content_kv and must hide the iii copy even mid-migration.
    await kv.delete(KV.sessions, "ses_iii");
    expect(await kv.get(KV.sessions, "ses_iii")).toBeNull();

    // After the scope is marked complete the read path is content-only.
    store.set(KV.summaries, new Map([["only_iii", { sessionId: "only_iii" }]]));
    expect(await kv.get(KV.summaries, "only_iii")).not.toBeNull(); // still migrating
    migration.markComplete(KV.summaries);
    expect(await kv.get(KV.summaries, "only_iii")).toBeNull(); // content-only now
  });

  it("migration-window list merges iii + content with content winning", async () => {
    const content = await openContent(dataDir);
    const { sdk, store } = makeFakeSdk();
    const migration = new ContentMigrationState();
    const kv = new ScopeRoutingKV(sdk, { content, migration });
    const scope = KV.observations("merge");

    store.set(
      scope,
      new Map([
        ["A", { id: "A", v: "iii" }],
        ["B", { id: "B", v: "iii" }],
        ["D", { id: "D", v: "iii" }],
      ]),
    );
    await content.set(scope, "B", { id: "B", v: "content" });
    await content.set(scope, "C", { id: "C", v: "content" });
    await kv.delete(scope, "D"); // tombstone D

    const rows = await kv.list<{ id: string; v: string }>(scope);
    const byId = new Map(rows.map((r) => [r.id, r.v]));
    expect(byId.get("A")).toBe("iii"); // iii-only survives
    expect(byId.get("B")).toBe("content"); // content overrides iii
    expect(byId.get("C")).toBe("content"); // content-only present
    expect(byId.has("D")).toBe(false); // tombstone hides iii row
  });

  it("findContentByKey resolves a bare obsId without a sessionId", async () => {
    const content = await openContent(dataDir);
    const { sdk } = makeFakeSdk();
    const migration = new ContentMigrationState();
    migration.markAllComplete();
    const kv = new ScopeRoutingKV(sdk, { content, migration });
    await kv.set(KV.observations("deep"), "obs_findme", { id: "obs_findme", ok: true });
    const hit = await kv.findContentByKey<{ ok: boolean }>("obs_findme", [KV.memories], ["mem:obs:"]);
    expect(hit?.value.ok).toBe(true);
  });
});

describe.skipIf(!lancedbAvailable)("backfillContentIfIncomplete", () => {
  it("copies iii content, is idempotent, and resumes", async () => {
    const dir = mkdtempSync(join(tmpdir(), "agentmemory-backfill-"));
    try {
      const content = await openContent(dir);
      const { sdk, store } = makeFakeSdk();
      const StateKVmod = await import("../src/state/kv.js");
      const source = new StateKVmod.StateKV(sdk);

      store.set(
        KV.memories,
        new Map([
          ["mem_a", { id: "mem_a" }],
          ["mem_b", { id: "mem_b" }],
        ]),
      );
      store.set(KV.sessions, new Map([["ses_1", { id: "ses_1" }]]));
      store.set(KV.summaries, new Map([["ses_1", { sessionId: "ses_1" }]]));
      store.set(KV.observations("ses_1"), new Map([["obs_1", { id: "obs_1" }]]));

      const migration = new ContentMigrationState();
      const report = await backfillContentIfIncomplete(
        source,
        content,
        migration,
        [KV.observations("ses_1")],
      );
      expect(report.totalCopied).toBe(5); // 2 mem + 1 ses + 1 sum + 1 obs
      expect(await content.get(KV.memories, "mem_a")).toEqual({ id: "mem_a" });
      expect(await content.get(KV.summaries, "ses_1")).toEqual({ sessionId: "ses_1" });
      expect(await content.get(KV.observations("ses_1"), "obs_1")).toEqual({ id: "obs_1" });

      // Second run is a manifest no-op (nothing re-copied).
      const report2 = await backfillContentIfIncomplete(
        source,
        content,
        new ContentMigrationState(),
        [KV.observations("ses_1")],
      );
      expect(report2.totalCopied).toBe(0);
      expect(report2.scopes.every((s) => s.skipped)).toBe(true);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
