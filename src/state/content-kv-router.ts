// Content-scope KV router.
//
// Memory/observation/session/summary CONTENT used to live in the iii engine's
// file-based KV (state::*). That store is write-back / in-memory and only
// reaches disk on a clean engine flush; once the LanceDB fork broke clean
// shutdown (SIGKILL on stop), every save since late May lived only in RAM and
// died on each restart. The vectors survived because LanceDB writes them
// through on every save.
//
// This module finishes the same migration for the last dataset still in the
// lazy iii KV: it generalizes the graph router into a `ScopeRoutingKV` that
// routes the content scopes to a write-through LanceDB-backed `content_kv`
// store (durable on every save, surviving a `kill -9`) while passing every
// other scope to the iii KV unchanged. It is additive routing on the existing
// seam, not a rewrite: all content read/write sites already funnel through the
// single shared `kv` instance, so none of them are touched.
import { readdirSync } from "node:fs";
import type { ISdk } from "iii-sdk";
import { StateKV } from "./kv.js";
import { KV } from "./schema.js";
import { GRAPH_SCOPES, type GraphKvStore } from "./graph-kv-router.js";

// ---------------------------------------------------------------------------
// Tombstone encoding
//
// A delete on a routed content scope writes a durable TOMBSTONE row rather than
// hard-removing, so a stale iii copy (read-fallback during migration, or the
// rollback export) can never resurrect a row that was deleted post-cutover. The
// tombstone is encoded as a sentinel string in the `value` column that
// JSON.stringify can never produce (a leading NUL byte), keeping the column a
// plain non-null Utf8 exactly like the graph_kv table — no extra schema column.
export const CONTENT_TOMBSTONE = "__CONTENT_TOMBSTONE__";

export interface ContentRawGet<T = unknown> {
  found: boolean;
  deleted: boolean;
  value: T | null;
}

export interface ContentRawRow<T = unknown> {
  key: string;
  value: T | null;
  deleted: boolean;
}

// The write-through content store contract. Extends the graph KV contract
// (get/set/delete/list/optimize) with the methods the migration machinery and
// bare-obsId resolution need. `delete` is tombstone-semantics here (see above),
// not a hard remove.
export interface ContentKvStore extends GraphKvStore {
  // Tri-state read for the migration read-fallback: distinguishes "absent"
  // (fall back to iii) from "tombstoned" (deleted; do NOT fall back).
  getRaw<T = unknown>(scope: string, key: string): Promise<ContentRawGet<T>>;
  // Every row for a scope INCLUDING tombstones — used by the migration-window
  // list merge and the rollback export.
  listRaw<T = unknown>(scope: string): Promise<Array<ContentRawRow<T>>>;
  // Resolve a bare key (e.g. a bare obsId) across all scopes whose name is one
  // of `scopeExact` or starts with one of `scopePrefixes`. Skips tombstones.
  findByKey<T = unknown>(
    key: string,
    scopeExact: string[],
    scopePrefixes: string[],
  ): Promise<{ scope: string; value: T } | null>;
  // Insert-only bulk upsert for backfill: rows whose composite pk already
  // exists (a newer write, or a tombstone) are PRESERVED, never overwritten by
  // stale iii data. Used ONLY for the one-shot backfill.
  bulkInsertIfAbsent(
    rows: Array<{ scope: string; key: string; value: unknown }>,
  ): Promise<void>;
  // Non-tombstone row count for a scope (verification / observability).
  countScope(scope: string): Promise<number>;
  // Every non-tombstone key across all scopes, for the boot ghost repair check
  // (indexed ids absent from this set have no content and cannot expand).
  allKeys(): Promise<Set<string>>;
  // Resolve only once any in-flight write/delete/backfill/optimize on
  // content_kv has settled - a shutdown drain. Content is already durable on
  // disk per write; this only avoids interrupting a merge mid-commit.
  drain(): Promise<void>;
}

// ---------------------------------------------------------------------------
// Scope membership
//
// Static content scopes are fixed strings; observation/enriched scopes are
// per-session and matched by prefix (`mem:obs:<id>`, `mem:enriched:<id>`).
export const CONTENT_STATIC_SCOPES: ReadonlySet<string> = new Set<string>([
  KV.memories,
  KV.sessions,
  KV.summaries,
]);

export const CONTENT_SCOPE_PREFIXES: readonly string[] = ["mem:obs:", "mem:enriched:"];

export function isContentScope(scope: string): boolean {
  if (CONTENT_STATIC_SCOPES.has(scope)) return true;
  return CONTENT_SCOPE_PREFIXES.some((p) => scope.startsWith(p));
}

// iii `state::list` returns stored VALUES, not keys, so backfill must
// reconstruct each row's key from the value's stable id field. Every routed
// scope keys on `value.id` EXCEPT summaries, which key on `value.sessionId`
// (summarize.ts writes `kv.set(KV.summaries, sessionId, summary)`).
export function contentKeyOf(scope: string, value: unknown): string | null {
  if (!value || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  if (scope === KV.summaries) {
    return typeof v.sessionId === "string" && v.sessionId.length > 0 ? v.sessionId : null;
  }
  return typeof v.id === "string" && v.id.length > 0 ? v.id : null;
}

// Enumerate the dynamic content scopes present on disk in the iii state_store
// directory. Files are named `<url-encoded-scope>.bin` (e.g.
// `mem%3Aobs%3A<sid>.bin` -> `mem:obs:<sid>`), so the inventory is the
// authoritative on-disk source for which observation/enriched scopes exist.
// Returns only prefix-matched content scopes; static scopes are added by the
// backfill explicitly. A missing/unreadable directory yields an empty list.
export function discoverContentScopesFromBin(stateStoreDir: string): string[] {
  let files: string[];
  try {
    files = readdirSync(stateStoreDir);
  } catch {
    return [];
  }
  const scopes = new Set<string>();
  for (const f of files) {
    if (!f.endsWith(".bin")) continue;
    let scope: string;
    try {
      scope = decodeURIComponent(f.slice(0, -4));
    } catch {
      continue;
    }
    if (CONTENT_SCOPE_PREFIXES.some((p) => scope.startsWith(p))) scopes.add(scope);
  }
  return Array.from(scopes).sort();
}

// ---------------------------------------------------------------------------
// Migration state
//
// Tracks which content scopes have been fully backfilled. Until a scope is
// complete the router reads content_kv FIRST and falls back to iii for any key
// not yet present, so a not-yet-copied iii row stays visible. Boot backfill
// runs to completion BEFORE traffic is accepted and then `markAllComplete()`
// is called, after which the steady-state read path is content-only.
export class ContentMigrationState {
  private readonly complete = new Set<string>();
  private all = false;

  markComplete(scope: string): void {
    this.complete.add(scope);
  }

  markAllComplete(): void {
    this.all = true;
  }

  isComplete(scope: string): boolean {
    return this.all || this.complete.has(scope);
  }
}

// ---------------------------------------------------------------------------
// Router
//
// Routes graph scopes to `graph`, content scopes to `content`, and everything
// else to the iii KV (super). Both backends are optional and routed
// independently; with the memory vector backend neither is present and this
// behaves exactly like a plain StateKV.
export interface ScopeRoutingOptions {
  graph?: GraphKvStore;
  graphScopes?: ReadonlySet<string>;
  content?: ContentKvStore;
  migration?: ContentMigrationState;
}

export class ScopeRoutingKV extends StateKV {
  constructor(
    sdk: ISdk,
    private readonly routes: ScopeRoutingOptions,
  ) {
    super(sdk);
  }

  private graphScope(scope: string): boolean {
    return (
      !!this.routes.graph &&
      (this.routes.graphScopes ?? GRAPH_SCOPES).has(scope)
    );
  }

  private contentScope(scope: string): boolean {
    return !!this.routes.content && isContentScope(scope);
  }

  private migrating(scope: string): boolean {
    return !!this.routes.migration && !this.routes.migration.isComplete(scope);
  }

  async get<T = unknown>(scope: string, key: string): Promise<T | null> {
    if (this.graphScope(scope)) return this.routes.graph!.get<T>(scope, key);
    if (this.contentScope(scope)) {
      const content = this.routes.content!;
      if (this.migrating(scope)) {
        const raw = await content.getRaw<T>(scope, key);
        if (raw.found) return raw.deleted ? null : raw.value;
        return super.get<T>(scope, key); // not yet copied — read iii
      }
      return content.get<T>(scope, key);
    }
    return super.get<T>(scope, key);
  }

  async set<T = unknown>(scope: string, key: string, value: T): Promise<T> {
    if (this.graphScope(scope)) return this.routes.graph!.set<T>(scope, key, value);
    if (this.contentScope(scope)) {
      await this.routes.content!.set<T>(scope, key, value);
      return value;
    }
    return super.set<T>(scope, key, value);
  }

  async delete(scope: string, key: string): Promise<void> {
    if (this.graphScope(scope)) return this.routes.graph!.delete(scope, key);
    if (this.contentScope(scope)) {
      // Durable tombstone in content_kv (the authoritative store). Then a
      // best-effort drop of the iii copy while it still exists, so the
      // migration read-fallback never serves a stale row; the tombstone
      // already guards reads even if the iii delete fails.
      await this.routes.content!.delete(scope, key);
      await super.delete(scope, key).catch(() => {});
      return;
    }
    return super.delete(scope, key);
  }

  async list<T = unknown>(scope: string): Promise<T[]> {
    if (this.graphScope(scope)) return this.routes.graph!.list<T>(scope);
    if (this.contentScope(scope)) {
      const content = this.routes.content!;
      if (this.migrating(scope)) {
        // Migration-window merge: content_kv (writes AND tombstones) overrides
        // iii per key, so a new write shows through, a deleted key is hidden,
        // and a not-yet-copied iii row remains visible.
        const rawRows = await content.listRaw<T>(scope);
        const byKey = new Map<string, ContentRawRow<T>>();
        for (const r of rawRows) byKey.set(r.key, r);
        const iiiItems = await super.list<T>(scope).catch(() => [] as T[]);
        const out: T[] = [];
        for (const item of iiiItems) {
          const k = contentKeyOf(scope, item);
          if (k !== null && byKey.has(k)) continue; // content wins
          out.push(item);
        }
        for (const r of byKey.values()) {
          if (!r.deleted && r.value !== null) out.push(r.value);
        }
        return out;
      }
      return content.list<T>(scope);
    }
    return super.list<T>(scope);
  }

  // Capability used by smart-search's findObservation to resolve a bare obsId
  // (no sessionId) in a single indexed lookup instead of an O(sessions) scan.
  // Returns null when no content store is routed (memory backend).
  async findContentByKey<T = unknown>(
    key: string,
    scopeExact: string[],
    scopePrefixes: string[],
  ): Promise<{ scope: string; value: T } | null> {
    if (!this.routes.content) return null;
    return this.routes.content.findByKey<T>(key, scopeExact, scopePrefixes);
  }
}

// Narrow runtime guard so consumers (smart-search) can opt into the bare-key
// capability without a hard import-time dependency on ScopeRoutingKV.
export function hasContentByKey(
  kv: unknown,
): kv is { findContentByKey: ScopeRoutingKV["findContentByKey"] } {
  return (
    !!kv &&
    typeof (kv as { findContentByKey?: unknown }).findContentByKey === "function"
  );
}

// ---------------------------------------------------------------------------
// Backfill
//
// Seeds content_kv from the iii KV on boot, BEFORE any writer accepts traffic,
// so a stale iii row can never overwrite a newer content_kv write. A per-scope
// manifest (stored in content_kv itself) records completion: a scope is marked
// done only after a full copy, and a crashed partial backfill re-runs the scope
// to completion. `bulkInsertIfAbsent` makes re-copy idempotent.
const MANIFEST_SCOPE = "mem:__content_backfill_manifest";

interface ManifestEntry {
  done: boolean;
  sourceCount: number;
  copied: number;
  at: string;
}

export interface ContentBackfillScopeReport {
  scope: string;
  source: number;
  copied: number;
  skipped: boolean;
}

export interface ContentBackfillReport {
  scopes: ContentBackfillScopeReport[];
  totalSource: number;
  totalCopied: number;
  staticScopes: number;
  dynamicScopes: number;
}

// Stage-A quiesced export overlay: scope -> key -> value. Captures the live
// daemon's RAM delta (saves not yet flushed to the on-disk .bin) so the Stage-B
// boot backfill seeds content_kv from the UNION of iii (.bin) and this overlay,
// with the overlay WINNING on conflict (it is the latest RAM state).
export type ContentExportOverlay = Record<string, Record<string, unknown>>;

export async function backfillContentIfIncomplete(
  source: StateKV,
  content: ContentKvStore,
  migration: ContentMigrationState,
  dynamicScopes: string[],
  log: (msg: string) => void = () => {},
  overlay?: ContentExportOverlay,
): Promise<ContentBackfillReport> {
  const staticScopes = [KV.memories, KV.sessions, KV.summaries];
  const overlayScopes = overlay ? Object.keys(overlay) : [];
  const dyn = Array.from(new Set([...dynamicScopes, ...overlayScopes]))
    .filter((s) => isContentScope(s) && !CONTENT_STATIC_SCOPES.has(s))
    .sort();
  const allScopes = [...staticScopes, ...dyn];

  const report: ContentBackfillReport = {
    scopes: [],
    totalSource: 0,
    totalCopied: 0,
    staticScopes: staticScopes.length,
    dynamicScopes: dyn.length,
  };

  for (const scope of allScopes) {
    const manifest = await content
      .get<ManifestEntry>(MANIFEST_SCOPE, scope)
      .catch(() => null);
    if (manifest?.done) {
      migration.markComplete(scope);
      report.scopes.push({
        scope,
        source: manifest.sourceCount ?? -1,
        copied: 0,
        skipped: true,
      });
      continue;
    }

    const items = await source
      .list<Record<string, unknown>>(scope)
      .catch(() => [] as Array<Record<string, unknown>>);

    // Merge iii (.bin) with the Stage-A overlay, overlay winning per key (it is
    // the newer RAM state). Keyed map collapses duplicates before the insert.
    const merged = new Map<string, unknown>();
    for (const item of items) {
      const key = contentKeyOf(scope, item);
      if (key !== null) merged.set(key, item);
    }
    const overlayRows = overlay?.[scope];
    if (overlayRows) {
      for (const [key, value] of Object.entries(overlayRows)) merged.set(key, value);
    }
    const rows = Array.from(merged, ([key, value]) => ({ scope, key, value }));
    if (rows.length > 0) await content.bulkInsertIfAbsent(rows);

    const entry: ManifestEntry = {
      done: true,
      sourceCount: merged.size,
      copied: rows.length,
      at: new Date().toISOString(),
    };
    await content.set(MANIFEST_SCOPE, scope, entry);
    migration.markComplete(scope);

    report.scopes.push({ scope, source: merged.size, copied: rows.length, skipped: false });
    report.totalSource += merged.size;
    report.totalCopied += rows.length;
  }

  await content.optimize?.();
  log(
    `Content backfill: ${report.totalCopied} rows across ${allScopes.length} scopes ` +
      `(${report.staticScopes} static + ${report.dynamicScopes} dynamic), ` +
      `source total ${report.totalSource}`,
  );
  return report;
}
