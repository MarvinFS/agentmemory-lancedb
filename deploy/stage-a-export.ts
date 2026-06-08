// Stage-A quiesced RAM export (cutover tooling - run against the OLD daemon).
//
// Recovers the live daemon's RAM delta - content saved since the last iii flush
// that exists only in the running process's memory and would be lost on stop.
// Reads every routed content scope from the live iii engine (read-only
// state::list, touches NO LanceDB table) and writes a Stage-B overlay JSON
// {scope:{key:value}}. Repeats until two consecutive passes are byte-identical
// (high-water mark) so an in-flight straggler is caught.
//
// IMPORTANT operational order (per the design doc): quiesce writes FIRST (block
// the observe/remember POST paths at nginx with 503 and confirm no active save
// session), THEN run this. Skipping Stage-A loses only the current session's
// small delta (re-saved after cutover); it is an OPTIONAL safety step, not a
// correctness dependency.
//
// Run (after approval, against the live engine):
//   ./node_modules/.bin/tsx deploy/stage-a-export.ts /home/marvinfs/aml-stage-a-overlay.json
//   ./node_modules/.bin/tsx deploy/stage-a-export.ts --check   # offline self-check, no engine
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { registerWorker } from "iii-sdk";
import { StateKV } from "../src/state/kv.js";
import { KV } from "../src/state/schema.js";
import {
  contentKeyOf,
  discoverContentScopesFromBin,
} from "../src/state/content-kv-router.js";

const STATIC_SCOPES = [KV.memories, KV.sessions, KV.summaries];

// Deterministic serialization (sorted scope + key) so the high-water-mark
// comparison is order-independent.
function stableStringify(overlay: Record<string, Record<string, unknown>>): string {
  const scopes = Object.keys(overlay).sort();
  const out: Record<string, Record<string, unknown>> = {};
  for (const s of scopes) {
    const keys = Object.keys(overlay[s]).sort();
    const inner: Record<string, unknown> = {};
    for (const k of keys) inner[k] = overlay[s][k];
    out[s] = inner;
  }
  return JSON.stringify(out);
}

function stateStoreDir(): string {
  return (
    process.env.AGENTMEMORY_STATE_STORE_DIR ||
    join(process.cwd(), "data", "state_store.db")
  );
}

async function discoverScopes(kv: StateKV): Promise<string[]> {
  const scopes = new Set<string>(STATIC_SCOPES);
  for (const s of discoverContentScopesFromBin(stateStoreDir())) scopes.add(s);
  const sessions = await kv
    .list<{ id?: string }>(KV.sessions)
    .catch(() => [] as Array<{ id?: string }>);
  for (const s of sessions) {
    if (s && typeof s.id === "string" && s.id.length > 0) {
      scopes.add(KV.observations(s.id));
      scopes.add(KV.enrichedChunks(s.id));
    }
  }
  return Array.from(scopes).sort();
}

async function exportOnce(
  kv: StateKV,
): Promise<Record<string, Record<string, unknown>>> {
  const scopes = await discoverScopes(kv);
  const overlay: Record<string, Record<string, unknown>> = {};
  for (const scope of scopes) {
    const items = await kv
      .list<Record<string, unknown>>(scope)
      .catch(() => [] as Array<Record<string, unknown>>);
    if (items.length === 0) continue;
    const inner: Record<string, unknown> = {};
    for (const item of items) {
      const key = contentKeyOf(scope, item);
      if (key !== null) inner[key] = item;
    }
    if (Object.keys(inner).length > 0) overlay[scope] = inner;
  }
  return overlay;
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args[0] === "--check") {
    // Offline self-check: validates the module loads and the .bin discovery
    // runs without touching the live engine.
    const scopes = discoverContentScopesFromBin(stateStoreDir());
    console.log(
      `stage-a-export --check OK: ${scopes.length} obs/enriched scopes on disk under ${stateStoreDir()}`,
    );
    process.exit(0);
  }

  const outPath = args[0] || join(process.cwd(), "aml-stage-a-overlay.json");
  const engineUrl = process.env.III_ENGINE_URL || "ws://localhost:49134";
  const maxPasses = parseInt(process.env.STAGE_A_MAX_PASSES || "8", 10) || 8;

  const sdk = registerWorker(engineUrl, {
    workerName: "agentmemory-stage-a-export",
    invocationTimeoutMs: 180000,
  });
  const kv = new StateKV(sdk);

  let prev = "";
  let pass = 0;
  let overlay: Record<string, Record<string, unknown>> = {};
  for (; pass < maxPasses; pass++) {
    overlay = await exportOnce(kv);
    const cur = stableStringify(overlay);
    const scopeCount = Object.keys(overlay).length;
    const rowCount = Object.values(overlay).reduce((n, m) => n + Object.keys(m).length, 0);
    console.log(`pass ${pass + 1}: ${scopeCount} scopes, ${rowCount} rows`);
    if (cur === prev) {
      console.log(`high-water mark reached after ${pass + 1} passes (two identical)`);
      break;
    }
    prev = cur;
  }

  writeFileSync(outPath, JSON.stringify(overlay, null, 0));
  const scopeCount = Object.keys(overlay).length;
  const rowCount = Object.values(overlay).reduce((n, m) => n + Object.keys(m).length, 0);
  console.log(`wrote ${rowCount} rows across ${scopeCount} scopes to ${outPath}`);
  process.exit(0);
}

main().catch((err) => {
  console.error(`stage-a-export failed: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
