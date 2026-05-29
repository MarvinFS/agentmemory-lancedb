# LanceDB-backed hybrid index + Adaptive Knowledge Lifecycle

Date: 2026-05-29
Status: Approved design, implementation in progress
Repo: private fork `agentmemory-lancedb` of `rohitg00/agentmemory` (Apache-2.0)
Branch: `feat/lancedb-hybrid-store`

## Problem

agentmemory stores its BM25 lexical index and embedding vectors as serialized blobs inside the iii
engine KV store (scope `mem:index:bm25`, keys `data` and `vectors`). Two stacked defects degrade
semantic recall to keyword-only:

1. The iii engine is a separate long-lived process holding the index in memory, so on every worker
   restart `bm25.size === 0` is never true and the only corpus re-embed path (`rebuildIndex`) never
   fires. After switching embeddings from Gemini 768d to Voyage `voyage-4-large` 1024d, the old
   vectors were dropped on the dimension check and the corpus was never re-embedded at 1024d.
2. When a rebuild is forced, `IndexPersistence.save()` writes the whole vector index as one ~27MB
   `state::set` value, exceeding the 180,000ms invocation timeout (a limit agentmemory sets on its own
   trigger calls in `src/index.ts` + `iii-config.yaml`; iii's default is 30s). Since #204 this fails
   silently. The BM25 `data` key is still a single ~11.5MB write - the same bug, latent.

The production workaround is a fragile string-replacement patch that chunks the vectors into <6MB
pieces, re-applied nightly. We are replacing it.

## Approach

Stop routing the index through iii KV. Give both the vectors and the BM25 text to LanceDB, which owns
its own files on disk (append-only Lance fragments + manifest, native ANN, native Tantivy full-text
search, `mergeInsert` upserts). This eliminates both monolithic-blob failures and, as a bonus,
unblocks us without the iii engine bump past 0.11.2 that upstream's own SQLite migration (#309) waits
on - we simply leave the KV.

The vector store is pluggable via the same `VECTOR_BACKEND` Strategy seam upstream PR #300 introduces,
so sqlite-vec (reserved) or a future SQL/iii-database backend drop in without touching fusion, MCP, or
REST. Memory also gains importance/recency/maturity-aware ranking (Adaptive Knowledge Lifecycle).

The architectural invariant: a self-persisting store MUST own its own files and MUST NOT be
re-serialized back through iii `state::set`, or the 180s timeout returns.

## Interfaces

We adopt PR #300's `VectorBackend` and extend it with `persistsExternally`. We add a sibling
`LexicalStore` for BM25. The LanceDB backend implements both over one table.

```ts
// src/state/vector-index.ts (from PR #300, extended)
interface VectorBackend {
  add(obsId, sessionId, embedding: Float32Array): Promise<void>;
  remove(obsId): Promise<void>;
  search(query: Float32Array, limit?): Promise<{obsId; sessionId; score}[]>;
  readonly size: number;                 // cached counter (sync) - lancedb keeps a live count
  clear(): Promise<void>;
  restoreFrom(serializedJson: string): Promise<void>;
  serialize(): Promise<string>;          // no-op ("" ) when persistsExternally
  validateDimensions(expected): Promise<{mismatches; seenDimensions}>;
  readonly persistsExternally: boolean;  // lancedb=true -> IndexPersistence skips KV
}

// src/state/lexical-store.ts (new)
interface LexicalStore {
  add(obsId, sessionId, text): Promise<void>;
  remove(obsId): Promise<void>;
  search(query: string, limit?): Promise<{obsId; sessionId; score}[]>;
  readonly size: number;
  clear(): Promise<void>;
  readonly persistsExternally: boolean;
}
```

Backend selection (reuse PR #300's env `VECTOR_BACKEND`): `memory` (default; existing in-memory
`SearchIndex` + `MemoryVectorIndex`; keeps standalone/tests working) | `lancedb` (production) |
`sqlite-vec` (reserved, PR #300) | `iii` (reserved, future SQL via iii database worker). When
`lancedb` is selected it owns BOTH legs over one table; otherwise BM25 stays the in-memory
`SearchIndex`.

Keeping `size` a synchronous getter (cached counter in the LanceDB backend) avoids async-widening the
two hot sync reads at `hybrid-search.ts:91` and `index.ts:393`.

## LanceDB table `memories` (at `${dataDir}/lancedb`, filesystem - never iii KV)

| column | type | purpose |
|--------|------|---------|
| `id` | utf8 (PK) | obsId; `mergeInsert` key for per-row upsert |
| `session_id` | utf8 | leg tuple + session diversification |
| `text` | utf8 | source text; native FTS (Tantivy BM25) index |
| `vector` | fixed_size_list<float32>[dim] | ANN index (IVF_PQ / HNSW) |
| `importance` | float32 (0-100) | lifecycle |
| `recency` | float32 (0-1) | lifecycle |
| `access_count` | int32 | lifecycle |
| `update_count` | int32 | lifecycle |
| `maturity` | utf8 (draft/validated/core) | lifecycle |
| `created_at` | int64 (epoch ms) | lifecycle |
| `updated_at` | int64 (epoch ms) | lifecycle decay anchor |

Vector ANN index created once the table passes a row threshold; FTS index on `text`. Both queries hit
this single table.

## Data flow (recall)

`HybridSearch.tripleStreamSearch` -> embed query (Voyage) -> `vectorStore.search()` +
`lexicalStore.search()` (two queries on the LanceDB table) + graph leg (iii KV, unchanged) -> RRF
fuse (k=60, unchanged) -> **lifecycle re-rank** (compound score x maturity tier boost) -> session
diversification -> KV enrich -> optional rerank. Access bumps accumulate in memory during the query
and flush once as a batched `mergeInsert`. RRF consumes rank, so LanceDB's distance ranking drops in
unchanged.

## Persistence rule (the fix)

When the active backend `persistsExternally === true`, `IndexPersistence.save()/load()` become no-ops
for BOTH `data` and `vectors` - LanceDB writes to disk via its own native I/O. iii KV retains only the
knowledge graph and memory bodies. The DROP_STALE / dimension guard at `index.ts:393-442` is preserved
by querying the LanceDB table's vector dimension via `validateDimensions()`.

## Adaptive Knowledge Lifecycle (clean-room; byterover is Elastic-2.0, NO code copy)

Pure functions in `src/state/lifecycle-scoring.ts`:
- Compound: `base = 0.6*rrfNorm + 0.2*(importance/100) + 0.2*recency`, then `* tierBoost`
  (core 1.15, validated 1.0, draft 0.85).
- Lazy decay at read: `recency' = exp(-days/30)`, `importance' = importance * 0.995^days`.
- Reinforcement: search hit +3 importance; save/update +5 and reset recency=1; cap 100.
- Maturity hysteresis: promote draft->validated >=65, validated->core >=85; demote core->validated <60,
  validated->draft <35.
- Deterministic GC (`src/functions/lifecycle-gc.ts`): flag `importance<35` OR stale-for-tier; never
  auto-delete `core`; emit candidates for review.

## Build / packaging

`@lancedb/lancedb@^0.30.0` + `apache-arrow@18.1.0` (pinned to LanceDB 0.30's peer ceiling
`>=15.0.0 <=18.1.0`) added to `optionalDependencies` and to `tsdown.config.ts` `external[]` (native
`.node` binaries must not be bundled - same treatment as onnxruntime). Lazy-loaded only when the
lancedb backend is active.

## Cutover (hard, after validation gate)

Flag-validate on the VM, then hard cutover: lancedb becomes the default, the chunk-persist patch is
removed from `agentmemory-autoupdate.sh`, and the stale `mem:index:bm25` keys are deleted. Pre-cutover
backup of the data directory (`~/data/`) is the rollback.

## Verification

`npm test` green on the Linux VM build (native binary loads on Debian x86_64); a write -> restart loop
shows "Loaded persisted vector index (N)" with rebuild-count 0; semantic-paraphrase queries with low
keyword overlap return the right memory; no `Invocation timeout after 180000ms: state::set` in logs;
`du -sh .../lancedb` grows incrementally; lifecycle raises importance/maturity on recall and surfaces
stale low-importance items as GC candidates while never flagging `core`.

## Out of scope

Graph migration (#309 graph half), sqlite-vec implementation (reserve key only), iii engine upgrade /
iii-database adoption, standalone MCP backend swap, AUTO_COMPRESS cost tuning.
