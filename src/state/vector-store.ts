import type { StateKV } from "./kv.js";
import { VectorIndex, MemoryVectorIndex, type VectorBackend } from "./vector-index.js";
import { KvIndexBlobStore, type IndexBlobStore } from "./index-blob-store.js";
import type { GraphKvStore } from "./graph-kv-router.js";

// Pluggable vector-store selector (upstream PR #300's VECTOR_BACKEND env).
//   memory     - in-memory Map + brute-force cosine (default; tests + standalone)
//   lancedb    - on-disk LanceDB (ANN vectors per-row + BM25 blob), self-persisting
//   sqlite-vec - reserved (upstream PR #300); not implemented in this build
//   iii        - reserved (future SQL via the iii database worker)
export type VectorBackendKind = "memory" | "lancedb" | "sqlite-vec" | "iii";

export interface PersistenceBackends {
  vector: VectorIndex;
  blobStore: IndexBlobStore;
  // Present only when the backend owns its own files (lancedb): the knowledge
  // graph's KV scopes are routed here instead of the iii engine KV. Undefined
  // for the memory backend, where the graph stays in iii KV.
  graphKv?: GraphKvStore;
}

export interface CreateBackendsOptions {
  kv: StateKV;
  dataDir: string;
  dimensions: number;
  backend: VectorBackendKind;
}

// Builds the vector index AND the BM25 blob store together, because a
// self-persisting backend (lancedb) shares one on-disk connection across
// both. Called only when an embedding provider is configured; with no
// provider there are no vectors and BM25 stays in the iii KV
// (KvIndexBlobStore), wired directly in src/index.ts.
export async function createPersistenceBackends(
  opts: CreateBackendsOptions,
): Promise<PersistenceBackends> {
  switch (opts.backend) {
    case "lancedb": {
      // Lazy import: @lancedb/lancedb is an optionalDependency carrying a
      // platform-specific native binary. Only load it when selected so
      // the default `memory` path never touches it. createLanceBackends
      // is responsible for awaiting its own init() (opening the table and
      // caching the row count so VectorIndex.size stays synchronous).
      const { createLanceBackends } = await import("./stores/lancedb-store.js");
      return createLanceBackends(opts);
    }
    case "sqlite-vec":
    case "iii":
      throw new Error(
        `[agentmemory] VECTOR_BACKEND="${opts.backend}" is reserved but not ` +
          `implemented in this build. Use "lancedb" or "memory".`,
      );
    case "memory":
    default: {
      const backend: VectorBackend = new MemoryVectorIndex();
      if (backend.init) await backend.init();
      return {
        vector: new VectorIndex(backend),
        blobStore: new KvIndexBlobStore(opts.kv),
      };
    }
  }
}
