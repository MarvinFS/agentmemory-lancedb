import type {
  CreateBackendsOptions,
  PersistenceBackends,
} from "../vector-store.js";

// Stub — implemented in WS C. Kept as a real, type-correct module so the
// factory's dynamic import resolves and `npm run build` stays green
// before the LanceDB backend lands. The real implementation returns a
// VectorIndex wrapping a LanceVectorBackend (per-row ANN, persistsExternally
// = true) plus a LanceIndexBlobStore for the BM25 blob, both sharing one
// LanceDB connection under `${dataDir}/lancedb`.
export async function createLanceBackends(
  _opts: CreateBackendsOptions,
): Promise<PersistenceBackends> {
  throw new Error(
    "[agentmemory] LanceDB backend not yet implemented (WS C). " +
      "Set VECTOR_BACKEND=memory.",
  );
}
