// Pass byteOffset + byteLength explicitly so the round-trip survives
// Node's Buffer pool. Buffer.from(b64, "base64") returns a slice of a
// shared 8KB pool (poolSize), and `new Float32Array(buf.buffer)` ignores
// the slice metadata — it would mint a 2048-element view over the whole
// pool. Same risk on the encode side if the input Float32Array is itself
// a sliced view. Reported as a phantom "2048 dimensions on disk" crash
// in #455 / #469 / #584 / #587.
function float32ToBase64(arr: Float32Array): string {
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString(
    "base64",
  );
}

function base64ToFloat32(b64: string): Float32Array {
  const buf = Buffer.from(b64, "base64");
  return new Float32Array(
    buf.buffer,
    buf.byteOffset,
    buf.byteLength / Float32Array.BYTES_PER_ELEMENT,
  );
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export interface VectorSearchHit {
  obsId: string;
  sessionId: string;
  score: number;
}

export interface DimensionReport {
  mismatches: Array<{ obsId: string; dim: number }>;
  seenDimensions: Set<number>;
}

// Adaptive Knowledge Lifecycle fields. Only backends that materialize a
// per-row table (LanceDB) track these; the in-memory backend does not,
// so the lifecycle methods on VectorBackend are optional and the wrapper
// degrades to no-ops. See src/state/lifecycle-scoring.ts.
export interface LifecycleFields {
  importance: number; // 0-100
  recency: number; // 0-1
  accessCount: number;
  updateCount: number;
  maturity: string; // draft | validated | core
  createdAt: number; // epoch ms
  updatedAt: number; // epoch ms
}

// The pluggable vector-store contract. Aligned with the Strategy pattern
// from upstream PR #300 (VECTOR_BACKEND), extended with:
//   - persistsExternally: when true, the backend owns its own on-disk
//     files (e.g. LanceDB) and IndexPersistence must NOT serialize it
//     back through the iii KV store (that is the ~27MB / 180s-timeout
//     failure this whole change exists to kill). serialize()/restoreFrom()
//     become no-ops for such backends.
//   - optional lifecycle methods (getLifecycle/bumpAccess/setLifecycle)
//     for the Adaptive Knowledge Lifecycle re-rank + reinforcement.
//   - init(): one-time async open (LanceDB opens its table and caches the
//     row count so `size` can stay a synchronous getter).
export interface VectorBackend {
  init?(): Promise<void>;
  add(obsId: string, sessionId: string, embedding: Float32Array): Promise<void>;
  remove(obsId: string): Promise<void>;
  search(query: Float32Array, limit?: number): Promise<VectorSearchHit[]>;
  readonly size: number;
  clear(): Promise<void>;
  restoreFrom(serializedJson: string): Promise<void>;
  serialize(): Promise<string>;
  validateDimensions(expected: number): Promise<DimensionReport>;
  readonly persistsExternally: boolean;
  // Adaptive Knowledge Lifecycle (optional; no-op when unsupported).
  getLifecycle?(obsIds: string[]): Promise<Map<string, LifecycleFields>>;
  bumpAccess?(obsIds: string[]): Promise<void>;
  setLifecycle?(obsId: string, fields: Partial<LifecycleFields>): Promise<void>;
}

// Thin wrapper consumers hold. Delegates to a swappable backend and
// provides safe lifecycle defaults so call sites never branch on backend
// type. `size` and `persistsExternally` are synchronous passthroughs.
export class VectorIndex {
  constructor(private backend: VectorBackend) {}

  add(
    obsId: string,
    sessionId: string,
    embedding: Float32Array,
  ): Promise<void> {
    return this.backend.add(obsId, sessionId, embedding);
  }

  remove(obsId: string): Promise<void> {
    return this.backend.remove(obsId);
  }

  search(query: Float32Array, limit = 20): Promise<VectorSearchHit[]> {
    return this.backend.search(query, limit);
  }

  get size(): number {
    return this.backend.size;
  }

  get persistsExternally(): boolean {
    return this.backend.persistsExternally;
  }

  clear(): Promise<void> {
    return this.backend.clear();
  }

  restoreFrom(serializedJson: string): Promise<void> {
    return this.backend.restoreFrom(serializedJson);
  }

  serialize(): Promise<string> {
    return this.backend.serialize();
  }

  validateDimensions(expected: number): Promise<DimensionReport> {
    return this.backend.validateDimensions(expected);
  }

  getLifecycle(obsIds: string[]): Promise<Map<string, LifecycleFields>> {
    return this.backend.getLifecycle?.(obsIds) ?? Promise.resolve(new Map());
  }

  bumpAccess(obsIds: string[]): Promise<void> {
    return this.backend.bumpAccess?.(obsIds) ?? Promise.resolve();
  }

  setLifecycle(obsId: string, fields: Partial<LifecycleFields>): Promise<void> {
    return this.backend.setLifecycle?.(obsId, fields) ?? Promise.resolve();
  }
}

// Default in-memory backend: a flat Map with brute-force cosine search.
// Identical behavior to the pre-refactor VectorIndex. persistsExternally
// is false, so IndexPersistence serializes it to the iii KV store exactly
// as before (this preserves the `memory` backend used by tests and the
// standalone MCP path).
export class MemoryVectorIndex implements VectorBackend {
  private vectors: Map<string, { embedding: Float32Array; sessionId: string }> =
    new Map();

  readonly persistsExternally = false;

  async add(
    obsId: string,
    sessionId: string,
    embedding: Float32Array,
  ): Promise<void> {
    this.vectors.set(obsId, { embedding, sessionId });
  }

  async remove(obsId: string): Promise<void> {
    this.vectors.delete(obsId);
  }

  async search(query: Float32Array, limit = 20): Promise<VectorSearchHit[]> {
    const results: VectorSearchHit[] = [];
    let minScore = -Infinity;

    for (const [obsId, entry] of this.vectors) {
      const score = cosineSimilarity(query, entry.embedding);
      if (results.length < limit) {
        results.push({ obsId, sessionId: entry.sessionId, score });
        if (results.length === limit) {
          results.sort((a, b) => a.score - b.score);
          minScore = results[0].score;
        }
      } else if (score > minScore) {
        results[0] = { obsId, sessionId: entry.sessionId, score };
        results.sort((a, b) => a.score - b.score);
        minScore = results[0].score;
      }
    }

    results.sort((a, b) => b.score - a.score);
    return results;
  }

  get size(): number {
    return this.vectors.size;
  }

  async clear(): Promise<void> {
    this.vectors.clear();
  }

  // Walks every stored vector and returns the obsIds whose dimension
  // doesn't match `expected`, plus the set of distinct dimensions seen.
  // Used by the persistence-restore guard in src/index.ts to refuse
  // loading any index containing wrong-dimension vectors.
  async validateDimensions(expected: number): Promise<DimensionReport> {
    const mismatches: Array<{ obsId: string; dim: number }> = [];
    const seenDimensions = new Set<number>();
    for (const [obsId, entry] of this.vectors) {
      const dim = entry.embedding.length;
      seenDimensions.add(dim);
      if (dim !== expected) {
        mismatches.push({ obsId, dim });
      }
    }
    return { mismatches, seenDimensions };
  }

  async restoreFrom(serializedJson: string): Promise<void> {
    let data: unknown;
    try {
      data = JSON.parse(serializedJson);
    } catch {
      return;
    }
    if (!Array.isArray(data)) return;
    this.vectors.clear();
    for (const row of data) {
      try {
        if (!Array.isArray(row) || row.length < 2) continue;
        const [obsId, entry] = row;
        if (
          typeof obsId !== "string" ||
          typeof entry?.embedding !== "string" ||
          typeof entry?.sessionId !== "string"
        )
          continue;
        this.vectors.set(obsId, {
          embedding: base64ToFloat32(entry.embedding),
          sessionId: entry.sessionId,
        });
      } catch {
        continue;
      }
    }
  }

  async serialize(): Promise<string> {
    const data: Array<[string, { embedding: string; sessionId: string }]> = [];
    for (const [obsId, entry] of this.vectors) {
      data.push([
        obsId,
        {
          embedding: float32ToBase64(entry.embedding),
          sessionId: entry.sessionId,
        },
      ]);
    }
    return JSON.stringify(data);
  }
}
