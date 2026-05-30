import { SearchIndex } from "./search-index.js";
import { VectorIndex } from "./vector-index.js";
import type { IndexBlobStore } from "./index-blob-store.js";
import { logger } from "../logger.js";

const DEBOUNCE_MS = 5000;
const FAILURE_LOG_THROTTLE_MS = 60_000;

export class IndexPersistence {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private lastFailureLogAt = 0;

  constructor(
    private blobStore: IndexBlobStore,
    private bm25: SearchIndex,
    private vector: VectorIndex | null,
  ) {}

  scheduleSave(): void {
    if (this.timer) clearTimeout(this.timer);
    // setTimeout discards the returned promise, so any rejection inside
    // save() would surface as unhandledRejection and crash the process
    // under sustained write timeouts (issue #204). Funnel rejections
    // through logFailure() instead.
    this.timer = setTimeout(() => {
      this.save().catch((err) => this.logFailure(err));
    }, DEBOUNCE_MS);
  }

  async save(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    try {
      // BM25 index is always a serialized blob. With the default (memory)
      // backend this lands in the iii KV; with lancedb it lands in a
      // LanceDB blob table — no 180s state::set ceiling either way.
      await this.blobStore.set("data", this.bm25.serialize());
      // Self-persisting backends (lancedb) own their vectors on disk
      // per-row; do NOT serialize the whole index back through the blob
      // store — that monolithic write is exactly the failure this change
      // eliminates. Only the in-memory backend round-trips vectors.
      if (
        this.vector &&
        !this.vector.persistsExternally &&
        this.vector.size > 0
      ) {
        await this.blobStore.set("vectors", await this.vector.serialize());
      }
    } catch (err) {
      this.logFailure(err);
    }
  }

  async load(): Promise<{
    bm25: SearchIndex | null;
    vectorJson: string | null;
  }> {
    let bm25: SearchIndex | null = null;

    const bm25Data = await this.blobStore.get("data");
    if (bm25Data) {
      bm25 = SearchIndex.deserialize(bm25Data);
    }

    // For self-persisting backends the live vector index already holds its
    // data (opened from disk at construction), so there is nothing to
    // restore from a blob. Only fetch the legacy "vectors" blob for the
    // in-memory backend.
    let vectorJson: string | null = null;
    if (this.vector && !this.vector.persistsExternally) {
      vectorJson = await this.blobStore.get("vectors");
    }

    return { bm25, vectorJson };
  }

  stop(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private logFailure(err: unknown): void {
    const now = Date.now();
    // Throttle: persistence failures under load arrive in bursts.
    if (now - this.lastFailureLogAt < FAILURE_LOG_THROTTLE_MS) return;
    this.lastFailureLogAt = now;
    const code = (err as { code?: string })?.code;
    const message = err instanceof Error ? err.message : String(err);
    logger.warn("index persistence: failed to save BM25/vector index", {
      code,
      message,
      hint:
        code === "TIMEOUT"
          ? "index store write timed out; recent index updates remain in memory and will retry on the next debounce flush"
          : undefined,
    });
  }
}
