import type { StateKV } from "./kv.js";
import { KV } from "./schema.js";

// Persistence target for serialized search-index blobs: the BM25 "data"
// key, and the legacy "vectors" key for backends that do NOT persist
// themselves. Abstracted so the lancedb backend can keep these blobs in
// its own on-disk store instead of the iii KV — escaping the ~27MB /
// 180,000ms state::set invocation timeout that motivated this change.
export interface IndexBlobStore {
  set(key: string, value: string): Promise<void>;
  get(key: string): Promise<string | null>;
}

// Default: store blobs in the iii KV under scope mem:index:bm25, keys
// "data" / "vectors". Byte-for-byte identical to the pre-existing
// behavior, so the `memory` backend and tests are unaffected.
export class KvIndexBlobStore implements IndexBlobStore {
  constructor(private kv: StateKV) {}

  async set(key: string, value: string): Promise<void> {
    await this.kv.set(KV.bm25Index, key, value);
  }

  async get(key: string): Promise<string | null> {
    const v = await this.kv.get<string>(KV.bm25Index, key).catch(() => null);
    return typeof v === "string" ? v : null;
  }
}
