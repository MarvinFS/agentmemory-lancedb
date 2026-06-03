// Persistence target for serialized search-index blobs: the BM25 "data"
// key, and the legacy "vectors" key for backends that do NOT persist
// themselves. Abstracted so the lancedb backend can keep these blobs in
// its own on-disk store instead of the iii KV — escaping the ~27MB /
// 180,000ms state::set invocation timeout that motivated this change.
// Only lancedb implements it (LanceIndexBlobStore); memory/iii use the
// upstream sharded IndexPersistence path instead.
export interface IndexBlobStore {
  set(key: string, value: string): Promise<void>;
  get(key: string): Promise<string | null>;
}
