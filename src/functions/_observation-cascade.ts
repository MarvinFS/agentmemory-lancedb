import type { ISdk } from "iii-sdk";
import type { StateKV } from "../state/kv.js";
import { recordAudit } from "./audit.js";
import { getSearchIndex, vectorIndexRemove } from "./search.js";
import { decrementImageRef } from "./image-refs.js";

// Shared observation-deletion cascade. The five hand-rolled call sites
// (evict low-importance + cap, observation-prune, auto-forget low-value, and
// the two mem::forget observation branches) all run the same set of secondary
// removals AFTER the row itself is deleted from KV. The kv.delete stays at the
// call site because each site couples it to its own control flow (try/continue,
// deletedOk gating, stat increments); this helper only runs the post-delete
// cascade so that control flow is preserved byte-for-byte.
//
// Three axes differ per site and are parameterized via the options arg:
//   (a) the audit reason/action string and its metadata, so each site keeps
//       its EXACT current audit semantics; an omitted `audit` skips the
//       per-id recordAudit entirely (mem::forget batches a single audit row
//       across all removed ids at the end of the call, so its observation
//       branches must NOT emit per-id rows);
//   (b) whether the BM25/vector index removals run at all — the evict path
//       historically does not touch the search/vector index, so removeFromIndex
//       is left false there; and
//   (c) the image-ref decrement, which is data-driven (imageData/imageRef) and
//       guarded identically at every site (`imageRef && imageRef !== imageData`),
//       so it lives here unconditionally.

export interface ObservationCascadeAudit {
  // Maps to recordAudit(kv, operation, functionId, [obsId], details).
  operation: Parameters<typeof recordAudit>[1];
  functionId: string;
  details: Record<string, unknown>;
}

export interface DeleteObservationCascadeOptions {
  obsId: string;
  imageData?: string;
  imageRef?: string;
  // When true, also drop the row from the BM25 and vector indexes. The evict
  // path leaves this false to match its current (index-untouched) behavior.
  removeFromIndex: boolean;
  // Omit to skip the per-id audit (mem::forget audits in one batch at the end).
  audit?: ObservationCascadeAudit;
}

export async function deleteObservationCascade(
  kv: StateKV,
  sdk: ISdk,
  opts: DeleteObservationCascadeOptions,
): Promise<void> {
  if (opts.removeFromIndex) {
    getSearchIndex().remove(opts.obsId);
    await vectorIndexRemove(opts.obsId);
  }
  if (opts.imageData) await decrementImageRef(kv, sdk, opts.imageData);
  if (opts.imageRef && opts.imageRef !== opts.imageData) {
    await decrementImageRef(kv, sdk, opts.imageRef);
  }
  if (opts.audit) {
    await recordAudit(kv, opts.audit.operation, opts.audit.functionId, [
      opts.obsId,
    ], opts.audit.details);
  }
}
