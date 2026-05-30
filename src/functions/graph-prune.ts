import type { ISdk } from "iii-sdk";
import type { StateKV } from "../state/kv.js";
import { KV } from "../state/schema.js";
import type {
  Session,
  Memory,
  CompressedObservation,
  GraphNode,
  GraphEdge,
} from "../types.js";
import { recordAudit } from "./audit.js";
import { logger } from "../logger.js";

// Hard-delete graph nodes/edges that mem::cascade-update only flagged
// stale. cascade-update sets node.stale/edge.stale=true when a
// superseded memory's source observations overlap, but never removes
// the rows — so after a bulk source-observation delete, orphaned graph
// rows accumulate with no live backing source. This prunes them.
//
// Liveness is resolved once up front into two id Sets:
//   - live observation ids: union of every session's observation scope
//     (KV.observations is sharded per-session, so we list each session
//     scope exactly once — O(sessions) list calls, not O(nodes)).
//   - live memory ids: all rows in KV.memories.
// Orphan detection is then O(1) per node/edge against those Sets.
//
// A node/edge's sourceObservationIds carry no session hint, so a
// sourceId counts as "live" if it resolves to EITHER a live observation
// (any session) OR a live memory. This is deliberately permissive on
// the liveness side to stay conservative on the delete side.

export function registerGraphPruneFunction(sdk: ISdk, kv: StateKV): void {
  sdk.registerFunction("mem::graph-prune",
    async (data: { dryRun?: boolean }) => {
      const dryRun = data?.dryRun === true;

      // Build the live-source id set once. Observations are sharded by
      // session scope, so list each session's scope a single time.
      const liveSourceIds = new Set<string>();

      const sessions = await kv.list<Session>(KV.sessions);
      for (const session of sessions) {
        const obs = await kv
          .list<CompressedObservation>(KV.observations(session.id))
          .catch(() => []);
        for (const o of obs) {
          if (o.id) liveSourceIds.add(o.id);
        }
      }

      const memories = await kv.list<Memory>(KV.memories);
      for (const m of memories) {
        if (m.id) liveSourceIds.add(m.id);
      }

      const nodes = await kv.list<GraphNode>(KV.graphNodes);
      const edges = await kv.list<GraphEdge>(KV.graphEdges);

      // A node is orphaned iff it is stale AND none of its source ids
      // still resolve to a live observation/memory. Never touch a node
      // that still has any live source, even if flagged stale.
      const hasLiveSource = (sourceIds: string[] | undefined): boolean =>
        (sourceIds ?? []).some((id) => liveSourceIds.has(id));

      const orphanNodes = nodes.filter(
        (n) => n.stale === true && !hasLiveSource(n.sourceObservationIds),
      );
      const orphanNodeIds = new Set(orphanNodes.map((n) => n.id));

      // After node deletion the surviving node id set is everything not
      // pruned. An edge is orphaned iff:
      //   (a) it is stale AND has no live source, OR
      //   (b) either endpoint no longer exists (already gone, or pruned
      //       in this pass).
      const survivingNodeIds = new Set(
        nodes.filter((n) => !orphanNodeIds.has(n.id)).map((n) => n.id),
      );

      const orphanEdges = edges.filter((e) => {
        const danglingEndpoint =
          !survivingNodeIds.has(e.sourceNodeId) ||
          !survivingNodeIds.has(e.targetNodeId);
        const staleOrphan =
          e.stale === true && !hasLiveSource(e.sourceObservationIds);
        return danglingEndpoint || staleOrphan;
      });

      if (dryRun) {
        return {
          dryRun: true,
          wouldDeleteNodes: orphanNodes.length,
          wouldDeleteEdges: orphanEdges.length,
        };
      }

      let deletedNodes = 0;
      let deletedEdges = 0;

      for (const node of orphanNodes) {
        try {
          await kv.delete(KV.graphNodes, node.id);
          deletedNodes++;
        } catch (err) {
          logger.warn("graph-prune node delete failed", {
            id: node.id,
            error: err instanceof Error ? err.message : String(err),
          });
        }
      }

      for (const edge of orphanEdges) {
        try {
          await kv.delete(KV.graphEdges, edge.id);
          deletedEdges++;
        } catch (err) {
          logger.warn("graph-prune edge delete failed", {
            id: edge.id,
            error: err instanceof Error ? err.message : String(err),
          });
        }
      }

      // Bulk-batched audit row per invocation (one for nodes, one for
      // edges), per the audit coverage policy for sweeps in audit.ts.
      if (deletedNodes > 0) {
        await recordAudit(
          kv,
          "delete",
          "mem::graph-prune",
          orphanNodes.map((n) => n.id),
          {
            resourceType: "GraphNode",
            reason: "orphaned_stale_node_no_live_source",
            deleted: deletedNodes,
          },
        );
      }
      if (deletedEdges > 0) {
        await recordAudit(
          kv,
          "delete",
          "mem::graph-prune",
          orphanEdges.map((e) => e.id),
          {
            resourceType: "GraphEdge",
            reason: "orphaned_edge_stale_or_dangling_endpoint",
            deleted: deletedEdges,
          },
        );
      }

      logger.info("Graph prune complete", {
        deletedNodes,
        deletedEdges,
      });

      return {
        dryRun: false,
        deletedNodes,
        deletedEdges,
      };
    },
  );
}
