// Graph-scope KV router.
//
// The knowledge graph is stored as plain KV data under three scopes
// (mem:graph:nodes, mem:graph:edges, mem:graph:edge-history) and accessed by
// ~12 call sites using only get/set/delete/list - the key/value subset, no
// state::update. That makes the graph cleanly relocatable WITHOUT touching any
// of those consumers: this subclass of StateKV intercepts the three graph
// scopes and routes them to a backend that owns its own files (LanceDB), while
// every other scope falls through to the iii engine's KV (state::*) unchanged.
//
// This is the same "give the index its own store instead of round-tripping a
// blob through the engine KV" move the vector + BM25 legs already made; the
// graph is the last index-like dataset still living in the iii KV.
import type { ISdk } from "iii-sdk";
import { StateKV } from "./kv.js";
import { KV } from "./schema.js";

// Minimal key/value contract the routed backend must satisfy. Matches the
// StateKV method shapes for exactly the operations the graph code uses.
export interface GraphKvStore {
  get<T = unknown>(scope: string, key: string): Promise<T | null>;
  set<T = unknown>(scope: string, key: string, value: T): Promise<T>;
  delete(scope: string, key: string): Promise<void>;
  list<T = unknown>(scope: string): Promise<T[]>;
  optimize?(): Promise<void>;
}

// The scopes that move to the routed backend. Everything else stays in iii KV.
export const GRAPH_SCOPES: ReadonlySet<string> = new Set<string>([
  KV.graphNodes,
  KV.graphEdges,
  KV.graphEdgeHistory,
]);

export class GraphRoutingKV extends StateKV {
  constructor(
    sdk: ISdk,
    private readonly graph: GraphKvStore,
  ) {
    super(sdk);
  }

  async get<T = unknown>(scope: string, key: string): Promise<T | null> {
    if (GRAPH_SCOPES.has(scope)) return this.graph.get<T>(scope, key);
    return super.get<T>(scope, key);
  }

  async set<T = unknown>(scope: string, key: string, value: T): Promise<T> {
    if (GRAPH_SCOPES.has(scope)) return this.graph.set<T>(scope, key, value);
    return super.set<T>(scope, key, value);
  }

  async delete(scope: string, key: string): Promise<void> {
    if (GRAPH_SCOPES.has(scope)) return this.graph.delete(scope, key);
    return super.delete(scope, key);
  }

  async list<T = unknown>(scope: string): Promise<T[]> {
    if (GRAPH_SCOPES.has(scope)) return this.graph.list<T>(scope);
    return super.list<T>(scope);
  }
}

// One-time migration of the existing graph from the iii KV into the routed
// backend. Idempotent: if the backend already holds graph rows it is a no-op,
// so a normal restart never re-copies. Reads come from `source` (the raw
// StateKV -> iii KV) because the routing KV would read the still-empty backend.
// Every graph record (node/edge/history) is keyed by its own `.id`.
export async function backfillGraphIfEmpty(
  source: StateKV,
  graph: GraphKvStore,
): Promise<{
  migrated: boolean;
  nodes: number;
  edges: number;
  history: number;
}> {
  const [haveNodes, haveEdges] = await Promise.all([
    graph.list(KV.graphNodes),
    graph.list(KV.graphEdges),
  ]);
  if (haveNodes.length > 0 || haveEdges.length > 0) {
    return {
      migrated: false,
      nodes: haveNodes.length,
      edges: haveEdges.length,
      history: 0,
    };
  }

  const counts = { nodes: 0, edges: 0, history: 0 };
  const scopes: Array<[string, keyof typeof counts]> = [
    [KV.graphNodes, "nodes"],
    [KV.graphEdges, "edges"],
    [KV.graphEdgeHistory, "history"],
  ];
  for (const [scope, bucket] of scopes) {
    const items = await source
      .list<{ id?: string }>(scope)
      .catch(() => [] as Array<{ id?: string }>);
    for (const item of items) {
      if (item && typeof item.id === "string" && item.id.length > 0) {
        await graph.set(scope, item.id, item);
        counts[bucket]++;
      }
    }
  }
  // Each set above created its own on-disk version; compact once so the freshly
  // migrated table starts tight rather than at one fragment per row.
  await graph.optimize?.();

  return { migrated: true, ...counts };
}
