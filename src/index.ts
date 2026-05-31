import { registerWorker } from "iii-sdk";
import {
  loadConfig,
  getEnvVar,
  loadEmbeddingConfig,
  loadFallbackConfig,
  loadClaudeBridgeConfig,
  loadTeamConfig,
  loadSnapshotConfig,
  isGraphExtractionEnabled,
  isAutoCompressEnabled,
  isConsolidationEnabled,
  isContextInjectionEnabled,
  isDropStaleIndexEnabled,
  getVectorBackendKind,
} from "./config.js";
import { loadSystemdCredentials } from "./load-credentials.js";
import {
  createProvider,
  createFallbackProvider,
  createEmbeddingProvider,
  createImageEmbeddingProvider,
} from "./providers/index.js";
import { StateKV } from "./state/kv.js";
import { KV } from "./state/schema.js";
import { VectorIndex } from "./state/vector-index.js";
import { createPersistenceBackends } from "./state/vector-store.js";
import {
  GraphRoutingKV,
  backfillGraphIfEmpty,
  type GraphKvStore,
} from "./state/graph-kv-router.js";
import {
  KvIndexBlobStore,
  type IndexBlobStore,
} from "./state/index-blob-store.js";
import { HybridSearch } from "./state/hybrid-search.js";
import { IndexPersistence } from "./state/index-persistence.js";
import { registerPrivacyFunction } from "./functions/privacy.js";
import { registerObserveFunction } from "./functions/observe.js";
import { registerImageQuotaCleanup } from "./functions/image-quota-cleanup.js";
import { registerVisionSearchFunctions } from "./functions/vision-search.js";
import { registerSlotsFunctions, isSlotsEnabled, isReflectEnabled } from "./functions/slots.js";
import { registerDiskSizeManager } from "./functions/disk-size-manager.js";
import { registerCompressFunction } from "./functions/compress.js";
import {
  registerSearchFunction,
  setHybridSearcher,
  rebuildIndex,
  getSearchIndex,
  setVectorIndex,
  setEmbeddingProvider,
  setIndexPersistence,
} from "./functions/search.js";
import { registerContextFunction } from "./functions/context.js";
import { registerSummarizeFunction } from "./functions/summarize.js";
import { registerMigrateFunction } from "./functions/migrate.js";
import { registerFileIndexFunction } from "./functions/file-index.js";
import { registerConsolidateFunction } from "./functions/consolidate.js";
import { registerPatternsFunction } from "./functions/patterns.js";
import { registerRememberFunction } from "./functions/remember.js";
import { registerLifecycleSweepFunction } from "./functions/lifecycle-sweep.js";
import { registerEvictFunction } from "./functions/evict.js";
import { registerRelationsFunction } from "./functions/relations.js";
import { registerTimelineFunction } from "./functions/timeline.js";
import { registerSmartSearchFunction } from "./functions/smart-search.js";
import { registerProfileFunction } from "./functions/profile.js";
import { registerAutoForgetFunction } from "./functions/auto-forget.js";
import { registerExportImportFunction } from "./functions/export-import.js";
import { registerEnrichFunction } from "./functions/enrich.js";
import { registerClaudeBridgeFunction } from "./functions/claude-bridge.js";
import { registerGraphFunction } from "./functions/graph.js";
import { registerConsolidationPipelineFunction } from "./functions/consolidation-pipeline.js";
import { registerTeamFunction } from "./functions/team.js";
import { registerGovernanceFunction } from "./functions/governance.js";
import { registerSnapshotFunction } from "./functions/snapshot.js";
import { registerActionsFunction } from "./functions/actions.js";
import { registerFrontierFunction } from "./functions/frontier.js";
import { registerLeasesFunction } from "./functions/leases.js";
import { registerRoutinesFunction } from "./functions/routines.js";
import { registerSignalsFunction } from "./functions/signals.js";
import { registerCheckpointsFunction } from "./functions/checkpoints.js";
import { registerFlowCompressFunction } from "./functions/flow-compress.js";
import { registerMeshFunction } from "./functions/mesh.js";
import { registerBranchAwareFunction } from "./functions/branch-aware.js";
import { registerSentinelsFunction } from "./functions/sentinels.js";
import { registerSketchesFunction } from "./functions/sketches.js";
import { registerCrystallizeFunction } from "./functions/crystallize.js";
import { registerDiagnosticsFunction } from "./functions/diagnostics.js";
import { registerFacetsFunction } from "./functions/facets.js";
import { registerVerifyFunction } from "./functions/verify.js";
import { registerCascadeFunction } from "./functions/cascade.js";
import { registerObservationPruneFunction } from "./functions/observation-prune.js";
import { registerGraphPruneFunction } from "./functions/graph-prune.js";
import { initLessonVectorStore } from "./state/lesson-vectors.js";
import { registerLessonsFunctions } from "./functions/lessons.js";
import { registerObsidianExportFunction } from "./functions/obsidian-export.js";
import { registerReflectFunctions } from "./functions/reflect.js";
import { registerWorkingMemoryFunctions } from "./functions/working-memory.js";
import { registerSkillExtractFunctions } from "./functions/skill-extract.js";
import { registerSlidingWindowFunction } from "./functions/sliding-window.js";
import { registerQueryExpansionFunction } from "./functions/query-expansion.js";
import { registerTemporalGraphFunctions } from "./functions/temporal-graph.js";
import { registerRetentionFunctions } from "./functions/retention.js";
import { registerCompressFileFunction } from "./functions/compress-file.js";
import { registerReplayFunctions } from "./functions/replay.js";
import { registerApiTriggers } from "./triggers/api.js";
import { registerEventTriggers } from "./triggers/events.js";
import { registerMcpEndpoints } from "./mcp/server.js";
import { getAllTools } from "./mcp/tools-registry.js";
import { startViewerServer } from "./viewer/server.js";
import { MetricsStore } from "./eval/metrics-store.js";
import { DedupMap } from "./functions/dedup.js";
import { registerHealthMonitor } from "./health/monitor.js";
import { initMetrics, OTEL_CONFIG } from "./telemetry/setup.js";
import { VERSION } from "./version.js";
import { bootLog } from "./logger.js";
import { mkdirSync, writeFileSync, unlinkSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";

// #640 + #474: the worker process (this file) is spawned by iii-exec
// inside the engine. When `agentmemory stop` kills only the engine pid,
// this worker can survive (detached spawn, signal not propagated, or a
// wrapper script keeps it running) and reconnects to the next engine as
// a duplicate worker. Write the worker pid alongside iii.pid so
// `agentmemory stop` can reap us too.
function workerPidfilePath(): string {
  return join(homedir(), ".agentmemory", "worker.pid");
}
function writeWorkerPidfile(): void {
  try {
    const p = workerPidfilePath();
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, `${process.pid}\n`, { encoding: "utf-8" });
  } catch {
    // best-effort; stop still has the engine pidfile + port scan fallback
  }
}
function clearWorkerPidfile(): void {
  try {
    unlinkSync(workerPidfilePath());
  } catch {}
}

function hasGetMeter(
  sdk: unknown,
): sdk is { getMeter: (name: string) => unknown } {
  return (
    typeof sdk === "object" &&
    sdk !== null &&
    "getMeter" in sdk &&
    typeof (sdk as { getMeter?: unknown }).getMeter === "function"
  );
}

// Top-level safety net for iii-engine invocation timeouts (issue #204).
// Under sustained write load (e.g. Claude Code hooks across many
// projects) `state::set` can occasionally exceed the SDK's 30s timeout.
// We don't want one such timeout to terminate the long-lived memory
// service — the rejection is surfaced to the relevant call site via
// .catch() where it matters; everything else is logged-and-continued.
// Throttle logs to avoid spamming on bursts.
let lastUnhandledLogAt = 0;
process.on("unhandledRejection", (reason) => {
  const now = Date.now();
  if (now - lastUnhandledLogAt < 60_000) return;
  lastUnhandledLogAt = now;
  const r = reason as { code?: string; function_id?: string; message?: string };
  console.warn(
    `[agentmemory] unhandledRejection (suppressed):`,
    r?.code ? `${r.code} ${r.function_id ?? ""} ${r.message ?? ""}`.trim() : reason,
  );
});

async function main() {
  // Pull any systemd-delivered secrets ($CREDENTIALS_DIRECTORY) into the
  // environment before config/providers read keys. Runs here in the worker
  // (whether in-process under cli.mjs or as the iii-exec'd dist/index.mjs)
  // so the values populate this process's env at runtime without ever
  // entering the exec environ block / /proc/<pid>/environ.
  loadSystemdCredentials();
  const config = loadConfig();
  const embeddingConfig = loadEmbeddingConfig();
  const fallbackConfig = loadFallbackConfig();

  const provider =
    fallbackConfig.providers.length > 0
      ? createFallbackProvider(config.provider, fallbackConfig)
      : createProvider(config.provider);

  const embeddingProvider = createEmbeddingProvider();
  const imageEmbeddingProvider = createImageEmbeddingProvider();

  bootLog(`Starting worker v${VERSION}...`);
  bootLog(`Engine: ${config.engineUrl}`);
  bootLog(
    `Provider: ${config.provider.provider} (${config.provider.model})`,
  );
  if (embeddingProvider) {
    bootLog(
      `Embedding provider: ${embeddingProvider.name} (${embeddingProvider.dimensions} dims)`,
    );
  } else {
    bootLog(`Embedding provider: none (BM25-only mode)`);
  }
  if (imageEmbeddingProvider) {
    bootLog(
      `Image embedding provider: ${imageEmbeddingProvider.name} (${imageEmbeddingProvider.dimensions} dims) — vision-search active`,
    );
  }
  bootLog(
    `REST API: http://localhost:${config.restPort}/agentmemory/*`,
  );
  bootLog(`Streams: ws://localhost:${config.streamsPort}`);

  const sdk = registerWorker(config.engineUrl, {
    workerName: "agentmemory",
    invocationTimeoutMs: 180000,
    otel: {
      serviceName: OTEL_CONFIG.serviceName,
      serviceVersion: OTEL_CONFIG.serviceVersion,
      metricsExportIntervalMs: OTEL_CONFIG.metricsExportIntervalMs,
    },
    // Explicit worker telemetry metadata. iii-sdk falls back to
    // auto-detection (cwd / package.json name / hostname) when this
    // is omitted, which produces inconsistent values per host —
    // `agentmemory`, `node`, `npm`, occasionally the user's home
    // directory basename. Pinning the value here gives every install
    // the same stable project identifier for downstream attribution
    // and grouping in the engine's metrics + traces output.
    telemetry: {
      project_name: "agentmemory",
      language: "node",
      framework: "iii-sdk",
    },
  });

  writeWorkerPidfile();

  const baseKv = new StateKV(sdk);
  const secret = getEnvVar("AGENTMEMORY_SECRET");
  const dedupMap = new DedupMap();

  // Vector + BM25 persistence backends. With an embedding provider the
  // selected VECTOR_BACKEND owns the vector index (and, for lancedb, the
  // BM25 blob + graph KV too); without one there are no vectors and BM25 stays
  // in the iii KV. createPersistenceBackends opens lancedb's tables and caches
  // its row count before returning, so VectorIndex.size is valid at boot below.
  let vectorIndex: VectorIndex | null = null;
  let indexBlobStore: IndexBlobStore;
  let graphKv: GraphKvStore | undefined;
  if (embeddingProvider) {
    const backends = await createPersistenceBackends({
      kv: baseKv,
      dataDir: config.dataDir,
      dimensions: embeddingProvider.dimensions,
      backend: getVectorBackendKind(),
    });
    vectorIndex = backends.vector;
    indexBlobStore = backends.blobStore;
    graphKv = backends.graphKv;
  } else {
    indexBlobStore = new KvIndexBlobStore(baseKv);
  }

  // When a self-persisting backend owns its files (lancedb), route the graph's
  // KV scopes to it - out of the iii KV - and migrate the existing graph once.
  // Otherwise the graph stays in the iii KV via the plain StateKV. Every other
  // scope always falls through to the iii KV regardless.
  const kv = graphKv ? new GraphRoutingKV(sdk, graphKv) : baseKv;
  const metricsStore = new MetricsStore(kv);
  if (graphKv) {
    try {
      const r = await backfillGraphIfEmpty(baseKv, graphKv);
      bootLog(
        r.migrated
          ? `Graph backend: LanceDB (migrated ${r.nodes} nodes, ${r.edges} edges, ${r.history} history from iii KV)`
          : `Graph backend: LanceDB (already populated: ${r.nodes} nodes, ${r.edges} edges)`,
      );
    } catch (err) {
      bootLog(
        `Graph backfill skipped (error): ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  setVectorIndex(vectorIndex);
  setEmbeddingProvider(embeddingProvider);

  if (embeddingProvider) {
    await initLessonVectorStore({
      dataDir: config.dataDir,
      dimensions: embeddingProvider.dimensions,
      backend: getVectorBackendKind(),
    });
  }

  const meterAccessor = hasGetMeter(sdk)
    ? (sdk.getMeter.bind(sdk) as (name: string) => unknown)
    : undefined;

  initMetrics(meterAccessor as ((name: string) => import("@opentelemetry/api").Meter) | undefined);

  registerPrivacyFunction(sdk);
  registerObserveFunction(sdk, kv, dedupMap, config.maxObservationsPerSession);
  registerImageQuotaCleanup(sdk, kv);
  registerVisionSearchFunctions(sdk, kv, imageEmbeddingProvider);
  if (isSlotsEnabled()) {
    registerSlotsFunctions(sdk, kv);
  }
  registerDiskSizeManager(sdk, kv);
  registerCompressFunction(sdk, kv, provider, metricsStore);
  registerSearchFunction(sdk, kv);
  registerContextFunction(sdk, kv, config.tokenBudget);
  registerSummarizeFunction(sdk, kv, provider, metricsStore);
  registerMigrateFunction(sdk, kv);
  registerFileIndexFunction(sdk, kv);
  registerConsolidateFunction(sdk, kv, provider);
  registerPatternsFunction(sdk, kv);
  registerRememberFunction(sdk, kv);
  registerEvictFunction(sdk, kv);
  registerObservationPruneFunction(sdk, kv);

  registerRelationsFunction(sdk, kv);
  registerTimelineFunction(sdk, kv);
  registerProfileFunction(sdk, kv);
  registerAutoForgetFunction(sdk, kv);
  registerExportImportFunction(sdk, kv);
  registerEnrichFunction(sdk, kv);

  const claudeBridgeConfig = loadClaudeBridgeConfig();
  if (claudeBridgeConfig.enabled) {
    registerClaudeBridgeFunction(sdk, kv, claudeBridgeConfig);
    bootLog(
      `Claude bridge: syncing to ${claudeBridgeConfig.memoryFilePath}`,
    );
  }

  if (isGraphExtractionEnabled()) {
    registerGraphFunction(sdk, kv, provider);
    bootLog(`Knowledge graph: extraction enabled`);
  }

  registerConsolidationPipelineFunction(sdk, kv, provider);
  bootLog(`Consolidation pipeline: registered (CONSOLIDATION_ENABLED=${isConsolidationEnabled() ? "true" : "false"})`);

  if (isAutoCompressEnabled()) {
    bootLog(
      `WARNING: AGENTMEMORY_AUTO_COMPRESS=true — every PostToolUse observation will be sent to your LLM provider for compression. This spends API tokens proportional to your session tool-use frequency (see #138). Set AGENTMEMORY_AUTO_COMPRESS=false to disable.`,
    );
  } else {
    bootLog(
      `Auto-compress: OFF (default, #138) — observations indexed via zero-LLM synthetic compression. Set AGENTMEMORY_AUTO_COMPRESS=true to opt-in to LLM-powered summaries (uses your API key).`,
    );
  }

  if (isContextInjectionEnabled()) {
    bootLog(
      `WARNING: AGENTMEMORY_INJECT_CONTEXT=true — the PreToolUse and SessionStart hooks will inject up to ~4000 chars of memory context into every tool turn. On Claude Pro this burns session tokens proportional to your tool-call frequency (see #143). Set AGENTMEMORY_INJECT_CONTEXT=false to disable.`,
    );
  } else {
    bootLog(
      `Context injection: OFF (default, #143) — hooks capture observations but do not inject context into Claude Code's conversation. Set AGENTMEMORY_INJECT_CONTEXT=true to opt-in (warning: expect your Claude Pro allocation to drain faster).`,
    );
  }

  const teamConfig = loadTeamConfig();
  if (teamConfig) {
    registerTeamFunction(sdk, kv, teamConfig);
    bootLog(
      `Team memory: ${teamConfig.teamId} (${teamConfig.mode})`,
    );
  }

  registerGovernanceFunction(sdk, kv);

  registerActionsFunction(sdk, kv);
  registerFrontierFunction(sdk, kv);
  registerLeasesFunction(sdk, kv);
  registerRoutinesFunction(sdk, kv);
  registerSignalsFunction(sdk, kv);
  registerCheckpointsFunction(sdk, kv);
  registerMeshFunction(sdk, kv, secret);
  registerBranchAwareFunction(sdk, kv);
  registerFlowCompressFunction(sdk, kv, provider);
  registerSentinelsFunction(sdk, kv);
  registerSketchesFunction(sdk, kv);
  registerCrystallizeFunction(sdk, kv, provider);
  registerDiagnosticsFunction(sdk, kv);
  registerFacetsFunction(sdk, kv);
  registerVerifyFunction(sdk, kv);
  registerLessonsFunctions(sdk, kv);
  registerObsidianExportFunction(sdk, kv);
  registerReflectFunctions(sdk, kv, provider);
  registerWorkingMemoryFunctions(sdk, kv, config.tokenBudget);
  registerSkillExtractFunctions(sdk, kv, provider);
  registerCascadeFunction(sdk, kv);
  registerGraphPruneFunction(sdk, kv);

  registerSlidingWindowFunction(sdk, kv, provider);
  registerQueryExpansionFunction(sdk, provider);
  registerTemporalGraphFunctions(sdk, kv, provider);
  registerRetentionFunctions(sdk, kv);
  registerCompressFileFunction(sdk, kv, provider);
  registerReplayFunctions(sdk, kv);
  registerLifecycleSweepFunction(sdk);
  bootLog(
    `v0.6 advanced retrieval: sliding-window, query-expansion, temporal-graph, retention-scoring`,
  );
  bootLog(
    `Orchestration layer: actions, frontier, leases, routines, signals, checkpoints, flow-compress, mesh, branch-aware, sentinels, sketches, crystallize, diagnostics, facets`,
  );
  if (isSlotsEnabled()) {
    bootLog(
      `Slots: enabled (pinned editable memory). Reflect on Stop hook: ${isReflectEnabled() ? "on" : "off"}`,
    );
  }

  const snapshotConfig = loadSnapshotConfig();
  if (snapshotConfig.enabled) {
    registerSnapshotFunction(sdk, kv, snapshotConfig.dir);
    bootLog(
      `Git snapshots: ${snapshotConfig.dir} (every ${snapshotConfig.interval}s)`,
    );
  }

  const bm25Index = getSearchIndex();
  const graphWeight = parseFloat(getEnvVar("AGENTMEMORY_GRAPH_WEIGHT") || "0.3");
  const hybridSearch = new HybridSearch(
    bm25Index,
    vectorIndex,
    embeddingProvider,
    kv,
    embeddingConfig.bm25Weight,
    embeddingConfig.vectorWeight,
    graphWeight,
  );

  // Route mem::search (memory_recall) through the same hybrid searcher so the
  // mandated recall tool gets BM25 + vector + graph + lifecycle ranking with
  // one-call full bodies, instead of bare BM25. Lazy wiring: registerSearchFunction
  // ran earlier, before hybridSearch existed.
  setHybridSearcher((query, limit) => hybridSearch.search(query, limit));

  registerSmartSearchFunction(sdk, kv, (query, limit) =>
    hybridSearch.search(query, limit),
  );

  registerApiTriggers(sdk, kv, secret, metricsStore, provider);
  registerEventTriggers(sdk, kv);
  registerMcpEndpoints(sdk, kv, secret);

  const healthMonitor = registerHealthMonitor(sdk, kv);

  const indexPersistence = new IndexPersistence(
    indexBlobStore,
    bm25Index,
    vectorIndex,
  );
  // Wire the persistence hook so delete paths can flush BM25/vector
  // index mutations to disk. Without this, an in-memory remove can be
  // lost across a hard process exit and the persisted snapshot
  // restores the deleted entry at next boot.
  setIndexPersistence(indexPersistence);

  const loaded = await indexPersistence.load().catch((err) => {
    console.warn(`[agentmemory] Failed to load persisted index:`, err);
    return null;
  });
  if (loaded?.bm25 && loaded.bm25.size > 0) {
    bm25Index.restoreFrom(loaded.bm25);
    bootLog(
      `Loaded persisted BM25 index (${bm25Index.size} docs)`,
    );
  }
  // Restore in-memory vectors from the blob (no-op for self-persisting
  // backends like lancedb, which already opened their data from disk at
  // construction; their vectorJson is null).
  if (loaded?.vectorJson && vectorIndex) {
    await vectorIndex.restoreFrom(loaded.vectorJson);
  }
  if (vectorIndex && vectorIndex.size > 0) {
    // Persisted vectors carry whatever dimension the provider had when
    // they were written. If the active provider declares a different
    // dimension — or if the on-disk index contains a mix of dimensions
    // (legacy indexes written before the live-API guard in this PR) —
    // restoring would silently corrupt search: cosineSimilarity returns
    // 0 on cross-dim pairs, so affected observations stop matching
    // anything and recall degrades without an error. Walk every stored
    // vector instead of trusting the first; refuse to load if anything
    // is off.
    const activeDim = embeddingProvider?.dimensions ?? 0;
    const { mismatches, seenDimensions } =
      activeDim > 0
        ? await vectorIndex.validateDimensions(activeDim)
        : { mismatches: [], seenDimensions: new Set<number>() };

    if (mismatches.length > 0) {
      const sample = mismatches
        .slice(0, 5)
        .map((m) => `${m.obsId} (dim=${m.dim})`)
        .join(", ");
      const distinct = Array.from(seenDimensions).sort((a, b) => a - b).join(", ");
      const dropStale = isDropStaleIndexEnabled();
      if (dropStale) {
        console.warn(
          `[agentmemory] Persisted vector index has ${mismatches.length} of ` +
            `${vectorIndex.size} vectors with the wrong dimension. Active ` +
            `provider (${embeddingProvider?.name}) declares ${activeDim}; ` +
            `dimensions seen on disk: ${distinct}. ` +
            `AGENTMEMORY_DROP_STALE_INDEX=true is set — discarding the persisted ` +
            `vectors. Live observations will rebuild the index over time.`,
        );
        await vectorIndex.clear();
      } else {
        throw new Error(
          `[agentmemory] Refusing to start: persisted vector index has ` +
            `${mismatches.length} of ${vectorIndex.size} vectors with the ` +
            `wrong dimension. Active provider (${embeddingProvider?.name}) ` +
            `declares ${activeDim}; dimensions seen on disk: ${distinct}. ` +
            `First mismatched obsIds: ${sample}. Loading would silently corrupt ` +
            `search (cross-dimension cosine returns 0). Choose one:\n` +
            `  - Re-embed the existing index against the new provider, then start.\n` +
            `  - Set AGENTMEMORY_DROP_STALE_INDEX=true to discard the persisted ` +
            `vectors and rebuild from live observations.\n` +
            `  - Switch the embedding provider back to the one that wrote the index.`,
        );
      }
    } else {
      bootLog(
        `Loaded persisted vector index (${vectorIndex.size} vectors)`,
      );
    }
  }

  // Rebuild when BM25 is empty, OR when an embedding provider is configured
  // but the vector index is empty while BM25 is not. The latter happens after
  // a DROP_STALE dimension-mismatch clear() (or any path that empties the
  // vector store while the BM25 blob survives): leaving it would strand
  // semantic search with zero vectors and no recovery. rebuildIndex re-embeds
  // the whole corpus (fire-and-forget below, so boot is not blocked).
  const needsRebuild =
    bm25Index.size === 0 ||
    (vectorIndex !== null &&
      embeddingProvider !== null &&
      vectorIndex.size === 0);

  if (needsRebuild) {
    // Fire-and-forget. rebuildIndex iterates every observation across
    // every session and AWAITS an embedding-provider call per record.
    // On a large corpus + rate-limited embedding endpoint that can
    // take HOURS; awaiting it here blocks every subsequent boot step
    // (including startViewerServer below, leaving the viewer port
    // unbound for the duration). The index lazily fills in over time
    // and search degrades gracefully — partial coverage > no viewer
    // for hours. Errors still surface via the inner .catch.
    void rebuildIndex(kv)
      .then((indexCount) => {
        if (indexCount > 0) {
          bootLog(`Search index rebuilt: ${indexCount} entries`);
          indexPersistence.scheduleSave();
        }
      })
      .catch((err) => {
        console.warn(`[agentmemory] Failed to rebuild search index:`, err);
      });
  } else {
    // Backfill memories into BM25 for users upgrading from <0.9.5: prior
    // versions of mem::remember never indexed memories, so the persisted
    // BM25 covers observations only and `memory_smart_search` returns
    // empty for everything saved via memory_save (#257). Walk KV.memories
    // and add the ones missing from the restored index. Idempotent on
    // re-runs because SearchIndex.has() short-circuits already-indexed
    // ids.
    try {
      const memories = await kv.list<import("./types.js").Memory>(KV.memories);
      let backfilled = 0;
      for (const memory of memories) {
        if (memory.isLatest === false) continue;
        if (!memory.title || !memory.content) continue;
        if (bm25Index.has(memory.id)) continue;
        bm25Index.add({
          id: memory.id,
          sessionId: memory.sessionIds?.[0] ?? "memory",
          timestamp: memory.createdAt,
          type: "decision",
          title: memory.title,
          facts: [memory.content],
          narrative: memory.content,
          concepts: memory.concepts,
          files: memory.files,
          importance: memory.strength,
        });
        backfilled++;
      }
      if (backfilled > 0) {
        bootLog(
          `Backfilled ${backfilled} memories into BM25 (legacy gap before #257)`,
        );
        indexPersistence.scheduleSave();
      }
    } catch (err) {
      console.warn(
        `[agentmemory] Failed to backfill memories into BM25:`,
        err,
      );
    }
  }

  // Periodic LanceDB compaction. Live writes each create a new on-disk
  // version; without periodic compaction the index accrues unbounded
  // fragments. Hourly optimize()+prune keeps it tight. No-op for the
  // in-memory backend; unref so it never holds the process open at exit.
  if (vectorIndex || graphKv) {
    const vi = vectorIndex;
    const gkv = graphKv;
    const optimizeTimer = setInterval(
      () => {
        vi?.optimize().catch((err) => {
          console.warn(`[agentmemory] vector index optimize failed:`, err);
        });
        gkv?.optimize?.().catch((err) => {
          console.warn(`[agentmemory] graph kv optimize failed:`, err);
        });
      },
      60 * 60 * 1000,
    );
    optimizeTimer.unref?.();
  }

  // Ready / Endpoints lines are emitted via `bootLog` so they're
  // buffered in quiet mode and printed verbatim under --verbose. The
  // CLI surfaces a compact summary when it sees the worker reach
  // ready state.
  bootLog(
    `Ready. ${embeddingProvider ? "Triple-stream (BM25+Vector+Graph)" : "BM25+Graph"} search active.`,
  );
  bootLog(
    `REST API: 128 endpoints at http://localhost:${config.restPort}/agentmemory/*`,
  );
  bootLog(
    `MCP surface (opt-in via \`npx @agentmemory/mcp\`): ${getAllTools().length} tools · 6 resources · 3 prompts`,
  );

  const viewerPort = config.restPort + 2;
  const viewerServer = startViewerServer(
    viewerPort,
    kv,
    sdk,
    secret,
    config.restPort,
  );

  const autoForgetIntervalMs = parseInt(process.env.AUTO_FORGET_INTERVAL_MS || "3600000", 10);
  const consolidationIntervalMs = parseInt(process.env.CONSOLIDATION_INTERVAL_MS || "7200000", 10);

  if (process.env.AUTO_FORGET_ENABLED !== "false") {
    const autoForgetTimer = setInterval(async () => {
      try {
        await sdk.trigger({ function_id: "mem::auto-forget", payload: { dryRun: false } });
      } catch {}
    }, autoForgetIntervalMs);
    autoForgetTimer.unref();
    bootLog(`Auto-forget: enabled (every ${autoForgetIntervalMs / 60000}m)`);
  }

  if (process.env.LESSON_DECAY_ENABLED !== "false") {
    const lessonDecayTimer = setInterval(async () => {
      try {
        await sdk.trigger({ function_id: "mem::lesson-decay-sweep", payload: {} });
      } catch {}
    }, 86400000);
    lessonDecayTimer.unref();
    bootLog(`Lesson decay sweep: enabled (every 24h)`);
  }

  if (process.env.INSIGHT_DECAY_ENABLED !== "false") {
    const insightDecayTimer = setInterval(async () => {
      try {
        await sdk.trigger({ function_id: "mem::insight-decay-sweep", payload: {} });
      } catch {}
    }, 86400000);
    insightDecayTimer.unref();
  }

  if (isConsolidationEnabled()) {
    const consolidationTimer = setInterval(async () => {
      try {
        await sdk.trigger({ function_id: "mem::consolidate-pipeline", payload: {} });
      } catch {}
    }, consolidationIntervalMs);
    consolidationTimer.unref();
    bootLog(`Auto-consolidation: enabled (every ${consolidationIntervalMs / 60000}m)`);
  }

  if (process.env.LIFECYCLE_SWEEP_ENABLED !== "false") {
    const parsedSweepInterval = parseInt(
      process.env.LIFECYCLE_SWEEP_INTERVAL_MS || "86400000",
      10,
    );
    // Guard against a non-numeric override (parseInt -> NaN) or a non-positive
    // value: setInterval(fn, NaN) coerces the delay to 0 and would spin the
    // sweep — a full listLifecycle scan plus tier writes — on every event-loop
    // tick. Fall back to the 24h default.
    const lifecycleSweepIntervalMs =
      Number.isFinite(parsedSweepInterval) && parsedSweepInterval > 0
        ? parsedSweepInterval
        : 86400000;
    const lifecycleSweepTimer = setInterval(async () => {
      try {
        await sdk.trigger({ function_id: "mem::lifecycle-sweep", payload: {} });
      } catch {}
    }, lifecycleSweepIntervalMs);
    lifecycleSweepTimer.unref();
    bootLog(`Lifecycle sweep: enabled (maturity demotion + GC review, every ${lifecycleSweepIntervalMs / 3600000}h)`);
  }

  const shutdown = async () => {
    console.log(`\n[agentmemory] Shutting down...`);
    healthMonitor.stop();
    dedupMap.stop();
    indexPersistence.stop();
    await new Promise<void>((resolve) => viewerServer.close(() => resolve()));
    await indexPersistence.save().catch((err) => {
      console.warn(`[agentmemory] Failed to save index on shutdown:`, err);
    });
    await sdk.shutdown();
    clearWorkerPidfile();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main().catch((err) => {
  console.error(`[agentmemory] Fatal:`, err);
  process.exit(1);
});
