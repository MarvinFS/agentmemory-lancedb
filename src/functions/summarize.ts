import type { ISdk } from "iii-sdk";
import type {
  CompressedObservation,
  SessionSummary,
  MemoryProvider,
  Session,
} from "../types.js";
import { KV } from "../state/schema.js";
import { StateKV } from "../state/kv.js";
import {
  SUMMARY_SYSTEM,
  buildSummaryPrompt,
  REDUCE_SYSTEM,
  buildReducePrompt,
} from "../prompts/summary.js";
import { getXmlTag, getXmlChildren } from "../prompts/xml.js";
import { SummaryOutputSchema } from "../eval/schemas.js";
import { validateOutput } from "../eval/validator.js";
import { scoreSummary } from "../eval/quality.js";
import type { MetricsStore } from "../eval/metrics-store.js";
import { safeAudit } from "./audit.js";
import { logger } from "../logger.js";

// Per-chunk observation budget when a session is too large to fit in one
// LLM call. Default ≈ 30k input tokens per chunk at ~110 tok/obs — fits
// comfortably in 128k-window models. Lowered from 400 to 250 (0.9.x): a
// smaller chunk is a shorter single completion, which cuts per-call latency
// and the odds of any one call hitting the SUMMARIZE_CALL_TIMEOUT_MS ceiling
// (see getCallTimeoutMs) and blowing the iii 180s budget. Override via
// SUMMARIZE_CHUNK_SIZE.
const CHUNK_SIZE_DEFAULT = 250;
// Concurrent in-flight chunk calls. 6 keeps a 100-chunk session under
// iii's 180s function-invocation timeout at ~8s/call while staying
// inside generous-but-not-unlimited provider rate limits (well below
// OpenAI free tier's 500 RPM). High-throughput providers
// (Novita / DeepInfra / DeepSeek) typically allow 100+ concurrent — set
// SUMMARIZE_CHUNK_CONCURRENCY higher to cover ~1000+ chunk sessions.
const CHUNK_CONCURRENCY_DEFAULT = 6;
// Per-LLM-call wall-time ceiling. The MemoryProvider interface exposes no
// AbortSignal, so we cap each call by racing it against a timer (the in-flight
// provider request keeps running but its result is discarded). A single hung
// call would otherwise consume the whole iii 180s invocation budget and kill
// the function — a fast local timeout lets the retry / synthetic-fallback
// path recover instead. Override via SUMMARIZE_CALL_TIMEOUT_MS.
const CALL_TIMEOUT_MS_DEFAULT = 45_000;
// Bail on the merged summary if more than this fraction of chunks produce
// NO content at all — neither an LLM parse nor a synthetic fallback. With the
// synthetic fallback in place this should only trip on pathological input
// (e.g. observations with empty titles); a half-blind narrative is worse than
// a clean error.
const MAX_SKIP_RATIO = 0.5;

function getChunkSize(): number {
  const raw = process.env.SUMMARIZE_CHUNK_SIZE;
  if (!raw) return CHUNK_SIZE_DEFAULT;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : CHUNK_SIZE_DEFAULT;
}

function getChunkConcurrency(): number {
  const raw = process.env.SUMMARIZE_CHUNK_CONCURRENCY;
  if (!raw) return CHUNK_CONCURRENCY_DEFAULT;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : CHUNK_CONCURRENCY_DEFAULT;
}

function getCallTimeoutMs(): number {
  const raw = process.env.SUMMARIZE_CALL_TIMEOUT_MS;
  if (!raw) return CALL_TIMEOUT_MS_DEFAULT;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : CALL_TIMEOUT_MS_DEFAULT;
}

// Race a provider call against a wall-time ceiling. Resolves with the call's
// result if it finishes first; rejects with a `summarize_call_timeout` Error
// if the timer wins, so callers treat it like any other call failure (retry,
// then synthetic fallback). The underlying request is not aborted — the
// MemoryProvider interface has no AbortSignal — but its eventual result is
// discarded. The timer is always cleared so a fast call leaves no dangling
// handle keeping the event loop alive.
async function withCallTimeout<T>(
  fn: () => Promise<T>,
  timeoutMs: number,
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`summarize_call_timeout after ${timeoutMs}ms`));
    }, timeoutMs);
  });
  try {
    return await Promise.race([fn(), timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

// One chunk call with retry-once, each attempt bounded by withCallTimeout.
// Returns null when both attempts fail — whether by parse failure, provider
// 4xx (content rejected by upstream filters), call timeout, or transient
// network/5xx errors that didn't recover on retry. All failure modes are
// equivalent at this layer: the LLM produced nothing usable for this chunk.
// The caller (produceSummaryXml) then salvages the chunk via a zero-LLM
// synthetic fallback, so a null here is a degrade signal, not an automatic
// skip; only a chunk the synthetic path also can't summarize is truly skipped.
async function summarizeChunkWithRetry(
  provider: MemoryProvider,
  chunk: CompressedObservation[],
  sessionId: string,
  project: string,
  idx: number,
  total: number,
): Promise<SessionSummary | null> {
  const timeoutMs = getCallTimeoutMs();
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const xml = await withCallTimeout(
        () => provider.summarize(SUMMARY_SYSTEM, buildSummaryPrompt(chunk)),
        timeoutMs,
      );
      const parsed = parseSummaryXml(xml, sessionId, project, chunk.length);
      if (parsed) return parsed;
      logger.warn("Summarize chunk parse failed", {
        sessionId,
        chunk: `${idx + 1}/${total}`,
        attempt,
      });
    } catch (err) {
      logger.warn("Summarize chunk LLM call failed", {
        sessionId,
        chunk: `${idx + 1}/${total}`,
        attempt,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }
  return null;
}

// Returns the final summary XML string. For sessions ≤ chunk size, this is
// a single LLM call (legacy behavior). For larger sessions, observations
// are split into chunks processed in parallel batches, each chunk retried
// once on parse failure, persistently-bad chunks skipped, and remaining
// partials merged via a reduce call.
async function produceSummaryXml(
  provider: MemoryProvider,
  compressed: CompressedObservation[],
  sessionId: string,
  project: string,
): Promise<{
  response: string;
  mode: "single" | "chunked";
  chunks: number;
  skipped?: number;
}> {
  const chunkSize = getChunkSize();
  const callTimeoutMs = getCallTimeoutMs();
  if (compressed.length <= chunkSize) {
    try {
      const response = await withCallTimeout(
        () => provider.summarize(SUMMARY_SYSTEM, buildSummaryPrompt(compressed)),
        callTimeoutMs,
      );
      if (response && response.trim() && parseSummaryXml(response, sessionId, project, compressed.length)) {
        return { response, mode: "single", chunks: 1 };
      }
      logger.warn("Summarize single-call empty/unparseable, using synthetic fallback", {
        sessionId,
        observationCount: compressed.length,
      });
    } catch (err) {
      logger.warn("Summarize single-call failed, using synthetic fallback", {
        sessionId,
        error: err instanceof Error ? err.message : String(err),
      });
    }
    // Single LLM call timed out / failed / produced no usable XML. Salvage the
    // session with a zero-LLM synthetic summary instead of returning a hard
    // failure — partial content beats none. Serialize it back to summary XML so
    // the caller's existing parse/validate/persist path is unchanged.
    const synthetic = buildSyntheticChunkSummary(compressed, sessionId, project);
    if (synthetic) {
      return { response: serializeSummaryXml(synthetic), mode: "single", chunks: 1 };
    }
    return { response: "", mode: "single", chunks: 1 };
  }

  const chunks: CompressedObservation[][] = [];
  for (let i = 0; i < compressed.length; i += chunkSize) {
    chunks.push(compressed.slice(i, i + chunkSize));
  }
  const concurrency = getChunkConcurrency();
  logger.info("Summarize chunking session", {
    sessionId,
    chunks: chunks.length,
    chunkSize,
    concurrency,
    totalObservations: compressed.length,
  });

  // Sparse array preserves chunk → index mapping after parallel resolution,
  // so the reduce step sees partials in chronological order even when some
  // were skipped.
  const partialByIdx: Array<SessionSummary | null> = new Array(chunks.length).fill(null);
  let degraded = 0; // chunks that fell back to synthetic (non-LLM) compression
  for (let batchStart = 0; batchStart < chunks.length; batchStart += concurrency) {
    const batch = chunks.slice(batchStart, batchStart + concurrency);
    await Promise.all(
      batch.map(async (chunk, j) => {
        const idx = batchStart + j;
        const llm = await summarizeChunkWithRetry(
          provider,
          chunk,
          sessionId,
          project,
          idx,
          chunks.length,
        );
        if (llm) {
          partialByIdx[idx] = llm;
          return;
        }
        // Both LLM attempts failed. Rather than drop the chunk (and risk the
        // MAX_SKIP_RATIO bailout), salvage its content with a zero-LLM
        // synthetic summary so it still feeds the reduce step. A chunk only
        // stays null when even the synthetic path has nothing to work with.
        const synthetic = buildSyntheticChunkSummary(chunk, sessionId, project);
        if (synthetic) {
          degraded += 1;
          partialByIdx[idx] = synthetic;
          logger.warn("Summarize chunk degraded to synthetic fallback", {
            sessionId,
            chunk: `${idx + 1}/${chunks.length}`,
          });
        }
      }),
    );
  }

  const skipped = partialByIdx.filter((p) => p === null).length;
  const partials = partialByIdx.filter((p): p is SessionSummary => p !== null);

  if (skipped > Math.floor(chunks.length * MAX_SKIP_RATIO)) {
    throw new Error(
      `too_many_chunks_skipped: ${skipped}/${chunks.length} chunks produced no content after retry and synthetic fallback`,
    );
  }
  if (skipped > 0 || degraded > 0) {
    logger.warn("Summarize chunks partially degraded", {
      sessionId,
      skipped,
      degraded,
      total: chunks.length,
    });
  }

  const reduceInput = partials.map((p) => {
    const originalIdx = partialByIdx.indexOf(p);
    return {
      title: p.title,
      narrative: p.narrative,
      keyDecisions: p.keyDecisions,
      filesModified: p.filesModified,
      concepts: p.concepts,
      obsRangeStart: originalIdx * chunkSize + 1,
      obsRangeEnd: Math.min((originalIdx + 1) * chunkSize, compressed.length),
    };
  });
  try {
    const response = await withCallTimeout(
      () => provider.summarize(REDUCE_SYSTEM, buildReducePrompt(reduceInput)),
      callTimeoutMs,
    );
    if (response && response.trim() && parseSummaryXml(response, sessionId, project, compressed.length)) {
      return { response, mode: "chunked", chunks: chunks.length, skipped };
    }
    logger.warn("Summarize reduce empty/unparseable, merging partials synthetically", {
      sessionId,
    });
  } catch (err) {
    logger.warn("Summarize reduce failed, merging partials synthetically", {
      sessionId,
      error: err instanceof Error ? err.message : String(err),
    });
  }
  // Reduce call timed out / failed. The chunk partials already hold the real
  // content, so merge them without a second LLM round-trip rather than throwing
  // away the whole chunked run. Serialize back to summary XML for the caller.
  return {
    response: serializeSummaryXml(mergePartialsSynthetically(partials, sessionId, project, compressed.length)),
    mode: "chunked",
    chunks: chunks.length,
    skipped,
  };
}

// Zero-LLM fallback summary for a single chunk whose LLM calls both failed.
// Reuses the already-compressed observations directly: their titles become a
// terse narrative, their files/concepts are unioned, and the highest-signal
// titles double as "decisions". This keeps the chunk contributing real content
// to the reduce step instead of being dropped. Returns null only when the
// chunk is empty (nothing to salvage). Style mirrors compress-synthetic.ts's
// heuristic, no-token philosophy.
function buildSyntheticChunkSummary(
  chunk: CompressedObservation[],
  sessionId: string,
  project: string,
): SessionSummary | null {
  if (chunk.length === 0) return null;

  const files = uniq(chunk.flatMap((o) => o.files ?? []));
  const concepts = uniq(chunk.flatMap((o) => o.concepts ?? []));
  // Surface the highest-importance observations as the chunk's decisions so
  // the reduce step sees the most load-bearing items first.
  const ranked = [...chunk].sort((a, b) => (b.importance ?? 0) - (a.importance ?? 0));
  const keyDecisions = uniq(ranked.slice(0, 5).map((o) => o.title).filter(Boolean));

  const titles = chunk.map((o) => o.title).filter(Boolean);
  const narrative =
    `Auto-summarized chunk of ${chunk.length} observations (LLM unavailable). ` +
    `Activity: ${titles.slice(0, 8).join("; ")}${titles.length > 8 ? "; …" : ""}`;

  return {
    sessionId,
    project,
    createdAt: new Date().toISOString(),
    title: `Session activity (${chunk.length} observations)`,
    narrative,
    keyDecisions,
    filesModified: files,
    concepts,
    observationCount: chunk.length,
  };
}

// Merge several chunk partials into one summary without an LLM round-trip.
// Used when the reduce call itself times out / fails: the partials already
// carry the salient content, so a deterministic union is a safe degraded path.
function mergePartialsSynthetically(
  partials: SessionSummary[],
  sessionId: string,
  project: string,
  obsCount: number,
): SessionSummary {
  const narratives = partials.map((p) => p.narrative).filter(Boolean);
  return {
    sessionId,
    project,
    createdAt: new Date().toISOString(),
    title: partials[0]?.title ?? "Session summary",
    narrative:
      narratives.join(" ").trim() ||
      `Merged summary of ${partials.length} chunks across ${obsCount} observations.`,
    keyDecisions: uniq(partials.flatMap((p) => p.keyDecisions ?? [])),
    filesModified: uniq(partials.flatMap((p) => p.filesModified ?? [])),
    concepts: uniq(partials.flatMap((p) => p.concepts ?? [])),
    observationCount: obsCount,
  };
}

function uniq(values: string[]): string[] {
  return [...new Set(values.filter((v) => typeof v === "string" && v.length > 0))];
}

// Neutralize the only characters that would break the regex tag matchers in
// xml.ts when this text is re-parsed: literal angle brackets. We deliberately
// do NOT entity-encode (&amp; etc.) because getXmlTag/getXmlChildren do not
// decode entities, so encoding would leave visible "&amp;" artifacts in the
// stored summary. Angle brackets are replaced with their fullwidth lookalikes,
// which preserves readability for the rare title/path that contains them.
function sanitizeTagText(s: string): string {
  return s.replace(/</g, "＜").replace(/>/g, "＞");
}

// Serialize a SessionSummary back into the exact <summary> XML shape the
// prompts ask for, so a synthetically-built summary flows through the caller's
// unchanged parse/validate/persist path.
function serializeSummaryXml(s: SessionSummary): string {
  const decisions = s.keyDecisions
    .map((d) => `    <decision>${sanitizeTagText(d)}</decision>`)
    .join("\n");
  const files = s.filesModified
    .map((f) => `    <file>${sanitizeTagText(f)}</file>`)
    .join("\n");
  const concepts = s.concepts
    .map((c) => `    <concept>${sanitizeTagText(c)}</concept>`)
    .join("\n");
  return `<summary>
  <title>${sanitizeTagText(s.title)}</title>
  <narrative>${sanitizeTagText(s.narrative)}</narrative>
  <decisions>
${decisions}
  </decisions>
  <files>
${files}
  </files>
  <concepts>
${concepts}
  </concepts>
</summary>`;
}

// Strip the common ways an LLM wraps or mangles the XML we asked for, so the
// strict tag matchers in xml.ts still find the structure. Handles: ```xml /
// ``` code fences, leading/trailing prose around the <summary> block, smart
// quotes that slip into otherwise-valid tags, and a bare unclosed <summary>
// (we only ever read inner tags, so a missing root close is harmless once the
// fences/prose are gone). Intentionally conservative — it normalizes the
// envelope, it does not try to rewrite malformed inner tags.
function sanitizeSummaryXml(xml: string): string {
  let s = xml;
  // Remove code fences (```xml ... ``` or ``` ... ```), keeping the body.
  s = s.replace(/```(?:xml|XML)?\s*/g, "").replace(/```/g, "");
  // Normalize smart quotes / non-breaking spaces that LLMs emit in prose and
  // occasionally inside tag text.
  s = s
    .replace(/[“”„‟]/g, '"')
    .replace(/[‘’‚‛]/g, "'")
    .replace(/ /g, " ");
  // If a <summary> root exists, keep only from its opening tag onward so any
  // leading prose ("Here is the summary:") is discarded. Tolerate a missing
  // closing </summary> by falling back to end-of-string.
  const open = s.indexOf("<summary>");
  if (open !== -1) {
    const close = s.indexOf("</summary>");
    s = close !== -1 ? s.slice(open, close + "</summary>".length) : s.slice(open);
  }
  return s.trim();
}

// Last-resort title recovery when no <title> tag survives even after
// sanitization: take a leading markdown heading ("# ...", "## ...") or a
// **bold** lead line, else the first non-empty, non-tag prose line. Bounded to
// 100 chars to match the prompt's title contract. Returns "" when nothing
// usable is found, so the caller still falls through to a hard parse failure.
function recoverTitleFromProse(s: string): string {
  for (const rawLine of s.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith("<")) continue; // skip stray tags
    const heading = line.match(/^#{1,6}\s+(.+)$/);
    if (heading) return heading[1].replace(/\*\*/g, "").trim().slice(0, 100);
    const bold = line.match(/^\*\*(.+?)\*\*$/);
    if (bold) return bold[1].trim().slice(0, 100);
    return line.slice(0, 100); // first real prose line
  }
  return "";
}

function parseSummaryXml(
  xml: string,
  sessionId: string,
  project: string,
  obsCount: number,
): SessionSummary | null {
  // Try the response as-is first (the happy path costs no extra work), then a
  // sanitized pass that recovers from common LLM envelope deviations, then a
  // prose-title last resort so a structurally-broken-but-non-empty response
  // still yields a usable (if thin) summary instead of being dropped.
  let title = getXmlTag(xml, "title");
  let source = xml;
  if (!title) {
    source = sanitizeSummaryXml(xml);
    title = getXmlTag(source, "title");
  }
  if (!title) {
    title = recoverTitleFromProse(source);
    if (!title) return null;
  }

  return {
    sessionId,
    project,
    createdAt: new Date().toISOString(),
    title,
    narrative: getXmlTag(source, "narrative"),
    keyDecisions: getXmlChildren(source, "decisions", "decision"),
    filesModified: getXmlChildren(source, "files", "file"),
    concepts: getXmlChildren(source, "concepts", "concept"),
    observationCount: obsCount,
  };
}

export function registerSummarizeFunction(
  sdk: ISdk,
  kv: StateKV,
  provider: MemoryProvider,
  metricsStore?: MetricsStore,
): void {
  sdk.registerFunction("mem::summarize", 
    async (data: { sessionId: string } | undefined) => {
      const startMs = Date.now();
      if (!data || typeof data.sessionId !== "string" || !data.sessionId.trim()) {
        return { success: false, error: "sessionId is required" };
      }
      const sessionId = data.sessionId.trim();

      const session = await kv.get<Session>(KV.sessions, sessionId);
      if (!session) {
        logger.warn("Session not found for summarize", {
          sessionId,
        });
        return { success: false, error: "session_not_found" };
      }

      const observations = await kv.list<CompressedObservation>(
        KV.observations(sessionId),
      );
      const compressed = observations.filter((o) => o.title);

      if (compressed.length === 0) {
        logger.info("No observations to summarize", {
          sessionId,
        });
        return { success: false, error: "no_observations" };
      }

      if (provider.name === "noop") {
        logger.info("Summarize skipped — no LLM provider configured", {
          sessionId,
        });
        return {
          success: false,
          error: "no_provider",
          reason:
            "No LLM provider key set; Summarize is a no-op. Set ANTHROPIC_API_KEY (or GEMINI/OPENROUTER/MINIMAX) in ~/.agentmemory/.env to enable.",
        };
      }

      try {
        const { response, mode, chunks } = await produceSummaryXml(
          provider,
          compressed,
          sessionId,
          session.project,
        );
        if (!response || !response.trim()) {
          const latencyMs = Date.now() - startMs;
          if (metricsStore) {
            await metricsStore.record("mem::summarize", latencyMs, false);
          }
          logger.warn("Empty provider response on summarize", {
            sessionId,
            provider: provider.name,
            mode,
            chunks,
            observationCount: compressed.length,
          });
          return { success: false, error: "empty_provider_response" };
        }
        const summary = parseSummaryXml(
          response,
          sessionId,
          session.project,
          compressed.length,
        );

        if (!summary) {
          const latencyMs = Date.now() - startMs;
          if (metricsStore) {
            await metricsStore.record("mem::summarize", latencyMs, false);
          }
          logger.warn("Failed to parse summary XML", {
            sessionId,
          });
          return { success: false, error: "parse_failed" };
        }

        const summaryForValidation = {
          title: summary.title,
          narrative: summary.narrative,
          keyDecisions: summary.keyDecisions,
          filesModified: summary.filesModified,
          concepts: summary.concepts,
        };
        const validation = validateOutput(
          SummaryOutputSchema,
          summaryForValidation,
          "mem::summarize",
        );

        if (!validation.valid) {
          const latencyMs = Date.now() - startMs;
          if (metricsStore) {
            await metricsStore.record("mem::summarize", latencyMs, false);
          }
          logger.warn("Summary validation failed", {
            sessionId,
            errors: validation.result.errors,
          });
          return { success: false, error: "validation_failed" };
        }

        const qualityScore = scoreSummary(summaryForValidation);

        await kv.set(KV.summaries, sessionId, summary);
        await safeAudit(kv, "compress", "mem::summarize", [sessionId], {
          title: summary.title,
          observationCount: compressed.length,
        });

        const latencyMs = Date.now() - startMs;
        if (metricsStore) {
          await metricsStore.record(
            "mem::summarize",
            latencyMs,
            true,
            qualityScore,
          );
        }

        logger.info("Session summarized", {
          sessionId,
          title: summary.title,
          decisions: summary.keyDecisions.length,
          qualityScore,
          valid: validation.valid,
        });

        return { success: true, summary, qualityScore };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        const latencyMs = Date.now() - startMs;
        if (metricsStore) {
          await metricsStore.record("mem::summarize", latencyMs, false);
        }
        logger.error("Summarize failed", {
          sessionId,
          error: msg,
        });
        return { success: false, error: msg };
      }
    },
  );
}
