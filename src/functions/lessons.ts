import type { ISdk } from "iii-sdk";
import type { StateKV } from "../state/kv.js";
import { KV, fingerprintId } from "../state/schema.js";
import type { Lesson } from "../types.js";
import { recordAudit } from "./audit.js";
import { getEmbeddingProvider } from "./search.js";
import {
  getLessonVectorStore,
  lessonVectorAddGuarded,
  lessonText,
} from "../state/lesson-vectors.js";
import { logger } from "../logger.js";

// English stopword set for lexical lesson recall. A natural-language query
// like "what did we do for the lancedb rework" otherwise matches unrelated
// lessons purely through high-frequency function words ("the/for/we/do"),
// which is the root cause of the ESXi/Veeam/Excel false positives. Stripping
// these leaves only the content-bearing terms ("lancedb", "rework").
const LESSON_STOPWORDS = new Set<string>([
  "a", "an", "the", "of", "for", "to", "we", "i", "do", "did", "does", "done",
  "what", "how", "why", "when", "where", "who", "which", "and", "or", "but",
  "in", "on", "at", "by", "is", "it", "this", "that", "these", "those", "be",
  "was", "were", "are", "am", "as", "with", "from", "into", "about", "our",
  "us", "you", "your", "they", "them", "their", "he", "she", "his", "her",
  "its", "my", "me", "can", "could", "would", "should", "will", "shall",
  "may", "might", "must", "have", "has", "had", "if", "then", "than", "so",
  "not", "no", "yes", "up", "out", "over", "again", "just", "any", "all",
  "some", "more", "most", "such", "use", "used", "using", "get", "got",
]);

// Minimum lexical relevance a lesson must reach to surface. The lexical leg
// scores in [0, ~confidence]; 0.3 drops the stopword-only false positives
// (which previously scored ~0.46-0.49 purely on function-word overlap) while
// keeping genuine single-term content matches. Used both here and gated again
// in smart-search.ts so weak lessons never reach the smart-search bucket.
export const LESSON_SCORE_FLOOR = 0.3;

// Tokenize a query into content-bearing, lowercased terms: split on
// non-word characters, drop stopwords and 1-char noise. Shared by the
// lexical scorer so query and stored text are tokenized identically.
function lessonQueryTerms(query: string): string[] {
  return query
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 1 && !LESSON_STOPWORDS.has(t));
}

// Word-boundary term match against pre-tokenized lesson text. The old code
// used text.includes(term), so "do" matched "domain" and "id" matched
// "video". Matching against a Set of whole tokens enforces boundaries
// without a per-term regex compile.
function lessonTokenSet(text: string): Set<string> {
  return new Set(
    text.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 0),
  );
}

// Lexical relevance of one lesson to a set of query terms, in [0, confidence].
// confidence * (matchedTerms / totalTerms) * recencyBoost — identical shape to
// the original scorer, but term matching is now whole-token (word-boundary)
// and stopword-free. Returns 0 when nothing content-bearing matches.
function lexicalLessonScore(lesson: Lesson, terms: string[]): number {
  if (terms.length === 0) return 0;
  const tokens = lessonTokenSet(lessonText(lesson));
  const matchCount = terms.filter((t) => tokens.has(t)).length;
  if (matchCount === 0) return 0;
  const relevance = matchCount / terms.length;
  const daysSinceReinforced = lesson.lastReinforcedAt
    ? (Date.now() - new Date(lesson.lastReinforcedAt).getTime()) /
      (1000 * 60 * 60 * 24)
    : (Date.now() - new Date(lesson.createdAt).getTime()) /
      (1000 * 60 * 60 * 24);
  const recencyBoost = 1 / (1 + daysSinceReinforced * 0.01);
  return lesson.confidence * relevance * recencyBoost;
}

function reinforceLesson(lesson: Lesson): void {
  const now = new Date().toISOString();
  lesson.reinforcements++;
  lesson.confidence = Math.min(
    1.0,
    lesson.confidence + 0.1 * (1 - lesson.confidence),
  );
  lesson.lastReinforcedAt = now;
  lesson.updatedAt = now;
}

export function registerLessonsFunctions(sdk: ISdk, kv: StateKV): void {
  sdk.registerFunction("mem::lesson-save", 
    async (data: {
      content: string;
      context?: string;
      confidence?: number;
      project?: string;
      tags?: string[];
      source?: "crystal" | "manual" | "consolidation";
      sourceIds?: string[];
    }) => {
      if (!data.content?.trim()) {
        return { success: false, error: "content is required" };
      }

      const fp = fingerprintId("lsn", data.content.trim().toLowerCase());
      const existing = await kv.get<Lesson>(KV.lessons, fp);

      if (existing && !existing.deleted) {
        reinforceLesson(existing);
        // Re-embedding is only needed when the embedded text actually changed.
        // The lesson id is a fingerprint of content, so content is identical on
        // a strengthen; only a newly-added context alters lessonText(). Track
        // that to avoid a wasted embed call on every reinforcement.
        let contextChanged = false;
        if (data.context && !existing.context) {
          existing.context = data.context;
          contextChanged = true;
        }
        await kv.set(KV.lessons, existing.id, existing);

        if (contextChanged) {
          await lessonVectorAddGuarded(existing);
        }

        try {
          await recordAudit(kv, "lesson_strengthen", "mem::lesson-save", [
            existing.id,
          ]);
        } catch {}

        return {
          success: true,
          action: "strengthened",
          lesson: existing,
        };
      }

      const confidence =
        typeof data.confidence === "number" &&
        data.confidence >= 0 &&
        data.confidence <= 1
          ? data.confidence
          : 0.5;

      const now = new Date().toISOString();
      const lesson: Lesson = {
        id: fp,
        content: data.content.trim(),
        context: data.context?.trim() || "",
        confidence,
        reinforcements: 0,
        source: data.source || "manual",
        sourceIds: data.sourceIds || [],
        project: data.project,
        tags: data.tags || [],
        createdAt: now,
        updatedAt: now,
        decayRate: 0.05,
      };

      await kv.set(KV.lessons, lesson.id, lesson);

      // Embed the lesson into the lessons vector table so paraphrased recall
      // works. Best-effort: a downed embedder or the memory backend (no lesson
      // table) leaves recall on the lexical leg. Mirrors remember.ts, which
      // also embeds after the KV write without blocking the save.
      await lessonVectorAddGuarded(lesson);

      try {
        await recordAudit(kv, "lesson_save", "mem::lesson-save", [lesson.id]);
      } catch {}

      return { success: true, action: "created", lesson };
    },
  );

  sdk.registerFunction("mem::lesson-recall", 
    async (data: {
      query: string;
      project?: string;
      minConfidence?: number;
      limit?: number;
    }) => {
      if (!data.query?.trim()) {
        return { success: false, error: "query is required" };
      }

      const minConfidence = data.minConfidence ?? 0.1;
      const limit = data.limit ?? 10;

      let lessons = await kv.list<Lesson>(KV.lessons);

      lessons = lessons.filter(
        (l) => !l.deleted && l.confidence >= minConfidence,
      );

      if (data.project) {
        lessons = lessons.filter((l) => l.project === data.project);
      }

      const byId = new Map<string, Lesson>(lessons.map((l) => [l.id, l]));

      // --- Lexical leg (Phase A, always on) ------------------------------
      // Stopword-stripped, word-boundary scoring with a relevance floor. This
      // leg is fully self-sufficient: if the vector leg is disabled or fails,
      // recall degrades to exactly this hardened lexical behavior.
      const terms = lessonQueryTerms(data.query);
      const lexical = lessons
        .map((l) => ({ lesson: l, score: lexicalLessonScore(l, terms) }))
        .filter((s) => s.score >= LESSON_SCORE_FLOOR);
      lexical.sort((a, b) => b.score - a.score);

      // --- Vector leg (Phase B, best-effort) -----------------------------
      // Semantic recall over the lessons LanceDB table. Paraphrased queries
      // ("rework" vs "migration") retrieve the right lesson here even when no
      // surface term overlaps. Embedding/search failures are swallowed so the
      // lexical leg above still answers.
      const vectorHits = await recallLessonVectors(data.query, byId);

      // --- Fuse via Reciprocal Rank Fusion (RRF, k=60) -------------------
      // Mirrors hybrid-search.ts's fusion of BM25 + vector. Each leg
      // contributes 1/(RRF_K + rank); a lesson found by both legs ranks
      // above one found by a single leg. We keep the lexical leg's raw score
      // as the surfaced `score` (back-compat with the prior contract) and use
      // RRF only for ordering. Pure-vector hits that never cleared the
      // lexical floor still appear, carrying their semantic similarity as the
      // surfaced score.
      const fused = fuseLessonLegs(lexical, vectorHits);
      fused.sort((a, b) => b.rrf - a.rrf);

      try {
        await recordAudit(kv, "lesson_recall", "mem::lesson-recall", [], {
          query: data.query,
          resultCount: fused.length,
        });
      } catch {}

      return {
        success: true,
        lessons: fused.slice(0, limit).map((s) => ({
          ...s.lesson,
          score: Math.round(s.score * 1000) / 1000,
        })),
      };
    },
  );

  sdk.registerFunction("mem::lesson-list", 
    async (data: {
      project?: string;
      source?: string;
      minConfidence?: number;
      limit?: number;
    }) => {
      const limit = data.limit ?? 50;
      const minConfidence = data.minConfidence ?? 0;
      let lessons = await kv.list<Lesson>(KV.lessons);

      lessons = lessons.filter(
        (l) => !l.deleted && l.confidence >= minConfidence,
      );

      if (data.project) {
        lessons = lessons.filter((l) => l.project === data.project);
      }
      if (data.source) {
        lessons = lessons.filter((l) => l.source === data.source);
      }

      lessons.sort((a, b) => b.confidence - a.confidence);

      return { success: true, lessons: lessons.slice(0, limit) };
    },
  );

  sdk.registerFunction("mem::lesson-strengthen", 
    async (data: { lessonId: string }) => {
      if (!data.lessonId) {
        return { success: false, error: "lessonId is required" };
      }

      const lesson = await kv.get<Lesson>(KV.lessons, data.lessonId);
      if (!lesson || lesson.deleted) {
        return { success: false, error: "lesson not found" };
      }

      reinforceLesson(lesson);

      await kv.set(KV.lessons, lesson.id, lesson);

      try {
        await recordAudit(kv, "lesson_strengthen", "mem::lesson-strengthen", [
          lesson.id,
        ]);
      } catch {}

      return { success: true, lesson };
    },
  );

  sdk.registerFunction("mem::lesson-decay-sweep", 
    async () => {
      const lessons = await kv.list<Lesson>(KV.lessons);
      let decayed = 0;
      let softDeleted = 0;
      const now = Date.now();
      const timestamp = new Date().toISOString();
      const dirty: Lesson[] = [];
      const auditEvents: Array<{
        id: string;
        action: "decay" | "soft-delete";
        beforeConfidence: number;
        afterConfidence: number;
        beforeDeleted: boolean;
        afterDeleted: boolean;
      }> = [];

      for (const lesson of lessons) {
        if (lesson.deleted) continue;

        const baseline = lesson.lastDecayedAt || lesson.lastReinforcedAt || lesson.createdAt;
        const weeksSinceBaseline =
          (now - new Date(baseline).getTime()) / (1000 * 60 * 60 * 24 * 7);

        if (weeksSinceBaseline < 1) continue;

        const decay = lesson.decayRate * weeksSinceBaseline;
        const newConfidence = Math.max(0.05, lesson.confidence - decay);

        if (newConfidence !== lesson.confidence) {
          const beforeConfidence = lesson.confidence;
          const beforeDeleted = !!lesson.deleted;
          lesson.confidence = Math.round(newConfidence * 1000) / 1000;
          lesson.lastDecayedAt = timestamp;
          lesson.updatedAt = timestamp;

          if (lesson.confidence <= 0.1 && lesson.reinforcements === 0) {
            lesson.deleted = true;
            softDeleted++;
          } else {
            decayed++;
          }

          dirty.push(lesson);
          auditEvents.push({
            id: lesson.id,
            action: lesson.deleted ? "soft-delete" : "decay",
            beforeConfidence,
            afterConfidence: lesson.confidence,
            beforeDeleted,
            afterDeleted: !!lesson.deleted,
          });
        }
      }

      await Promise.all(dirty.map((l) => kv.set(KV.lessons, l.id, l)));

      // Drop vectors for soft-deleted lessons so they stop occupying the
      // lessons table. Recall already filters l.deleted out via byId, so this
      // is housekeeping, not correctness; best-effort, never blocks the sweep.
      const removedIds = auditEvents
        .filter((e) => e.action === "soft-delete")
        .map((e) => e.id);
      if (removedIds.length > 0) {
        const store = getLessonVectorStore();
        if (store) {
          await Promise.all(
            removedIds.map((id) =>
              store.remove(id).catch((err: unknown) => {
                logger.warn("lesson-decay-sweep: vector remove failed", {
                  lessonId: id,
                  error: err instanceof Error ? err.message : String(err),
                });
              }),
            ),
          );
        }
      }

      await Promise.all(
        auditEvents.map((event) =>
          recordAudit(kv, "lesson_strengthen", "mem::lesson-decay-sweep", [event.id], {
            action: event.action,
            actor: "system",
            reason: "decay-sweep",
            before: {
              confidence: event.beforeConfidence,
              deleted: event.beforeDeleted,
            },
            after: {
              confidence: event.afterConfidence,
              deleted: event.afterDeleted,
            },
          }),
        ),
      );

      return { success: true, decayed, softDeleted, total: lessons.length };
    },
  );

  // One-time (and idempotent) backfill: embed every existing non-deleted
  // lesson into the lessons vector table. Lessons predate Phase B, so their
  // vectors don't exist until this runs. Uses a single clear() + one bulk
  // insert per batch (NOT per-row add) to avoid LanceDB write amplification,
  // mirroring rebuildIndex()/vectorIndexAddBatchGuarded in search.ts. Safe to
  // re-run: it clears the lessons table and re-embeds from KV.lessons, so the
  // table always reflects current lesson content. No-op (success, embedded:0)
  // when no embedding provider or no lessons vector backend is configured.
  sdk.registerFunction("mem::lesson-embed-backfill",
    async (data?: { batchSize?: number }) => {
      const store = getLessonVectorStore();
      const provider = getEmbeddingProvider();
      if (!store || !provider) {
        return {
          success: true,
          embedded: 0,
          failed: 0,
          skipped: "no lessons vector backend or embedding provider",
        };
      }

      const batchSize =
        typeof data?.batchSize === "number" &&
        Number.isInteger(data.batchSize) &&
        data.batchSize > 0
          ? data.batchSize
          : 32;

      const all = await kv.list<Lesson>(KV.lessons);
      const live = all.filter((l) => !l.deleted);

      // Clear first so the table reflects current content on every run and
      // every row is a fresh bulk insert (no O(n) existence check per row).
      try {
        await store.clear();
      } catch (err) {
        logger.warn("lesson-embed-backfill: clear failed", {
          error: err instanceof Error ? err.message : String(err),
        });
        return { success: false, embedded: 0, failed: live.length };
      }

      let embedded = 0;
      let failed = 0;
      for (let i = 0; i < live.length; i += batchSize) {
        const chunk = live.slice(i, i + batchSize);
        let embeddings: Float32Array[];
        try {
          embeddings = await provider.embedBatch(chunk.map(lessonText));
        } catch (err) {
          logger.warn("lesson-embed-backfill: embedBatch failed — skipping batch", {
            batchSize: chunk.length,
            error: err instanceof Error ? err.message : String(err),
          });
          failed += chunk.length;
          continue;
        }
        if (embeddings.length !== chunk.length) {
          failed += chunk.length;
          continue;
        }
        const rows: Array<{ lessonId: string; embedding: Float32Array }> = [];
        for (let j = 0; j < chunk.length; j++) {
          if (embeddings[j].length !== provider.dimensions) {
            failed++;
            continue;
          }
          rows.push({ lessonId: chunk[j].id, embedding: embeddings[j] });
        }
        try {
          await store.bulkAdd(rows);
          embedded += rows.length;
        } catch (err) {
          logger.warn("lesson-embed-backfill: bulk insert failed — skipping batch", {
            batchSize: rows.length,
            error: err instanceof Error ? err.message : String(err),
          });
          failed += rows.length;
        }
      }

      try {
        await store.optimize();
      } catch {}

      logger.info("Lesson embed backfill complete", {
        embedded,
        failed,
        total: live.length,
      });
      return { success: true, embedded, failed, total: live.length };
    },
  );
}

// Run the lessons vector leg of recall: embed the query, ANN-search the
// lessons table, and map each hit back to its in-scope Lesson. Returns
// rank-ordered { lesson, score } where score is the cosine similarity. Drops
// hits whose lesson is missing from `byId` (soft-deleted / filtered out by
// project or confidence). Best-effort: any failure (no backend, downed
// embedder) yields an empty array so the lexical leg stands alone.
async function recallLessonVectors(
  query: string,
  byId: Map<string, Lesson>,
): Promise<Array<{ lesson: Lesson; score: number }>> {
  const store = getLessonVectorStore();
  const provider = getEmbeddingProvider();
  if (!store || !provider || store.size === 0) return [];
  try {
    const embedding = await provider.embed(query);
    if (embedding.length !== provider.dimensions) return [];
    // Over-fetch a little so project/confidence filtering still leaves a
    // useful number of in-scope semantic hits.
    const hits = await store.search(embedding, 20);
    const out: Array<{ lesson: Lesson; score: number }> = [];
    for (const h of hits) {
      const lesson = byId.get(h.lessonId);
      if (lesson) out.push({ lesson, score: h.score });
    }
    return out;
  } catch (err) {
    logger.warn("lesson-recall: vector leg failed — lexical only", {
      error: err instanceof Error ? err.message : String(err),
    });
    return [];
  }
}

const LESSON_RRF_K = 60;

// Fuse the lexical and vector legs with Reciprocal Rank Fusion, mirroring
// hybrid-search.ts (1/(K+rank) per leg). The surfaced `score` stays the
// lexical raw score where available (back-compat with the prior recall
// contract) and falls back to the vector similarity for pure-semantic hits.
// `rrf` is the ordering key only.
function fuseLessonLegs(
  lexical: Array<{ lesson: Lesson; score: number }>,
  vector: Array<{ lesson: Lesson; score: number }>,
): Array<{ lesson: Lesson; score: number; rrf: number }> {
  const acc = new Map<
    string,
    { lesson: Lesson; rrf: number; lexScore?: number; vecScore?: number }
  >();

  lexical.forEach((s, i) => {
    acc.set(s.lesson.id, {
      lesson: s.lesson,
      rrf: 1 / (LESSON_RRF_K + i + 1),
      lexScore: s.score,
    });
  });

  vector.forEach((s, i) => {
    const contrib = 1 / (LESSON_RRF_K + i + 1);
    const existing = acc.get(s.lesson.id);
    if (existing) {
      existing.rrf += contrib;
      existing.vecScore = s.score;
    } else {
      acc.set(s.lesson.id, { lesson: s.lesson, rrf: contrib, vecScore: s.score });
    }
  });

  return Array.from(acc.values()).map((e) => ({
    lesson: e.lesson,
    score: e.lexScore ?? e.vecScore ?? 0,
    rrf: e.rrf,
  }));
}
