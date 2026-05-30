import type { Lesson, EmbeddingProvider } from "../types.js";
import { getEmbeddingProvider, clipEmbedInput } from "../functions/search.js";
import { logger } from "../logger.js";

// ---------------------------------------------------------------------------
// Lessons vector store (Phase B of the lesson-recall semantic upgrade).
//
// Lessons live in KV.lessons and, until now, were NEVER embedded — recall was
// lexical-only, so paraphrased queries missed the right lesson. This module
// adds a SEPARATE LanceDB vector table ("lessons") mirroring the memories
// vector path. A separate table (rather than reusing "memories") is the clean
// choice: lessons have their own id space (lsn_*) and the memories table has
// no type column to distinguish them.
//
// The backend is a singleton, set once at boot (initLessonVectorStore, wired
// from src/index.ts on the lancedb path). On the in-memory backend it stays
// null and lesson recall degrades to the hardened lexical leg — which is fully
// self-sufficient. All lesson-write paths (save/strengthen/backfill) and the
// recall vector leg consult getLessonVectorStore() and no-op when it is null.
// ---------------------------------------------------------------------------

// Minimal vector-store contract for lessons. Deliberately narrower than the
// memories VectorBackend: lessons need no lifecycle columns, no dimension
// validation (the table is rebuilt by backfill, not persistence-restored), and
// no serialize/restore (self-persisting on disk). Implemented by
// LanceLessonVectorBackend in stores/lancedb-store.ts.
export interface LessonVectorBackend {
  init(): Promise<void>;
  add(lessonId: string, embedding: Float32Array): Promise<void>;
  bulkAdd(
    items: Array<{ lessonId: string; embedding: Float32Array }>,
  ): Promise<void>;
  remove(lessonId: string): Promise<void>;
  search(
    query: Float32Array,
    limit?: number,
  ): Promise<Array<{ lessonId: string; score: number }>>;
  clear(): Promise<void>;
  optimize(): Promise<void>;
  readonly size: number;
}

let lessonVectorStore: LessonVectorBackend | null = null;

export function setLessonVectorStore(store: LessonVectorBackend | null): void {
  lessonVectorStore = store;
}

export function getLessonVectorStore(): LessonVectorBackend | null {
  return lessonVectorStore;
}

// Boot wiring. Only the lancedb backend gets a lessons vector table — the
// memory backend (tests / standalone) stays lexical-only. Best-effort: a
// failure here (missing native module, disk error) logs and leaves the store
// null so lesson recall still works lexically. Idempotent: a second call
// re-opens and replaces the singleton.
export async function initLessonVectorStore(opts: {
  dataDir: string;
  dimensions: number;
  backend: string;
}): Promise<void> {
  if (opts.backend !== "lancedb") {
    setLessonVectorStore(null);
    return;
  }
  try {
    const { createLanceLessonVectorBackend } = await import(
      "./stores/lancedb-store.js"
    );
    const backend = await createLanceLessonVectorBackend({
      dataDir: opts.dataDir,
      dimensions: opts.dimensions,
    });
    setLessonVectorStore(backend);
    logger.info("Lessons vector table ready", {
      backend: opts.backend,
      dimensions: opts.dimensions,
      size: backend.size,
    });
  } catch (err) {
    setLessonVectorStore(null);
    logger.warn(
      "initLessonVectorStore: failed — lesson recall stays lexical-only",
      { error: err instanceof Error ? err.message : String(err) },
    );
  }
}

// The text embedded for a lesson. Mirrors the lexical recall text
// (content + context + tags) so the semantic and lexical legs see the same
// surface, and mirrors remember.ts's "title + content" composition pattern.
export function lessonText(lesson: Lesson): string {
  const tags = lesson.tags?.length ? " " + lesson.tags.join(" ") : "";
  const context = lesson.context ? " " + lesson.context : "";
  return `${lesson.content}${context}${tags}`;
}

// Single guarded lesson-vector write. Mirrors search.ts::vectorIndexAddGuarded:
// soft-fails on a missing store / missing provider / dimension mismatch / embed
// throw so a downed embedder never breaks the lesson save. Returns true only on
// a committed vector write.
export async function lessonVectorAddGuarded(lesson: Lesson): Promise<boolean> {
  const store = lessonVectorStore;
  const provider: EmbeddingProvider | null = getEmbeddingProvider();
  if (!store || !provider) return false;
  try {
    const embedding = await provider.embed(clipEmbedInput(lessonText(lesson)));
    if (embedding.length !== provider.dimensions) {
      logger.warn("lesson-vector add: dimension mismatch — skipping", {
        lessonId: lesson.id,
        provider: provider.name,
        expected: provider.dimensions,
        received: embedding.length,
      });
      return false;
    }
    await store.add(lesson.id, embedding);
    return true;
  } catch (err) {
    logger.warn("lesson-vector add: embed failed — skipping", {
      lessonId: lesson.id,
      provider: provider.name,
      error: err instanceof Error ? err.message : String(err),
    });
    return false;
  }
}
