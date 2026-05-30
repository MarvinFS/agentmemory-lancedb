import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

vi.mock("../src/state/schema.js", () => ({
  KV: {
    sessions: "sessions",
    summaries: "summaries",
    observations: (sessionId: string) => `obs:${sessionId}`,
    audit: "audit",
  },
}));

vi.mock("../src/eval/schemas.js", () => ({
  SummaryOutputSchema: {},
}));

// Real-ish validator: enforce the one constraint the synthetic fallbacks must
// satisfy to be persisted (narrative length), so the test proves the fallback
// content is actually usable, not just that the code path runs.
vi.mock("../src/eval/validator.js", () => ({
  validateOutput: (_schema: unknown, value: any) => {
    const ok =
      typeof value?.title === "string" &&
      value.title.length >= 1 &&
      typeof value?.narrative === "string" &&
      value.narrative.length >= 20;
    return { valid: ok, result: { errors: ok ? [] : ["narrative too short"] } };
  },
}));

vi.mock("../src/eval/quality.js", () => ({
  scoreSummary: () => 100,
}));

vi.mock("../src/functions/audit.js", () => ({
  safeAudit: vi.fn(),
}));

import { registerSummarizeFunction } from "../src/functions/summarize.js";
import type {
  CompressedObservation,
  Session,
  MemoryProvider,
} from "../src/types.js";

function mockKV() {
  const store = new Map<string, Map<string, unknown>>();
  return {
    store,
    get: async <T>(scope: string, key: string): Promise<T | null> =>
      (store.get(scope)?.get(key) as T) ?? null,
    set: async <T>(scope: string, key: string, data: T): Promise<T> => {
      if (!store.has(scope)) store.set(scope, new Map());
      store.get(scope)!.set(key, data);
      return data;
    },
    delete: async (scope: string, key: string): Promise<void> => {
      store.get(scope)?.delete(key);
    },
    list: async <T>(scope: string): Promise<T[]> => {
      const entries = store.get(scope);
      return entries ? (Array.from(entries.values()) as T[]) : [];
    },
  };
}

function mockSdk() {
  const functions = new Map<string, Function>();
  return {
    functions,
    registerFunction: (id: string, handler: Function) => {
      functions.set(id, handler);
    },
    registerTrigger: () => {},
    trigger: async () => ({}),
  };
}

function makeObs(i: number, sessionId: string): CompressedObservation {
  return {
    id: `obs_${i}`,
    sessionId,
    timestamp: new Date().toISOString(),
    type: "conversation",
    title: `obs ${i}`,
    facts: [`fact ${i}`],
    narrative: `narrative for obs ${i}`,
    concepts: [`concept-${i}`],
    files: [`src/file_${i}.ts`],
    importance: 5,
  };
}

function summaryXml(title: string): string {
  return `<summary>
<title>${title}</title>
<narrative>a sufficiently long narrative for validation purposes</narrative>
<decisions></decisions>
<files></files>
<concepts></concepts>
</summary>`;
}

async function setupHandler(opts: {
  sessionId: string;
  obsCount: number;
  provider: MemoryProvider;
}) {
  const sdk = mockSdk();
  const kv = mockKV();
  const session: Session = {
    id: opts.sessionId,
    project: "test-project",
    cwd: "/tmp",
    startedAt: new Date().toISOString(),
    status: "completed",
    observationCount: opts.obsCount,
  };
  await kv.set("sessions", opts.sessionId, session);
  for (let i = 0; i < opts.obsCount; i++) {
    const o = makeObs(i, opts.sessionId);
    await kv.set(`obs:${opts.sessionId}`, o.id, o);
  }
  registerSummarizeFunction(sdk as any, kv as any, opts.provider);
  const handler = sdk.functions.get("mem::summarize")!;
  return { handler, kv };
}

describe("mem::summarize resilience", () => {
  const ORIGINAL_ENV = { ...process.env };

  beforeEach(() => {
    delete process.env.SUMMARIZE_CHUNK_SIZE;
    delete process.env.SUMMARIZE_CHUNK_CONCURRENCY;
    delete process.env.SUMMARIZE_CALL_TIMEOUT_MS;
  });

  afterEach(() => {
    process.env = { ...ORIGINAL_ENV };
  });

  describe("tolerant XML parsing", () => {
    it("recovers from a ```xml code-fenced response with leading prose", async () => {
      const fenced =
        "Here is the summary you requested:\n\n```xml\n" +
        summaryXml("Fenced session") +
        "\n```\nLet me know if you need anything else.";
      const provider: MemoryProvider = {
        name: "test",
        compress: async () => "",
        summarize: async () => fenced,
      };
      const { handler, kv } = await setupHandler({
        sessionId: "ses_fenced",
        obsCount: 5,
        provider,
      });

      const result: any = await handler({ sessionId: "ses_fenced" });

      expect(result.success).toBe(true);
      const stored: any = await kv.get("summaries", "ses_fenced");
      expect(stored?.title).toBe("Fenced session");
    });

    it("recovers a response whose only structure is a stray <title> buried in prose with a missing root close", async () => {
      // No <summary> wrapper close, surrounded by prose. The strict pass still
      // finds the well-formed inner tags; the run succeeds rather than returning
      // parse_failed. Documents that well-formed inner tags survive a mangled
      // envelope.
      const messy =
        "Sure! Here's the session summary:\n" +
        "<title>Buried title</title>\n" +
        "<narrative>a narrative that comfortably clears the length bar</narrative>\n" +
        "<decisions><decision>decided X</decision></decisions>\n" +
        "<files><file>src/x.ts</file></files>\n" +
        "<concepts><concept>alpha</concept></concepts>";
      const provider: MemoryProvider = {
        name: "test",
        compress: async () => "",
        summarize: async () => messy,
      };
      const { handler, kv } = await setupHandler({
        sessionId: "ses_messy",
        obsCount: 5,
        provider,
      });

      const result: any = await handler({ sessionId: "ses_messy" });

      expect(result.success).toBe(true);
      const stored: any = await kv.get("summaries", "ses_messy");
      expect(stored?.title).toBe("Buried title");
      expect(stored?.keyDecisions).toEqual(["decided X"]);
    });

    it("recovers a title from a markdown heading when no <title> tag exists", async () => {
      // Model ignored the format and answered in markdown. The prose-title
      // last resort lifts the heading; a <narrative> tag is still present so
      // validation (narrative >= 20 chars) passes.
      const md =
        "## Refactored the auth layer\n\n" +
        "<narrative>Swapped the session store and tightened token checks across the board</narrative>\n" +
        "<files><file>src/auth.ts</file></files>";
      const provider: MemoryProvider = {
        name: "test",
        compress: async () => "",
        summarize: async () => md,
      };
      const { handler, kv } = await setupHandler({
        sessionId: "ses_md",
        obsCount: 5,
        provider,
      });

      const result: any = await handler({ sessionId: "ses_md" });

      expect(result.success).toBe(true);
      const stored: any = await kv.get("summaries", "ses_md");
      expect(stored?.title).toBe("Refactored the auth layer");
      expect(stored?.filesModified).toEqual(["src/auth.ts"]);
    });
  });

  describe("per-call timeout", () => {
    it("a hung single call hits SUMMARIZE_CALL_TIMEOUT_MS and falls back to a synthetic summary", async () => {
      process.env.SUMMARIZE_CALL_TIMEOUT_MS = "30"; // fail fast
      const provider: MemoryProvider = {
        name: "test",
        compress: async () => "",
        // Never resolves within the timeout window.
        summarize: () => new Promise<string>((r) => setTimeout(() => r(summaryXml("late")), 5_000)),
      };
      const { handler, kv } = await setupHandler({
        sessionId: "ses_hang",
        obsCount: 5,
        provider,
      });

      const result: any = await handler({ sessionId: "ses_hang" });

      // Synthetic fallback means the session still summarizes successfully
      // rather than blowing the iii 180s budget.
      expect(result.success).toBe(true);
      const stored: any = await kv.get("summaries", "ses_hang");
      expect(stored?.observationCount).toBe(5);
      expect(stored?.filesModified.length).toBeGreaterThan(0);
    });
  });

  describe("synthetic chunk fallback (no skip)", () => {
    it("a persistently-broken chunk degrades to synthetic content instead of being skipped", async () => {
      process.env.SUMMARIZE_CHUNK_SIZE = "100";
      process.env.SUMMARIZE_CHUNK_CONCURRENCY = "1";
      let i = 0;
      const provider: MemoryProvider = {
        name: "test",
        compress: async () => "",
        summarize: async (system: string) => {
          i += 1;
          if (i === 2 || i === 3) return "<garbage/>"; // chunk 2 both attempts fail
          return summaryXml(system.includes("merging") ? "merged" : `chunk ${i}`);
        },
      };
      // 250 obs / 100 => 3 chunks; chunk 2 fails both attempts but is salvaged.
      const { handler, kv } = await setupHandler({
        sessionId: "ses_degraded",
        obsCount: 250,
        provider,
      });

      const result: any = await handler({ sessionId: "ses_degraded" });

      expect(result.success).toBe(true);
      const stored: any = await kv.get("summaries", "ses_degraded");
      expect(stored?.title).toBe("merged");
    });

    it("reduce-call failure merges partials synthetically instead of failing the run", async () => {
      process.env.SUMMARIZE_CHUNK_SIZE = "100";
      process.env.SUMMARIZE_CHUNK_CONCURRENCY = "1";
      let i = 0;
      const provider: MemoryProvider = {
        name: "test",
        compress: async () => "",
        summarize: async (system: string) => {
          i += 1;
          if (system.includes("merging")) throw new Error("reduce 500");
          return summaryXml(`chunk ${i}`);
        },
      };
      const { handler, kv } = await setupHandler({
        sessionId: "ses_reduce_fail",
        obsCount: 250,
        provider,
      });

      const result: any = await handler({ sessionId: "ses_reduce_fail" });

      expect(result.success).toBe(true);
      const stored: any = await kv.get("summaries", "ses_reduce_fail");
      // Title is taken from the first surviving partial.
      expect(stored?.title).toBe("chunk 1");
      expect(stored?.observationCount).toBe(250);
    });
  });
});
