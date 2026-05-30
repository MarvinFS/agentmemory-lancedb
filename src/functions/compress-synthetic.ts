import type {
  RawObservation,
  CompressedObservation,
  ObservationType,
} from "../types.js";

// Zero-LLM compression path. Converts a RawObservation into a
// CompressedObservation using only heuristics — no Claude call, no token
// spend. This is the default as of 0.8.8 (#138); users who want richer
// LLM-generated summaries set AGENTMEMORY_AUTO_COMPRESS=true.

// Breadcrumb observation types. These are the low-signal, high-volume
// per-tool-call records the auto-capture hooks emit (file reads, edits,
// searches, command runs, etc.). They are deliberately given LOW importance
// so the lifecycle GC ages them out, are EXCLUDED from access reinforcement
// in hybrid-search (so recall does not climb them into the maturity tiers),
// and are the default target set for mem::observations-prune. Higher-signal
// types (decision, discovery, image) and curated Memory records are NOT
// breadcrumbs and keep their normal lifecycle. `error` is a breadcrumb for
// importance/reinforcement purposes but is excluded from the default prune
// set (errors are worth keeping for postmortems).
export const BREADCRUMB_TYPES: ReadonlySet<ObservationType> = new Set([
  "file_read",
  "file_write",
  "file_edit",
  "command_run",
  "search",
  "web_fetch",
  "conversation",
  "subagent",
  "task",
  "notification",
  "error",
  "other",
]);

export function isBreadcrumbType(type: ObservationType): boolean {
  return BREADCRUMB_TYPES.has(type);
}

// Heuristic importance (0-5 scale of CompressedObservation.importance) for a
// synthetic breadcrumb. Mirrors inferType's tool->type mapping so the value
// tracks the observation's own type. Low values let mem::auto-forget
// (importance<=2) and mem::evict (importance<threshold) prune them, instead
// of the old hardcoded 5 which made every breadcrumb un-prunable.
function inferImportance(
  toolName: string | undefined,
  hookType: string,
): number {
  switch (inferType(toolName, hookType)) {
    case "file_read":
    case "search":
    case "web_fetch":
    case "subagent":
    case "task":
    case "notification":
      return 2;
    case "command_run":
    case "file_write":
    case "file_edit":
    case "conversation":
      return 3;
    case "decision":
      return 4;
    default:
      return 5;
  }
}

function inferType(
  toolName: string | undefined,
  hookType: string,
): ObservationType {
  if (hookType === "post_tool_failure") return "error";
  if (hookType === "prompt_submit") return "conversation";
  if (hookType === "subagent_stop" || hookType === "task_completed")
    return "subagent";
  if (hookType === "notification") return "notification";

  if (!toolName) return "other";
  // Normalize camelCase and kebab-case into word chunks so we can match
  // substrings like "WebFetch" -> "web" / "fetch".
  const n = toolName
    .replace(/([a-z])([A-Z])/g, "$1_$2")
    .replace(/[-\s]+/g, "_")
    .toLowerCase();
  const hasWord = (word: string) =>
    new RegExp(`(^|_)${word}(_|$)`).test(n) ||
    n === word ||
    n.endsWith(word) ||
    n.startsWith(word);
  if (["fetch", "http", "web"].some(hasWord)) return "web_fetch";
  if (["grep", "search", "glob", "find"].some(hasWord)) return "search";
  if (["bash", "shell", "exec", "run"].some(hasWord)) return "command_run";
  if (["edit", "update", "patch", "replace"].some(hasWord)) return "file_edit";
  if (["write", "create"].some(hasWord)) return "file_write";
  if (["read", "view"].some(hasWord)) return "file_read";
  if (["task", "agent"].some(hasWord)) return "subagent";
  return "other";
}

function extractFiles(input: unknown): string[] {
  if (!input || typeof input !== "object") return [];
  const o = input as Record<string, unknown>;
  const out = new Set<string>();
  for (const key of [
    "file_path",
    "filepath",
    "path",
    "filePath",
    "file",
    "pattern",
  ]) {
    const v = o[key];
    if (typeof v === "string" && v.length > 0 && v.length < 512) out.add(v);
  }
  return [...out];
}

function stringifyForNarrative(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "\u2026" : s;
}

export function buildSyntheticCompression(
  raw: RawObservation,
): CompressedObservation {
  const toolName = raw.toolName ?? raw.hookType;
  const inputStr = stringifyForNarrative(raw.toolInput);
  const outputStr = stringifyForNarrative(raw.toolOutput);
  const promptStr = raw.userPrompt ?? "";

  const narrativeParts = [promptStr, inputStr, outputStr].filter(
    (s) => s.length > 0,
  );

  const result: CompressedObservation = {
    id: raw.id,
    sessionId: raw.sessionId,
    timestamp: raw.timestamp,
    type: inferType(toolName, raw.hookType),
    title: truncate(toolName || "observation", 80),
    subtitle: inputStr ? truncate(inputStr, 120) : undefined,
    facts: [],
    narrative: truncate(narrativeParts.join(" | "), 400),
    concepts: [],
    files: extractFiles(raw.toolInput),
    importance: inferImportance(toolName, raw.hookType),
    confidence: 0.3,
  };
  if (raw.modality) result.modality = raw.modality;
  if (raw.imageData) result.imageData = raw.imageData;
  if (raw.agentId) result.agentId = raw.agentId;
  return result;
}
