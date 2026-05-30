import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { execSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { resolveProject } from "../src/hooks/_project.js";

describe("resolveProject — hook project basename resolver", () => {
  const originalEnv = process.env.AGENTMEMORY_PROJECT_NAME;

  // Derive the expected name from the real git toplevel the same way
  // resolveProject does, so the suite passes regardless of the checkout's
  // directory name (e.g. upstream "agentmemory" vs a fork "agentmemory-lancedb").
  const repoBasename = basename(
    execSync("git rev-parse --show-toplevel").toString().trim(),
  );

  beforeEach(() => {
    delete process.env.AGENTMEMORY_PROJECT_NAME;
  });

  afterEach(() => {
    if (originalEnv === undefined) {
      delete process.env.AGENTMEMORY_PROJECT_NAME;
    } else {
      process.env.AGENTMEMORY_PROJECT_NAME = originalEnv;
    }
  });

  it("AGENTMEMORY_PROJECT_NAME env wins over everything", () => {
    process.env.AGENTMEMORY_PROJECT_NAME = "my-override";
    expect(resolveProject("/var/log")).toBe("my-override");
    expect(resolveProject(process.cwd())).toBe("my-override");
  });

  it("trims whitespace on env override", () => {
    process.env.AGENTMEMORY_PROJECT_NAME = "  spaced  ";
    expect(resolveProject("/var/log")).toBe("spaced");
  });

  it("ignores empty env override", () => {
    process.env.AGENTMEMORY_PROJECT_NAME = "   ";
    expect(resolveProject(process.cwd())).toBe(repoBasename);
  });

  it("returns git toplevel basename when cwd is inside a repo", () => {
    const top = resolveProject(process.cwd());
    expect(top).toBe(repoBasename);
  });

  it("returns git toplevel basename from a nested subdir", () => {
    const nested = join(process.cwd(), "src", "hooks");
    expect(resolveProject(nested)).toBe(repoBasename);
  });

  it("falls back to basename(cwd) when not in a git repo", () => {
    const dir = mkdtempSync(join(tmpdir(), "amem-noproj-"));
    try {
      expect(resolveProject(dir)).toBe(dir.split("/").pop());
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("defaults to process.cwd() when no cwd argument given", () => {
    expect(resolveProject()).toBe(repoBasename);
  });

  it("defaults to process.cwd() when cwd argument is empty", () => {
    expect(resolveProject("")).toBe(repoBasename);
    expect(resolveProject("   ")).toBe(repoBasename);
  });
});
