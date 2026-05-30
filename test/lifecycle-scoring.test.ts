import { describe, it, expect } from "vitest";
import {
  defaultLifecycle,
  applyDecay,
  reinforceOnAccess,
  reinforceOnUpdate,
  tierBoost,
  nextMaturity,
  compoundScore,
} from "../src/state/lifecycle-scoring.js";
import type { LifecycleFields } from "../src/state/vector-index.js";

const MS_PER_DAY = 86_400_000;

// Reference LifecycleFields builder so each test starts from a known shape
// and overrides only what it cares about.
function lf(overrides: Partial<LifecycleFields> = {}): LifecycleFields {
  return {
    importance: 50,
    recency: 1,
    accessCount: 0,
    updateCount: 0,
    maturity: "draft",
    createdAt: 0,
    updatedAt: 0,
    ...overrides,
  };
}

describe("defaultLifecycle", () => {
  it("returns the fresh-observation defaults stamped at nowMs", () => {
    const now = 1_700_000_000_000;
    const f = defaultLifecycle(now);
    expect(f.importance).toBe(50);
    expect(f.recency).toBe(1);
    expect(f.accessCount).toBe(0);
    expect(f.updateCount).toBe(0);
    expect(f.maturity).toBe("draft");
    expect(f.createdAt).toBe(now);
    expect(f.updatedAt).toBe(now);
  });
});

describe("applyDecay", () => {
  it("decays recency as exp(-days/30) and importance as *0.995^days", () => {
    const updatedAt = 0;
    const now = 30 * MS_PER_DAY;
    const f = lf({ importance: 50, recency: 1, updatedAt });

    const out = applyDecay(f, now);

    // recency = exp(-30/30) = exp(-1) ≈ 0.3678794
    expect(out.recency).toBeCloseTo(Math.exp(-1), 6);
    // importance = 50 * 0.995^30 ≈ 43.00292
    expect(out.importance).toBeCloseTo(50 * Math.pow(0.995, 30), 4);
    // updatedAt is NOT advanced by decay.
    expect(out.updatedAt).toBe(updatedAt);
    // Other fields carried through unchanged.
    expect(out.accessCount).toBe(0);
    expect(out.maturity).toBe("draft");
  });

  it("does not compound across repeated calls (recency recomputed from updatedAt)", () => {
    const f = lf({ importance: 80, recency: 1, updatedAt: 0 });
    const now = 10 * MS_PER_DAY;
    const once = applyDecay(f, now);
    // Re-decaying the ORIGINAL record at the same now yields the same result.
    const again = applyDecay(f, now);
    expect(again.recency).toBeCloseTo(once.recency, 10);
    expect(again.importance).toBeCloseTo(once.importance, 10);
    expect(once.recency).toBeCloseTo(Math.exp(-10 / 30), 6);
    expect(once.importance).toBeCloseTo(80 * Math.pow(0.995, 10), 4);
  });

  it("clamps importance to >= 0 and never amplifies for a future updatedAt", () => {
    // updatedAt in the future relative to nowMs -> days clamped to 0, so
    // recency = exp(0) = 1 and importance is unchanged (0.995^0 = 1). No
    // amplification.
    const f = lf({ importance: 70, recency: 0.5, updatedAt: 100 * MS_PER_DAY });
    const out = applyDecay(f, 0);
    expect(out.recency).toBe(1);
    expect(out.importance).toBeCloseTo(70, 10);
  });
});

describe("reinforceOnAccess", () => {
  it("adds +3 importance, increments accessCount, resets recency, stamps updatedAt", () => {
    const now = 12_345;
    const f = lf({ importance: 50, accessCount: 2, recency: 0.1, updatedAt: 0 });
    const out = reinforceOnAccess(f, now);
    expect(out.importance).toBe(53);
    expect(out.accessCount).toBe(3);
    expect(out.recency).toBe(1);
    expect(out.updatedAt).toBe(now);
    expect(out.updateCount).toBe(0);
  });

  it("caps importance at 100", () => {
    const out = reinforceOnAccess(lf({ importance: 99 }), 1);
    expect(out.importance).toBe(100);
  });
});

describe("reinforceOnUpdate", () => {
  it("adds +5 importance, increments updateCount, resets recency, stamps updatedAt", () => {
    const now = 67_890;
    const f = lf({ importance: 40, updateCount: 1, recency: 0.2, updatedAt: 0 });
    const out = reinforceOnUpdate(f, now);
    expect(out.importance).toBe(45);
    expect(out.updateCount).toBe(2);
    expect(out.recency).toBe(1);
    expect(out.updatedAt).toBe(now);
    expect(out.accessCount).toBe(0);
  });

  it("caps importance at 100", () => {
    const out = reinforceOnUpdate(lf({ importance: 98 }), 1);
    expect(out.importance).toBe(100);
  });
});

describe("tierBoost", () => {
  it("returns the per-tier multiplier", () => {
    expect(tierBoost("core")).toBe(1.15);
    expect(tierBoost("validated")).toBe(1.0);
    expect(tierBoost("draft")).toBe(0.85);
  });

  it("falls back to neutral 1.0 for unknown/legacy maturity", () => {
    expect(tierBoost("unknown")).toBe(1.0);
    expect(tierBoost("")).toBe(1.0);
  });
});

describe("nextMaturity hysteresis", () => {
  it("promotes draft -> validated at importance >= 65", () => {
    expect(nextMaturity(lf({ maturity: "draft", importance: 65 }))).toBe(
      "validated",
    );
    expect(nextMaturity(lf({ maturity: "draft", importance: 64 }))).toBe(
      "draft",
    );
  });

  it("promotes validated -> core at importance >= 85", () => {
    expect(nextMaturity(lf({ maturity: "validated", importance: 85 }))).toBe(
      "core",
    );
    expect(nextMaturity(lf({ maturity: "validated", importance: 84 }))).toBe(
      "validated",
    );
  });

  it("demotes core -> validated at importance < 60", () => {
    expect(nextMaturity(lf({ maturity: "core", importance: 59 }))).toBe(
      "validated",
    );
    expect(nextMaturity(lf({ maturity: "core", importance: 60 }))).toBe("core");
  });

  it("demotes validated -> draft at importance < 35", () => {
    expect(nextMaturity(lf({ maturity: "validated", importance: 34 }))).toBe(
      "draft",
    );
    expect(nextMaturity(lf({ maturity: "validated", importance: 35 }))).toBe(
      "validated",
    );
  });

  it("stays unchanged in the dead bands between thresholds", () => {
    // draft below promote threshold
    expect(nextMaturity(lf({ maturity: "draft", importance: 50 }))).toBe(
      "draft",
    );
    // validated between demote (35) and promote (85) thresholds
    expect(nextMaturity(lf({ maturity: "validated", importance: 50 }))).toBe(
      "validated",
    );
    // core above demote threshold
    expect(nextMaturity(lf({ maturity: "core", importance: 90 }))).toBe("core");
  });

  it("returns unknown maturity values unchanged", () => {
    expect(nextMaturity(lf({ maturity: "legacy", importance: 99 }))).toBe(
      "legacy",
    );
  });
});

describe("compoundScore", () => {
  it("returns the RRF score untouched when no lifecycle record is present", () => {
    expect(compoundScore(0.42, undefined)).toBe(0.42);
    expect(compoundScore(0, undefined)).toBe(0);
  });

  it("blends 0.6*rrf + 0.2*(imp/100) + 0.2*recency, scaled by tier boost (validated)", () => {
    const f = lf({ importance: 80, recency: 0.5, maturity: "validated" });
    const rrf = 0.7;
    const base = 0.6 * rrf + 0.2 * (80 / 100) + 0.2 * 0.5;
    expect(compoundScore(rrf, f)).toBeCloseTo(base * 1.0, 10);
  });

  it("applies the core tier boost (1.15) to the additive base", () => {
    const f = lf({ importance: 100, recency: 1, maturity: "core" });
    const rrf = 0.5;
    const base = 0.6 * rrf + 0.2 * 1 + 0.2 * 1;
    expect(compoundScore(rrf, f)).toBeCloseTo(base * 1.15, 10);
  });

  it("applies the draft tier penalty (0.85)", () => {
    const f = lf({ importance: 0, recency: 0, maturity: "draft" });
    const rrf = 0.3;
    const base = 0.6 * rrf + 0.2 * 0 + 0.2 * 0;
    expect(compoundScore(rrf, f)).toBeCloseTo(base * 0.85, 10);
  });
});
