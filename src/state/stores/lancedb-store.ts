import type {
  CreateBackendsOptions,
  PersistenceBackends,
} from "../vector-store.js";
import {
  VectorIndex,
  type VectorBackend,
  type VectorSearchHit,
  type VectorAddItem,
  type DimensionReport,
  type LifecycleFields,
} from "../vector-index.js";
import type { IndexBlobStore } from "../index-blob-store.js";
import type { GraphKvStore } from "../graph-kv-router.js";

// ---------------------------------------------------------------------------
// LanceDB storage backend (VECTOR_BACKEND=lancedb).
//
// Both the vector index and the BM25 blob store live in ONE on-disk LanceDB
// database under `${dataDir}/lancedb`, sharing a single connection. This is
// the whole point of the fork: vectors are materialized per-row on disk
// (persistsExternally=true) so IndexPersistence never round-trips a ~27MB
// monolithic blob through the iii KV's 180s state::set ceiling (#204).
//
// All @lancedb/lancedb / apache-arrow imports are lazy and confined to this
// module — they are optionalDependencies carrying a platform-specific native
// binary, so the default `memory` path must never touch them.
//
// API verified against @lancedb/lancedb v0.30.0 source (tag v0.30.0):
//   nodejs/lancedb/{index,connection,table,query,merge}.ts.
// apache-arrow is pinned to 18.1.0; schemas are built explicitly with
// arrow.Schema / Field / Float32 / Int32 / Int64 / Utf8 / FixedSizeList
// because vector-dimension inference is impossible on an empty table.
// ---------------------------------------------------------------------------

// Minimal structural types for the lazily-imported native modules. We keep
// these local and loose (no compile-time dependency on @lancedb/lancedb's
// .d.ts, which may be absent when the optionalDependency isn't installed)
// while still matching the v0.30.0 method shapes we actually call.
interface LanceMergeBuilder {
  whenMatchedUpdateAll(options?: { where: string }): LanceMergeBuilder;
  whenNotMatchedInsertAll(): LanceMergeBuilder;
  execute(data: Array<Record<string, unknown>>): Promise<unknown>;
}

interface LanceVectorQuery {
  distanceType(distanceType: string): LanceVectorQuery;
  limit(limit: number): LanceVectorQuery;
  where(predicate: string): LanceVectorQuery;
  select(columns: string[]): LanceVectorQuery;
  toArray(): Promise<Array<Record<string, unknown>>>;
}

interface LanceQuery {
  where(predicate: string): LanceQuery;
  select(columns: string[]): LanceQuery;
  limit(limit: number): LanceQuery;
  toArray(): Promise<Array<Record<string, unknown>>>;
}

interface LanceTable {
  countRows(filter?: string): Promise<number>;
  add(data: Array<Record<string, unknown>>): Promise<unknown>;
  optimize(options?: { cleanupOlderThan?: Date }): Promise<unknown>;
  update(opts: {
    where?: string;
    values: Record<string, unknown>;
  }): Promise<unknown>;
  delete(predicate: string): Promise<unknown>;
  mergeInsert(on: string | string[]): LanceMergeBuilder;
  // search(vector) returns VectorQuery | Query in 0.30; we always pass a
  // number[] so it is a VectorQuery, but the declared union forces a cast.
  search(query: number[]): LanceVectorQuery | LanceQuery;
  query(): LanceQuery;
  schema(): Promise<{ fields: Array<{ name: string; type: unknown }> }>;
}

interface LanceConnection {
  tableNames(): Promise<string[]>;
  openTable(name: string): Promise<LanceTable>;
  createEmptyTable(
    name: string,
    schema: unknown,
    options?: { mode?: "create" | "overwrite" },
  ): Promise<LanceTable>;
  dropTable(name: string): Promise<void>;
}

// Shape of the apache-arrow constructors we use. Loaded lazily so the
// pinned 18.1.0 build is only required on the lancedb path.
interface ArrowModule {
  Schema: new (fields: unknown[]) => unknown;
  Field: new (name: string, type: unknown, nullable?: boolean) => unknown;
  Float32: new () => unknown;
  Int32: new () => unknown;
  Int64: new () => unknown;
  Utf8: new () => unknown;
  FixedSizeList: new (listSize: number, child: unknown) => unknown;
}

interface LanceModule {
  connect(uri: string): Promise<LanceConnection>;
}

const MEMORIES_TABLE = "memories";
const BLOBS_TABLE = "index_blobs";
const GRAPH_KV_TABLE = "graph_kv";

// Default Adaptive Knowledge Lifecycle values for a freshly-added vector.
// Mirrors LifecycleFields; written only on a genuine insert and PRESERVED
// across re-adds (see add()).
const DEFAULT_IMPORTANCE = 50; // 0-100
const DEFAULT_RECENCY = 1; // 0-1
const DEFAULT_MATURITY = "draft"; // draft | validated | core

// Escape a string for embedding in a LanceDB SQL filter literal. LanceDB
// filters are SQL strings; obsIds/sessionIds are app-controlled but may
// contain apostrophes, so single-quote them safely.
function sqlString(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

// int64 columns arrive from Arrow as BigInt; coerce to JS number. Epoch-ms
// timestamps are well within Number.MAX_SAFE_INTEGER, so this is lossless.
function toNumber(v: unknown): number {
  if (typeof v === "bigint") return Number(v);
  if (typeof v === "number") return v;
  return Number(v ?? 0);
}

// Map a raw LanceDB row (columns named per the schema) to LifecycleFields.
function rowToLifecycle(row: Record<string, unknown>): LifecycleFields {
  return {
    importance: toNumber(row.importance),
    recency: toNumber(row.recency),
    accessCount: toNumber(row.access_count),
    updateCount: toNumber(row.update_count),
    maturity: typeof row.maturity === "string" ? row.maturity : DEFAULT_MATURITY,
    createdAt: toNumber(row.created_at),
    updatedAt: toNumber(row.updated_at),
  };
}

class LanceVectorBackend implements VectorBackend {
  readonly persistsExternally = true;

  // Cached row count so VectorIndex.size can stay a synchronous getter.
  // Seeded from countRows() in init() (the boot-time accuracy that the
  // daemon relies on to decide whether to rebuild), then adjusted
  // best-effort on add/remove. clear() resets it to 0.
  private _count = 0;
  private table: LanceTable | null = null;

  constructor(
    private readonly conn: LanceConnection,
    private readonly arrow: ArrowModule,
    private readonly dimensions: number,
  ) {}

  // Build the explicit Arrow schema. Dimension is baked into the vector
  // FixedSizeList, which is exactly why an empty-table schema cannot be
  // inferred and must be passed to createEmptyTable.
  private buildSchema(): unknown {
    const a = this.arrow;
    return new a.Schema([
      new a.Field("id", new a.Utf8(), false),
      new a.Field("session_id", new a.Utf8(), true),
      new a.Field(
        "vector",
        new a.FixedSizeList(
          this.dimensions,
          new a.Field("item", new a.Float32(), true),
        ),
        true,
      ),
      new a.Field("importance", new a.Float32(), true),
      new a.Field("recency", new a.Float32(), true),
      new a.Field("access_count", new a.Int32(), true),
      new a.Field("update_count", new a.Int32(), true),
      new a.Field("maturity", new a.Utf8(), true),
      new a.Field("created_at", new a.Int64(), true),
      new a.Field("updated_at", new a.Int64(), true),
    ]);
  }

  async init(): Promise<void> {
    const names = await this.conn.tableNames();
    if (names.includes(MEMORIES_TABLE)) {
      this.table = await this.conn.openTable(MEMORIES_TABLE);
    } else {
      this.table = await this.conn.createEmptyTable(
        MEMORIES_TABLE,
        this.buildSchema(),
      );
    }
    // Accurate boot count is the critical correctness point: index.ts reads
    // size>0 to decide whether to rebuild and whether to validateDimensions.
    this._count = await this.table.countRows();
  }

  private requireTable(): LanceTable {
    if (!this.table) {
      throw new Error("[agentmemory] LanceVectorBackend used before init()");
    }
    return this.table;
  }

  async add(
    obsId: string,
    sessionId: string,
    embedding: Float32Array,
  ): Promise<void> {
    const table = this.requireTable();
    const vector = Array.from(embedding);
    const now = Date.now();

    // Lifecycle-preservation logic. LanceDB's mergeInsert whenMatchedUpdateAll
    // does a FULL ROW REPLACE from the source row (verified in merge.ts:
    // "replacing the old row with the corresponding matching row") — it cannot
    // partially update columns. So a blind upsert would reset importance /
    // recency / maturity / access_count / created_at on every re-add. We must
    // branch on existence:
    //   - not matched -> insert a full row with DEFAULT lifecycle
    //   - matched     -> update ONLY vector, session_id, updated_at and leave
    //                    every lifecycle column untouched
    const existing = await table.countRows(`id = ${sqlString(obsId)}`);

    if (existing > 0) {
      // Re-add: refresh only the vector + session + updated_at. The narrow
      // update(...) preserves all lifecycle columns by definition (it only
      // touches the keys it is given).
      await table.update({
        where: `id = ${sqlString(obsId)}`,
        values: {
          vector,
          session_id: sessionId,
          updated_at: now,
        },
      });
      // No count change on an update.
      return;
    }

    // Genuine insert. whenNotMatchedInsertAll() (without whenMatchedUpdateAll)
    // is an idempotent "insert if absent": if a concurrent writer raced us and
    // inserted the same id, the matched branch is a no-op rather than a
    // lifecycle-clobbering replace. This keeps re-add safe under concurrency.
    await table
      .mergeInsert("id")
      .whenNotMatchedInsertAll()
      .execute([
        {
          id: obsId,
          session_id: sessionId,
          vector,
          importance: DEFAULT_IMPORTANCE,
          recency: DEFAULT_RECENCY,
          access_count: 0,
          update_count: 0,
          maturity: DEFAULT_MATURITY,
          created_at: now,
          updated_at: now,
        },
      ]);
    // Best-effort: increment only on a genuine insert path.
    this._count += 1;
  }

  // Bulk insert for the rebuild/backfill path. rebuildIndex clear()s the table
  // first, so every row is new and a single table.add() of the whole batch is
  // ONE on-disk commit — vs the per-row add() above, which writes a Lance
  // version per vector (fine for occasional live writes, but on a full-corpus
  // rebuild that means thousands of fragments + an O(n) existence check each).
  // Assumes new ids (no upsert); the caller guarantees a prior clear().
  async bulkAdd(items: VectorAddItem[]): Promise<void> {
    if (items.length === 0) return;
    const table = this.requireTable();
    const now = Date.now();
    const rows = items.map((it) => ({
      id: it.obsId,
      session_id: it.sessionId,
      vector: Array.from(it.embedding),
      importance: DEFAULT_IMPORTANCE,
      recency: DEFAULT_RECENCY,
      access_count: 0,
      update_count: 0,
      maturity: DEFAULT_MATURITY,
      created_at: now,
      updated_at: now,
    }));
    await table.add(rows);
    this._count += rows.length;
  }

  async remove(obsId: string): Promise<void> {
    const table = this.requireTable();
    await table.delete(`id = ${sqlString(obsId)}`);
    // Best-effort decrement; never go below zero.
    if (this._count > 0) this._count -= 1;
  }

  async search(query: Float32Array, limit = 20): Promise<VectorSearchHit[]> {
    const table = this.requireTable();
    if (this._count === 0) return [];

    const queryVec = Array.from(query);
    // search(number[]) returns a VectorQuery in 0.30; cast through the union.
    // distanceType("cosine") matches the in-memory backend's cosine ranking;
    // results carry a `_distance` column (lower = closer).
    // _distance is requested explicitly alongside id/session_id. On a vector
    // query LanceDB normally auto-appends the _distance system column, but
    // whether select() preserves it is version-sensitive; listing it makes the
    // projection self-contained either way (harmless if already auto-added).
    const rows = await (table.search(queryVec) as LanceVectorQuery)
      .distanceType("cosine")
      .limit(limit)
      .select(["id", "session_id", "_distance"])
      .toArray();

    // Convert distance -> similarity so higher = better, matching
    // VectorSearchHit semantics and the in-memory backend. Cosine distance is
    // in [0, 2]; 1/(1+d) is monotonic decreasing in d, keeping best-first
    // order intact. toArray() already returns rows in ascending `_distance`.
    return rows.map((row) => {
      const distance = toNumber(row._distance);
      return {
        obsId: String(row.id),
        sessionId: typeof row.session_id === "string" ? row.session_id : "",
        score: 1 / (1 + distance),
      } satisfies VectorSearchHit;
    });
  }

  get size(): number {
    return this._count;
  }

  async clear(): Promise<void> {
    // DROP_STALE recovery after a dimension change: drop and recreate the
    // table empty at the CURRENT opts.dimensions so the next add() writes
    // correctly-sized vectors.
    if (this.table) {
      const names = await this.conn.tableNames();
      if (names.includes(MEMORIES_TABLE)) {
        await this.conn.dropTable(MEMORIES_TABLE);
      }
    }
    this.table = await this.conn.createEmptyTable(
      MEMORIES_TABLE,
      this.buildSchema(),
    );
    this._count = 0;
  }

  // Compact small fragments and prune superseded versions. Per-row add() and
  // the live write stream each create a new Lance version; without periodic
  // compaction the table's file count and on-disk size grow unbounded. We keep
  // no index version history (external tar backups cover recovery), so prune
  // everything older than now.
  async optimize(): Promise<void> {
    const table = this.requireTable();
    await table.optimize({ cleanupOlderThan: new Date() });
  }

  async validateDimensions(expected: number): Promise<DimensionReport> {
    // Empty table: nothing to validate, no dimensions seen.
    if (this._count === 0) {
      return { mismatches: [], seenDimensions: new Set<number>() };
    }
    const table = this.requireTable();
    // The vector dimension is a property of the SCHEMA (FixedSizeList listSize),
    // not of individual rows — every row in a Lance table shares it. So a
    // single schema read tells us the on-disk dimension for all vectors.
    const tableDim = await this.readVectorListSize(table);

    if (tableDim === null) {
      // Could not determine; treat as a mismatch so the boot guard refuses to
      // silently load an index of unknown shape.
      return {
        mismatches: [{ obsId: "<lancedb:memories>", dim: -1 }],
        seenDimensions: new Set<number>([-1]),
      };
    }
    if (tableDim !== expected) {
      return {
        mismatches: [{ obsId: "<lancedb:memories>", dim: tableDim }],
        seenDimensions: new Set<number>([tableDim]),
      };
    }
    return { mismatches: [], seenDimensions: new Set<number>([expected]) };
  }

  // Read the vector column's FixedSizeList listSize from the table schema.
  // Arrow's FixedSizeList exposes the fixed length as `listSize`; we probe a
  // couple of property names defensively because the exact field name on the
  // arrow type instance is not contractually guaranteed across arrow builds.
  private async readVectorListSize(table: LanceTable): Promise<number | null> {
    try {
      const schema = await table.schema();
      const field = schema.fields.find((f) => f.name === "vector");
      if (!field) return null;
      const type = field.type as {
        listSize?: number;
        fixedSize?: number;
        byteWidth?: number;
      };
      if (typeof type.listSize === "number") return type.listSize;
      if (typeof type.fixedSize === "number") return type.fixedSize;
      return null;
    } catch {
      return null;
    }
  }

  // Self-persisting: vectors already live on disk. Nothing to restore from a
  // serialized blob, and nothing to serialize back through the KV.
  async restoreFrom(_serializedJson: string): Promise<void> {
    // no-op
  }

  async serialize(): Promise<string> {
    return "";
  }

  async getLifecycle(
    obsIds: string[],
  ): Promise<Map<string, LifecycleFields>> {
    const result = new Map<string, LifecycleFields>();
    if (obsIds.length === 0 || this._count === 0) return result;
    const table = this.requireTable();

    // id IN (...) over the requested ids. Filter strings cap out in length, so
    // chunk to keep each predicate reasonable on very large id sets.
    const CHUNK = 500;
    for (let i = 0; i < obsIds.length; i += CHUNK) {
      const chunk = obsIds.slice(i, i + CHUNK);
      const inList = chunk.map(sqlString).join(", ");
      const rows = await table
        .query()
        .where(`id IN (${inList})`)
        .select([
          "id",
          "importance",
          "recency",
          "access_count",
          "update_count",
          "maturity",
          "created_at",
          "updated_at",
        ])
        .toArray();
      for (const row of rows) {
        result.set(String(row.id), rowToLifecycle(row));
      }
    }
    return result;
  }

  async listLifecycle(): Promise<Map<string, LifecycleFields>> {
    const result = new Map<string, LifecycleFields>();
    if (this._count === 0) return result;
    const table = this.requireTable();
    // Full scan of id + lifecycle columns (vector column excluded to keep the
    // scan light). A plain query() has NO default row limit in 0.30 — it
    // returns every row — so we deliberately do NOT call .limit() here. Using
    // the best-effort _count as a cap would truncate the scan if the cached
    // count had drifted low. (Verified: query.ts limit() jsdoc — "By default,
    // a plain search has no limit ... every valid row from the table will be
    // returned." The default-10 limit applies only to vector search.)
    const rows = await table
      .query()
      .select([
        "id",
        "importance",
        "recency",
        "access_count",
        "update_count",
        "maturity",
        "created_at",
        "updated_at",
      ])
      .toArray();
    for (const row of rows) {
      result.set(String(row.id), rowToLifecycle(row));
    }
    return result;
  }

  async setLifecycle(
    obsId: string,
    fields: Partial<LifecycleFields>,
  ): Promise<void> {
    const table = this.requireTable();
    // Map the LifecycleFields camelCase keys to the snake_case schema columns,
    // including only the keys the caller actually provided (partial update).
    const values: Record<string, unknown> = {};
    if (fields.importance !== undefined) values.importance = fields.importance;
    if (fields.recency !== undefined) values.recency = fields.recency;
    if (fields.accessCount !== undefined)
      values.access_count = fields.accessCount;
    if (fields.updateCount !== undefined)
      values.update_count = fields.updateCount;
    if (fields.maturity !== undefined) values.maturity = fields.maturity;
    if (fields.createdAt !== undefined) values.created_at = fields.createdAt;
    if (fields.updatedAt !== undefined) values.updated_at = fields.updatedAt;

    if (Object.keys(values).length === 0) return;

    // If the caller mutated reinforcement signals (update_count/importance/
    // recency) without explicitly bumping updated_at, stamp it now so
    // recency-decay scoring sees the touch.
    if (
      values.updated_at === undefined &&
      (fields.updateCount !== undefined ||
        fields.importance !== undefined ||
        fields.recency !== undefined)
    ) {
      values.updated_at = Date.now();
    }

    await table.update({ where: `id = ${sqlString(obsId)}`, values });
  }
}

class LanceIndexBlobStore implements IndexBlobStore {
  private table: LanceTable | null = null;

  constructor(
    private readonly conn: LanceConnection,
    private readonly arrow: ArrowModule,
  ) {}

  // The BM25 "data" blob can be ~11-30MB. Arrow's 32-bit Utf8 offsets cap a
  // single string column's TOTAL bytes at 2GB, and per-value length is also
  // well within range at tens of MB, so plain Utf8 is sufficient here — no
  // LargeUtf8 needed. (LargeUtf8 would only matter past the 2GB aggregate
  // boundary, which a handful of index blobs never approach.)
  private buildSchema(): unknown {
    const a = this.arrow;
    return new a.Schema([
      new a.Field("key", new a.Utf8(), false),
      new a.Field("value", new a.Utf8(), true),
    ]);
  }

  async init(): Promise<void> {
    const names = await this.conn.tableNames();
    if (names.includes(BLOBS_TABLE)) {
      this.table = await this.conn.openTable(BLOBS_TABLE);
    } else {
      this.table = await this.conn.createEmptyTable(
        BLOBS_TABLE,
        this.buildSchema(),
      );
    }
  }

  private requireTable(): LanceTable {
    if (!this.table) {
      throw new Error("[agentmemory] LanceIndexBlobStore used before init()");
    }
    return this.table;
  }

  async set(key: string, value: string): Promise<void> {
    const table = this.requireTable();
    // Upsert keyed on "key": replace the whole row on match (only two columns,
    // both supplied), insert when absent. Full-row replace is correct here
    // because there are no preserved columns — unlike the memories table.
    await table
      .mergeInsert("key")
      .whenMatchedUpdateAll()
      .whenNotMatchedInsertAll()
      .execute([{ key, value }]);
  }

  async get(key: string): Promise<string | null> {
    const table = this.requireTable();
    const rows = await table
      .query()
      .where(`key = ${sqlString(key)}`)
      .select(["value"])
      .limit(1)
      .toArray();
    if (rows.length === 0) return null;
    const v = rows[0].value;
    return typeof v === "string" ? v : null;
  }
}

// Knowledge-graph KV backend: a generic (scope, key, value) table that gives
// the graph its own on-disk files instead of the iii KV. Each node/edge/history
// record is one row; the value column holds the JSON-serialized object, so the
// stored shapes (GraphNode/GraphEdge) round-trip unchanged. The composite
// identity is `scope + " " + key` in the `pk` column (neither a graph scope
// nor a generated id contains a space), used ONLY as the mergeInsert match
// key; get/delete filter on `scope AND key`, list filters on `scope`. Graph
// ids collide across scopes (an edge and its history share a "ge_" id), so a
// composite key is required.
class LanceGraphKvStore implements GraphKvStore {
  private table: LanceTable | null = null;

  constructor(
    private readonly conn: LanceConnection,
    private readonly arrow: ArrowModule,
  ) {}

  private buildSchema(): unknown {
    const a = this.arrow;
    return new a.Schema([
      new a.Field("pk", new a.Utf8(), false),
      new a.Field("scope", new a.Utf8(), false),
      new a.Field("key", new a.Utf8(), false),
      new a.Field("value", new a.Utf8(), true),
    ]);
  }

  async init(): Promise<void> {
    const names = await this.conn.tableNames();
    this.table = names.includes(GRAPH_KV_TABLE)
      ? await this.conn.openTable(GRAPH_KV_TABLE)
      : await this.conn.createEmptyTable(GRAPH_KV_TABLE, this.buildSchema());
  }

  private requireTable(): LanceTable {
    if (!this.table) {
      throw new Error("[agentmemory] LanceGraphKvStore used before init()");
    }
    return this.table;
  }

  private pk(scope: string, key: string): string {
    return `${scope} ${key}`;
  }

  async set<T = unknown>(scope: string, key: string, value: T): Promise<T> {
    const table = this.requireTable();
    await table
      .mergeInsert("pk")
      .whenMatchedUpdateAll()
      .whenNotMatchedInsertAll()
      .execute([
        { pk: this.pk(scope, key), scope, key, value: JSON.stringify(value) },
      ]);
    return value;
  }

  async get<T = unknown>(scope: string, key: string): Promise<T | null> {
    const table = this.requireTable();
    const rows = await table
      .query()
      .where(`scope = ${sqlString(scope)} AND key = ${sqlString(key)}`)
      .select(["value"])
      .limit(1)
      .toArray();
    if (rows.length === 0) return null;
    return this.parse<T>(rows[0].value);
  }

  async delete(scope: string, key: string): Promise<void> {
    const table = this.requireTable();
    await table.delete(
      `scope = ${sqlString(scope)} AND key = ${sqlString(key)}`,
    );
  }

  async list<T = unknown>(scope: string): Promise<T[]> {
    const table = this.requireTable();
    // No .limit(): a plain query returns every matching row (the default-10
    // cap applies only to vector search). The graph is small (hundreds of
    // rows), so a full per-scope scan is cheap.
    const rows = await table
      .query()
      .where(`scope = ${sqlString(scope)}`)
      .select(["value"])
      .toArray();
    const out: T[] = [];
    for (const r of rows) {
      const parsed = this.parse<T>(r.value);
      if (parsed !== null) out.push(parsed);
    }
    return out;
  }

  async optimize(): Promise<void> {
    await this.requireTable().optimize({ cleanupOlderThan: new Date() });
  }

  private parse<T>(v: unknown): T | null {
    if (typeof v !== "string") return null;
    try {
      return JSON.parse(v) as T;
    } catch {
      return null;
    }
  }
}

export async function createLanceBackends(
  opts: CreateBackendsOptions,
): Promise<PersistenceBackends> {
  // Lazy native imports. Both modules are optionalDependencies with platform
  // binaries; surface a clear, actionable error if absent rather than the
  // opaque MODULE_NOT_FOUND the dynamic import would otherwise throw.
  let lancedb: LanceModule;
  let arrow: ArrowModule;
  try {
    lancedb = (await import("@lancedb/lancedb")) as unknown as LanceModule;
  } catch (err) {
    throw new Error(
      "[agentmemory] @lancedb/lancedb not installed. The lancedb vector " +
        "backend requires the @lancedb/lancedb native module (an " +
        "optionalDependency). Install it, or set VECTOR_BACKEND=memory. " +
        `Underlying error: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  try {
    arrow = (await import("apache-arrow")) as unknown as ArrowModule;
  } catch (err) {
    throw new Error(
      "[agentmemory] apache-arrow not installed. The lancedb vector backend " +
        "requires apache-arrow (pinned 18.1.0) to build table schemas. " +
        `Underlying error: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  // One connection shared by both the vector table and the blob table, opened
  // under `${dataDir}/lancedb`.
  const conn = await lancedb.connect(`${opts.dataDir}/lancedb`);

  const vectorBackend = new LanceVectorBackend(conn, arrow, opts.dimensions);
  await vectorBackend.init();

  const blobStore = new LanceIndexBlobStore(conn, arrow);
  await blobStore.init();

  const graphKv = new LanceGraphKvStore(conn, arrow);
  await graphKv.init();

  return {
    vector: new VectorIndex(vectorBackend),
    blobStore,
    graphKv,
  };
}
