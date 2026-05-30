// Always-on MCP-over-HTTP (stateless Streamable HTTP) endpoint.
//
// Remote clients (Codex, Claude Code) POST MCP JSON-RPC to a single URL and
// receive a JSON-RPC response, so they can talk to the always-on daemon
// directly instead of spawning the per-thread local stdio shim (which Codex
// killed mid-thread, losing every tool). See standalone.ts:447-491 for the
// stdio shim's equivalent method switch — this module mirrors that surface
// but over HTTP request/response.
//
// Stateless: each POST is self-contained. We respond application/json for a
// single request (no SSE stream, no Mcp-Session-Id, no server-initiated
// messages — a request/response tool server needs none of that). This is a
// conformant Streamable HTTP server per the 2025-03-26 / 2025-06-18 spec,
// which explicitly permits returning application/json instead of text/event-
// stream when the server does not need to stream.

import { getVisibleTools } from "./tools-registry.js";
import { VERSION } from "../version.js";

// Protocol versions we understand. We echo the client's requested version
// when it's one of these, otherwise we negotiate down to the latest we know.
const SUPPORTED_PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"];
const DEFAULT_PROTOCOL_VERSION = "2025-06-18";

export type ToolCallResult = {
  status_code: number;
  body: unknown;
};

// Dispatch a single tools/call to the shared tool implementation. Provided by
// server.ts so we reuse the exact same switch that backs mcp::tools::call
// (zero duplication of tool logic).
export type ToolDispatcher = (
  name: string,
  args: Record<string, unknown>,
) => Promise<ToolCallResult>;

type JsonRpcId = string | number | null;

type JsonRpcRequest = {
  jsonrpc?: string;
  id?: JsonRpcId;
  method?: unknown;
  params?: unknown;
};

type JsonRpcResponse = {
  jsonrpc: "2.0";
  id: JsonRpcId;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
};

function isNotification(req: JsonRpcRequest): boolean {
  // Per JSON-RPC 2.0 §4 a Notification is a request WITHOUT an id member.
  // An explicit `id: null` is a (discouraged but legal) request and must
  // receive a response, so only an absent id counts as a notification.
  return req.id === undefined;
}

function negotiateProtocolVersion(params: Record<string, unknown>): string {
  const requested = params["protocolVersion"];
  if (
    typeof requested === "string" &&
    SUPPORTED_PROTOCOL_VERSIONS.includes(requested)
  ) {
    return requested;
  }
  return DEFAULT_PROTOCOL_VERSION;
}

function rpcResult(id: JsonRpcId, result: unknown): JsonRpcResponse {
  return { jsonrpc: "2.0", id, result };
}

function rpcError(
  id: JsonRpcId,
  code: number,
  message: string,
): JsonRpcResponse {
  return { jsonrpc: "2.0", id, error: { code, message } };
}

// Handle one JSON-RPC message. Returns a JsonRpcResponse to send back, or null
// for notifications (which per JSON-RPC 2.0 §4 / the MCP transport contract
// MUST NOT receive a response body).
async function handleOne(
  req: JsonRpcRequest,
  dispatch: ToolDispatcher,
): Promise<JsonRpcResponse | null> {
  // A batch element can be any JSON value (null, a number, a string). Reject
  // non-objects up front: reading `.id` / `.method` off `null` throws and would
  // crash the whole batch handler; a primitive is simply an Invalid Request.
  if (!req || typeof req !== "object") {
    return rpcError(null, -32600, "Invalid Request");
  }
  const rawId = req.id;
  const id: JsonRpcId =
    typeof rawId === "string" || typeof rawId === "number" ? rawId : null;

  if (req.jsonrpc !== "2.0" || typeof req.method !== "string") {
    if (isNotification(req)) return null;
    return rpcError(id, -32600, "Invalid Request");
  }

  const method = req.method;
  const params =
    req.params && typeof req.params === "object"
      ? (req.params as Record<string, unknown>)
      : {};
  const notification = isNotification(req);

  // notifications/* carry no id and expect no response (e.g.
  // notifications/initialized). Mirrors standalone.ts:459-460.
  if (method.startsWith("notifications/")) {
    return null;
  }

  switch (method) {
    case "initialize":
      return rpcResult(id, {
        protocolVersion: negotiateProtocolVersion(params),
        serverInfo: { name: "agentmemory", version: VERSION },
        capabilities: { tools: {} },
      });

    case "tools/list":
      if (notification) return null;
      return rpcResult(id, { tools: getVisibleTools() });

    case "tools/call": {
      if (notification) return null;
      const name = params["name"];
      if (typeof name !== "string" || !name.trim()) {
        return rpcError(id, -32602, "Invalid params: name is required");
      }
      const args =
        params["arguments"] && typeof params["arguments"] === "object"
          ? (params["arguments"] as Record<string, unknown>)
          : {};
      const result = await dispatch(name, args);
      // The shared dispatcher returns the REST-shaped { status_code, body }.
      // On success body is the MCP { content:[...] } envelope, which is
      // exactly the tools/call result shape. On a tool-level validation error
      // (4xx) surface it as JSON-RPC content with isError so the client sees
      // the message rather than a transport failure.
      if (result.status_code >= 200 && result.status_code < 300) {
        return rpcResult(id, result.body);
      }
      const errBody = result.body as { error?: unknown };
      const msg =
        typeof errBody?.error === "string" ? errBody.error : "Tool call failed";
      return rpcResult(id, {
        content: [{ type: "text", text: `Error: ${msg}` }],
        isError: true,
      });
    }

    default:
      if (notification) return null;
      return rpcError(id, -32601, `Method not found: ${method}`);
  }
}

// Process a parsed JSON-RPC body (single object or batch array) and produce the
// value to serialize back. Returns null when there is nothing to send (a body
// of only notifications, or a single notification) — the caller should reply
// 202 with no body in that case.
export async function handleJsonRpcBody(
  body: unknown,
  dispatch: ToolDispatcher,
): Promise<JsonRpcResponse | JsonRpcResponse[] | null> {
  // Batch: an array of requests. Per JSON-RPC 2.0 §6 the server returns an
  // array of responses for the non-notification members, and nothing if every
  // member was a notification.
  if (Array.isArray(body)) {
    if (body.length === 0) {
      return rpcError(null, -32600, "Invalid Request");
    }
    const responses: JsonRpcResponse[] = [];
    for (const item of body) {
      const res = await handleOne(item as JsonRpcRequest, dispatch);
      if (res) responses.push(res);
    }
    return responses.length ? responses : null;
  }

  if (!body || typeof body !== "object") {
    return rpcError(null, -32700, "Parse error");
  }

  return handleOne(body as JsonRpcRequest, dispatch);
}
