/**
 * SSE client — fetch + ReadableStream，自己解析 SSE 帧。
 *
 * 为什么不用 EventSource：后端 SSE 端点（/chat/sessions/.../messages、
 * /rag/stream、/agent/stream、/insurance/.../stream）全是 POST + JSON
 * body。EventSource 只支持 GET 且无法塞 Authorization header。fetch
 * + ReadableStream 是唯一干净的路径。
 *
 * 协议（docs §5）：
 *   event: <name>\n           （ASCII，单词，runner 限词表）
 *   data: <one-line JSON>\n   （ensure_ascii=False，CJK 直出）
 *   \n                         （帧分隔）
 *
 * Heartbeat: `: keepalive\n\n`（comment 帧）。我们在解析层吞掉，
 * 不暴露给上层。中转站慢时若 ≥15s 没事件后端会自动发 heartbeat
 * 让代理（nginx 60s 默认 timeout）保连。
 *
 * 终止帧约定：`done` 永远是最后一帧；`error` 之后必有 `done`。
 *
 * 跨帧拼接：fetch 给的每个 chunk 不一定刚好对齐 `\n\n`，所以维护
 * `buffer` 字符串，按 `\n\n` 切，最后一段（半帧）回填到 buffer。
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { SSEEvent, SSEEventName, SSEStatus } from "@/lib/sse-types";
import { useAuthStore } from "@/stores/auth";

// ============================================================
// 低层：parseSSEStream — async generator，吐解析后的事件
// ============================================================

interface RawFrame {
  event: string;
  /** raw data string — 调用方负责 JSON.parse。 */
  data: string;
}

/**
 * 把一个 SSE 帧字符串（不含末尾的 \n\n 分隔符）解析成 {event, data}。
 * 协议约定每帧只有一行 `event:` 和一行 `data:`，但兼容多行 data
 * 的标准（拼接成 `\n` 串）以防后端某天改实现。
 *
 * CRLF 兼容：跨代理可能把 `\n` 改成 `\r\n`；我们按 `\r?\n` 切行，
 * 单独的 `\r` 也会被 trim 掉。否则 event 名会变成 `"done\r"`，
 * 终止判断失效。
 */
function parseFrame(frame: string): RawFrame | null {
  let event = "message"; // SSE 默认 event 名（虽然我们后端总会发 event:）
  const dataLines: string[] = [];

  for (const rawLine of frame.split(/\r?\n/)) {
    const line = rawLine.replace(/\r$/, "");
    if (!line) continue;
    if (line.startsWith(":")) continue; // comment 行（heartbeat / keepalive）
    const colon = line.indexOf(":");
    if (colon < 0) continue;
    const field = line.slice(0, colon);
    // SSE 规定 colon 后可有一个可选空格，吃掉它
    const value =
      colon + 1 < line.length && line.charAt(colon + 1) === " "
        ? line.slice(colon + 2)
        : line.slice(colon + 1);
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
    // 其它字段（id / retry）当前后端不发，忽略
  }

  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

/**
 * 把一个 ReadableStream<Uint8Array>（fetch.body）转成 RawFrame 流。
 * 跨 chunk buffer + UTF-8 增量解码。
 *
 * 取消语义：finally 总会 cancel reader —— 调用方提前 break（done 帧
 * 提前退出）时，reader.cancel() 会触发 server-side response
 * generator 收到 disconnect → EventBus.is_closed=True → 释放 LLM
 * stream，避免我们已经不需要的 token 还在烧钱。
 */
async function* streamFrames(
  body: ReadableStream<Uint8Array>,
  signal: AbortSignal,
): AsyncGenerator<RawFrame> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let buffer = "";

  // \r?\n\r?\n 兼容 LF / CRLF 帧分隔。
  const SEP_RE = /\r?\n\r?\n/;

  try {
    while (true) {
      if (signal.aborted) return;
      const { value, done } = await reader.read();
      if (done) {
        const tail = buffer + decoder.decode();
        if (tail.trim()) {
          const frame = parseFrame(tail);
          if (frame) yield frame;
        }
        return;
      }
      buffer += decoder.decode(value, { stream: true });

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const m = SEP_RE.exec(buffer);
        if (!m) break;
        const frameStr = buffer.slice(0, m.index);
        buffer = buffer.slice(m.index + m[0].length);
        const frame = parseFrame(frameStr);
        if (frame) yield frame;
      }
    }
  } finally {
    // 无论是消费方提前 break / done 退出 / abort / 异常，都尝试 cancel。
    // 已被 cancel 的 reader 再 cancel 是 no-op；releaseLock 在 cancel
    // 后调用会抛 — 我们主动 cancel 就不用 releaseLock 了。
    try {
      await reader.cancel();
    } catch {
      // 已 cancel / 已 released — 忽略。
    }
  }
}

// ============================================================
// 高层：fetchSSE — 对外的 typed event 流
// ============================================================

/**
 * 把 baseURL 跟 endpoint 合并成最终 URL。规则：
 *   - endpoint 是 absolute URL（http(s)://）→ 原样返回
 *   - baseURL 是 absolute URL（带 origin）→ new URL(endpoint, baseURL)
 *   - baseURL 是 path（如 "/api"）→ "/api" + endpoint
 *   - 已经以 baseURL 开头的 endpoint 视为完整路径，不重复拼
 */
function resolveURL(endpoint: string, baseURL: string): string {
  if (/^https?:\/\//i.test(endpoint)) return endpoint;

  // baseURL 是 origin (http://host[:port][/path])
  if (/^https?:\/\//i.test(baseURL)) {
    const base = baseURL.endsWith("/") ? baseURL : baseURL + "/";
    const ep = endpoint.startsWith("/") ? endpoint.slice(1) : endpoint;
    return new URL(ep, base).toString();
  }

  // baseURL 是 path（"/api" 类）
  const basePath = baseURL.replace(/\/+$/, ""); // 去尾 /
  const ep = endpoint.startsWith("/") ? endpoint : "/" + endpoint;
  if (basePath && ep.startsWith(basePath + "/")) return ep; // 已含前缀
  if (basePath && ep === basePath) return ep;
  return basePath + ep;
}

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

export interface FetchSSEOptions {
  /** body 直接被 JSON.stringify。 */
  body?: unknown;
  /** override request method, default POST. */
  method?: "POST" | "GET";
  /** 额外 header（不会覆盖 Authorization / Content-Type）。 */
  headers?: Record<string, string>;
  signal: AbortSignal;
}

/**
 * 低阶 typed event 流 — 给非 React 调用方。React 组件请用 useSSE。
 *
 * 协议解析：
 *   - heartbeat 由 streamFrames 在 parseFrame 层吞（comment lines）
 *   - 每帧 data 走 JSON.parse；解析失败 console.warn 不中断流
 *   - 收到 done 后 break；server-side EventBus 会观察到 cancel 释
 *     放 LLM stream
 *   - 收到 error 不 break，等下一帧 done（docs §5 契约）
 */
export async function* fetchSSE(
  url: string,
  opts: FetchSSEOptions,
): AsyncGenerator<SSEEvent> {
  const { body, method = "POST", headers = {}, signal } = opts;

  const token = useAuthStore.getState().token;
  const fullUrl = resolveURL(url, API_BASE);

  const resp = await fetch(fullUrl, {
    method,
    headers: {
      Accept: "text/event-stream",
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal,
  });

  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const text = await resp.text();
      if (text) detail = `${detail} — ${text.slice(0, 256)}`;
    } catch {
      // ignore
    }
    throw new SSEHttpError(resp.status, detail);
  }

  if (!resp.body) {
    throw new Error("SSE response has no body");
  }

  for await (const frame of streamFrames(resp.body, signal)) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(frame.data);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn("[sse] bad JSON in frame, dropping:", frame.event, frame.data, e);
      continue;
    }

    yield {
      event: frame.event as SSEEventName,
      data: parsed,
    } as SSEEvent;

    if (frame.event === "done") {
      // streamFrames 的 finally 会 cancel reader，server 会感知。
      return;
    }
  }
}

export class SSEHttpError extends Error {
  // 显式赋值而不是 parameter property —— tsconfig 启用了
  // erasableSyntaxOnly，不允许 parameter property 这种 emit-time
  // syntax。
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "SSEHttpError";
    this.status = status;
  }
}

// ============================================================
// React hook
// ============================================================

export interface UseSSEResult {
  status: SSEStatus;
  /** 全部已收到的事件，按到达顺序。done / error 都在这里面。 */
  events: SSEEvent[];
  /** 最后的错误（来自 SSEHttpError、network、SSE error 帧）。 */
  error: Error | null;
  /** 启动一次新流；会先 abort 之前未结束的流。 */
  start: (url: string, body?: unknown) => void;
  /** 主动中断（用户离开页面 / 切 mode）。 */
  abort: () => void;
  /** 清空 events + 重置 status 到 idle。 */
  reset: () => void;
}

/**
 * Stability contract for callbacks:
 *
 * ``onEvent`` / ``onDone`` are captured into a ref and refreshed via
 * ``useEffect`` (not during render) to avoid the
 * ``react-hooks/refs-not-during-render`` lint and to behave correctly
 * under StrictMode's double render. This means the ref lags by **one
 * render**: a callback fired between two consecutive renders sees the
 * previous callback identity.
 *
 * **Semantic stability is enough**: existing callsites pass inline
 * arrow functions but their bodies only touch refs / dispatchers /
 * setState whose identity is stable, so the one-render lag is
 * invisible. You only need to wrap in ``useCallback`` if your callback
 * closes over a value that genuinely differs between renders AND the
 * callback may fire in that exact in-between window. For chat/agent
 * stream patterns (token append into a ref, dispatch to a reducer),
 * inline is fine.
 *
 * ``dropTokenFrames`` and ``abortOnUnmount`` are also read off the
 * latest ref each event, so toggling them mid-stream takes effect on
 * the next event after the next commit.
 */
export interface UseSSEOptions {
  /**
   * 每收到一个事件触发的 hook。比 react state 提前一拍 —— 用于
   * 把 token 增量直接拼到外部状态（避免每帧都触发组件 re-render
   * 全部历史）。**保持语义稳定**（见上方 contract，inline arrow
   * 闭包到稳定 ref/dispatch 即可）。
   */
  onEvent?: (ev: SSEEvent) => void;
  /**
   * 流自然 done 时触发，传完整事件快照（包含 token / error 等）。
   * 调用方可以遍历快照判断 "是 error→done 还是 clean done"。
   * **保持语义稳定**（见上方 contract）。
   */
  onDone?: (events: SSEEvent[]) => void;
  /**
   * 是否在 hook unmount 时自动 abort。默认 true —— 用户离开页面
   * 不应该继续付 LLM token 钱。
   */
  abortOnUnmount?: boolean;
  /**
   * 当为 true 时，``token`` 帧仅经 ``onEvent`` 派发到外部累加器
   * （如 ChatPage 的 reducer 拼到 ``answer`` 字符串），不再追加进
   * ``events`` 数组、也不再进入 onDone 的快照。
   *
   * 默认 false 保持原行为：fraud/agent 模式 GraphPage 直接遍历
   * ``sse.events`` 取 token 拼字符串，需要保留全帧。
   *
   * 为什么要 opt-in：长流式回答里 token 帧的数量 ≈ 答案 token 数
   * （动辄几千），全保留 + onDone 时再二次遍历 ``allEvents``，Web
   * 多轮对话连续累积是 Chrome tab OOM 的核心来源（用户实测两轮即崩）。
   * Chat 路径已经把 token delta 即时拼到 turn.answer，没人再读
   * ``sse.events`` 里的 token 帧——可以放心丢。
   */
  dropTokenFrames?: boolean;
}

/**
 * React hook — 单 SSE 连接的生命周期管理。
 *
 * 设计要点：
 *   1) `runIdRef` 防 race：用户连点发送时，旧流的回调用过期 runId
 *      直接吞，不污染新流的 events。
 *   2) abort 用 AbortController + DOMException("AbortError") 识别。
 *   3) sawErrorFrame：docs §5 contract 是 `error→done`。如果只看
 *      最后一帧 done 就 setStatus("done") 会丢 error 信号。这里
 *      跟踪是否在流过程中见过 error 帧，done 时按需 setStatus("error")。
 *   4) allEvents：onDone 的回调拿到完整列表（用户回调可以遍历）。
 *      React state 仍走微批 16ms flush，避免 token 风暴 N 次 render。
 */
export function useSSE(opts: UseSSEOptions = {}): UseSSEResult {
  const [status, setStatus] = useState<SSEStatus>("idle");
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [error, setError] = useState<Error | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const runIdRef = useRef(0);
  // ``opts`` is captured once on mount via useRef and refreshed via a
  // dedicated effect. Mutating ``optsRef.current = opts`` directly in
  // the render body would trip ``react-hooks/refs-not-during-render``
  // (and is genuinely unsafe under Strict Mode's double-render). The
  // effect runs after commit so callbacks fired between two consecutive
  // renders see the previous opts — acceptable trade-off because
  // opts.onEvent / onDone are normally stable across renders.
  const optsRef = useRef(opts);
  useEffect(() => {
    optsRef.current = opts;
  }, [opts]);

  const abort = useCallback(() => {
    // 推进 runIdRef，让所有还在 microtask 队列里的 flush() 看到的
    // myRunId 都是过期的 → 直接丢弃缓冲事件，避免 abort 后 buffered
    // 帧又流回 sse.events 触发下游 effect 当作"新事件"处理。典型现
    // 场：GraphPage agent 中途点"中止"，对应 SSE 已 buffer 了一帧
    // graph_subgraph，若不在这里 bump runId，下一次微任务 tick 就把
    // 它 flush 进 events，effect 拿来又发一次 /graph/expand，把刚清
    // 的 overlay 又盖回去。
    runIdRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const reset = useCallback(() => {
    abort();
    setStatus("idle");
    setEvents([]);
    setError(null);
  }, [abort]);

  const start = useCallback(
    (url: string, body?: unknown) => {
      // 用 abort() 而不是直接 abortRef.current?.abort() —— 让 runIdRef
      // 在 cancel old run 的同时 bump 一次，buffered flush 自然失效。
      // 紧接着 ++runIdRef 再前进一格，这才是"新 run"的 myRunId。
      abort();
      const myRunId = ++runIdRef.current;
      const controller = new AbortController();
      abortRef.current = controller;

      setStatus("connecting");
      setEvents([]);
      setError(null);

      void (async () => {
        // 完整事件历史（onDone 用）。React state 仍走 microbatch。
        const allEvents: SSEEvent[] = [];
        let sawErrorFrame: ErrorEventLike | null = null;
        const buffered: SSEEvent[] = [];
        let flushTimer: ReturnType<typeof setTimeout> | null = null;
        const flush = () => {
          flushTimer = null;
          if (runIdRef.current !== myRunId) return;
          if (buffered.length === 0) return;
          const drained = buffered.splice(0, buffered.length);
          setEvents((prev) => prev.concat(drained));
        };
        const scheduleFlush = () => {
          if (flushTimer != null) return;
          flushTimer = setTimeout(flush, 16);
        };

        try {
          let firstStreaming = true;
          for await (const ev of fetchSSE(url, {
            body,
            signal: controller.signal,
          })) {
            if (runIdRef.current !== myRunId) break;

            if (firstStreaming) {
              firstStreaming = false;
              setStatus("streaming");
            }
            if (ev.event === "error") {
              sawErrorFrame = ev.data;
            }
            // Lean storage path: token frames bypass both allEvents
            // and the React state buffer when opted in. ``onEvent``
            // (below) still fires so the consumer's reducer can
            // append the delta to its own answer string.
            const skipStorage =
              optsRef.current.dropTokenFrames === true && ev.event === "token";
            if (!skipStorage) {
              allEvents.push(ev);
              buffered.push(ev);
              scheduleFlush();
            }

            try {
              optsRef.current.onEvent?.(ev);
            } catch (e) {
              // eslint-disable-next-line no-console
              console.error("[sse] onEvent threw:", e);
            }

            if (ev.event === "done") {
              if (flushTimer != null) clearTimeout(flushTimer);
              flush();
              if (runIdRef.current === myRunId) {
                if (sawErrorFrame) {
                  // docs §5: error 帧之后必有 done。流"做完了但有错"。
                  setStatus("error");
                  setError(
                    new Error(
                      sawErrorFrame.message ||
                        `SSE stream errored (${sawErrorFrame.type ?? "unknown"})`,
                    ),
                  );
                } else {
                  setStatus("done");
                }
                try {
                  optsRef.current.onDone?.(allEvents);
                } catch (e) {
                  // eslint-disable-next-line no-console
                  console.error("[sse] onDone threw:", e);
                }
              }
              return;
            }
          }

          // 流自然结束但没收到 done —— 视为不完整结束。
          if (runIdRef.current === myRunId) {
            if (flushTimer != null) {
              clearTimeout(flushTimer);
              flush();
            }
            if (controller.signal.aborted) {
              setStatus("aborted");
            } else if (sawErrorFrame) {
              setStatus("error");
              setError(
                new Error(
                  sawErrorFrame.message ||
                    `SSE stream errored (${sawErrorFrame.type ?? "unknown"})`,
                ),
              );
            } else {
              setStatus("error");
              setError(new Error("SSE stream ended without `done` frame"));
            }
          }
        } catch (e: unknown) {
          if (runIdRef.current !== myRunId) return;
          if (
            (e as DOMException)?.name === "AbortError" ||
            controller.signal.aborted
          ) {
            setStatus("aborted");
            return;
          }
          if (flushTimer != null) {
            clearTimeout(flushTimer);
            flush();
          }
          setStatus("error");
          setError(e instanceof Error ? e : new Error(String(e)));
        }
      })();
    },
    // status 故意不进 deps —— start 不应该因为 status 变化而重建。
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  // unmount 时自动 abort
  useEffect(() => {
    return () => {
      if (opts.abortOnUnmount !== false) {
        abortRef.current?.abort();
      }
    };
    // 只在 mount/unmount 跑
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { status, events, error, start, abort, reset };
}

interface ErrorEventLike {
  message: string;
  type?: string;
}
