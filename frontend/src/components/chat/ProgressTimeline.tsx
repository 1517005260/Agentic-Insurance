import { useMemo, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Loader2,
  Check,
  AlertTriangle,
  Wrench,
  X as XIcon,
} from "lucide-react";

import { cn } from "@/lib/utils";
import type { AssistantStatus } from "./types";
import type { SSEEvent, ToolCallEvent, ToolResultEvent } from "@/lib/sse-types";

interface Props {
  events: SSEEvent[];
  status: AssistantStatus;
  /** 父组件控制：第一个 token 到达时翻 true，自动折叠成 1 行 summary。 */
  autoCollapsed: boolean;
  /**
   * Turn 标识。同一 turn 内用户手动展开/折叠后不应该再被 autoCollapse
   * 重置；新 turn 时 reset 用户偏好。
   */
  turnKey: string;
}

// ============================================================
// 行渲染：把 tool_call / tool_result 合并成一行；其它事件原样
// ============================================================

interface MergedRow {
  kind: "tool" | "event" | "thought";
  /** 时间顺序里的 leading event index（唯一 key）。 */
  index: number;
  /** for "tool" kind */
  loop?: number;
  toolName?: string;
  /** Args 缩略（仅文本类型；id 等数值/数组省略）。 */
  argsBlurb?: string;
  /** 结果状态：pending = 还在等 result */
  toolStatus?: "pending" | "ok" | "error";
  toolError?: string;
  /** for "thought" kind */
  thoughtText?: string;
  thoughtLoop?: number;
  /** for "event" kind */
  raw?: SSEEvent;
}

function mergeRows(events: SSEEvent[]): MergedRow[] {
  const out: MergedRow[] = [];
  // (loop, name, occurrenceIdx) → index in `out`
  const seenCallByLoop = new Map<string, number>();
  // 每个 loop 的 thought 行索引 — status phase=thinking 创建占位行；
  // 同 loop 的 thought event 后到来时把 text 写进去。
  const thoughtRowByLoop = new Map<number, number>();

  for (let i = 0; i < events.length; i++) {
    const ev = events[i];

    // status phase=thinking 只用来"开槽"，本身不渲染 —— 大多数模型在
     // 工具调用前不会输出 reasoning 文本（仅 tool_calls），强行渲染一行
     // "模型未输出推理文本" 占位会让时间线拥挤。一旦有 thought event 真
     // 来了，下面的分支会就地新增 thought 行。
    if (ev.event === "status" && (ev.data as { phase?: string }).phase === "thinking") {
      continue;
    }

    if (ev.event === "tool_call") {
      const key = `${ev.data.loop}:${ev.data.name}:${countKey(seenCallByLoop, ev.data.loop, ev.data.name)}`;
      const row: MergedRow = {
        kind: "tool",
        index: i,
        loop: ev.data.loop,
        toolName: ev.data.name,
        argsBlurb: blurbArgs(ev.data.args),
        toolStatus: "pending",
      };
      out.push(row);
      seenCallByLoop.set(key, out.length - 1);
      continue;
    }
    if (ev.event === "tool_result") {
      // 配对：找最近的 same (loop, name) pending 行
      const key = pickPendingKey(seenCallByLoop, ev.data.loop, ev.data.name);
      if (key != null) {
        const idx = seenCallByLoop.get(key);
        if (idx != null && out[idx]?.kind === "tool" && out[idx].toolStatus === "pending") {
          out[idx].toolStatus = ev.data.error ? "error" : "ok";
          out[idx].toolError = ev.data.error;
          seenCallByLoop.delete(key);
          continue;
        }
      }
      // 没配上 — 不显示（避免把孤儿 result 单独成行污染时间线）
      continue;
    }
    if (ev.event === "token") {
      // tokens 不进时间线（已经体现在 answer 主区，重复显示无意义）
      continue;
    }
    if (ev.event === "thought") {
      const loop = ev.data.loop;
      const existing = thoughtRowByLoop.get(loop);
      if (existing != null && out[existing]?.kind === "thought") {
        // 同一 loop 已有 thought 行：拼接而不是覆盖（一个 loop 内
        // 模型可能多次产出 reasoning 文本，覆盖会丢前一段）。
        const prev = out[existing].thoughtText ?? "";
        out[existing].thoughtText = prev
          ? `${prev}\n\n${ev.data.text}`
          : ev.data.text;
        continue;
      }
      out.push({
        kind: "thought",
        index: i,
        thoughtLoop: loop,
        thoughtText: ev.data.text,
      });
      thoughtRowByLoop.set(loop, out.length - 1);
      continue;
    }
    out.push({ kind: "event", index: i, raw: ev });
  }
  return out;
}

/** 计算同 (loop, name) 第几次调用，用于 result 配对去重。 */
function countKey(map: Map<string, number>, loop: number, name: string): number {
  let n = 0;
  for (const k of map.keys()) {
    if (k.startsWith(`${loop}:${name}:`)) n++;
  }
  return n;
}

function pickPendingKey(map: Map<string, number>, loop: number, name: string): string | null {
  // 取序号最小的 pending（FIFO）
  let chosen: string | null = null;
  let chosenSeq = Infinity;
  for (const k of map.keys()) {
    if (!k.startsWith(`${loop}:${name}:`)) continue;
    const seq = Number(k.split(":")[2]);
    if (seq < chosenSeq) {
      chosen = k;
      chosenSeq = seq;
    }
  }
  return chosen;
}

/**
 * 把 tool_call.args 渲染成单行短摘要。
 *
 * 设计目标：用户能"一眼看懂这次调用问了什么"。所以保留：
 *   - query / pattern / filename_regex / question / text 等"问题类"字段
 *   - file_id / file_ids（缩写成尾部 8 字符）—— "读了哪个文件"
 *   - page_numbers / page_id（数字数组）—— "哪几页"
 *   - 小数组的 channels / unit_type 等枚举
 *
 * 抽象的内部 id（hash_id / node_id / passage_id / observation_id）
 * 不展示 —— 用户看了也没意义。
 */
function blurbArgs(args: Record<string, unknown> | undefined | null): string {
  if (!args || typeof args !== "object") return "";
  // 完全省略的 id 字段：内部技术 ID，用户读不出含义。
  const HIDDEN_KEYS = new Set([
    "hash_id",
    "node_id",
    "passage_id",
    "table_row_id",
    "observation_id",
    "page_id",
  ]);
  const parts: string[] = [];
  for (const [k, v] of Object.entries(args)) {
    if (HIDDEN_KEYS.has(k)) continue;
    if (v == null) continue;
    const rendered = renderArgValue(k, v);
    if (rendered != null) parts.push(rendered);
  }
  return truncate(parts.join("  "), 110);
}

function renderArgValue(key: string, v: unknown): string | null {
  // file_id / file_ids 缩成尾 8 字符（保留尾部辨识度，前端 chip 也是这套规则）
  if (key === "file_id" && typeof v === "string") {
    return `file=${shortFileId(v)}`;
  }
  if (key === "file_ids" && Array.isArray(v)) {
    if (v.length === 0) return null;
    const ids = v
      .filter((x): x is string => typeof x === "string")
      .map(shortFileId);
    if (ids.length === 0) return null;
    return `files=[${ids.slice(0, 3).join(",")}${ids.length > 3 ? `,…+${ids.length - 3}` : ""}]`;
  }
  if (typeof v === "string") {
    const s = v.trim();
    if (!s) return null;
    return `${key}=${truncate(s, 48)}`;
  }
  if (typeof v === "number" || typeof v === "boolean") {
    return `${key}=${v}`;
  }
  if (Array.isArray(v)) {
    if (v.length === 0) return null;
    if (v.length <= 8 && v.every((x) => typeof x === "string" || typeof x === "number")) {
      return `${key}=[${v.map(String).slice(0, 8).join(",")}]`;
    }
    return `${key}(${v.length})`;
  }
  // object 嵌套不展示
  return null;
}

function shortFileId(id: string): string {
  // 文件 id 形式 "<人类前缀>_<8位 hash>"；取最后 8 字节作辨识。
  // 如果不是这种形式就头/尾各取 4 字符。
  const m = id.match(/_([0-9a-f]{8,})$/i);
  if (m) return m[1].slice(-8);
  if (id.length <= 12) return id;
  return `${id.slice(0, 4)}…${id.slice(-4)}`;
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n - 1) + "…";
}

// ============================================================
// 配色 / icon
// ============================================================

const EVENT_TONE: Record<string, string> = {
  status: "text-info",
  preprocess: "text-ink-muted",
  retrieval: "text-primary-700",
  reranked: "text-primary-700",
  obligation: "text-success",
  claim: "text-success",
  gap: "text-danger",
  citations: "text-accent-700",
  final: "text-success",
  error: "text-danger",
};

function summarize(ev: SSEEvent): string {
  switch (ev.event) {
    case "status":
      return `阶段 → ${ev.data.phase}`;
    case "preprocess":
      return `${ev.data.step} ${ev.data.phase}${ev.data.elapsed_ms != null ? ` · ${ev.data.elapsed_ms}ms` : ""}`;
    case "retrieval":
      return ev.data.channel === "web"
        ? `web · ${ev.data.n_results} 来源`
        : `${ev.data.channel} · ${ev.data.elapsed_ms}ms · ${ev.data.hits.length} 命中`;
    case "reranked":
      return `重排 → ${ev.data.pages.length} 页 · ${ev.data.elapsed_ms}ms`;
    case "obligation":
      return `${ev.data.id} ${ev.data.kind} → ${ev.data.status}`;
    case "claim":
      return `${ev.data.id} ${ev.data.kind}${ev.data.status === "REMOVED" ? " (已撤)" : ""}`;
    case "gap":
      return `${ev.data.id} ${ev.data.kind} → ${ev.data.status}`;
    case "citations":
      return `${ev.data.items.length} 条引用`;
    case "final":
      return "已完成";
    case "error":
      return ev.data.message;
    default:
      return ev.event;
  }
}

/**
 * 一行 summary：检索通道命中 / 重排页数 / 用了几个工具。
 *
 * 折叠态显示，让用户一眼看到 "AI 干了什么" 而不是堆 20 行细节。
 */
function buildOneLineSummary(events: SSEEvent[]): string {
  const retrievals = events.filter(
    (e): e is Extract<SSEEvent, { event: "retrieval" }> => e.event === "retrieval",
  );
  const totalHits = retrievals.reduce(
    (n, e) => n + (e.data.channel === "web" ? e.data.n_results : e.data.hits.length),
    0,
  );

  const reranked = events.find(
    (e): e is Extract<SSEEvent, { event: "reranked" }> => e.event === "reranked",
  );
  const toolCalls = events.filter((e): e is ToolCallEvent => e.event === "tool_call").length;
  const toolFails = events.filter(
    (e): e is ToolResultEvent => e.event === "tool_result" && !!e.data.error,
  ).length;
  // 计 loop 数：每条 status phase=thinking 算一个 loop，已经在 mergeRows
  // 里去重过；这里只统计有真实 content 的 thought event。
  const thoughts = events.filter((e) => e.event === "thought").length;
  const loops = events.filter(
    (e): e is Extract<SSEEvent, { event: "status" }> =>
      e.event === "status" && (e.data as { phase?: string }).phase === "thinking",
  ).length;
  const obligations = new Set(
    events
      .filter((e): e is Extract<SSEEvent, { event: "obligation" }> => e.event === "obligation")
      .map((e) => e.data.id),
  ).size;

  const parts: string[] = [];
  if (retrievals.length) {
    parts.push(
      totalHits > 0
        ? `检索 ${retrievals.length} 通道 · ${totalHits} 命中`
        : `检索 ${retrievals.length} 通道，无召回`,
    );
  }
  if (reranked) parts.push(`重排 ${reranked.data.pages.length} 页`);
  if (loops) {
    parts.push(thoughts ? `${loops} 轮思考 (${thoughts} 次有 reasoning)` : `${loops} 轮思考`);
  } else if (thoughts) {
    parts.push(`${thoughts} 次思考`);
  }
  if (toolCalls) {
    parts.push(
      toolFails > 0 ? `调用 ${toolCalls} 次工具 (${toolFails} 失败)` : `调用 ${toolCalls} 次工具`,
    );
  }
  if (obligations) parts.push(`${obligations} 个 obligation`);
  return parts.join(" · ") || "处理中";
}

const STATUS_ICON: Record<AssistantStatus, React.ReactNode> = {
  connecting: <Loader2 className="h-3.5 w-3.5 animate-spin text-ink-muted" />,
  streaming: <Loader2 className="h-3.5 w-3.5 animate-spin text-primary-600" />,
  done: <Check className="h-3.5 w-3.5 text-success" />,
  error: <AlertTriangle className="h-3.5 w-3.5 text-danger" />,
  aborted: <AlertTriangle className="h-3.5 w-3.5 text-ink-subtle" />,
};

const STATUS_LABEL: Record<AssistantStatus, string> = {
  connecting: "连接中…",
  streaming: "思考中",
  done: "完成",
  error: "出错",
  aborted: "已中止",
};

const TOOL_STATUS_ICON: Record<NonNullable<MergedRow["toolStatus"]>, React.ReactNode> = {
  pending: <Loader2 className="h-3 w-3 animate-spin text-warning" />,
  ok: <Check className="h-3 w-3 text-success" />,
  error: <XIcon className="h-3 w-3 text-danger" />,
};

export function ProgressTimeline({ events, status, autoCollapsed, turnKey }: Props) {
  const [userOverride, setUserOverride] = useState<boolean | null>(null);
  // 只在 turn 切换时清 userOverride —— 这样同一 turn 内用户手动展开
  // 后，第一帧 token 的 autoCollapse 不会把它强行折回去。React 官方
  // 推荐"prev prop ref + render-time setState"模式重置派生 state
  // (https://react.dev/reference/react/useState#storing-information-from-previous-renders)，
  // 比 useEffect 少一次 commit-after-render 抖动，也不触发
  // ``react-hooks/set-state-in-effect``。
  // React 官方推荐"prev prop ref + render-time setState"模式重置派生 state
  // (https://react.dev/reference/react/useState#storing-information-from-previous-renders)。
  // ESLint 的 react-hooks/refs rule 不识别这个合法 pattern (它把 render
  // 期写 ref 一律视为风险)，需要 disable —— 这正是 React docs 推荐的
  // 做法。Strict Mode 双 render 下也安全：第一次写 ref + setState 入队，
  // 第二次 render 看到 ref 已更新就跳过。
  const lastTurnKeyRef = useRef(turnKey);
  // eslint-disable-next-line react-hooks/refs
  if (lastTurnKeyRef.current !== turnKey) {
    // eslint-disable-next-line react-hooks/refs
    lastTurnKeyRef.current = turnKey;
    setUserOverride(null);
  }
  const collapsed = userOverride != null ? userOverride : autoCollapsed;

  const oneLine = useMemo(() => buildOneLineSummary(events), [events]);
  const rows = useMemo(() => mergeRows(events), [events]);

  if (rows.length === 0 && status === "connecting") {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-muted">
        {STATUS_ICON[status]}
        <span>{STATUS_LABEL[status]}</span>
      </div>
    );
  }

  return (
    <div className="rounded-md border border-ink-line/70 bg-surface-raised/60">
      <button
        type="button"
        onClick={() => setUserOverride(!collapsed)}
        className="flex w-full items-center gap-2 px-3 py-2 text-sm text-left hover:bg-surface-sunk/60 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-600/30"
      >
        {collapsed ? (
          <ChevronRight className="h-3.5 w-3.5 text-ink-subtle" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5 text-ink-subtle" />
        )}
        {STATUS_ICON[status]}
        <span className="font-medium text-ink">{STATUS_LABEL[status]}</span>
        {collapsed && <span className="text-ink-muted truncate">· {oneLine}</span>}
        <span className="ml-auto text-[11px] text-ink-subtle font-mono">{rows.length}</span>
      </button>
      {!collapsed && (
        <ol className="px-3 pb-2 pt-1 space-y-1 border-t border-ink-line/50">
          {rows.map((row) => (
            <li
              key={row.index}
              className="flex items-start gap-2 text-[13px] leading-relaxed"
            >
              {row.kind === "tool" ? (
                <ToolRow row={row} />
              ) : row.kind === "thought" ? (
                <ThoughtRow loop={row.thoughtLoop ?? 0} text={row.thoughtText ?? ""} />
              ) : (
                <EventRow ev={row.raw!} />
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function ToolRow({ row }: { row: MergedRow }) {
  const status = row.toolStatus ?? "pending";
  const isErr = status === "error";
  return (
    <>
      <span className="shrink-0 mt-0.5 w-[42px] inline-flex items-center gap-1 font-mono text-[11px] text-ink-subtle">
        <span className="text-warning">L{row.loop}</span>
      </span>
      <span className="shrink-0 mt-0.5 inline-flex items-center justify-center w-3.5 h-3.5">
        {TOOL_STATUS_ICON[status]}
      </span>
      <span className="shrink-0 mt-0.5 inline-flex items-center gap-1 text-warning text-[12px] font-medium">
        <Wrench className="h-3 w-3" />
        {row.toolName}
      </span>
      <span
        className={cn(
          "min-w-0 flex-1 text-ink-muted break-words font-mono text-[12px] mt-0.5",
          isErr && "text-danger",
        )}
      >
        {isErr ? `失败：${row.toolError ?? "unknown"}` : (row.argsBlurb || "—")}
      </span>
    </>
  );
}

function ThoughtRow({ loop, text }: { loop: number; text: string }) {
  // thought content 可能很长（数百字），整段放进时间线会很拥挤。
  // 先展示前 ~280 字 + 折叠展开按钮；用户想看完整 reasoning 时点开。
  const [expanded, setExpanded] = useState(false);
  const isLong = text.length > 280;
  const shown = expanded || !isLong ? text : text.slice(0, 280) + "…";
  return (
    <>
      <span className="shrink-0 mt-0.5 w-[42px] inline-flex items-center font-mono text-[11px] text-ink-subtle">
        L{loop}
      </span>
      <span className="shrink-0 mt-0.5 inline-flex items-center justify-center w-3.5 h-3.5">
        <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-info" />
      </span>
      <span className="shrink-0 mt-0.5 text-info text-[12px] font-medium">思考</span>
      <span className="min-w-0 flex-1 text-ink break-words text-[13px] leading-6 whitespace-pre-wrap">
        {shown}
        {isLong && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="ml-1 align-baseline text-[11px] text-accent-700 hover:text-accent-800 underline-offset-2 hover:underline"
          >
            {expanded ? "收起" : "展开"}
          </button>
        )}
      </span>
    </>
  );
}

function EventRow({ ev }: { ev: SSEEvent }) {
  return (
    <>
      <span
        className={cn(
          "shrink-0 inline-flex w-[78px] font-mono text-[11px] mt-0.5",
          EVENT_TONE[ev.event] ?? "text-ink-subtle",
        )}
      >
        {ev.event}
      </span>
      <span className="text-ink-muted break-words flex-1">{summarize(ev)}</span>
    </>
  );
}
