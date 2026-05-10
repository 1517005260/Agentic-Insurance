import type { CitationItem, SSEEvent } from "@/lib/sse-types";

/** 一对发送（user）+ 回答（assistant）= 两条 turn。 */
export type Turn = UserTurn | AssistantTurn;

export interface UserTurn {
  id: string;
  role: "user";
  content: string;
  ts: number;
}

export type AssistantStatus =
  | "connecting"
  | "streaming"
  | "done"
  | "error"
  | "aborted";

/**
 * 一条 AI 回答 turn。
 *
 * 事件流分两层：
 *   - `events`         — 全量原始 SSE 事件（除 token），用于 ProgressTimeline 渲染
 *   - `answer`         — token 累积后的纯文本（最终走 markdown 渲染）
 *
 * `hasStartedAnswering` 是 ProgressTimeline 的折叠触发：第一个 token
 * 帧到达时翻 true，progress 折成单行 summary。
 */
export interface AssistantTurn {
  id: string;
  role: "assistant";
  ts: number;
  status: AssistantStatus;

  /** mode label, e.g. "RAG" / "Base Agent" / "Web Agent" / "Web RAG"。 */
  modeLabel: string;
  endpoint: string;

  /** 非 token 事件，时间顺序。 */
  progressEvents: SSEEvent[];
  answer: string;
  hasStartedAnswering: boolean;

  citations?: CitationItem[];
  finalSummary?: Record<string, unknown>;
  errorMessage?: string;
}
