import type { CitationItem, SSEEvent } from "@/lib/sse-types";
import type { AssistantStatus } from "@/components/chat/types";

/**
 * 工作台一次提交的累积状态。
 *
 * 跟 ChatPage 里的 AssistantTurn 几乎一样，但有两点差异：
 *   1. 工作台不需要 user/assistant 区分（左侧表单本身就是"问"）；
 *   2. id 可空 —— 还没第一次提交时 turn 处在 "empty" 态。
 */
export interface WorkbenchTurn {
  id: string | null;
  status: AssistantStatus;
  modeLabel: string;
  endpoint: string;
  progressEvents: SSEEvent[];
  answer: string;
  hasStartedAnswering: boolean;
  citations?: CitationItem[];
  finalSummary?: Record<string, unknown>;
  errorMessage?: string;
}

export type WorkbenchAction =
  | { type: "send"; id: string; modeLabel: string; endpoint: string }
  | { type: "event"; id: string; ev: SSEEvent }
  | {
      type: "finish";
      id: string;
      status: AssistantStatus;
      errorMessage?: string;
    }
  | { type: "reset"; modeLabel: string };

export function initialTurn(modeLabel: string): WorkbenchTurn {
  return {
    id: null,
    status: "done",
    modeLabel,
    endpoint: "",
    progressEvents: [],
    answer: "",
    hasStartedAnswering: false,
  };
}

export function workbenchReducer(
  state: WorkbenchTurn,
  action: WorkbenchAction,
): WorkbenchTurn {
  switch (action.type) {
    case "send":
      return {
        id: action.id,
        status: "connecting",
        modeLabel: action.modeLabel,
        endpoint: action.endpoint,
        progressEvents: [],
        answer: "",
        hasStartedAnswering: false,
      };
    case "event":
      if (state.id !== action.id) return state;
      return applyEvent(state, action.ev);
    case "finish":
      if (state.id !== action.id) return state;
      return {
        ...state,
        status: action.status,
        errorMessage: action.errorMessage ?? state.errorMessage,
      };
    case "reset":
      return initialTurn(action.modeLabel);
  }
}

function applyEvent(t: WorkbenchTurn, ev: SSEEvent): WorkbenchTurn {
  const next: WorkbenchTurn = { ...t, status: "streaming" };
  switch (ev.event) {
    case "token":
      return {
        ...next,
        answer: t.answer + ev.data.delta,
        hasStartedAnswering: true,
      };
    case "citations":
      return {
        ...next,
        citations: ev.data.items as CitationItem[],
        progressEvents: [...t.progressEvents, ev],
      };
    case "final":
      return {
        ...next,
        finalSummary: ev.data,
        // BaseAgent / ProofAgent 工作台答案在 final.answer 里（runner 的
        // result_payload 透过 final 事件透出）。
        answer:
          typeof ev.data.answer === "string" && !t.hasStartedAnswering
            ? ev.data.answer
            : t.answer,
        hasStartedAnswering:
          t.hasStartedAnswering ||
          (typeof ev.data.answer === "string" && ev.data.answer.length > 0),
        progressEvents: [...t.progressEvents, ev],
      };
    case "error":
      return {
        ...next,
        errorMessage: ev.data.message,
        progressEvents: [...t.progressEvents, ev],
      };
    case "done":
      return next;
    default:
      return {
        ...next,
        progressEvents: [...t.progressEvents, ev],
      };
  }
}
