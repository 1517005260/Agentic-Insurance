import { useCallback, useReducer, useRef } from "react";

import { useSSE } from "@/lib/sse";
import type { SSEEvent } from "@/lib/sse-types";
import { initialTurn, workbenchReducer, type WorkbenchTurn } from "./turn";

/**
 * 工作台流式提交的复用 hook。
 *
 * 把 useSSE + reducer + 抢占式 send 拢在一起，让具体 workbench 页只需
 * 组合 RHF + buildBody 就能复用。返回值字段命名跟 useSSE 对齐（status /
 * abort），多出 `turn` / `runStream` 两个 workbench 维度的语义。
 *
 * 设计选择：把"开新 turn → 写 reducer"和"调用 useSSE.start"绑成一个
 * `runStream(endpoint, body)` 调用，避免每个 workbench 重复写 dispatch
 * + activeIdRef 的 boilerplate。
 */
export interface UseWorkbenchStreamReturn {
  turn: WorkbenchTurn;
  status: ReturnType<typeof useSSE>["status"];
  busy: boolean;
  runStream: (endpoint: string, body: unknown) => void;
  abort: () => void;
  reset: () => void;
}

export function useWorkbenchStream(modeLabel: string): UseWorkbenchStreamReturn {
  const [turn, dispatch] = useReducer(workbenchReducer, initialTurn(modeLabel));
  const activeIdRef = useRef<string | null>(null);

  const { status, start, abort: sseAbort } = useSSE({
    onEvent: (ev: SSEEvent) => {
      const id = activeIdRef.current;
      if (!id) return;
      dispatch({ type: "event", id, ev });
    },
    onDone: (events) => {
      const id = activeIdRef.current;
      if (!id) return;
      const errEv = events.find(
        (e): e is Extract<typeof e, { event: "error" }> => e.event === "error",
      );
      dispatch({
        type: "finish",
        id,
        status: errEv ? "error" : "done",
        errorMessage: errEv?.data.message,
      });
      activeIdRef.current = null;
    },
  });

  const runStream = useCallback(
    (endpoint: string, body: unknown) => {
      const id = `wb_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
      activeIdRef.current = id;
      dispatch({ type: "send", id, modeLabel, endpoint });
      start(endpoint, body);
    },
    [modeLabel, start],
  );

  const abort = useCallback(() => {
    sseAbort();
    const id = activeIdRef.current;
    if (id) {
      dispatch({ type: "finish", id, status: "aborted" });
      activeIdRef.current = null;
    }
  }, [sseAbort]);

  const reset = useCallback(() => {
    sseAbort();
    activeIdRef.current = null;
    dispatch({ type: "reset", modeLabel });
  }, [sseAbort, modeLabel]);

  return {
    turn,
    status,
    busy: status === "connecting" || status === "streaming",
    runStream,
    abort,
    reset,
  };
}
