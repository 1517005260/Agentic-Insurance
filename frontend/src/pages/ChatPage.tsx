import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { ChatComposer, type ComposerMode } from "@/components/chat/ChatComposer";
import { MessageList } from "@/components/chat/MessageList";
import { SessionSidebar } from "@/components/chat/SessionSidebar";
import type { AssistantTurn, Turn } from "@/components/chat/types";
import {
  useCreateSession,
  useSessionDetail,
  type AgentKind,
  type MessageRow,
  type SessionMode,
  type SessionRow,
} from "@/hooks/useSessions";
import { useSSE } from "@/lib/sse";
import type { CitationItem, SSEEvent } from "@/lib/sse-types";
import { useSessionStore } from "@/stores/session";

/** mode → 后端端点 + body 构造 + 给 UI 显示的 label。
 *
 * 两条路径共存：
 *  - **stateless**：currentId === null，走 /rag/stream / /agent/stream /
 *    /web-rag/stream，body 含 query；不持久化对话。
 *  - **session-aware**：currentId 设值，走 /chat/sessions/{id}/messages
 *    body={content}；后端持久化 user + assistant；多轮 history 由后端
 *    从 trace 拼。
 */
function specFor(
  mode: ComposerMode,
  query: string,
  sessionId: number | null,
) {
  if (sessionId !== null) {
    // mode 在 session 里已经锁定，只发 content
    return {
      url: `/chat/sessions/${sessionId}/messages`,
      body: { content: query },
      label: labelFromMode(mode),
    };
  }
  if (mode.agent && mode.web) {
    return {
      url: "/agent/stream",
      body: { query, kind: "base", web: true },
      label: "Web Agent",
    };
  }
  if (mode.agent) {
    return {
      url: "/agent/stream",
      body: { query, kind: "base", web: false },
      label: "Base Agent",
    };
  }
  if (mode.web) {
    return {
      url: "/web-rag/stream",
      body: { query },
      label: "Web RAG",
    };
  }
  return {
    url: "/rag/stream",
    body: { query },
    label: "RAG",
  };
}

function labelFromMode(mode: ComposerMode): string {
  if (mode.agent && mode.web) return "Web Agent";
  if (mode.agent) return "Base Agent";
  if (mode.web) return "Web RAG";
  return "RAG";
}

function modeFromSession(s: SessionRow): ComposerMode {
  return { web: s.web, agent: s.mode === "agent" };
}

/** Session create body derived from composer mode + first query. */
function sessionCreateFromMode(
  mode: ComposerMode,
  query: string,
): {
  mode: SessionMode;
  agent_kind?: AgentKind | null;
  web: boolean;
  title: string;
} {
  const title = (query.trim().slice(0, 60) || "新对话").trim();
  if (mode.agent) {
    return { mode: "agent", agent_kind: "base", web: mode.web, title };
  }
  return { mode: "rag", web: mode.web, title };
}

// ---- reducer ----------------------------------------------------------

type Action =
  | {
      type: "send";
      userId: string;
      assistantId: string;
      query: string;
      modeLabel: string;
      endpoint: string;
    }
  | { type: "event"; assistantId: string; ev: SSEEvent }
  | {
      type: "finish";
      assistantId: string;
      status: AssistantTurn["status"];
      errorMessage?: string;
    }
  /** 切换 session / 新对话：直接换 turn list（用 historic detail
   * 转换得到 / 或 [] 表示空对话）。 */
  | { type: "reset"; turns: Turn[] };

function reducer(state: Turn[], action: Action): Turn[] {
  if (action.type === "reset") return action.turns;
  switch (action.type) {
    case "send": {
      const ts = Date.now();
      return [
        ...state,
        {
          id: action.userId,
          role: "user",
          content: action.query,
          ts,
        },
        {
          id: action.assistantId,
          role: "assistant",
          ts,
          status: "connecting",
          modeLabel: action.modeLabel,
          endpoint: action.endpoint,
          progressEvents: [],
          answer: "",
          hasStartedAnswering: false,
        },
      ];
    }
    case "event": {
      return state.map((t) => {
        if (t.id !== action.assistantId || t.role !== "assistant") return t;
        return applyEvent(t, action.ev);
      });
    }
    case "finish": {
      return state.map((t) => {
        if (t.id !== action.assistantId || t.role !== "assistant") return t;
        return {
          ...t,
          status: action.status,
          errorMessage: action.errorMessage ?? t.errorMessage,
        };
      });
    }
  }
}

/**
 * 把一帧事件折进 AssistantTurn。所有非 token 事件压栈给 progress
 * 时间轴；token 拼到 answer，第一帧 token 翻 hasStartedAnswering
 * 触发 ProgressTimeline 自动收起。
 */
function applyEvent(t: AssistantTurn, ev: SSEEvent): AssistantTurn {
  const next: AssistantTurn = { ...t, status: "streaming" };

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
        // Base Agent / Web Agent 的最终答案在 final.answer 里（不是
        // token 流），这里兜底拼上。RAG / Web RAG 的 final 没有
        // answer 字段，跳过。
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
      // done 只是终止信号；status 由 finish action 显式 set，避免
      // 这里跟 onDone 双写竞态。
      return next;
    default:
      return {
        ...next,
        progressEvents: [...t.progressEvents, ev],
      };
  }
}

// ---- component --------------------------------------------------------

export default function ChatPage() {
  const [turns, dispatch] = useReducer(reducer, [] as Turn[]);

  const [draft, setDraft] = useReducerState("");
  const [mode, setMode] = useReducerState<ComposerMode>({ web: false, agent: false });

  // 当前流绑定的 assistant turn id；onEvent / onDone 时找它写。
  const activeAssistantIdRef = useRef<string | null>(null);
  /**
   * The session id of the *in-flight* stream, if any. Set right
   * before ``start()``, cleared on finish/abort. The
   * currentSessionId-change effect uses this to skip its own
   * abort/reset path when the change came from us just creating a
   * session and starting the first message — without this guard the
   * detail-query-resolves → reset-turns path would tear down our
   * own freshly-started SSE.
   */
  const streamingSessionIdRef = useRef<number | null>(null);

  const { status, start, abort } = useSSE({
    // ChatPage's reducer pulls token deltas out of every event and
    // concats them onto turn.answer. ``dropTokenFrames`` tells
    // useSSE to skip retaining those frames in its internal events
    // array (and the onDone snapshot) — we don't read them
    // anywhere downstream and they are the dominant heap allocation
    // on long web-RAG turns. See useSSE for the OOM rationale.
    dropTokenFrames: true,
    onEvent: (ev) => {
      const id = activeAssistantIdRef.current;
      if (!id) return;
      dispatch({ type: "event", assistantId: id, ev });
    },
    onDone: (events) => {
      const id = activeAssistantIdRef.current;
      if (!id) return;
      // error 帧之后必有 done。useSSE 已经把 status 设成
      // "error"，但 ChatPage 的 reducer 是另一套 state；这里
      // 必须根据快照里有没有 error 帧来决定 finish 状态，否则
      // useStatusFinisher 看到 done 不动就被 onDone 覆盖成 done。
      const errEv = events.find(
        (e): e is Extract<typeof e, { event: "error" }> => e.event === "error",
      );
      dispatch({
        type: "finish",
        assistantId: id,
        status: errEv ? "error" : "done",
        errorMessage: errEv?.data.message,
      });
      activeAssistantIdRef.current = null;
      streamingSessionIdRef.current = null;
    },
  });

  // useSSE 不直接通知 error / aborted —— 用 effect 兜底。
  // 简单做法：onSend 启动后用 status 兜底处理（status 变 error/aborted 时 finish）。
  // 这里通过 ref 比较：当 useSSE.status 进入 error 但 assistantId 还活着，发 finish。
  // 实际实现见下方 useEffect。
  useStatusFinisher(status, activeAssistantIdRef, streamingSessionIdRef, dispatch);

  const busy = status === "connecting" || status === "streaming";

  // ----- session 协调 -----
  const currentSessionId = useSessionStore((s) => s.currentId);
  const setCurrentSession = useSessionStore((s) => s.setCurrent);
  const createSessionMu = useCreateSession();
  const detailQ = useSessionDetail(currentSessionId);

  // 切 session：清 turns + 把已有 messages 灌进 reducer。null 也清空。
  // detailQ.data?.session 同时锁 ChatComposer mode 显示。
  //
  // **不动**正在 stream 的 session：onSend 创 session 后 setCurrentSession
  // 会触发本 effect，但 detailQ 拿到的是空 messages（assistant 还没落库）。
  // 用 streamingSessionIdRef 守一个：当 effect 看到的 sid 等于在跑流的
  // sid，就跳过 reset/abort —— 那是我们自己刚启的流，不能拆。
  useEffect(() => {
    if (currentSessionId == null) {
      activeAssistantIdRef.current = null;
      streamingSessionIdRef.current = null;
      abort();
      dispatch({ type: "reset", turns: [] });
      return;
    }
    if (streamingSessionIdRef.current === currentSessionId) {
      // 这是我们自己刚启流的 session — 不要拆。
      return;
    }
    if (detailQ.data?.session && detailQ.data.session.id === currentSessionId) {
      const restored = messagesToTurns(detailQ.data.messages, detailQ.data.session);
      activeAssistantIdRef.current = null;
      streamingSessionIdRef.current = null;
      abort();
      dispatch({ type: "reset", turns: restored });
      setMode(modeFromSession(detailQ.data.session));
    }
    // 想监听的是 currentSessionId 切换 + detailQ 数据 ready
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSessionId, detailQ.data]);

  const sessionLocked = currentSessionId != null;

  const onSend = useCallback(async () => {
    const q = draft.trim();
    if (!q || busy) return;

    // session-aware：没 session 时先建一个，然后用其 id 发 message。
    let sid = currentSessionId;
    if (sid == null) {
      try {
        const created = await createSessionMu.mutateAsync(
          sessionCreateFromMode(mode, q),
        );
        sid = created.id;
        setCurrentSession(created.id);
      } catch (e) {
        // 不阻塞用户：fall back 到无状态路径
        console.warn("[chat] create session failed; falling back to stateless", e);
        sid = null;
      }
    }

    const spec = specFor(mode, q, sid);
    const userId = `u_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    const assistantId = `a_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    activeAssistantIdRef.current = assistantId;
    streamingSessionIdRef.current = sid;  // null in stateless path is fine

    dispatch({
      type: "send",
      userId,
      assistantId,
      query: q,
      modeLabel: spec.label,
      endpoint: spec.url,
    });
    setDraft("");
    start(spec.url, spec.body);
  }, [
    draft,
    busy,
    mode,
    start,
    setDraft,
    currentSessionId,
    createSessionMu,
    setCurrentSession,
  ]);

  const onAbort = useCallback(() => {
    abort();
    const id = activeAssistantIdRef.current;
    if (id) {
      dispatch({ type: "finish", assistantId: id, status: "aborted" });
      activeAssistantIdRef.current = null;
    }
    streamingSessionIdRef.current = null;
  }, [abort]);

  // 当前 mode 标签（即便没在发送，也显示 composer 上）
  const currentMode = specFor(mode, "", currentSessionId).label;

  return (
    <div className="flex h-full overflow-hidden">
      <SessionSidebar />
      <div className="flex h-full flex-1 flex-col overflow-hidden">
        <MessageList turns={turns} />

        <div className="px-4 pb-6 pt-2">
          <div className="mx-auto max-w-3xl w-full">
            <ChatComposer
              value={draft}
              onChange={setDraft}
              mode={mode}
              onModeChange={sessionLocked ? () => {} : setMode}
              onSend={onSend}
              onAbort={onAbort}
              busy={busy}
              modeLabel={currentMode}
            />
            {sessionLocked && (
              <div className="mt-1 text-[11px] text-ink-subtle text-center">
                会话已锁定 mode；新对话可点左侧"+ 新对话"。
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Convert backend MessageRow[] to ChatPage Turn[] for replay.
 *
 *  - user → UserTurn
 *  - assistant → AssistantTurn{status:'done', citations from metadata.citations}
 *  - tool / system 行（如有）→ 跳过（chat UI 不渲染）
 */
function messagesToTurns(messages: MessageRow[], session: SessionRow): Turn[] {
  const label = labelFromMode(modeFromSession(session));
  const turns: Turn[] = [];
  for (const m of messages) {
    const ts = new Date(m.created_at).getTime();
    if (m.role === "user") {
      turns.push({ id: `u_${m.id}`, role: "user", content: m.content, ts });
    } else if (m.role === "assistant") {
      const meta = m.metadata ?? {};
      const citations = (meta.citations as CitationItem[] | undefined) ?? undefined;
      const errorMessage =
        typeof meta.error === "string" ? (meta.error as string) : undefined;
      turns.push({
        id: `a_${m.id}`,
        role: "assistant",
        ts,
        status: errorMessage ? "error" : "done",
        modeLabel: label,
        endpoint: `/chat/sessions/${session.id}/messages`,
        progressEvents: [],
        answer: m.content,
        hasStartedAnswering: true,
        citations,
        errorMessage,
      });
    }
  }
  return turns;
}

// ---- tiny helpers ------------------------------------------------------

/** 给 useState 起个更明显的名字，避免阅读时和 reducer 混淆。 */
function useReducerState<T>(initial: T): [T, (v: T) => void] {
  const [v, setV] = useState<T>(initial);
  return [v, setV];
}

/**
 * useSSE 没暴露 onError / onAborted；用 status 兜底：进入 error /
 * aborted 时把 active assistant turn finish 掉。
 *
 * 写成单独 hook 是为了 ChatPage 里不混进 useEffect。
 */
function useStatusFinisher(
  status: ReturnType<typeof useSSE>["status"],
  activeRef: React.MutableRefObject<string | null>,
  streamingSessionIdRef: React.MutableRefObject<number | null>,
  dispatch: React.Dispatch<Action>,
) {
  useEffect(() => {
    if (status === "error" || status === "aborted") {
      const id = activeRef.current;
      if (!id) return;
      dispatch({ type: "finish", assistantId: id, status });
      activeRef.current = null;
      streamingSessionIdRef.current = null;
    }
  }, [status, activeRef, streamingSessionIdRef, dispatch]);
}
