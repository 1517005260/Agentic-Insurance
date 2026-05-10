import { useEffect, useRef } from "react";

import { AssistantTurn } from "./AssistantTurn";
import { UserBubble } from "./UserBubble";
import type { Turn } from "./types";

interface Props {
  turns: Turn[];
}

/**
 * 消息流：user 右、assistant 左。
 *
 * Auto-scroll 策略：
 *   - 默认 stick to bottom；用户主动往上滚 (>96px) 时停止跟随
 *   - 新一轮 user turn 进来时强制恢复 stick（用户提问理应想看回答）
 *   - token 风暴下用 rAF 合并多帧 scroll 写入，避免每个 token 触
 *     发一次 layout reflow
 */
export function MessageList({ turns }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);
  const lastLenRef = useRef(0);
  const rafRef = useRef<number | null>(null);

  // 监听用户手动滚动；脱离底部 96px 即解除 stick
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onScroll = () => {
      const dist = el.scrollHeight - (el.scrollTop + el.clientHeight);
      stickRef.current = dist < 96;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  // 内容变化时按 rAF 合并刷新；新 user turn 强制 stick。
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;

    // ChatPage 的 send 一次性 dispatch user + assistant 两条，所
    // 以 turns 长度会 +2；用这个跳变作为"用户发了新问题"的信号
    // 强制恢复 stick。token 风暴只会让 turns 内容变（length 不
    // 变），不会触发恢复。
    const len = turns.length;
    if (len >= lastLenRef.current + 2) {
      stickRef.current = true;
    }
    lastLenRef.current = len;

    if (!stickRef.current) return;
    if (rafRef.current != null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      if (!stickRef.current) return;
      el.scrollTop = el.scrollHeight;
    });
  });

  useEffect(
    () => () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    },
    [],
  );

  return (
    <div ref={wrapRef} className="flex-1 min-h-0 overflow-y-auto scrollbar-thin">
      <div className="mx-auto max-w-3xl w-full px-4 py-8 space-y-7">
        {turns.length === 0 ? (
          <EmptyState />
        ) : (
          turns.map((t) =>
            t.role === "user" ? (
              <UserBubble key={t.id} turn={t} />
            ) : (
              <AssistantTurn key={t.id} turn={t} />
            ),
          )
        )}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center text-center pt-16">
      <div className="font-serif text-3xl text-primary-700 tracking-tight">
        Agentic Insurance
      </div>
      <p className="mt-3 text-sm text-ink-muted max-w-md">
        基于本地保单文档 + 知识图谱 + 联网法规检索的智能问答助手。
        左下角两个图标可切换<strong className="font-medium text-ink"> 联网 </strong>
        与 <strong className="font-medium text-ink">Agent</strong> 模式。
      </p>
    </div>
  );
}
