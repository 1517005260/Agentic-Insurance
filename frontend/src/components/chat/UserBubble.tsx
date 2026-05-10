import type { UserTurn } from "./types";

/** 用户消息 — 右对齐，主色淡填。 */
export function UserBubble({ turn }: { turn: UserTurn }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-xl rounded-tr-sm bg-primary-600 text-surface-raised px-4 py-2.5 text-[15px] leading-relaxed shadow-sm whitespace-pre-wrap break-words">
        {turn.content}
      </div>
    </div>
  );
}
