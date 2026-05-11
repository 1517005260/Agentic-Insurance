import { memo } from "react";

import type { UserTurn } from "./types";

/**
 * 用户消息 — 右对齐，主色淡填。
 *
 * memo 化：长会话下 turns 数组每次更新（如 token 流写入新 assistant
 * turn）会让 MessageList 重渲染整个列表；但 UserBubble 的内容
 * (turn.content) 在该 turn 创建后不会再变，引用稳定即可跳过 re-render。
 */
export const UserBubble = memo(function UserBubble({ turn }: { turn: UserTurn }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-xl rounded-tr-sm bg-primary-600 text-surface-raised px-4 py-2.5 text-[15px] leading-relaxed shadow-sm whitespace-pre-wrap break-words">
        {turn.content}
      </div>
    </div>
  );
});
