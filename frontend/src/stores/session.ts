/**
 * stores/session.ts — current chat session id (zustand).
 *
 * 设计选择：
 *  - 不持久化到 localStorage —— 切账号 / 刷新就重新选；避免在 reload
 *    时跨用户混（与 token 切换走 cache.clear 同样的理由）。
 *  - 只放 currentId；session 列表 / 详情走 react-query 独立缓存
 *    (`hooks/useSessions.ts`)，避免 store 和 query 双源。
 *  - currentId === null 表示"无状态模式"：composer 走 /rag/stream 等
 *    smoke 端点，不持久化对话。
 */
import { create } from "zustand";

interface SessionState {
  currentId: number | null;
  setCurrent: (id: number | null) => void;
  /** 切回 null 的语义：开新一轮无状态对话；侧栏 "新对话" 也走它再 +
   *  按 mode 创 session（异步路径在 ChatPage 处理）。 */
  reset: () => void;
}

export const useSessionStore = create<SessionState>((set) => ({
  currentId: null,
  setCurrent: (id) => set({ currentId: id }),
  reset: () => set({ currentId: null }),
}));
