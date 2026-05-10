import { create } from "zustand";

import type { CitationItem } from "@/lib/sse-types";

/**
 * 全站 citation 抽屉的全局通道。
 *
 * 两路触发：
 *   1. RAG / Web RAG / 业务 workbench 答案里 [^k] 上标点击 → 解析后传 sup 索引到 items[] 找命中。
 *   2. Agent base/proof/graph chat：tool_result is_evidence=true 的 read/proof_scan 反推
 *      file_id + page_number，构造一个临时 LocalCitation 直接传 target。
 *
 * 抽屉只关心 target；items 数组用于"上一条/下一条"翻页（同 turn 内多个引用）。
 */
export interface CitationDrawerState {
  open: boolean;
  /** 当前展示的引用条目（local 走 PDF / web 走外链）。 */
  target: CitationItem | null;
  /** 当前 turn 的全部引用，用于左右切换。 */
  items: CitationItem[];

  /** 打开抽屉：传一组引用 + 当前焦点（默认 items[0]）。 */
  open_: (items: CitationItem[], focus?: CitationItem) => void;
  /** 已打开时切换焦点（不重置 items）。 */
  focus: (target: CitationItem) => void;
  /** 翻到下一条（同 items 数组内）。 */
  next: () => void;
  prev: () => void;
  close: () => void;
}

export const useCitationStore = create<CitationDrawerState>((set, get) => ({
  open: false,
  target: null,
  items: [],

  open_: (items, focus) => {
    if (items.length === 0) return;
    set({ open: true, items, target: focus ?? items[0] });
  },
  focus: (target) => set({ target }),
  next: () => {
    const { items, target } = get();
    if (!target) return;
    const idx = items.findIndex((c) => isSameCitation(c, target));
    if (idx < 0) return;
    const nextItem = items[(idx + 1) % items.length];
    set({ target: nextItem });
  },
  prev: () => {
    const { items, target } = get();
    if (!target) return;
    const idx = items.findIndex((c) => isSameCitation(c, target));
    if (idx < 0) return;
    const prevItem = items[(idx - 1 + items.length) % items.length];
    set({ target: prevItem });
  },
  close: () => set({ open: false, target: null, items: [] }),
}));

// Dev-only：把 store 挂到 window，方便 playwright eval / devtools 调试。
// Production build 由 Vite tree-shake 掉 import.meta.env.DEV 分支。
if (import.meta.env.DEV) {
  (globalThis as unknown as { __citationStore?: typeof useCitationStore }).__citationStore =
    useCitationStore;
}

/**
 * 两条引用是否同一条目。
 *
 * 关键约束：identity 必须能在 page_number=undefined 的多条 evidence 之间
 * 区分（agent 反推时同一文件多个 read 不同页但 args 仅给 file_ids 时 page
 * 都缺失）。所以总是把 sup（evidence chip 也是递增编号）和 observation_id
 * 加进 identity，避免被 (file_id, undefined) 整体折叠成一条。
 */
export function isSameCitation(a: CitationItem, b: CitationItem): boolean {
  if (a.sup !== b.sup) return false;
  const aWeb = "kind" in a && a.kind === "web";
  const bWeb = "kind" in b && b.kind === "web";
  if (aWeb !== bWeb) return false;
  if (aWeb && bWeb) return (a as { url: string }).url === (b as { url: string }).url;
  const al = a as { file_id: string; page_number?: number; observation_id?: string };
  const bl = b as { file_id: string; page_number?: number; observation_id?: string };
  if (al.file_id !== bl.file_id) return false;
  // observation_id（proof 路径）若两侧都有则必须一致；否则 page_number 兜底
  if (al.observation_id || bl.observation_id) {
    return al.observation_id === bl.observation_id;
  }
  return al.page_number === bl.page_number;
}
