import { FileText, Globe } from "lucide-react";

import type { CitationItem, LocalCitation, WebCitation } from "@/lib/sse-types";

/**
 * 工作台答案下方的引用 chip 列表。
 *
 * 跟 chat AssistantTurn 的 chip 样式一致 —— 工作台路径只有 "引用"
 * 标签（不会出现 "证据"，因为 workbench runner 直接发 citations 事件，
 * 不走 evidence 反推）。
 */
export function CitationChipRow({
  items,
  onOpen,
}: {
  items: CitationItem[];
  onOpen: (target: CitationItem) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-ink-line/60">
      <span className="text-[11px] uppercase tracking-[0.16em] text-ink-subtle font-mono mr-1">
        引用
      </span>
      {items.map((c, i) => {
        const isWeb = "kind" in c && c.kind === "web";
        return (
          <button
            key={`${isWeb ? "w" : "l"}_${i}_${c.sup}`}
            type="button"
            onClick={() => onOpen(c)}
            className="inline-flex items-center gap-1 rounded-sm border border-accent-200 bg-accent-50 hover:bg-accent-100 hover:border-accent-300 px-1.5 py-0.5 text-[11px] text-accent-700 hover:text-accent-800 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/40"
            title={
              isWeb
                ? ((c as WebCitation).snippet ?? (c as WebCitation).url)
                : ((c as LocalCitation).page_preview ?? "")
            }
          >
            {isWeb ? <Globe className="h-3 w-3" /> : <FileText className="h-3 w-3" />}
            <span className="font-mono">[{c.sup}]</span>
            {isWeb ? (
              <span className="truncate max-w-[200px]">{(c as WebCitation).title}</span>
            ) : (
              <>
                <span className="truncate max-w-[160px]">
                  {(c as LocalCitation).file_id.slice(0, 14)}…
                </span>
                <span className="text-accent-600 font-mono">
                  p.{(c as LocalCitation).page_number ?? "?"}
                </span>
              </>
            )}
          </button>
        );
      })}
    </div>
  );
}
