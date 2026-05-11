import { memo, useMemo } from "react";
import { Sparkles, AlertTriangle, FileText, Globe } from "lucide-react";

import { ProgressTimeline } from "./ProgressTimeline";
import { MarkdownWithSup } from "./MarkdownWithSup";
import { extractEvidenceCitations } from "./evidence";
import type { AssistantTurn as AssistantTurnT } from "./types";
import { useCitationStore } from "@/stores/citation";
import type { CitationItem, LocalCitation, WebCitation } from "@/lib/sse-types";

interface Props {
  turn: AssistantTurnT;
}

/**
 * memo 化：长会话下 MessageList 的 turns 数组在 token 流期间频繁
 * 替换，但只有"当前 streaming"那一条 turn 内容实际变化。把
 * AssistantTurn memo 化后，已 done 的历史 turn 引用稳定就直接跳过
 * 重渲，避免每次 token 都把整个对话重新走一遍 ProgressTimeline /
 * MarkdownWithSup 的子树。
 */
export const AssistantTurn = memo(function AssistantTurn({ turn }: Props) {
  const open_ = useCitationStore((s) => s.open_);
  const showProgress =
    turn.progressEvents.length > 0 || turn.status === "connecting";

  // 引用渲染策略按 modeLabel 显式分支（不用 endpoint —— Web Agent 和
  // Base/Proof/Graph Agent 共享 /agent/stream，但前者需要 sup 解析，后者
  // 是纯 evidence 反推；endpoint 单字段判定会把 Web Agent 错归为
  // evidence-only）：
  //
  //   "RAG" / "Web RAG"     → SSE citations 事件 + 答案 [^k] 上标
  //   "Web Agent"           → 答案 [^k] 上标（web_system prompt 强制）；
  //                           citations 事件 backend 暂未发，sup 渲染但 muted；
  //                           同时降级显示 read/proof_scan evidence chip
  //   "Base/Proof/Graph"    → 不解析 [^k]；纯 evidence chip 反推
  const SUP_MODES = new Set(["RAG", "Web RAG", "Web Agent"]);
  const LOCAL_AGENT_MODES = new Set(["Base Agent", "Proof Agent", "Graph Agent"]);
  const supParsable = SUP_MODES.has(turn.modeLabel);
  const showEvidence =
    LOCAL_AGENT_MODES.has(turn.modeLabel) ||
    (turn.modeLabel === "Web Agent" && (turn.citations?.length ?? 0) === 0);

  // Agent 路径 evidence chip 列表（memoize 避免 progressEvents 每帧重算）。
  const evidenceCitations = useMemo<LocalCitation[]>(
    () => (showEvidence ? extractEvidenceCitations(turn.progressEvents) : []),
    [showEvidence, turn.progressEvents],
  );

  // 底部 chip 显示哪一组：sup 模式用 turn.citations；local-agent 用反推
  // evidence；Web Agent 在 citations 缺失时也显示 evidence 兜底。
  const chipItems: CitationItem[] = showEvidence
    ? evidenceCitations
    : turn.citations ?? [];

  return (
    <div className="flex gap-3">
      <div className="shrink-0 mt-0.5">
        <div className="h-7 w-7 rounded-md bg-primary-50 border border-primary-200 flex items-center justify-center">
          <Sparkles className="h-3.5 w-3.5 text-primary-700" />
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-2.5">
        {showProgress && (
          <ProgressTimeline
            events={turn.progressEvents}
            status={turn.status}
            autoCollapsed={turn.hasStartedAnswering}
            turnKey={turn.id}
          />
        )}

        {turn.answer && (
          <div className="break-words">
            <MarkdownWithSup
              content={turn.answer}
              citations={turn.citations}
              parseSup={supParsable}
            />
            {turn.status === "streaming" && turn.hasStartedAnswering && (
              <span
                aria-hidden
                className="inline-block w-1.5 h-4 -mb-0.5 ml-0.5 bg-primary-600/70 animate-pulse align-middle"
              />
            )}
          </div>
        )}

        {turn.errorMessage && (
          // 后端 contract: error 帧之后必有 done 帧（docs §5）。done 抵
          // 达时 status 会变 "done"，所以仅判 status==='error' 会让
          // error 文案被吃掉。这里凡是 errorMessage 存在就显示。
          <div className="flex items-start gap-2 rounded-md bg-danger-soft border border-danger/20 px-3 py-2 text-sm text-danger">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <span className="break-words">{turn.errorMessage}</span>
          </div>
        )}

        {chipItems.length > 0 && (
          <CitationChipRow
            items={chipItems}
            isAgent={showEvidence}
            onOpen={(target) => open_(chipItems, target)}
          />
        )}
      </div>
    </div>
  );
});

// ---------------------------------------------------------- chip row

function CitationChipRow({
  items,
  isAgent,
  onOpen,
}: {
  items: CitationItem[];
  isAgent: boolean;
  onOpen: (target: CitationItem) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 pt-1">
      <span className="text-[11px] uppercase tracking-[0.16em] text-ink-subtle font-mono mr-1">
        {isAgent ? "证据" : "引用"}
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
                ? (c as WebCitation).snippet ?? (c as WebCitation).url
                : (c as LocalCitation).page_preview ?? ""
            }
          >
            {isWeb ? (
              <Globe className="h-3 w-3" />
            ) : (
              <FileText className="h-3 w-3" />
            )}
            <span className="font-mono">[{c.sup}]</span>
            {isWeb ? (
              <span className="truncate max-w-[180px]">{(c as WebCitation).title}</span>
            ) : (
              <>
                <span className="truncate max-w-[140px]">
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
