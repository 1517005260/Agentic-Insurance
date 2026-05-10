import { type ReactNode } from "react";
import { Send, Square, RotateCcw, Sparkles, AlertTriangle, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ProgressTimeline } from "@/components/chat/ProgressTimeline";
import { MarkdownWithSup } from "@/components/chat/MarkdownWithSup";
import type { CitationItem, FinalEvent } from "@/lib/sse-types";
import { useCitationStore } from "@/stores/citation";
import { CitationChipRow } from "./CitationChipRow";
import type { WorkbenchTurn } from "./turn";

/**
 * 工作台共用版面 — 左 form / 右 answer + ProgressTimeline。
 *
 * 状态由调用页通过 `useWorkbenchStream` 拿到（turn/busy/abort/reset），
 * 这里只负责 UI 布局；表单提交（含 zod 校验）由调用页的 onSubmit
 * 显式触发，scaffold 的"发送"按钮只是 wrapper。
 *
 * 这种切分比"scaffold 自己跑 RHF.handleSubmit"灵活：每个 workbench 的
 * RHF 状态/默认值/验证流程都不一样，让调用页全权管理。
 */
export interface WorkbenchScaffoldProps {
  title: string;
  description?: string;
  modeLabel: string;
  turn: WorkbenchTurn;
  busy: boolean;
  canSubmit: boolean;
  /** 用户点"发送"时调用 — 调用页用 form.handleSubmit 包出来，valid 时调 runStream。 */
  onSubmit: () => void;
  onAbort: () => void;
  onReset: () => void;
  renderForm: () => ReactNode;
  /** 答案区自定义。默认用 MarkdownWithSup。 */
  renderAnswer?: (turn: WorkbenchTurn) => ReactNode;
  /** 在 ProgressTimeline 和答案之间插入额外面板（如 Exclusion 的 Proof 看板）。 */
  renderExtras?: (turn: WorkbenchTurn) => ReactNode;
  parseSup?: boolean;
  /** 默认提交按钮文案 */
  submitLabel?: string;
  /** 隐藏底部 chip 行（PolicyCalc 在自定义 renderAnswer 里自管 chip 时用）。 */
  hideCitationChips?: boolean;
}

export function WorkbenchScaffold({
  title,
  description,
  modeLabel,
  turn,
  busy,
  canSubmit,
  onSubmit,
  onAbort,
  onReset,
  renderForm,
  renderAnswer,
  renderExtras,
  parseSup = true,
  submitLabel = "发送",
  hideCitationChips = false,
}: WorkbenchScaffoldProps) {
  const open_ = useCitationStore((s) => s.open_);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="border-b border-ink-line px-6 py-3 flex items-center gap-3 shrink-0">
        <Sparkles className="h-4 w-4 text-primary-700" />
        <div className="min-w-0 flex-1">
          <h1 className="text-base font-semibold text-ink leading-tight">{title}</h1>
          {description && <p className="text-xs text-ink-muted truncate">{description}</p>}
        </div>
        <span className="text-[11px] uppercase tracking-[0.16em] text-ink-subtle font-mono">
          {modeLabel}
        </span>
      </header>

      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[420px_1fr] divide-y lg:divide-y-0 lg:divide-x divide-ink-line">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            onSubmit();
          }}
          className="overflow-y-auto scrollbar-thin px-5 py-5 space-y-5 max-h-full"
        >
          {renderForm()}

          <div className="flex items-center gap-2 pt-1">
            {!busy ? (
              <Button type="submit" disabled={!canSubmit} size="md">
                <Send className="h-3.5 w-3.5" />
                {submitLabel}
              </Button>
            ) : (
              <Button type="button" variant="secondary" size="md" onClick={onAbort}>
                <Square className="h-3.5 w-3.5" />
                中止
              </Button>
            )}
            <Button
              type="button"
              variant="ghost"
              size="md"
              onClick={onReset}
              disabled={busy}
            >
              <RotateCcw className="h-3.5 w-3.5" />
              重置
            </Button>
          </div>
        </form>

        <div className="overflow-y-auto scrollbar-thin px-6 py-5">
          {turn.id == null ? (
            <EmptyState busy={busy} />
          ) : (
            <article className="mx-auto max-w-3xl space-y-3">
              <ProgressTimeline
                events={turn.progressEvents}
                status={turn.status}
                autoCollapsed={turn.hasStartedAnswering}
                turnKey={turn.id}
              />

              {renderExtras?.(turn)}

              {turn.answer && (
                <div className="break-words pt-1">
                  {renderAnswer ? (
                    renderAnswer(turn)
                  ) : (
                    <MarkdownWithSup
                      content={turn.answer}
                      citations={turn.citations}
                      parseSup={parseSup}
                    />
                  )}
                  {turn.status === "streaming" && turn.hasStartedAnswering && (
                    <span
                      aria-hidden
                      className="inline-block w-1.5 h-4 -mb-0.5 ml-0.5 bg-primary-600/70 animate-pulse align-middle"
                    />
                  )}
                </div>
              )}

              {turn.errorMessage && (
                <div className="flex items-start gap-2 rounded-md bg-danger-soft border border-danger/20 px-3 py-2 text-sm text-danger">
                  <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                  <span className="break-words">{turn.errorMessage}</span>
                </div>
              )}

              {!hideCitationChips && turn.citations && turn.citations.length > 0 && (
                <CitationChipRow
                  items={turn.citations}
                  onOpen={(target) => open_(turn.citations as CitationItem[], target)}
                />
              )}
            </article>
          )}
        </div>
      </div>
    </div>
  );
}

function EmptyState({ busy }: { busy: boolean }) {
  return (
    <div className="flex h-full items-center justify-center text-sm text-ink-subtle">
      <div className="text-center space-y-1">
        {busy ? (
          <div className="inline-flex items-center gap-2 text-ink-muted">
            <Loader2 className="h-4 w-4 animate-spin" />
            连接中…
          </div>
        ) : (
          <>
            <div>填好左侧表单后点击"发送"</div>
            <div className="text-[11px] font-mono text-ink-subtle/80">SSE · 实时进度</div>
          </>
        )}
      </div>
    </div>
  );
}

export type WorkbenchFinal = FinalEvent["data"];
