import { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  X,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  FileText,
  Globe,
} from "lucide-react";

import PdfPageViewer from "./PdfPageViewer";
import { isSameCitation, useCitationStore } from "@/stores/citation";
import { cn } from "@/lib/utils";

const MAX_DRAWER_WIDTH = 640;

/**
 * 全站全局抽屉，挂在 LayoutShell。两种 target：
 *   - LocalCitation: 渲染 react-pdf 单页，header 带 file_id + page 序号 + 上下条切换
 *   - WebCitation:   不嵌 iframe（X-Frame-Options 大概率拒），渲染 title +
 *                    snippet + "在新标签打开"按钮
 *
 * 翻页只在 items.length > 1 时显示。Esc 关闭。点 backdrop 关闭。
 */
export function CitationDrawer() {
  const { open, target, items, prev, next, close } = useCitationStore();

  // PDF 渲染宽度跟随抽屉容器实时变化（窄屏 100vw、宽屏 max 640）。
  const [pageWidth, setPageWidth] = useState(MAX_DRAWER_WIDTH - 32);
  const containerRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  // 容器宽度变化（窗口 resize / 移动端旋转）实时同步给 react-pdf
  useLayoutEffect(() => {
    if (!open) return;
    const el = containerRef.current;
    if (!el) return;
    const sync = () => setPageWidth(Math.max(120, el.clientWidth - 32));
    sync();
    const ro = new ResizeObserver(sync);
    ro.observe(el);
    return () => ro.disconnect();
  }, [open]);

  // 打开时把 focus 拨进抽屉（关闭按钮 = 安全的初始落点）；关闭时归还焦点
  useEffect(() => {
    if (!open) return;
    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;
    // 等本帧 paint 完再 focus，避免 ResizeObserver 还没量好时 focus 跳屏
    const id = requestAnimationFrame(() => closeBtnRef.current?.focus());
    return () => {
      cancelAnimationFrame(id);
      previouslyFocusedRef.current?.focus?.();
    };
  }, [open]);

  // Esc 关 + 箭头翻页
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
      if (e.key === "ArrowLeft" && items.length > 1) prev();
      if (e.key === "ArrowRight" && items.length > 1) next();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, close, prev, next, items.length]);

  if (!open || !target) return null;

  const isWeb = "kind" in target && target.kind === "web";
  // header idx 用值相等（同 store.next/prev 的 isSameCitation），否则
  // 调用方传等值但非同对象 target 时会出 -1 → 0/N
  const idx = items.findIndex((c) => isSameCitation(c, target));
  const total = items.length;

  return (
    <>
      {/* backdrop — 点空白关闭，但不要黑屏（金融风格不该突兀）*/}
      <div
        className="fixed inset-0 z-40 bg-ink/10 animate-fade-in"
        onClick={close}
        aria-hidden
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="引用预览"
        className={cn(
          "fixed right-0 top-0 bottom-0 z-50",
          "bg-surface-raised border-l border-ink-line shadow-pop",
          "flex flex-col animate-slide-in-r",
          // 窄屏 100vw、宽屏封顶 640；ResizeObserver 听容器实际宽度同步给 PDF
          "w-screen sm:w-[min(640px,100vw)]",
        )}
        style={{ maxWidth: MAX_DRAWER_WIDTH }}
      >
        {/* header */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-ink-line shrink-0">
          {isWeb ? (
            <Globe className="h-4 w-4 text-accent-600" />
          ) : (
            <FileText className="h-4 w-4 text-primary-700" />
          )}
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium text-ink truncate">
              {isWeb
                ? (target as { title: string }).title
                : `${(target as { file_id: string }).file_id}`}
            </div>
            <div className="text-[11px] text-ink-subtle font-mono">
              {isWeb
                ? new URL((target as { url: string }).url).hostname
                : `第 ${(target as { page_number?: number }).page_number ?? "?"} 页`}
              {target.sup ? ` · [^${target.sup}]` : ""}
            </div>
          </div>

          {total > 1 && (
            <>
              <button
                type="button"
                onClick={prev}
                className="h-7 w-7 inline-flex items-center justify-center rounded-md hover:bg-surface-sunk text-ink-muted hover:text-ink"
                aria-label="上一条引用"
                title="上一条 (←)"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <span className="text-[11px] text-ink-subtle font-mono select-none">
                {idx + 1} / {total}
              </span>
              <button
                type="button"
                onClick={next}
                className="h-7 w-7 inline-flex items-center justify-center rounded-md hover:bg-surface-sunk text-ink-muted hover:text-ink"
                aria-label="下一条引用"
                title="下一条 (→)"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </>
          )}

          <button
            ref={closeBtnRef}
            type="button"
            onClick={close}
            className="h-7 w-7 inline-flex items-center justify-center rounded-md hover:bg-surface-sunk text-ink-muted hover:text-ink"
            aria-label="关闭"
            title="关闭 (Esc)"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* preview snippet */}
        {isWeb ? (
          <WebPreview target={target as WebTarget} />
        ) : (
          <LocalPreview
            target={target as LocalTarget}
            containerRef={containerRef}
            pageWidth={pageWidth}
          />
        )}
      </aside>
    </>
  );
}

// ----------------------------------------------------- preview branches

interface LocalTarget {
  file_id: string;
  page_number?: number;
  page_preview?: string;
  observation_id?: string;
  sup?: number;
}
interface WebTarget {
  title: string;
  url: string;
  snippet?: string;
  sup?: number;
  published_date?: string | null;
}

function LocalPreview({
  target,
  containerRef,
  pageWidth,
}: {
  target: LocalTarget;
  containerRef: React.RefObject<HTMLDivElement | null>;
  pageWidth: number;
}) {
  // 早先在 PDF 渲染上方贴了一段 ``page_preview`` 文本（来自 OCR 抽出的
  // markdown 截断），但那段文字常常带 ``<table border=1>`` 之类的原始
  // HTML 噪音 —— 见 CitationBuilder.from_reranked_pages 抽 ``text_markdown``
  // 的方式。这条信息在下方 PDF 已经能直接读到，去掉减少干扰。
  return (
    <div
      ref={containerRef}
      className="flex-1 min-h-0 overflow-y-auto scrollbar-thin px-4 py-4 bg-surface-sunk/30"
    >
      {target.page_number != null ? (
        <PdfPageViewer
          fileId={target.file_id}
          pageNumber={target.page_number}
          width={pageWidth}
        />
      ) : (
        <div className="text-sm text-ink-muted text-center py-12">
          该引用未提供页码 — 仅来源文件 <code>{target.file_id}</code>
        </div>
      )}
    </div>
  );
}

function WebPreview({ target }: { target: WebTarget }) {
  return (
    <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin px-4 py-4 space-y-3">
      <a
        href={target.url}
        target="_blank"
        rel="noreferrer"
        className="inline-flex items-center gap-1.5 text-sm text-accent-700 hover:text-accent-800 break-all"
      >
        <ExternalLink className="h-3.5 w-3.5 shrink-0" />
        <span className="break-all">{target.url}</span>
      </a>
      {target.published_date && (
        <div className="text-[11px] text-ink-subtle font-mono">
          发布：{target.published_date}
        </div>
      )}
      {target.snippet && (
        // Tavily 返回的是搜索摘要片段，不是 LLM 输出。早先用
        // MarkdownWithSup 渲染会把行内的 ``### `` 字符当 markdown
        // heading 解析失败，最终字面输出，看起来"没渲染"。snippet
        // 是非结构化文本，保守按 plain text 显示反而更可读。
        <div className="text-ink whitespace-pre-wrap break-words leading-6 text-[14px]">
          {target.snippet}
        </div>
      )}
      <p className="text-[11px] text-ink-subtle pt-3 border-t border-ink-line/60">
        网页内容因 <code>X-Frame-Options</code> 多被禁止内嵌；此处仅显示
        Tavily 摘要 + 外链。点击上方链接在新标签打开原始页面。
      </p>
    </div>
  );
}
