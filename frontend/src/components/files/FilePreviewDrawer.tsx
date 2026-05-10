/**
 * FilePreviewDrawer — full-document PDF browser opened from a file card.
 *
 * The CitationDrawer is for *citation* navigation (jump to one specific
 * page tied to a [^k] marker). When the user clicks a file card from
 * /files they want to *browse the whole document*, not just page 1.
 *
 * Implementation: react-pdf <Document> with one <Page> per page, an
 * IntersectionObserver to track which page is in view, and a header
 * toolbar (current page / total / jump / zoom). The PDF arraybuffer
 * is fetched once via /files/{id}/download (the same auth-bearing
 * helper PdfPageViewer uses).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Loader2,
  Minus,
  Plus,
  X,
} from "lucide-react";
import { Document, Page, pdfjs } from "react-pdf";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

// pdfjs worker shared with PdfPageViewer; keep the import here so
// react-pdf works even if FilePreviewDrawer is the first consumer.
import pdfWorkerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";
pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

import type { AxiosResponse } from "axios";

import { api } from "@/api/client";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface Props {
  fileId: string;
  displayName: string;
  /** Total page count from the FileRow (server-side ground truth). 0/null is OK — UI hides paging until known. */
  pageCount?: number | null;
  onClose: () => void;
}

const DRAWER_MAX_WIDTH = 880;
const ZOOM_LEVELS = [0.6, 0.8, 1.0, 1.25, 1.5];

export default function FilePreviewDrawer({
  fileId,
  displayName,
  pageCount,
  onClose,
}: Props) {
  // Pair the buffer with its fileId so a stale result from a previous
  // fileId never sneaks into the active render. This is also why we
  // *don't* call setState from inside the effect synchronously —
  // ``derivedBuf`` masks any lingering data for the wrong fileId.
  const [data, setData] = useState<{ fileId: string; buf: ArrayBuffer } | null>(
    null,
  );
  const [error, setError] = useState<{ fileId: string; message: string } | null>(
    null,
  );
  const [numPages, setNumPages] = useState<number | null>(pageCount ?? null);
  const [currentPage, setCurrentPage] = useState(1);
  const [zoomIdx, setZoomIdx] = useState(2); // index into ZOOM_LEVELS, default 1.0
  const [pageInputValue, setPageInputValue] = useState("1");

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  /**
   * Suppress IntersectionObserver-driven currentPage updates briefly
   * after a user-initiated jump (toolbar input / arrow keys). Without
   * this the smooth-scroll animation triggers every page it crosses
   * and the input value flickers through the intermediate pages.
   */
  const ignoreObserverUntilRef = useRef<number>(0);

  // ---- fetch PDF blob once per fileId (auth-bearing) --------------------
  // No synchronous setState in the effect body — both setData and
  // setError are deferred to the async resolution path. ``cancelled``
  // guards a switch-fileId-mid-fetch race where the older request
  // resolves after the newer one already mounted.
  useEffect(() => {
    if (data?.fileId === fileId) return;
    let cancelled = false;
    const ctrl = new AbortController();
    api
      .get(`/files/${encodeURIComponent(fileId)}/download`, {
        responseType: "arraybuffer",
        signal: ctrl.signal,
      })
      .then((res: AxiosResponse<ArrayBuffer>) => {
        if (cancelled) return;
        setData({ fileId, buf: res.data });
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const e = err as { code?: string; name?: string; message?: string };
        if (e?.code === "ERR_CANCELED" || e?.name === "CanceledError") return;
        setError({ fileId, message: e?.message ?? "PDF 加载失败" });
      });
    return () => {
      cancelled = true;
      ctrl.abort();
    };
  }, [fileId, data?.fileId]);

  // Mask stale results from a previous fileId (data only valid when
  // fileId matches; same for the inline error message).
  const activeBuf = data?.fileId === fileId ? data.buf : null;
  const activeError = error?.fileId === fileId ? error.message : null;

  // The same transferable-buffer guard PdfPageViewer uses: react-pdf
  // detaches the ArrayBuffer when handing it to the worker, so each
  // memo result is a fresh slice that survives parent re-renders.
  const docFile = useMemo(() => (activeBuf ? activeBuf.slice(0) : null), [activeBuf]);

  // ---- ESC closes ---------------------------------------------------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // ---- IntersectionObserver: which page is centered ----------------------
  useEffect(() => {
    if (!numPages) return;
    const root = scrollRef.current;
    if (!root) return;
    const obs = new IntersectionObserver(
      (entries) => {
        if (Date.now() < ignoreObserverUntilRef.current) return;
        // Pick the entry with the largest intersection ratio that's
        // currently visible. Using a single rootMargin band would
        // miss the first / last page on documents shorter than the
        // viewport.
        let bestPage = currentPage;
        let bestRatio = 0;
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          if (entry.intersectionRatio > bestRatio) {
            bestRatio = entry.intersectionRatio;
            const dataPage = (entry.target as HTMLElement).dataset.page;
            if (dataPage) bestPage = Number(dataPage);
          }
        }
        if (bestPage !== currentPage) {
          setCurrentPage(bestPage);
          setPageInputValue(String(bestPage));
        }
      },
      {
        root,
        threshold: [0.25, 0.5, 0.75],
      },
    );
    for (const el of pageRefs.current.values()) obs.observe(el);
    return () => obs.disconnect();
  }, [numPages, currentPage]);

  const jumpToPage = useCallback(
    (target: number) => {
      if (!numPages) return;
      const clamped = Math.min(numPages, Math.max(1, target));
      const el = pageRefs.current.get(clamped);
      if (!el) return;
      // 600ms smooth scroll covers ~10 pages of distance comfortably;
      // suppress the observer for that duration so we don't bounce
      // the input value through intermediate pages.
      ignoreObserverUntilRef.current = Date.now() + 800;
      setCurrentPage(clamped);
      setPageInputValue(String(clamped));
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    },
    [numPages],
  );

  const onPageInputCommit = () => {
    const n = Number(pageInputValue);
    if (Number.isFinite(n) && n >= 1) jumpToPage(Math.trunc(n));
    else setPageInputValue(String(currentPage));
  };

  const zoomLevel = ZOOM_LEVELS[zoomIdx] ?? 1.0;

  return (
    <>
      <div
        className="fixed inset-0 z-40 bg-ink/10 animate-fade-in"
        onClick={onClose}
        aria-hidden
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={`${displayName} 全文件预览`}
        className={cn(
          "fixed right-0 top-0 bottom-0 z-50 w-screen sm:w-[min(720px,100vw)]",
          "bg-surface-raised border-l border-ink-line shadow-pop",
          "flex flex-col animate-slide-in-r",
        )}
        style={{ maxWidth: DRAWER_MAX_WIDTH }}
      >
        <header className="flex items-center justify-between gap-2 px-4 py-2 border-b border-ink-line shrink-0">
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium text-ink truncate">{displayName}</div>
            <div className="text-[11px] text-ink-subtle font-mono">
              {fileId.slice(-12)} · 全文件预览
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭"
            title="关闭 (Esc)"
            className="h-7 w-7 inline-flex items-center justify-center rounded-md hover:bg-surface-sunk text-ink-muted hover:text-ink"
          >
            <X className="h-4 w-4" />
          </button>
        </header>

        {/* toolbar */}
        <div className="flex items-center gap-2 px-4 py-2 border-b border-ink-line shrink-0 bg-surface-sunk/40">
          <div className="flex items-center gap-1 text-[12px] text-ink-muted">
            <button
              type="button"
              onClick={() => jumpToPage(currentPage - 1)}
              disabled={!numPages || currentPage <= 1}
              className="h-7 w-7 inline-flex items-center justify-center rounded hover:bg-surface-sunk disabled:opacity-30"
              title="上一页"
            >
              <ChevronUp className="h-4 w-4" />
            </button>
            <Input
              type="text"
              value={pageInputValue}
              onChange={(e) => setPageInputValue(e.target.value.replace(/[^0-9]/g, ""))}
              onBlur={onPageInputCommit}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  onPageInputCommit();
                }
              }}
              className="h-7 w-14 text-center text-[12px] font-mono"
              aria-label="跳转到页码"
            />
            <span className="font-mono">/ {numPages ?? "?"}</span>
            <button
              type="button"
              onClick={() => jumpToPage(currentPage + 1)}
              disabled={!numPages || currentPage >= numPages}
              className="h-7 w-7 inline-flex items-center justify-center rounded hover:bg-surface-sunk disabled:opacity-30"
              title="下一页"
            >
              <ChevronDown className="h-4 w-4" />
            </button>
          </div>
          <div className="flex-1" />
          <div className="flex items-center gap-1 text-[12px] text-ink-muted">
            <button
              type="button"
              onClick={() => setZoomIdx((i) => Math.max(0, i - 1))}
              disabled={zoomIdx <= 0}
              className="h-7 w-7 inline-flex items-center justify-center rounded hover:bg-surface-sunk disabled:opacity-30"
              title="缩小"
            >
              <Minus className="h-4 w-4" />
            </button>
            <span className="w-12 text-center font-mono">{Math.round(zoomLevel * 100)}%</span>
            <button
              type="button"
              onClick={() => setZoomIdx((i) => Math.min(ZOOM_LEVELS.length - 1, i + 1))}
              disabled={zoomIdx >= ZOOM_LEVELS.length - 1}
              className="h-7 w-7 inline-flex items-center justify-center rounded hover:bg-surface-sunk disabled:opacity-30"
              title="放大"
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* scroll area */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-auto bg-surface-sunk px-4 py-4 space-y-4"
        >
          {activeError && (
            <div className="flex items-start gap-2 rounded-md bg-danger-soft border border-danger/20 px-3 py-2 mx-4 my-6 text-sm text-danger">
              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
              <span className="break-words">{activeError}</span>
            </div>
          )}
          {!activeError && !docFile && (
            <div className="flex flex-col items-center justify-center py-16 text-ink-muted">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span className="mt-3 text-sm">加载 PDF…</span>
            </div>
          )}
          {docFile && (
            <Document
              file={docFile}
              loading={
                <div className="flex items-center justify-center py-16 text-ink-muted">
                  <Loader2 className="h-5 w-5 animate-spin" />
                </div>
              }
              onLoadSuccess={({ numPages: n }) => {
                setNumPages(n);
                if (currentPage > n) setCurrentPage(1);
              }}
              onLoadError={(err) =>
                setError({ fileId, message: err?.message ?? "PDF 解析失败" })
              }
              className="flex flex-col items-center"
            >
              {Array.from({ length: numPages ?? 0 }, (_, i) => i + 1).map((pno) => (
                <div
                  key={pno}
                  data-page={pno}
                  ref={(el) => {
                    if (el) pageRefs.current.set(pno, el);
                    else pageRefs.current.delete(pno);
                  }}
                  className="mb-4 last:mb-0"
                >
                  <Page
                    pageNumber={pno}
                    scale={zoomLevel}
                    renderAnnotationLayer={false}
                    renderTextLayer
                    loading={
                      <div className="flex items-center justify-center py-16 text-ink-muted">
                        <Loader2 className="h-5 w-5 animate-spin" />
                      </div>
                    }
                    className="shadow-card border border-ink-line/60 rounded-sm bg-white"
                  />
                </div>
              ))}
            </Document>
          )}
        </div>
      </aside>
    </>
  );
}
