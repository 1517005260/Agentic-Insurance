import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  FileText,
  Loader2,
  MoreVertical,
  Plus,
  Regex,
  RefreshCw,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { api } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { IngestProgress } from "@/components/files/IngestProgress";
import { UploadDialog } from "@/components/files/UploadDialog";
import FilePreviewDrawer from "@/components/files/FilePreviewDrawer";
import { useFiles, type FileRow } from "@/hooks/useFiles";
import { useAuthStore } from "@/stores/auth";
import { useIngestQueue, type IngestEntry } from "@/stores/ingestQueue";
import { cn } from "@/lib/utils";

/**
 * /files —— 文件浏览页（读写双角色）。
 *
 * 设计取舍：
 *  - analyst：列表 + 子串/正则筛选 + 点卡片预览 PDF（CitationDrawer 复用）。
 *  - admin：在 analyst 基础上多 3 个操作 —— 顶部"上传"按钮 / 卡片右上 "⋯"
 *    菜单（重新索引 / 删除）。**不拆双页** —— inline RBAC 显隐，少一倍维护。
 *  - 索引中 / 失败的卡片：不可点击预览，点 "⋯ → 查看进度" 弹 IngestProgress
 *    抽屉看 stage 时间线（admin 才看）。
 *  - 搜索体验：默认子串，可切正则；正则编译失败自动回落子串。
 *  - 分页一页 20，整数倍于 4/5 列网格。
 */

const PAGE_SIZE = 20;

function compileFilter(query: string, useRegex: boolean): (row: FileRow) => boolean {
  const q = query.trim();
  if (!q) return () => true;
  if (useRegex) {
    try {
      const re = new RegExp(q, "i");
      return (r) =>
        re.test(r.display_name) ||
        re.test(r.original_filename) ||
        re.test(r.file_id);
    } catch {
      // 用户还在敲半截 regex，回落到子串而不是把列表清空
      return substringMatcher(q);
    }
  }
  return substringMatcher(q);
}

function substringMatcher(q: string): (row: FileRow) => boolean {
  const lower = q.toLowerCase();
  return (r) =>
    r.display_name.toLowerCase().includes(lower) ||
    r.original_filename.toLowerCase().includes(lower) ||
    r.file_id.toLowerCase().includes(lower);
}

export default function FilesPage() {
  const { data, isLoading, isError, error } = useFiles();
  const isAdmin = useAuthStore((s) => s.isAdmin());
  const qc = useQueryClient();
  const [query, setQuery] = useState("");
  const [useRegex, setUseRegex] = useState(false);
  const [page, setPage] = useState(0);
  const [uploadOpen, setUploadOpen] = useState(false);
  const ingestEntries = useIngestQueue((s) => s.entries);
  const dequeueIngest = useIngestQueue((s) => s.dequeue);
  // syncWithFiles is now called from useFiles itself so every /files
  // refetch (including from workbench FileMultiSelect) drains the
  // queue — see hooks/useFiles.ts.
  const [chipMenuOpen, setChipMenuOpen] = useState(false);
  /**
   * 弹"查看进度"抽屉的目标。带 variant 区分 fresh / reingest（reingest
   * 多 purge 一段）；row 是当时拿到的快照（drawer 不依赖 row.status 判
   * 断 dismissible，靠 IngestProgress 的 onStreamClosed/onTerminal）。
   */
  const [progressFor, setProgressFor] = useState<{
    row: FileRow;
    variant: "fresh" | "reingest";
  } | null>(null);
  const [previewFor, setPreviewFor] = useState<FileRow | null>(null);

  const reingestMu = useMutation({
    mutationFn: async (fileId: string) =>
      api.post(`/files/${encodeURIComponent(fileId)}/reingest`),
    onSuccess: (_data, fileId) => {
      qc.invalidateQueries({ queryKey: ["files"] });
      // 立刻打开进度抽屉看 reingest stage
      const row = (data ?? []).find((r) => r.file_id === fileId);
      if (row) setProgressFor({ row, variant: "reingest" });
      // 任何同 fileId 的旧 minimized 条目都过期了
      dequeueIngest(fileId);
    },
  });

  const deleteMu = useMutation({
    mutationFn: async (fileId: string) =>
      api.delete(`/files/${encodeURIComponent(fileId)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["files"] });
    },
  });

  const onReingest = useCallback(
    (row: FileRow) => {
      if (!confirm(`重新索引 ${row.display_name}？\n这会清空现有索引并从缓存重建。`)) return;
      reingestMu.mutate(row.file_id);
    },
    [reingestMu],
  );

  const onDelete = useCallback(
    (row: FileRow) => {
      if (!confirm(`确认删除 ${row.display_name}？\n此操作不可撤销。`)) return;
      deleteMu.mutate(row.file_id);
    },
    [deleteMu],
  );

  const matcher = useMemo(() => compileFilter(query, useRegex), [query, useRegex]);

  const filtered = useMemo(() => {
    const all = data ?? [];
    return all.filter(matcher);
  }, [data, matcher]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  // page 越界（搜索把当前页空了）→ 自动回到 0；分页按钮也兜底
  const effectivePage = Math.min(page, totalPages - 1);
  const slice = filtered.slice(
    effectivePage * PAGE_SIZE,
    (effectivePage + 1) * PAGE_SIZE,
  );

  const onSubmitFilter = (next: string) => {
    setQuery(next);
    setPage(0);
  };

  return (
    <div className="flex flex-col gap-4 p-6">
      <header className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-medium text-ink">文件浏览</h1>
          <p className="text-[12px] text-ink-subtle mt-0.5">
            已索引文档共 {data?.length ?? 0} 个；点击卡片预览首页 PDF。
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative w-72">
            <Search className="absolute left-2 top-2 h-4 w-4 text-ink-muted" />
            <Input
              value={query}
              onChange={(e) => onSubmitFilter(e.target.value)}
              placeholder={
                useRegex
                  ? "正则模式：如 ^AXA.*盛利"
                  : "按名称 / 原文件名 / file_id 子串过滤"
              }
              className="pl-8 pr-8"
            />
            {query && (
              <button
                type="button"
                onClick={() => onSubmitFilter("")}
                aria-label="清除筛选"
                className="absolute right-2 top-2 text-ink-muted hover:text-ink"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
          <Button
            type="button"
            variant={useRegex ? "secondary" : "ghost"}
            size="md"
            onClick={() => setUseRegex((v) => !v)}
            aria-pressed={useRegex}
            title="切换正则匹配（不区分大小写）"
          >
            <Regex className="h-4 w-4" />
            <span>正则</span>
          </Button>
          {isAdmin && (
            <Button
              type="button"
              variant="primary"
              size="md"
              onClick={() => setUploadOpen(true)}
            >
              <Plus className="h-4 w-4" /> 上传 PDF
            </Button>
          )}
        </div>
      </header>

      {isAdmin && Object.keys(ingestEntries).length > 0 && (
        <IngestQueueChip
          entries={Object.values(ingestEntries)}
          open={chipMenuOpen}
          onToggle={setChipMenuOpen}
          onPickFile={(entry) => {
            setChipMenuOpen(false);
            // ProgressDrawer 期望 row + variant；优先从 useFiles 列表
            // 取行（保证 status 实时），否则用 queue 里的 displayName 兜底
            const row =
              (data ?? []).find((r) => r.file_id === entry.fileId) ?? {
                file_id: entry.fileId,
                display_name: entry.displayName,
                original_filename: entry.displayName,
                suffix: ".pdf",
                byte_size: 0,
                page_count: null,
                status: "indexing",
              };
            setProgressFor({ row, variant: entry.variant });
          }}
        />
      )}

      {isLoading && (
        <div className="flex items-center justify-center py-16 gap-2 text-ink-muted">
          <Loader2 className="h-4 w-4 animate-spin" /> 加载文件列表…
        </div>
      )}
      {isError && (
        <div className="flex items-start gap-2 px-3 py-3 text-sm text-danger">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <div>加载失败：{(error as Error)?.message ?? "未知错误"}</div>
        </div>
      )}

      {!isLoading && !isError && filtered.length === 0 && (
        <div className="rounded border border-dashed border-ink-line bg-surface-raised py-16 text-center text-ink-subtle">
          {(data?.length ?? 0) === 0
            ? "尚无已索引文件"
            : "无匹配项 —— 试试关闭正则或清空筛选"}
        </div>
      )}

      {slice.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5 gap-3">
          {slice.map((row) => (
            <FileCard
              key={row.file_id}
              row={row}
              highlight={query}
              showAdmin={isAdmin}
              onPreview={() => setPreviewFor(row)}
              onReingest={() => onReingest(row)}
              onDelete={() => onDelete(row)}
              onShowProgress={() => setProgressFor({ row, variant: "fresh" })}
            />
          ))}
        </div>
      )}

      {filtered.length > PAGE_SIZE && (
        <Pagination
          page={effectivePage}
          totalPages={totalPages}
          totalItems={filtered.length}
          onPage={(p) => setPage(p)}
        />
      )}

      {isAdmin && uploadOpen && (
        <UploadDialog onClose={() => setUploadOpen(false)} />
      )}
      {previewFor && (
        <FilePreviewDrawer
          fileId={previewFor.file_id}
          displayName={previewFor.display_name}
          pageCount={previewFor.page_count}
          onClose={() => setPreviewFor(null)}
        />
      )}
      {isAdmin && progressFor && (
        <ProgressDrawer
          row={progressFor.row}
          variant={progressFor.variant}
          onTerminal={(s) => {
            qc.invalidateQueries({ queryKey: ["files"] });
            // 任何终态都把这条 file 从 minimized queue 里清掉
            dequeueIngest(progressFor.row.file_id);
            void s;
          }}
          onClose={() => setProgressFor(null)}
        />
      )}
    </div>
  );
}

// --------------------------------------------------------------- chip

function IngestQueueChip({
  entries,
  open,
  onToggle,
  onPickFile,
}: {
  entries: IngestEntry[];
  open: boolean;
  onToggle: (v: boolean) => void;
  onPickFile: (entry: IngestEntry) => void;
}) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      const t = e.target as Node;
      if (wrapperRef.current && !wrapperRef.current.contains(t)) onToggle(false);
    };
    window.addEventListener("mousedown", onClick);
    return () => window.removeEventListener("mousedown", onClick);
  }, [open, onToggle]);

  return (
    <div ref={wrapperRef} className="relative -mt-2">
      <button
        type="button"
        onClick={() => onToggle(!open)}
        aria-expanded={open}
        className={cn(
          "inline-flex items-center gap-2 rounded border px-3 py-1.5 text-[12px]",
          "bg-accent-50 border-accent-300 text-accent-800",
          "hover:bg-accent-100",
        )}
      >
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        索引中（{entries.length}）
      </button>
      {open && (
        <div
          role="menu"
          className={cn(
            "absolute left-0 mt-1 w-[min(360px,calc(100vw-32px))] rounded-md border border-ink-line",
            "bg-surface-raised shadow-pop py-1 text-[13px] z-20",
          )}
        >
          {entries
            .sort((a, b) => b.startedAt - a.startedAt)
            .map((entry) => (
              <button
                key={entry.fileId}
                type="button"
                onClick={() => onPickFile(entry)}
                className="w-full flex items-center justify-between gap-2 px-3 py-2 text-left hover:bg-surface-sunk"
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-ink">{entry.displayName}</span>
                  <span className="block text-[11px] text-ink-subtle font-mono">
                    {entry.fileId.slice(-12)} · {fmtAgo(entry.startedAt)}
                  </span>
                </span>
                <span className="text-[10px] uppercase tracking-[0.16em] text-accent-700">
                  {entry.variant === "reingest" ? "reindex" : "ingest"}
                </span>
              </button>
            ))}
        </div>
      )}
    </div>
  );
}

function fmtAgo(epochMs: number): string {
  const sec = Math.max(0, Math.floor((Date.now() - epochMs) / 1000));
  if (sec < 60) return `${sec}s 前`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m 前`;
  return `${Math.floor(sec / 3600)}h 前`;
}

// --------------------------------------------------------------- card

interface FileCardProps {
  row: FileRow;
  highlight: string;
  showAdmin: boolean;
  onPreview: () => void;
  onReingest: () => void;
  onDelete: () => void;
  /** admin 看 indexing/failed 时点开看 stage 时间线。 */
  onShowProgress: () => void;
}

function FileCard({
  row,
  highlight,
  showAdmin,
  onPreview,
  onReingest,
  onDelete,
  onShowProgress,
}: FileCardProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const isReady = row.status === "ready";
  const isIngesting =
    row.status === "pending" ||
    row.status === "parsing" ||
    row.status === "indexing";
  const isDeleting = row.status === "deleting";
  // ⋯ 菜单的"重新索引 / 删除"在任一过渡态都禁用，避免明显的 409 触发；
  // 但"查看进度"只在 ingest 过渡态有意义（delete 没有 stage 进度可看）。
  const isInFlight = isIngesting || isDeleting;
  const previewUrl = useAuthedBlobUrl(
    isReady ? `/files/${encodeURIComponent(row.file_id)}/preview` : null,
  );

  return (
    <div
      className={cn(
        "group relative flex flex-col rounded-md border bg-surface-raised text-left overflow-hidden",
        "border-ink-line transition-colors",
        isReady && "hover:border-primary-300 hover:shadow-sm",
        !isReady && "opacity-90",
      )}
      title={row.original_filename}
    >
      <button
        type="button"
        onClick={isReady ? onPreview : isIngesting ? onShowProgress : undefined}
        disabled={!isReady && !showAdmin}
        className={cn(
          "block w-full text-left",
          isReady &&
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30",
          !isReady && !isIngesting && "cursor-not-allowed",
        )}
      >
        <div className="aspect-[3/4] w-full bg-surface-sunk flex items-center justify-center overflow-hidden">
          {!isReady && (
            <div className="flex flex-col items-center gap-1 text-ink-subtle">
              <FileText className="h-8 w-8" />
              <span
                className={cn(
                  "text-[10px] uppercase tracking-[0.16em]",
                  row.status === "failed" && "text-danger",
                  (isIngesting || isDeleting) && "text-accent-700",
                )}
              >
                {row.status}
              </span>
            </div>
          )}
          {isReady && previewUrl.url && (
            <img
              src={previewUrl.url}
              alt=""
              decoding="async"
              className="h-full w-full object-cover object-top group-hover:scale-[1.02] transition-transform"
            />
          )}
          {isReady && previewUrl.failed && <ThumbnailFallback />}
        </div>
        <div className="px-2 py-2 space-y-1">
          <div className="text-[13px] text-ink leading-snug line-clamp-2">
            <Highlighted text={row.display_name} match={highlight} />
          </div>
          <div className="flex items-center gap-1.5 text-[11px] text-ink-subtle font-mono">
            <span className="truncate">{row.file_id.slice(-8)}</span>
            <span>·</span>
            <span>{row.suffix.replace(/^\./, "")}</span>
            <span>·</span>
            <span>{row.page_count ?? "?"}p</span>
          </div>
        </div>
      </button>

      {showAdmin && (
        <CardMenu
          open={menuOpen}
          onToggle={setMenuOpen}
          // 重新索引 / 删除 在任何过渡态都禁用
          actionsDisabled={isInFlight}
          // 仅 ingest 过渡态有 stage 进度可看
          showProgressItem={isIngesting}
          onReingest={onReingest}
          onDelete={onDelete}
          onShowProgress={onShowProgress}
        />
      )}
    </div>
  );
}

// --------------------------------------------------------------- card menu

function CardMenu({
  open,
  onToggle,
  actionsDisabled,
  showProgressItem,
  onReingest,
  onDelete,
  onShowProgress,
}: {
  open: boolean;
  onToggle: (v: boolean) => void;
  actionsDisabled: boolean;
  showProgressItem: boolean;
  onReingest: () => void;
  onDelete: () => void;
  onShowProgress: () => void;
}) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  // outside-click 关菜单。用 ref + contains 避免拿 file_id 拼 CSS
  // selector — file_id 含 CJK / 中括号等 selector-special 字符。
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      const t = e.target as Node;
      if (wrapperRef.current && !wrapperRef.current.contains(t)) onToggle(false);
    };
    window.addEventListener("mousedown", onClick);
    return () => window.removeEventListener("mousedown", onClick);
  }, [open, onToggle]);

  return (
    <div
      ref={wrapperRef}
      className="absolute top-1 right-1"
      onClick={(e) => e.stopPropagation()}
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onToggle(!open);
        }}
        aria-label="文件操作菜单"
        aria-expanded={open}
        className={cn(
          "h-7 w-7 inline-flex items-center justify-center rounded",
          "bg-surface-raised/80 backdrop-blur border border-ink-line/60",
          "text-ink-muted hover:text-ink hover:bg-surface-raised",
          "opacity-0 group-hover:opacity-100",
          open && "opacity-100",
        )}
      >
        <MoreVertical className="h-4 w-4" />
      </button>
      {open && (
        <div
          role="menu"
          className={cn(
            "absolute right-0 mt-1 w-44 rounded-md border border-ink-line",
            "bg-surface-raised shadow-pop py-1 text-[13px] z-10",
          )}
        >
          {showProgressItem && (
            <MenuItem
              onClick={() => {
                onToggle(false);
                onShowProgress();
              }}
            >
              <Loader2 className="h-3.5 w-3.5" /> 查看进度
            </MenuItem>
          )}
          <MenuItem
            onClick={() => {
              onToggle(false);
              onReingest();
            }}
            disabled={actionsDisabled}
          >
            <RefreshCw className="h-3.5 w-3.5" /> 重新索引
          </MenuItem>
          <MenuItem
            onClick={() => {
              onToggle(false);
              onDelete();
            }}
            disabled={actionsDisabled}
            danger
          >
            <Trash2 className="h-3.5 w-3.5" /> 删除
          </MenuItem>
        </div>
      )}
    </div>
  );
}

function MenuItem({
  children,
  onClick,
  disabled,
  danger,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "w-full flex items-center gap-2 px-3 py-1.5 text-left",
        disabled && "opacity-40 cursor-not-allowed",
        !disabled && (danger
          ? "text-danger hover:bg-danger/5"
          : "text-ink hover:bg-surface-sunk"),
      )}
    >
      {children}
    </button>
  );
}

// --------------------------------------------------------------- progress drawer

function ProgressDrawer({
  row,
  variant,
  onClose,
  onTerminal,
}: {
  row: FileRow;
  variant: "fresh" | "reingest";
  onClose: () => void;
  onTerminal?: (status: "ready" | "failed") => void;
}) {
  // 右侧抽屉。关闭策略升级：除了"流结束后允许关"之外，**进行中也允许
  // 隐藏到 header chip** —— minimize 路径 push 到 useIngestQueue，关掉
  // 当前 SSE 订阅但不阻碍后端 bg task。用户从 chip 点回来时重新订阅，
  // 后端 single-consumer 守卫在重连时给 fresh slot。
  const [streamFinished, setStreamFinished] = useState(false);
  const enqueueIngest = useIngestQueue((s) => s.enqueue);

  const minimize = () => {
    enqueueIngest({
      fileId: row.file_id,
      displayName: row.display_name,
      variant,
      startedAt: Date.now(),
    });
    onClose();
  };

  // 终态：直接关，因为 chip 已经被 onTerminal 那条 dequeue 清掉了。
  // 进行中：minimize 路径。两条路径都允许，所以不再有"必须等 stream
  // 结束才能关"的体验。
  const dismissHandler = streamFinished ? onClose : minimize;
  const dismissLabel = streamFinished ? "关闭" : "隐藏（后台继续，header chip 可恢复）";

  return (
    <>
      <div
        className="fixed inset-0 z-40 bg-ink/10 animate-fade-in"
        onClick={dismissHandler}
        aria-hidden
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={`${row.display_name} 索引进度`}
        className={cn(
          "fixed right-0 top-0 bottom-0 z-50 w-screen sm:w-[min(440px,100vw)]",
          "bg-surface-raised border-l border-ink-line shadow-pop",
          "flex flex-col animate-slide-in-r",
        )}
      >
        <header className="flex items-center justify-between gap-2 px-4 py-3 border-b border-ink-line">
          <div className="min-w-0">
            <div className="text-sm font-medium text-ink truncate">
              {row.display_name}
            </div>
            <div className="text-[11px] text-ink-subtle font-mono">
              {row.file_id.slice(-12)} · 索引进度
            </div>
          </div>
          <button
            type="button"
            onClick={dismissHandler}
            aria-label={dismissLabel}
            title={dismissLabel}
            className="h-7 w-7 inline-flex items-center justify-center rounded hover:bg-surface-sunk text-ink-muted hover:text-ink"
          >
            <X className="h-4 w-4" />
          </button>
        </header>
        <div className="px-4 py-4 overflow-auto">
          <IngestProgress
            fileId={row.file_id}
            variant={variant}
            onTerminal={(s) => onTerminal?.(s)}
            onStreamClosed={() => setStreamFinished(true)}
          />
        </div>
      </aside>
    </>
  );
}

/**
 * 拉一个需要 Bearer 鉴权的图片，转成 object URL。
 *
 * 浏览器 <img src> 不会带 axios interceptor 加的 Authorization header，
 * 所以 /files/{id}/preview 必须经过 axios 走完整鉴权链路再 blob → URL。
 * 卸载时 revoke，避免泄漏。
 */
function useAuthedBlobUrl(path: string | null): { url: string | null; failed: boolean } {
  // 用 path 当 useEffect 的 trigger，但只有 fetch 成功 / 失败 走 setState。
  // path 变 null 时不再 setState；改靠 useMemo 派生 url：path 为 null →
  // url 一定 null。这样 effect 体内不再有"清理性 setState"，规避
  // react-hooks/set-state-in-effect。
  const [state, setState] = useState<{ url: string | null; failed: boolean }>({
    url: null,
    failed: false,
  });

  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    let objectUrl: string | null = null;
    api
      .get<Blob>(path, { responseType: "blob" })
      .then((res) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(res.data);
        setState({ url: objectUrl, failed: false });
      })
      .catch(() => {
        if (cancelled) return;
        setState({ url: null, failed: true });
      });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [path]);

  // path 切回 null 时把上一次的 state 屏蔽（不 setState 也能让消费者拿到
  // null —— 卡片状态从 ready 退回 indexing 极少见但理论可能）
  return path ? state : { url: null, failed: false };
}

function ThumbnailFallback() {
  return (
    <div className="h-full w-full flex flex-col items-center justify-center gap-1 text-ink-subtle">
      <FileText className="h-8 w-8" />
      <span className="text-[10px] uppercase tracking-[0.16em]">no preview</span>
    </div>
  );
}

// --------------------------------------------------------------- helpers

function Highlighted({ text, match }: { text: string; match: string }) {
  const m = match.trim();
  if (!m) return <>{text}</>;
  // Highlight 用纯子串，不跟正则模式同步：正则高亮的开销 + 边界 case
  // 太多（如负向预查），简单 substring 命中已经足够指明匹配。
  const lower = text.toLowerCase();
  const needle = m.toLowerCase();
  const idx = lower.indexOf(needle);
  if (idx < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, idx)}
      <mark className="bg-primary-100 text-primary-800 rounded-sm px-0.5">
        {text.slice(idx, idx + m.length)}
      </mark>
      {text.slice(idx + m.length)}
    </>
  );
}

function Pagination({
  page,
  totalPages,
  totalItems,
  onPage,
}: {
  page: number;
  totalPages: number;
  totalItems: number;
  onPage: (p: number) => void;
}) {
  return (
    <div className="flex items-center justify-between text-[12px] text-ink-muted">
      <span>
        共 {totalItems} 个 · 第 {page + 1} / {totalPages} 页
      </span>
      <div className="flex items-center gap-1">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => onPage(Math.max(0, page - 1))}
          disabled={page === 0}
        >
          上一页
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => onPage(Math.min(totalPages - 1, page + 1))}
          disabled={page >= totalPages - 1}
        >
          下一页
        </Button>
      </div>
    </div>
  );
}
