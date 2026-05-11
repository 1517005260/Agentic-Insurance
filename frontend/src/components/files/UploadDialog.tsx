/**
 * UploadDialog — admin-only modal: pick one or more PDFs, POST /files
 * for each, then mount IngestProgress per file.
 *
 * 设计要点：
 *  - 单 / 多文件：input ``multiple`` 接收数组；后端串行 INGEST_LOCK +
 *    并发 PARSE_SEM 自行调度，前端只需要并行起 N 个 POST 即可
 *  - 三态：select → uploading → progress（每个文件一行 IngestProgress）
 *  - 只接受 .pdf（后端能跑 paddle ocr 的最常见格式；其他格式后续再放）
 *  - dialog 在上传/索引中点关闭：把所有进行中的 file 入 ingestQueue 走
 *    minimize 路径，不 abort 后端 bg task —— 那是已经接受的 work
 *  - 完成后 invalidate ['files']，让 FilesPage 列表立刻刷新
 *  - 错误：upload POST 409 / 400 在文件行内显示；索引阶段失败靠
 *    IngestProgress 自己的红条（共一个错误显示通道）
 */
import { useEffect, useRef, useState } from "react";
import { AlertTriangle, FileText, Upload, X } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { isAxiosError } from "axios";

import { api } from "@/api/client";
import { Button } from "@/components/ui/button";
import { IngestProgress } from "@/components/files/IngestProgress";
import { useIngestQueue } from "@/stores/ingestQueue";
import { cn } from "@/lib/utils";
import { schedule } from "@/lib/fetchScheduler";

interface UploadResponse {
  file_id: string;
  display_name: string;
  original_filename: string;
  status: string;
}

interface PendingItem {
  /** 本地稳定 id（File 没有稳定标识，name 可能重复）。 */
  key: string;
  file: File;
  /** Server-side file_id，POST 成功后写入。 */
  fileId: string | null;
  status: "queued" | "uploading" | "indexing" | "ready" | "failed";
  /** 失败原因（upload POST 阶段或 IngestProgress 报上来）。 */
  error: string | null;
}

type Phase = "select" | "uploading" | "progress" | "done";

interface Props {
  onClose: () => void;
}

/** 每次上传批次的硬上限。超过后 onPickFiles 会丢弃尾部文件。 */
const MAX_BATCH = 8;

/**
 * 这个组件不持有 open prop —— 完全由父组件 mount/unmount 控制生命周期，
 * 这样进入即"select"，关闭即销毁，省去一段 reset-on-open 的 effect 副
 * 作用（也是 eslint react-hooks/set-state-in-effect 的根因）。
 */
export function UploadDialog({ onClose }: Props) {
  const [items, setItems] = useState<PendingItem[]>([]);
  const [phase, setPhase] = useState<Phase>("select");
  // 当 onPickFiles 因 MAX_BATCH 截断丢弃了文件时记一条警告，UI 顶部
  // 显示。带 ``nonce`` 是为了让"连续两次同文案"也触发 React state
  // 更新（否则 setLimitWarning(sameString) 会被跳过 → 5s timer 不
  // 重置 → 用户在新一次操作几秒后就看不到提示）。
  const [limitWarning, setLimitWarning] = useState<{
    msg: string;
    nonce: number;
  } | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const qc = useQueryClient();
  const enqueueIngest = useIngestQueue((s) => s.enqueue);
  const dequeueIngest = useIngestQueue((s) => s.dequeue);

  // mount 时一次性把焦点送进 dialog；rAF 等动画就位再 focus
  useEffect(() => {
    const id = requestAnimationFrame(() => dialogRef.current?.focus());
    return () => cancelAnimationFrame(id);
  }, []);

  const updateItem = (key: string, patch: Partial<PendingItem>) => {
    setItems((prev) => prev.map((it) => (it.key === key ? { ...it, ...patch } : it)));
  };

  const removeItem = (key: string) => {
    setItems((prev) => prev.filter((it) => it.key !== key));
  };

  const onPickFiles = (next: File[]) => {
    let dropped = 0;
    let dupes = 0;
    setItems((prev) => {
      // 合并现有 + 新选；按 (name+size+lastModified) 去重，不让同一
      // 份 PDF 加两次。``lastModified`` 防止"用户改了内容但同名同大小"
      // 的边界 case 误判为重复。
      // 一次最多保留 ``MAX_BATCH`` 个：浏览器并行 multipart POST 上限
      // (HTTP/1.1 6/host) + 后端一次性 ``await file.read()`` 内存压力，
      // 8 个已经接近一台 8 GB WSL 的安全线。
      const dedupeKey = (f: File) => `${f.name}::${f.size}::${f.lastModified}`;
      const seen = new Set(prev.map((it) => dedupeKey(it.file)));
      const merged = [...prev];
      for (const f of next) {
        const key = dedupeKey(f);
        if (seen.has(key)) {
          dupes += 1;
          continue;
        }
        if (merged.length >= MAX_BATCH) {
          dropped += 1;
          continue;
        }
        seen.add(key);
        merged.push({
          key: `${key}::${Date.now()}::${Math.random().toString(36).slice(2, 7)}`,
          file: f,
          fileId: null,
          status: "queued",
          error: null,
        });
      }
      return merged;
    });
    // 用户有可见反馈：超过 MAX_BATCH 的文件被丢、或选了同名同大小同
    // mtime 的重复文件
    if (dropped > 0 || dupes > 0) {
      const parts: string[] = [];
      if (dropped > 0) {
        parts.push(`已超出单批上限 ${MAX_BATCH} 个，丢弃了 ${dropped} 个文件`);
      }
      if (dupes > 0) {
        parts.push(`已忽略 ${dupes} 个重复文件（同名/大小/修改时间一致）`);
      }
      // nonce 强制 state 引用变化，覆盖同文案场景下的 5s timer 重置
      setLimitWarning({
        msg: parts.join("；") + "。可分批继续上传。",
        nonce: Date.now(),
      });
    } else {
      setLimitWarning(null);
    }
  };

  // limitWarning 5s 后自动消，避免在上传/索引阶段还残留
  useEffect(() => {
    if (!limitWarning) return;
    const id = setTimeout(() => setLimitWarning(null), 5000);
    return () => clearTimeout(id);
  }, [limitWarning]);

  /**
   * "Minimize" 路径：用户在 progress 阶段点关闭/ESC 时调用。把所有
   * 仍在跑的 fileId 推入 useIngestQueue，FilesPage header 会显示一个
   * "索引中(n)" chip 用来重新打开 ProgressDrawer。后端 bg task 不受影响
   * —— 我们只是断掉所有 SSE 订阅。
   */
  const minimizeAndClose = () => {
    if (phase === "progress") {
      for (const it of items) {
        if (it.fileId && it.status !== "ready" && it.status !== "failed") {
          enqueueIngest({
            fileId: it.fileId,
            displayName: it.file.name,
            variant: "fresh",
            startedAt: Date.now(),
          });
        }
      }
    }
    onClose();
  };

  // ESC 关闭：select / done 直接关；uploading 阶段拒绝中断 multipart
  // POST 防 partial blob；progress 阶段走 minimize 路径。
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (phase === "select" || phase === "done") onClose();
      else if (phase === "progress") minimizeAndClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, items]);

  /**
   * 多文件提交：把每个文件丢进 ``upload`` 队列（cap=2），由
   * fetchScheduler 限制 in-flight 并发，避免一次 8 个 multipart POST
   * 把浏览器 6/host 连接池抢光（导致 /auth/me / chat 列表被 Queued
   * 数秒）。等待中的项保持 ``queued`` 状态，被调度起跑后再翻
   * ``uploading``。
   *
   * 所有文件状态各自独立；任一文件失败不影响其它文件。所有任务结束
   * （无论成败）后进 progress 阶段，挂 IngestProgress 列表。
   */
  const onSubmit = async () => {
    if (items.length === 0) return;
    setPhase("uploading");
    await Promise.allSettled(
      items.map((it) =>
        schedule("upload", async () => {
          updateItem(it.key, { status: "uploading", error: null });
          const fd = new FormData();
          fd.append("file", it.file);
          try {
            const { data } = await api.post<UploadResponse>("/files", fd);
            updateItem(it.key, {
              fileId: data.file_id,
              status: "indexing",
            });
          } catch (e: unknown) {
            const msg =
              extractAxiosErr(e) ?? (e instanceof Error ? e.message : String(e));
            updateItem(it.key, { status: "failed", error: msg });
          }
        }),
      ),
    );
    qc.invalidateQueries({ queryKey: ["files"] });
    // 哪怕全部上传失败，也进 progress 让用户看到错误列表 + 自己关
    setPhase("progress");
  };

  // 自动关闭定时器需要在 unmount / 用户手动 close 时清理，否则在
  // 800ms 内重新打开 dialog 会被旧 timer 误关。
  const autoCloseRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    return () => {
      if (autoCloseRef.current) clearTimeout(autoCloseRef.current);
    };
  }, []);

  // 任一文件终态都触发：从 ingest queue 清掉这条，更新状态 + 错误。
  // 不在 updater 内做 phase/auto-close 副作用——那会在 React 18 严格
  // 模式 dev 双跑下被调两次。聚合判断改放到下方 useEffect([items]) 里。
  const onItemTerminal = (
    key: string,
    fileId: string,
    status: "ready" | "failed",
    errorMsg?: string | null,
  ) => {
    updateItem(key, { status, error: errorMsg ?? null });
    qc.invalidateQueries({ queryKey: ["files"] });
    dequeueIngest(fileId);
  };

  // items 变到全终态时收口 phase + 触发 800 ms 自动关。放在 effect 里
  // 是为了让 ``items`` updater 保持纯函数；也避免 setTimeout 在 render
  // 期被多次 schedule。
  // ``setPhase("done")`` 是必要的状态翻转，对 lint 的 set-state-in-effect
  // 规则 disable —— 这里不是"派生 state 应直接计算"的场景，而是"全部
  // 终态后的一次性 phase 转换"，必须用 setState 触发后续 UI 更新。
  useEffect(() => {
    if (phase !== "progress" || items.length === 0) return;
    const allDone = items.every(
      (it) => it.status === "ready" || it.status === "failed",
    );
    if (!allDone) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPhase("done");
    const anyFailed = items.some((it) => it.status === "failed");
    if (!anyFailed) {
      if (autoCloseRef.current) clearTimeout(autoCloseRef.current);
      autoCloseRef.current = setTimeout(onClose, 800);
    }
    // onClose ref-stable 由父保证（mount/unmount）；items 引用变才重判
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, phase]);

  // progress 阶段也允许关闭：走 minimize 路径而不是 onClose 直连，确保
  // 队列条目被记录，FilesPage 才能后续恢复进度查看。
  const allowDismiss = phase === "select" || phase === "progress" || phase === "done";
  const dismissHandler = phase === "progress" ? minimizeAndClose : onClose;

  return (
    <>
      <div
        className="fixed inset-0 z-40 bg-ink/20 animate-fade-in"
        onClick={() => allowDismiss && dismissHandler()}
        aria-hidden
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-dialog-title"
        tabIndex={-1}
        className={cn(
          "fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2",
          "w-[min(640px,calc(100vw-32px))] max-h-[90vh] overflow-auto",
          "bg-surface-raised border border-ink-line rounded-md shadow-pop",
          "flex flex-col animate-fade-in focus-visible:outline-none",
        )}
      >
        <header className="flex items-center justify-between gap-2 px-4 py-3 border-b border-ink-line">
          <h2 id="upload-dialog-title" className="text-sm font-medium text-ink">
            {phase === "select" && `上传 PDF（已选 ${items.length} 个）`}
            {phase === "uploading" && "上传中…"}
            {phase === "progress" && "索引中（可隐藏，header 索引 chip 可恢复）"}
            {phase === "done" && "已完成"}
          </h2>
          {allowDismiss && (
            <button
              type="button"
              onClick={dismissHandler}
              aria-label={phase === "progress" ? "隐藏到 header chip" : "关闭对话框"}
              title={phase === "progress" ? "隐藏（后台继续索引，header chip 可恢复查看）" : "关闭"}
              className="h-7 w-7 inline-flex items-center justify-center rounded hover:bg-surface-sunk text-ink-muted hover:text-ink"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </header>

        <div className="px-4 py-4 space-y-4">
          {limitWarning && (
            <div
              role="status"
              aria-live="polite"
              className="flex items-start gap-2 rounded border border-warning/30 bg-warning-soft/60 px-3 py-2 text-[12px] text-warning"
            >
              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
              <span className="break-words">{limitWarning.msg}</span>
            </div>
          )}

          {phase === "select" && (
            <SelectForm items={items} onPickFiles={onPickFiles} onRemove={removeItem} />
          )}

          {(phase === "uploading" || phase === "progress" || phase === "done") && (
            <ul className="space-y-3">
              {items.map((it) => (
                <li
                  key={it.key}
                  className="rounded border border-ink-line bg-surface-raised p-3"
                >
                  <div className="flex items-center gap-2 text-[13px] mb-2">
                    <FileText className="h-4 w-4 text-ink-muted shrink-0" />
                    <span className="text-ink truncate flex-1" title={it.file.name}>
                      {it.file.name}
                    </span>
                    <span className="text-[11px] text-ink-subtle font-mono">
                      {fmtSize(it.file.size)}
                    </span>
                  </div>
                  {it.status === "uploading" && (
                    <div className="flex items-center gap-2 text-[12px] text-ink-muted">
                      <span className="inline-block h-2 w-2 rounded-full bg-primary-500 animate-pulse" />
                      上传中…
                    </div>
                  )}
                  {it.status === "queued" && (
                    <div className="text-[12px] text-ink-subtle">排队中</div>
                  )}
                  {it.status === "failed" && it.error && (
                    <div
                      role="alert"
                      className="flex items-start gap-2 rounded border border-danger/30 bg-danger/5 px-2 py-1.5 text-[12px] text-danger"
                    >
                      <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                      <span className="break-words">{it.error}</span>
                    </div>
                  )}
                  {/*
                   * Failed 时仍要渲染 IngestProgress —— 让用户看到 stage
                   * 时间线 + 红条错误（前面的状态切换已经把 it.error
                   * 写好；IngestProgress 内部也会展示 final.error /
                   * log_tail，双保险）。
                   */}
                  {(it.status === "indexing" ||
                    it.status === "ready" ||
                    it.status === "failed") &&
                    it.fileId && (
                      <IngestProgress
                        fileId={it.fileId}
                        onTerminal={(s, errMsg) =>
                          onItemTerminal(it.key, it.fileId!, s, errMsg)
                        }
                        // 兜底：SSE 因 503/EOF 断而拿不到 final 帧时
                        // 仍要把 ``indexing`` 翻成 ``failed``，避免
                        // dialog 永远停在"索引中"且自动关条件
                        // (allDone) 永远不成立。
                        onStreamClosed={() => {
                          setItems((cur) =>
                            cur.map((row) =>
                              row.key === it.key && row.status === "indexing"
                                ? {
                                    ...row,
                                    status: "failed",
                                    error:
                                      row.error ??
                                      "SSE 流意外结束，请刷新文件页查看实际状态。",
                                  }
                                : row,
                            ),
                          );
                          qc.invalidateQueries({ queryKey: ["files"] });
                        }}
                      />
                    )}
                </li>
              ))}
            </ul>
          )}
        </div>

        {phase === "select" && (
          <footer className="flex items-center justify-end gap-2 px-4 py-3 border-t border-ink-line">
            <Button variant="ghost" onClick={onClose}>
              取消
            </Button>
            <Button
              variant="primary"
              onClick={onSubmit}
              disabled={items.length === 0}
              loading={false}
            >
              <Upload className="h-4 w-4" /> 开始上传 {items.length > 0 ? `(${items.length})` : ""}
            </Button>
          </footer>
        )}
        {phase === "done" && (
          <footer className="flex items-center justify-end gap-2 px-4 py-3 border-t border-ink-line">
            <Button variant="primary" onClick={onClose}>
              关闭
            </Button>
          </footer>
        )}
      </div>
    </>
  );
}

// --------------------------------------------------------------- form

function SelectForm({
  items,
  onPickFiles,
  onRemove,
}: {
  items: PendingItem[];
  onPickFiles: (files: File[]) => void;
  onRemove: (key: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <div className="space-y-3">
      <label className="block">
        <span className="block text-[12px] text-ink-muted mb-1">
          PDF 文件{" "}
          <span className="text-ink-subtle">
            （支持多选 / 多次添加，单批最多 {MAX_BATCH} 个）
          </span>
        </span>
        {/*
         * <label> already forwards clicks (and Enter via the focused
         * <input>) to the contained <input>. Adding a manual
         * onClick={inputRef.click()} on a wrapper used to fire the
         * native picker twice — once via the label's implicit
         * association and once via the explicit dispatch — which the
         * user observed as "select a PDF, then the picker pops up
         * again before the upload starts". Keep the visual dropzone
         * styling on the inner div but let the label handle activation.
         */}
        <div
          className={cn(
            "rounded border border-dashed border-ink-line p-4 text-center",
            "hover:border-primary-400 cursor-pointer",
            items.length > 0 && "border-primary-400 bg-primary-50/40",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            multiple
            className="hidden"
            onChange={(e) => {
              const next = Array.from(e.target.files ?? []);
              if (next.length > 0) onPickFiles(next);
              // reset 让用户连续点同一个文件也能重新触发
              if (e.target) e.target.value = "";
            }}
          />
          <div className="text-sm text-ink-muted">
            <Upload className="h-5 w-5 mx-auto mb-1 text-ink-line" />
            点击选择一个或多个 PDF
          </div>
        </div>
      </label>

      {items.length > 0 && (
        <ul className="space-y-1">
          {items.map((it) => (
            <li
              key={it.key}
              className="flex items-center gap-2 text-[12px] rounded border border-ink-line/60 bg-surface-sunk/40 px-2 py-1.5"
            >
              <FileText className="h-3.5 w-3.5 text-ink-muted shrink-0" />
              <span className="flex-1 truncate" title={it.file.name}>
                {it.file.name}
              </span>
              <span className="text-[11px] text-ink-subtle font-mono">
                {fmtSize(it.file.size)}
              </span>
              <button
                type="button"
                aria-label={`移除 ${it.file.name}`}
                onClick={() => onRemove(it.key)}
                className="text-ink-muted hover:text-danger"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// --------------------------------------------------------------- helpers

function fmtSize(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function extractAxiosErr(e: unknown): string | null {
  if (!isAxiosError(e)) return null;
  const detail = e.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (detail) return JSON.stringify(detail);
  if (e.message) return e.message;
  return null;
}
