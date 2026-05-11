/**
 * IngestProgress — live stage timeline driven by GET /files/{id}/jobs/stream.
 *
 * 设计要点：
 *  - 直接用 fetchSSE 而不是 useSSE：useSSE 默认 POST，加 method 选项
 *    会扩散到所有 caller；只一处 GET SSE，单点封装更干净。
 *  - 5/6 个 stage 顺序固定（parse → page_assets → text_dense →
 *    vision_dense → bm25 → graph，reingest 多一段 purge 在 parse 之前），
 *    pipeline serial 模式按 input order emit。第一次见 stage:start 入队，
 *    stage:done 收尾；这样错过中间 frame 也不会卡死。
 *  - `final` 帧来自后端 EventBus，携带 timings_ms / status / error；
 *    这是判定整体 ready / failed 的权威，不靠 status 字段推断。
 *  - error 帧（终止前一帧）携带 message，组件展示在最下方红条。
 *  - 卸载 abort：用户关 dialog 立刻断流，后端 EventBus.is_closed=true
 *    阻止后续 push（但 ingest 本身不会因此 abort —— bg task 跑完为止）。
 */
import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  Loader2,
} from "lucide-react";

import { fetchSSE, SSEHttpError } from "@/lib/sse";
import type {
  ErrorEvent as SSEErrorEvent,
  FinalEvent,
  IngestStageName,
  StageEvent,
} from "@/lib/sse-types";
import { acquireSlot } from "@/lib/fetchScheduler";
import { cn } from "@/lib/utils";

type StageStatus = "pending" | "running" | "done" | "error";

interface StageRow {
  name: string;
  status: StageStatus;
  elapsed_ms?: number;
  items?: number;
  skipped_reason?: string | null;
  error?: string;
}

// 标准顺序（serial mode 触发顺序）；purge 只有 reingest 才有，按需插入。
const FRESH_ORDER: IngestStageName[] = [
  "parse",
  "page_assets",
  "text_dense",
  "vision_dense",
  "bm25",
  "graph",
];
const REINGEST_ORDER: IngestStageName[] = ["purge", ...FRESH_ORDER];

const STAGE_LABEL: Record<string, string> = {
  parse: "解析 PDF",
  page_assets: "切页 / 抽文",
  text_dense: "文本向量索引",
  vision_dense: "视觉向量索引",
  bm25: "BM25 关键词索引",
  graph: "知识图谱构建",
  purge: "清理旧索引",
};

interface Props {
  fileId: string;
  /** "fresh" 默认顺序（6 stage）；"reingest" 7 stage（多 purge）。 */
  variant?: "fresh" | "reingest";
  /**
   * 后端 final.status 落地（ready / failed）时调用一次。``errorMsg``
   * 在 failed 时携带后端错误（``final.error`` / ``final.log_tail`` /
   * ``error`` 帧之一），父组件用它在外层显示错误信息——之前 onTerminal
   * 只传 status 导致父组件 failed 行什么也显示不出来。
   */
  onTerminal?: (status: "ready" | "failed", errorMsg?: string | null) => void;
  /**
   * 流以任何理由结束时调用一次：包括 final.status、503 重试用尽、EOF
   * 没拿到 done、网络异常等。父组件用它解锁 dismissible —— 否则 SSE
   * 异常断流（无 final 帧）会让 dialog 永久卡住无法关闭。
   */
  onStreamClosed?: () => void;
}

export function IngestProgress({
  fileId,
  variant = "fresh",
  onTerminal,
  onStreamClosed,
}: Props) {
  const initialOrder = variant === "reingest" ? REINGEST_ORDER : FRESH_ORDER;
  const [stages, setStages] = useState<StageRow[]>(() =>
    initialOrder.map((name) => ({ name, status: "pending" })),
  );
  const [final, setFinal] = useState<FinalEvent["data"] | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [streamState, setStreamState] = useState<"connecting" | "streaming" | "closed">(
    "connecting",
  );
  const onTerminalRef = useRef(onTerminal);
  const onStreamClosedRef = useRef(onStreamClosed);
  // 同步最新的 callback 进 ref（不能在 render body 写）
  useEffect(() => {
    onTerminalRef.current = onTerminal;
    onStreamClosedRef.current = onStreamClosed;
  }, [onTerminal, onStreamClosed]);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    let releaseSlot: (() => void) | null = null;

    /**
     * 一次完整的 fetchSSE 循环。返回 true 表示流以 `done` 帧正常收尾，
     * 要么终态成功要么终态失败 —— 无需重连。返回 false 表示流意外
     * EOF（拿不到 done），由外层决定是否重试。
     *
     * 503: 后端 wait_for_bus 超时但 bg task 还没起；重试 Retry-After
     * 后再连，最多 maxAttempts 次。
     */
    const runOnce = async (): Promise<boolean> => {
      let sawDone = false;
      for await (const ev of fetchSSE(`/files/${encodeURIComponent(fileId)}/jobs/stream`, {
        method: "GET",
        signal: controller.signal,
      })) {
        if (cancelled) return true;
        setStreamState((prev) => (prev === "connecting" ? "streaming" : prev));

        if (ev.event === "stage") {
          setStages((prev) => mergeStage(prev, ev as StageEvent));
        } else if (ev.event === "final") {
          setFinal((ev as FinalEvent).data);
        } else if (ev.event === "error") {
          setErrorMsg((ev as SSEErrorEvent).data.message || "ingest stream errored");
        } else if (ev.event === "done") {
          sawDone = true;
          break;
        }
      }
      return sawDone;
    };

    const maxRetries = 3;

    (async () => {
      // 排队拿 ingest_sse 槽位（cap=2）—— 这是 audit 里"4 文件 ingest
      // SSE 把 HTTP/1.1 6/host 连接池干光"的 root cause 修复。8 个
      // file 同时索引时，前 2 路开 SSE，剩下 6 路在这里 await 直到
      // 前面的进度终态后接力。等待期间 streamState 还是 "connecting"，
      // UI 显示"等待后端进度…"，与未拿槽位语义一致。
      const release = await acquireSlot("ingest_sse");
      if (cancelled) {
        release();
        return;
      }
      releaseSlot = release;
      try {
        let attempt = 0;
        while (!cancelled && attempt <= maxRetries) {
          try {
            const sawDone = await runOnce();
            if (cancelled) return;
            if (sawDone) {
              setStreamState("closed");
              return;
            }
            // 拿到正常 EOF 但没 done — 后端 bus 提前关或代理截断。
            // Don't retry; treat as failure so user sees a closed stream.
            setStreamState("closed");
            setErrorMsg((prev) =>
              prev ?? "ingest stream ended without `done` frame",
            );
            return;
          } catch (e) {
            if (cancelled) return;
            if ((e as DOMException)?.name === "AbortError") return;
            if (e instanceof SSEHttpError && e.status === 503 && attempt < maxRetries) {
              // bg task 还没起 / 短暂调度抖动；按 Retry-After 退避重试
              attempt += 1;
              await new Promise((r) => setTimeout(r, 2000));
              continue;
            }
            setErrorMsg(e instanceof Error ? e.message : String(e));
            setStreamState("closed");
            return;
          }
        }
      } finally {
        // 任何路径退出（cancelled / done / error / 503 用尽）都要归还
        // 槽位，让排队的下一行接力。release 幂等，cleanup 再调一次也安全。
        release();
        releaseSlot = null;
      }
    })();

    return () => {
      cancelled = true;
      controller.abort();
      releaseSlot?.();
    };
  }, [fileId]);

  // 终态触发外部 callback（一次）。failed 时把错误文案一并传出去，
  // 让父组件能在外层（例如 UploadDialog 的失败行）展示原因——否则父
  // 组件只知道 failed 但不知道为什么。
  useEffect(() => {
    if (!final) return;
    const status = final.status;
    if (status === "ready" || status === "failed") {
      const msg =
        status === "failed"
          ? (final.error ?? errorMsg ?? final.log_tail ?? null)
          : null;
      onTerminalRef.current?.(status, msg);
    }
  }, [final, errorMsg]);

  // 流任意原因结束（final 落地、503 用尽、EOF 无 done、网络异常）都
  // 通知父组件 —— 让 dialog 解锁 dismissible，避免 SSE 异常断流时
  // 永久卡住。
  useEffect(() => {
    if (streamState === "closed") {
      onStreamClosedRef.current?.();
    }
  }, [streamState]);

  const overall = computeOverall(final, errorMsg, streamState);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm">
          <OverallBadge state={overall} />
          <span className="text-ink-muted text-[12px]">
            {streamState === "connecting" && "等待后端进度…"}
            {streamState === "streaming" && overall === "running" && "正在索引…"}
            {streamState === "closed" && overall === "ready" && "索引完成"}
            {streamState === "closed" && overall === "failed" && "索引失败"}
          </span>
        </div>
        {final?.timings_ms && (
          <span className="text-[11px] text-ink-subtle font-mono">
            总耗时 {fmtMs(sumTimings(final.timings_ms))}
          </span>
        )}
      </div>

      <ol className="space-y-1.5">
        {stages.map((s) => (
          <StageItem key={s.name} stage={s} />
        ))}
      </ol>

      {(errorMsg || final?.error || final?.log_tail) && overall === "failed" && (
        <div
          role="alert"
          className="flex items-start gap-2 rounded border border-danger/30 bg-danger/5 px-3 py-2 text-[12px] text-danger"
        >
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <span className="break-words whitespace-pre-wrap">
            {errorMsg ?? final?.error ?? final?.log_tail}
          </span>
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------------------- stage row

function StageItem({ stage }: { stage: StageRow }) {
  const label = STAGE_LABEL[stage.name] ?? stage.name;
  const tail =
    stage.status === "done" || stage.status === "error"
      ? fmtTail(stage)
      : stage.status === "running"
        ? "运行中"
        : "等待";
  return (
    <li
      className={cn(
        "flex items-center gap-2 rounded px-2 py-1.5 text-[13px]",
        stage.status === "running" && "bg-primary-50",
        stage.status === "error" && "bg-danger/5",
      )}
    >
      <StageIcon status={stage.status} />
      <span className="flex-1 text-ink">{label}</span>
      <span
        className={cn(
          "text-[11px] font-mono tabular-nums",
          stage.status === "error" ? "text-danger" : "text-ink-muted",
        )}
      >
        {tail}
      </span>
    </li>
  );
}

function StageIcon({ status }: { status: StageStatus }) {
  switch (status) {
    case "running":
      return <Loader2 className="h-4 w-4 text-primary-600 animate-spin" />;
    case "done":
      return <CheckCircle2 className="h-4 w-4 text-primary-600" />;
    case "error":
      return <AlertTriangle className="h-4 w-4 text-danger" />;
    default:
      return <CircleDashed className="h-4 w-4 text-ink-line" />;
  }
}

function fmtTail(s: StageRow): string {
  const parts: string[] = [];
  if (s.elapsed_ms != null) parts.push(fmtMs(s.elapsed_ms));
  if (s.items != null && s.items > 0) parts.push(`${s.items} items`);
  if (s.skipped_reason) parts.push(`skipped: ${s.skipped_reason}`);
  if (s.error) parts.push(`✗ ${s.error.slice(0, 80)}`);
  return parts.join(" · ") || (s.status === "done" ? "ok" : "");
}

function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m${s.toString().padStart(2, "0")}s`;
}

function sumTimings(t: Record<string, number>): number {
  return Object.values(t).reduce((acc, v) => acc + (v || 0), 0);
}

// --------------------------------------------------------------- merge

function mergeStage(prev: StageRow[], ev: StageEvent): StageRow[] {
  const stage = ev.data.stage;
  const phase = ev.data.phase;
  const idx = prev.findIndex((s) => s.name === stage);

  // Unknown stage (e.g. "purge" arrived in fresh variant): append it.
  if (idx < 0) {
    return [
      ...prev,
      {
        name: stage,
        status: phase === "done" ? (ev.data.error ? "error" : "done") : "running",
        elapsed_ms: ev.data.elapsed_ms,
        items: ev.data.items,
        skipped_reason: ev.data.skipped_reason ?? null,
        error: ev.data.error,
      },
    ];
  }

  return prev.map((row, i) => {
    if (i !== idx) return row;
    if (phase === "start") return { ...row, status: "running" };
    return {
      ...row,
      status: ev.data.error ? "error" : "done",
      elapsed_ms: ev.data.elapsed_ms,
      items: ev.data.items,
      skipped_reason: ev.data.skipped_reason ?? null,
      error: ev.data.error,
    };
  });
}

// --------------------------------------------------------------- overall

type OverallState = "running" | "ready" | "failed";

function computeOverall(
  final: FinalEvent["data"] | null,
  errorMsg: string | null,
  stream: "connecting" | "streaming" | "closed",
): OverallState {
  if (final?.status === "ready") return "ready";
  if (final?.status === "failed" || errorMsg || final?.error) return "failed";
  if (stream === "closed") return "failed"; // 流断了但没 final
  return "running";
}

function OverallBadge({ state }: { state: OverallState }) {
  if (state === "ready") {
    return (
      <span className="inline-flex items-center gap-1 rounded bg-primary-100 text-primary-800 px-2 py-0.5 text-[11px] font-medium">
        <CheckCircle2 className="h-3 w-3" /> 完成
      </span>
    );
  }
  if (state === "failed") {
    return (
      <span className="inline-flex items-center gap-1 rounded bg-danger/10 text-danger px-2 py-0.5 text-[11px] font-medium">
        <AlertTriangle className="h-3 w-3" /> 失败
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded bg-accent-100 text-accent-800 px-2 py-0.5 text-[11px] font-medium">
      <Loader2 className="h-3 w-3 animate-spin" /> 进行中
    </span>
  );
}
