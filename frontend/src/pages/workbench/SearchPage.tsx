import { useState, useCallback } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  Send,
  Loader2,
  RotateCcw,
  Search as SearchIcon,
  AlertTriangle,
  FileText,
  Hash,
} from "lucide-react";

import { api, explainAxiosError } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FormField, FormSection } from "@/components/forms/FormPrimitives";
import { FileMultiSelect } from "@/components/forms/FileMultiSelect";
import { useCitationStore } from "@/stores/citation";
import type { LocalCitation } from "@/lib/sse-types";
import { cn } from "@/lib/utils";

/**
 * G3 条款检索 — POST /search 同步 JSON。
 *
 * 三粒度（page/passage/table_row）× 通道子集（semantic/bm25/graph_ppr，
 * regex 在 no-LLM 路径 422 拒）× 文件 / 页码 / 后缀过滤。
 *
 * 命中卡片 [^k] 角标点击 → CitationDrawer 渲染 PDF 单页（与 RAG 路径一致）。
 */

const CHANNELS = [
  { value: "semantic", label: "语义" },
  { value: "bm25", label: "BM25" },
  { value: "graph_ppr", label: "图谱 PPR" },
] as const;

const GRANULARITIES = [
  { value: "page", label: "整页" },
  { value: "passage", label: "段落" },
  { value: "table_row", label: "表格行" },
] as const;

const FormSchema = z.object({
  query: z.string().min(1, "请输入检索词").max(2000, "≤ 2000 字"),
  granularity: z.enum(["page", "passage", "table_row"]),
  channels: z
    .array(z.enum(["semantic", "bm25", "graph_ppr"]))
    .min(1, "至少选择一个通道"),
  file_ids: z.array(z.string()).optional(),
  page_lo: z.number().int().min(1).optional().nullable(),
  page_hi: z.number().int().min(1).optional().nullable(),
  rerank: z.boolean(),
  top_n: z.number().int().min(1).max(100).optional().nullable(),
}).refine(
  (v) => {
    if (v.page_lo == null && v.page_hi == null) return true;
    if (v.page_lo == null || v.page_hi == null) return false;
    return v.page_lo <= v.page_hi;
  },
  {
    message: "页码区间需同时填写且起始 ≤ 结束",
    path: ["page_hi"],
  },
);

type FormValues = z.infer<typeof FormSchema>;

interface SearchHit {
  file_id: string;
  page_id: string;
  page_number?: number | null;
  passage_id?: string | null;
  table_row_id?: string | null;
  score: number;
  channel_scores: Record<string, number>;
  channels_hit: string[];
  snippet: string;
  rerank_score?: number | null;
}

interface SearchResponse {
  query: string;
  granularity: "page" | "passage" | "table_row";
  channels_run: string[];
  filters_applied: Record<string, unknown>;
  hits: SearchHit[];
  n_total: number;
  n_returned: number;
  timings_ms: Record<string, number>;
  used_rrf: boolean;
  used_rerank: boolean;
  rrf_k?: number | null;
  rrf_top_m?: number | null;
  post_filter_overfetched: boolean;
  n_pre_filter: number;
}

export default function SearchPage() {
  const [busy, setBusy] = useState(false);
  const [resp, setResp] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const open_ = useCitationStore((s) => s.open_);

  const form = useForm<FormValues>({
    resolver: zodResolver(FormSchema),
    mode: "onChange",
    defaultValues: {
      query: "",
      granularity: "page",
      channels: ["semantic", "bm25"],
      file_ids: [],
      page_lo: null,
      page_hi: null,
      rerank: false,
      top_n: null,
    },
  });

  const submit = form.handleSubmit(async (values) => {
    setBusy(true);
    setError(null);
    setResp(null);
    try {
      const filters: Record<string, unknown> = {};
      if (values.file_ids && values.file_ids.length > 0) {
        filters.file_ids = values.file_ids;
      }
      if (values.page_lo != null && values.page_hi != null) {
        filters.page_range = [values.page_lo, values.page_hi];
      }
      const body: Record<string, unknown> = {
        query: values.query,
        granularity: values.granularity,
        channels: values.channels,
        rerank: values.rerank,
      };
      if (Object.keys(filters).length > 0) body.filters = filters;
      if (values.top_n) body.top_n = values.top_n;

      const { data } = await api.post<SearchResponse>("/search", body);
      setResp(data);
    } catch (e) {
      setError(explainAxiosError(e));
    } finally {
      setBusy(false);
    }
  });

  const reset = useCallback(() => {
    setResp(null);
    setError(null);
    form.reset();
  }, [form]);

  const openDrawerForHit = useCallback(
    (hitIdx: number) => {
      if (!resp) return;
      // 把当前命中列表整体当 items（用 hit index + 1 当 sup，纯 UI 编号），
      // 让 Drawer 的左右翻页能在多个命中之间切换；CitationDrawer 不强制
      // sup 单调，items 是命中数组本身。
      const items: LocalCitation[] = resp.hits.map((h, i) => ({
        sup: i + 1,
        kind: "local",
        file_id: h.file_id,
        page_id: h.page_id,
        page_number: h.page_number ?? undefined,
        page_preview: h.snippet,
      }));
      open_(items, items[hitIdx]);
    },
    [resp, open_],
  );

  const channels = form.watch("channels");
  const toggleChannel = (ch: (typeof CHANNELS)[number]["value"]) => {
    const next = channels.includes(ch)
      ? channels.filter((c) => c !== ch)
      : [...channels, ch];
    form.setValue("channels", next, { shouldValidate: true });
  };

  const granularity = form.watch("granularity");

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="border-b border-ink-line px-6 py-3 flex items-center gap-3 shrink-0">
        <SearchIcon className="h-4 w-4 text-primary-700" />
        <div className="min-w-0 flex-1">
          <h1 className="text-base font-semibold text-ink leading-tight">条款检索</h1>
          <p className="text-xs text-ink-muted truncate">
            三粒度 × 多通道 × 文件 / 页码 / 后缀过滤；不走 LLM。
          </p>
        </div>
        <span className="text-[11px] uppercase tracking-[0.16em] text-ink-subtle font-mono">Search</span>
      </header>

      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[420px_1fr] divide-y lg:divide-y-0 lg:divide-x divide-ink-line">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
          className="overflow-y-auto scrollbar-thin px-5 py-5 space-y-5 max-h-full"
        >
          <FormSection title="检索词">
            <FormField
              label="关键词 / 自然语言"
              htmlFor="search-query"
              required
              error={form.formState.errors.query?.message}
            >
              <Input
                id="search-query"
                placeholder="例：等待期、premium financing、IRR 计算"
                {...form.register("query")}
                aria-invalid={form.formState.errors.query ? true : undefined}
              />
            </FormField>
          </FormSection>

          <FormSection title="粒度">
            <div role="radiogroup" className="flex gap-1.5">
              {GRANULARITIES.map((g) => {
                const active = granularity === g.value;
                return (
                  <button
                    key={g.value}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    onClick={() => form.setValue("granularity", g.value, { shouldValidate: true })}
                    className={cn(
                      "rounded-sm border px-3 py-1 text-sm transition-colors",
                      active
                        ? "border-primary-300 bg-primary-50 text-primary-800"
                        : "border-ink-line text-ink-muted hover:border-primary-300 hover:bg-primary-50 hover:text-primary-700",
                    )}
                  >
                    {g.label}
                  </button>
                );
              })}
            </div>
          </FormSection>

          <FormSection title="通道">
            <FormField
              label="至少一个"
              error={form.formState.errors.channels?.message}
              hint="多通道自动 RRF 融合"
            >
              <div className="flex flex-wrap gap-1.5">
                {CHANNELS.map((c) => {
                  const active = channels.includes(c.value);
                  return (
                    <button
                      key={c.value}
                      type="button"
                      onClick={() => toggleChannel(c.value)}
                      className={cn(
                        "rounded-sm border px-2.5 py-1 text-sm transition-colors",
                        active
                          ? "border-primary-300 bg-primary-50 text-primary-800"
                          : "border-ink-line text-ink-muted hover:border-primary-300 hover:bg-primary-50 hover:text-primary-700",
                      )}
                      aria-pressed={active}
                    >
                      {c.label}
                    </button>
                  );
                })}
              </div>
            </FormField>
          </FormSection>

          <FormSection title="过滤">
            <FormField label="限定文件 (可选)" htmlFor="search-files" hint="不选 = 全部已索引文件">
              <FileMultiSelect
                id="search-files"
                value={form.watch("file_ids") ?? []}
                onChange={(v) => form.setValue("file_ids", v, { shouldValidate: true })}
                maxSelected={64}
              />
            </FormField>
            <div className="grid grid-cols-2 gap-3">
              <FormField label="起始页" htmlFor="page-lo">
                <Input
                  id="page-lo"
                  type="number"
                  min={1}
                  inputMode="numeric"
                  placeholder="不限"
                  {...form.register("page_lo", {
                    setValueAs: (v: unknown) => (v === "" || v == null ? null : Number(v)),
                  })}
                />
              </FormField>
              <FormField
                label="结束页"
                htmlFor="page-hi"
                error={form.formState.errors.page_hi?.message}
              >
                <Input
                  id="page-hi"
                  type="number"
                  min={1}
                  inputMode="numeric"
                  placeholder="不限"
                  {...form.register("page_hi", {
                    setValueAs: (v: unknown) => (v === "" || v == null ? null : Number(v)),
                  })}
                />
              </FormField>
            </div>
          </FormSection>

          <FormSection title="精排 / 输出">
            <div className="flex items-center gap-2">
              <input
                id="rerank"
                type="checkbox"
                {...form.register("rerank")}
                className="h-4 w-4 accent-primary-600"
              />
              <label htmlFor="rerank" className="text-sm text-ink">
                Rerank 模型重排
              </label>
              <span className="text-[11px] text-ink-subtle">(增加延迟和成本)</span>
            </div>
            <FormField label="返回前 N 条" htmlFor="top-n" hint="留空 = 后端默认值">
              <Input
                id="top-n"
                type="number"
                min={1}
                max={100}
                inputMode="numeric"
                placeholder="默认"
                {...form.register("top_n", {
                  setValueAs: (v: unknown) => (v === "" || v == null ? null : Number(v)),
                })}
              />
            </FormField>
          </FormSection>

          <div className="flex items-center gap-2 pt-1">
            {!busy ? (
              // 不再用 ``form.formState.isValid`` 门禁。用户改 granularity /
              // file_ids 时我们用了 ``shouldValidate: true`` 触发重算，但
              // mode="onChange" 下 isValid 仍可能短暂落后于真实状态，导致
              // 用户改完粒度按钮还是 disabled，看起来"请求一直发不出去"。
              // ``handleSubmit`` 自己会在提交时校验，校验失败会高亮错误
              // 字段，所以放心放行。
              <Button type="submit" size="md">
                <Send className="h-3.5 w-3.5" />
                检索
              </Button>
            ) : (
              <Button type="button" variant="secondary" size="md" disabled>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                检索中…
              </Button>
            )}
            <Button
              type="button"
              variant="ghost"
              size="md"
              onClick={reset}
              disabled={busy}
            >
              <RotateCcw className="h-3.5 w-3.5" />
              重置
            </Button>
          </div>
        </form>

        <div className="overflow-y-auto scrollbar-thin px-6 py-5">
          {!resp && !busy && !error && (
            <div className="flex h-full items-center justify-center text-sm text-ink-subtle">
              <div className="text-center space-y-1">
                <div>填好左侧条件后点击"检索"</div>
                <div className="text-[11px] font-mono text-ink-subtle/80">无 LLM · 通道命中即返</div>
              </div>
            </div>
          )}

          {busy && (
            <div className="flex h-full items-center justify-center text-sm text-ink-muted gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              检索中…
            </div>
          )}

          {error && !busy && (
            <div className="mx-auto max-w-3xl">
              <div className="flex items-start gap-2 rounded-md bg-danger-soft border border-danger/20 px-3 py-2 text-sm text-danger">
                <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                <span className="break-words">{error}</span>
              </div>
            </div>
          )}

          {resp && !busy && (
            <article className="mx-auto max-w-3xl space-y-3">
              <SummaryBar resp={resp} />
              {resp.hits.length === 0 ? (
                <div className="rounded-md border border-ink-line/70 bg-surface-raised/60 px-4 py-6 text-center text-sm text-ink-subtle">
                  没有命中 — 可放宽通道或粒度后再试
                </div>
              ) : (
                <ol className="space-y-2">
                  {resp.hits.map((h, i) => (
                    <HitCard
                      key={`${h.file_id}_${h.page_id}_${i}`}
                      idx={i}
                      hit={h}
                      granularity={resp.granularity}
                      onOpen={() => openDrawerForHit(i)}
                    />
                  ))}
                </ol>
              )}
            </article>
          )}
        </div>
      </div>
    </div>
  );
}

function SummaryBar({ resp }: { resp: SearchResponse }) {
  return (
    <div className="rounded-md border border-ink-line/70 bg-surface-raised/60 px-3 py-2 text-[12px] text-ink-muted flex flex-wrap items-center gap-x-4 gap-y-1">
      <span>
        粒度 <b className="text-ink">{resp.granularity}</b>
      </span>
      <span>
        通道 <b className="text-ink">{resp.channels_run.join("·")}</b>
      </span>
      <span>
        命中 <b className="text-ink">{resp.n_returned}</b> /{" "}
        {resp.post_filter_overfetched ? `${resp.n_pre_filter} 预过滤` : resp.n_total}
      </span>
      {resp.used_rrf && resp.rrf_k != null && (
        <span className="font-mono">RRF k={resp.rrf_k}</span>
      )}
      {resp.used_rerank && <span className="font-mono">+rerank</span>}
      {Object.keys(resp.timings_ms).length > 0 && (
        <span className="font-mono truncate" title={JSON.stringify(resp.timings_ms)}>
          {Object.entries(resp.timings_ms)
            .map(([k, v]) => `${k}:${v}ms`)
            .slice(0, 3)
            .join(" ")}
        </span>
      )}
    </div>
  );
}

function HitCard({
  idx,
  hit,
  granularity,
  onOpen,
}: {
  idx: number;
  hit: SearchHit;
  granularity: "page" | "passage" | "table_row";
  onOpen: () => void;
}) {
  return (
    <li className="rounded-md border border-ink-line/70 hover:border-primary-300 hover:bg-primary-50/30 transition-colors p-3">
      <div className="flex items-start gap-2.5">
        <button
          type="button"
          onClick={onOpen}
          className="shrink-0 inline-flex flex-col items-center gap-0.5 rounded-md bg-accent-50 hover:bg-accent-100 text-accent-700 px-2 py-1.5 text-[11px] font-mono"
          title="打开 PDF 单页预览"
        >
          <span>{idx + 1}</span>
          <FileText className="h-3 w-3" />
        </button>
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[12px] text-ink-muted">
            <span className="font-mono text-ink truncate max-w-[280px]" title={hit.file_id}>
              {hit.file_id}
            </span>
            {hit.page_number != null && (
              <span className="inline-flex items-center gap-0.5 font-mono">
                <Hash className="h-3 w-3" />p.{hit.page_number}
              </span>
            )}
            {granularity === "passage" && hit.passage_id && (
              <span className="font-mono text-[11px] text-ink-subtle truncate max-w-[140px]">
                ¶ {hit.passage_id}
              </span>
            )}
            {granularity === "table_row" && hit.table_row_id && (
              <span className="font-mono text-[11px] text-ink-subtle truncate max-w-[140px]">
                ▦ {hit.table_row_id}
              </span>
            )}
            <span className="ml-auto inline-flex items-center gap-1 text-[11px]">
              <span className="font-mono text-ink">score {hit.score.toFixed(3)}</span>
              {hit.rerank_score != null && (
                <span className="font-mono text-accent-700">
                  · rerank {hit.rerank_score.toFixed(3)}
                </span>
              )}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-1">
            {hit.channels_hit.map((c) => (
              <span
                key={c}
                className="rounded-sm bg-primary-50 text-primary-700 px-1.5 py-0.5 text-[10px] font-mono"
                title={`${c} score: ${(hit.channel_scores[c] ?? 0).toFixed(3)}`}
              >
                {c}
              </span>
            ))}
          </div>

          {hit.snippet && (
            <p className="text-[13px] leading-6 text-ink whitespace-pre-wrap line-clamp-5">
              {hit.snippet}
            </p>
          )}
        </div>
      </div>
    </li>
  );
}
