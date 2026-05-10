/**
 * /admin/config —— 配置中心（30 项）
 *
 * 设计取舍：
 *  - 按 entry.group 分段（rag / agent.* / prompt.* / tavily / citation 等）
 *  - 类型分发：
 *      int           → number input + min/max
 *      str (短)      → text input
 *      str (长 / 名 prompt.*) → textarea，且独立"重置默认"按钮
 *  - 改动是 dirty diff —— 只 PATCH 改过的 key，all-or-422 后端兜底；422
 *    报哪个 key 报错就行内红框
 *  - prompt 区段每行单独"重置"按钮（DELETE /admin/config/{key}）—— 千字
 *    prompt 不可能让 admin paste 默认回去
 *  - 刚保存成功后，dirty 清空 + react-query refetch；snapshot 用最新值
 */
import { useMemo, useState } from "react";
import { isAxiosError } from "axios";
import {
  AlertTriangle,
  Loader2,
  RotateCcw,
  Save,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  useAdminConfig,
  useAdminConfigPatch,
  useAdminConfigReset,
  type ConfigEntry,
  type ConfigValue,
} from "@/stores/config";
import { cn } from "@/lib/utils";

// 大文本字段：跟 entry.type === "str" 撞上时切到 textarea + 重置按钮
const TEXTAREA_GROUPS = new Set(["prompt"]);

function isTextarea(entry: ConfigEntry): boolean {
  if (entry.type !== "str") return false;
  if (TEXTAREA_GROUPS.has(entry.group.split(".")[0])) return true;
  // 兜底：默认值很长（>200 字符）也用 textarea
  return typeof entry.default === "string" && entry.default.length > 200;
}

function fmtAxios(e: unknown): string {
  if (isAxiosError(e)) {
    const detail = e.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (detail) return JSON.stringify(detail);
    return e.message;
  }
  return e instanceof Error ? e.message : String(e);
}

export default function ConfigPage() {
  const { data, isLoading, isError, error } = useAdminConfig();
  const patchMu = useAdminConfigPatch();
  const resetMu = useAdminConfigReset();

  /** 编辑中的脏值；key → 用户当前输入（int 已强转 number、bool 已强转 boolean）。 */
  const [dirty, setDirty] = useState<Record<string, ConfigValue>>({});
  const [saveError, setSaveError] = useState<string | null>(null);

  const grouped = useMemo(() => {
    if (!data) return new Map<string, ConfigEntry[]>();
    const m = new Map<string, ConfigEntry[]>();
    for (const e of data.schema) {
      const top = e.group.split(".")[0] || "misc";
      if (!m.has(top)) m.set(top, []);
      m.get(top)!.push(e);
    }
    return m;
  }, [data]);

  const onChange = (entry: ConfigEntry, raw: ConfigValue) => {
    setSaveError(null);
    setDirty((prev) => ({ ...prev, [entry.key]: raw }));
  };

  const onSave = () => {
    if (Object.keys(dirty).length === 0) return;
    patchMu.mutate(dirty, {
      onSuccess: () => {
        setDirty({});
        setSaveError(null);
      },
      onError: (e) => setSaveError(fmtAxios(e)),
    });
  };

  const onReset = (key: string) => {
    if (!confirm(`重置 ${key} 为默认值？`)) return;
    resetMu.mutate(key, {
      onSuccess: () => {
        // 重置成功后把这个 key 的 dirty 也清掉，否则用户的本地编辑会
        // 立刻盖回 server 的默认值（PATCH 比 reset 后的 refetch 早）
        setDirty((prev) => {
          const { [key]: _drop, ...rest } = prev;
          void _drop;
          return rest;
        });
        setSaveError(null);
      },
      onError: (e) => setSaveError(fmtAxios(e)),
    });
  };

  const dirtyCount = Object.keys(dirty).length;

  return (
    <div className="flex flex-col gap-4 p-6 max-w-[960px] mx-auto">
      {/*
       * sticky header notes:
       *  - z-30 lifts above section headers (which use the page-shell's
       *    default sub-z) so scrolled-up content can't shine through;
       *  - bg-surface-raised + shadow-sm + border-b draws an opaque
       *    floor so the header reads as a proper toolbar instead of a
       *    translucent ghost that "follows" the scrolled content
       *    (the reported "保存栏挡视线" symptom);
       *  - dropping the -mx-6 px-6 trick keeps the header inside the
       *    sticky-context container — without it the header tried to
       *    extend past the centered max-w container and visually
       *    floated independently of the scroll surface.
       */}
      <header className="flex items-center justify-between gap-4 sticky top-0 z-30 bg-surface-raised shadow-sm py-3 px-3 -mx-3 border-b border-ink-line rounded-b">
        <div>
          <h1 className="text-xl font-medium text-ink">系统配置</h1>
          <p className="text-[12px] text-ink-subtle mt-0.5">
            共 {data?.schema.length ?? 0} 项 · 已修改 {dirtyCount} 项 ·
            修改后点保存批量提交，全有效或全失败。
          </p>
        </div>
        <div className="flex items-center gap-2">
          {dirtyCount > 0 && (
            <Button
              variant="ghost"
              size="md"
              onClick={() => setDirty({})}
              disabled={patchMu.isPending}
            >
              丢弃修改
            </Button>
          )}
          <Button
            variant="primary"
            size="md"
            onClick={onSave}
            disabled={dirtyCount === 0 || patchMu.isPending}
            loading={patchMu.isPending}
          >
            <Save className="h-4 w-4" /> 保存（{dirtyCount}）
          </Button>
        </div>
      </header>

      {saveError && (
        <div
          role="alert"
          className="flex items-start gap-2 rounded border border-danger/30 bg-danger/5 px-3 py-2 text-[12px] text-danger"
        >
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <div className="break-words whitespace-pre-wrap">{saveError}</div>
        </div>
      )}

      {isLoading && (
        <div className="flex items-center justify-center py-16 gap-2 text-ink-muted">
          <Loader2 className="h-4 w-4 animate-spin" /> 加载配置…
        </div>
      )}

      {isError && (
        <div className="flex items-start gap-2 px-3 py-3 text-sm text-danger">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <div>加载失败：{(error as Error)?.message ?? "未知错误"}</div>
        </div>
      )}

      {data && (
        <div className="space-y-6">
          {[...grouped.entries()].map(([groupKey, entries]) => (
            <section
              key={groupKey}
              className="border border-ink-line rounded overflow-hidden"
            >
              <header className="bg-surface-sunk px-3 py-2 text-[12px] font-medium text-ink-muted uppercase tracking-[0.16em]">
                {groupKey}
              </header>
              <div className="divide-y divide-ink-line">
                {entries.map((entry) => (
                  <ConfigRow
                    key={entry.key}
                    entry={entry}
                    serverValue={data.snapshot[entry.key]}
                    dirtyValue={dirty[entry.key]}
                    onChange={(raw) => onChange(entry, raw)}
                    onReset={() => onReset(entry.key)}
                    resetting={
                      resetMu.isPending && resetMu.variables === entry.key
                    }
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------------------- row

function ConfigRow({
  entry,
  serverValue,
  dirtyValue,
  onChange,
  onReset,
  resetting,
}: {
  entry: ConfigEntry;
  serverValue: ConfigValue;
  dirtyValue: ConfigValue | undefined;
  onChange: (raw: ConfigValue) => void;
  onReset: () => void;
  resetting: boolean;
}) {
  const value = dirtyValue ?? serverValue;
  const isDirty = dirtyValue !== undefined;
  const isDefault = serverValue === entry.default;
  const useTextarea = isTextarea(entry);

  return (
    <div
      className={cn(
        "px-3 py-3 grid gap-2",
        useTextarea ? "grid-cols-1" : "md:grid-cols-[280px_1fr_auto]",
        isDirty && "bg-accent-50/40",
      )}
    >
      <div className={cn("space-y-0.5", useTextarea && "flex items-center gap-2")}>
        <div className="flex items-center gap-1.5">
          <span className="text-[13px] font-mono text-ink">{entry.key}</span>
          {isDirty && (
            <span className="text-[10px] uppercase tracking-[0.16em] text-accent-700">
              dirty
            </span>
          )}
          {!isDefault && !isDirty && (
            <span className="text-[10px] uppercase tracking-[0.16em] text-ink-muted">
              custom
            </span>
          )}
        </div>
        <div className="text-[11px] text-ink-subtle">{entry.description}</div>
      </div>

      <div>
        {useTextarea ? (
          <textarea
            value={String(value)}
            onChange={(e) => onChange(e.target.value)}
            className={cn(
              "w-full rounded border border-ink-line bg-surface-raised text-[12px] font-mono",
              "px-2 py-1.5 min-h-[140px]",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30",
            )}
            spellCheck={false}
          />
        ) : entry.type === "bool" ? (
          // bool: 切换按钮代替 checkbox（更显眼，admin 一眼看清当前值）
          <button
            type="button"
            onClick={() => onChange(!value)}
            className={cn(
              "inline-flex items-center gap-2 rounded border px-3 py-1.5 text-[13px]",
              value
                ? "bg-primary-100 text-primary-800 border-primary-300"
                : "bg-ink-line/30 text-ink-muted border-ink-line",
            )}
            aria-pressed={Boolean(value)}
          >
            <span
              className={cn(
                "h-2 w-2 rounded-full",
                value ? "bg-primary-600" : "bg-ink-muted",
              )}
            />
            {value ? "true" : "false"}
          </button>
        ) : entry.type === "int" ? (
          <Input
            type="number"
            value={String(value)}
            min={entry.min ?? undefined}
            max={entry.max ?? undefined}
            step={1}
            onChange={(e) => {
              const n = Number(e.target.value);
              onChange(Number.isNaN(n) ? 0 : Math.trunc(n));
            }}
          />
        ) : entry.type === "float" ? (
          <Input
            type="number"
            value={String(value)}
            min={entry.min ?? undefined}
            max={entry.max ?? undefined}
            step="any"
            onChange={(e) => {
              const n = Number(e.target.value);
              onChange(Number.isNaN(n) ? 0 : n);
            }}
          />
        ) : (
          <Input
            type="text"
            value={String(value)}
            onChange={(e) => onChange(e.target.value)}
            maxLength={entry.max_length ?? undefined}
          />
        )}
        {(entry.type === "int" || entry.type === "float") &&
          (entry.min != null || entry.max != null) && (
            <div className="text-[11px] text-ink-subtle mt-1 font-mono">
              范围：{entry.min ?? "-∞"} ~ {entry.max ?? "+∞"} · 默认 {String(entry.default)}
            </div>
          )}
      </div>

      <div className={cn("flex items-start", useTextarea && "justify-end mt-2")}>
        <button
          type="button"
          onClick={onReset}
          disabled={resetting || isDefault}
          className={cn(
            "h-7 inline-flex items-center gap-1 px-2 rounded text-[11px]",
            isDefault
              ? "text-ink-line cursor-not-allowed"
              : "text-ink-muted hover:text-ink hover:bg-surface-sunk",
          )}
          title={isDefault ? "已是默认值" : "重置为默认值"}
        >
          {resetting ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <RotateCcw className="h-3 w-3" />
          )}
          重置
        </button>
      </div>
    </div>
  );
}
