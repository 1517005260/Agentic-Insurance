import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Loader2, Search, X, Check, AlertTriangle } from "lucide-react";

import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useReadyFiles, type FileRow } from "@/hooks/useFiles";
import { cn } from "@/lib/utils";

/**
 * 多文件 / 单文件下拉多选。
 *
 * 数据源：useReadyFiles（已过滤 status==='ready'）。
 *
 * - `mode="multi"` —— Compare / ClaimCheck 用，最少 / 最多由父级 zod 校验
 * - `mode="single"` —— Exclusion / PolicyCalc 用；勾一个就 close 下拉
 *
 * 选中态以 file_id 数组形式向上传，避免 caller 自己再 dedup。
 */
export function FileMultiSelect({
  id,
  mode = "multi",
  value,
  onChange,
  invalid,
  describedBy,
  // FormField 用 cloneElement 注入 `aria-describedby`，但 FileMultiSelect
  // 是自定义组件 —— React 不会把这个属性自动透到内部 button。这里显式接
  // 收并 merge 进 describedBy。
  "aria-describedby": ariaDescribedBy,
  maxSelected,
  placeholder = "点击选择已索引的保单文档",
}: {
  id?: string;
  mode?: "multi" | "single";
  value: string[];
  onChange: (next: string[]) => void;
  invalid?: boolean;
  describedBy?: string;
  "aria-describedby"?: string;
  maxSelected?: number;
  placeholder?: string;
}) {
  const { data: files, isLoading, isError, error } = useReadyFiles();
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const wrapRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  // outside click / Esc / 完成按钮 三种关闭路径需要区分焦点行为：
  //   - Esc / 完成 / single-mode 自动关 → 把焦点还给 trigger（键盘流畅）
  //   - outside click 关 → 不要抢焦点（用户已经点了别的控件）
  // 用 shouldRestoreFocusRef 在每个"主动关闭"分支前置位，effect 里读后清。
  const shouldRestoreFocusRef = useRef(false);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) {
        // outside click：关，但不归还焦点
        shouldRestoreFocusRef.current = false;
        setOpen(false);
      }
    };
    window.addEventListener("mousedown", onDocClick);
    return () => window.removeEventListener("mousedown", onDocClick);
  }, [open]);

  const wasOpenRef = useRef(false);
  useEffect(() => {
    if (wasOpenRef.current && !open) {
      if (shouldRestoreFocusRef.current) {
        triggerRef.current?.focus();
      }
      shouldRestoreFocusRef.current = false;
    }
    wasOpenRef.current = open;
  }, [open]);

  const filtered = useMemo<FileRow[]>(() => {
    const all = files ?? [];
    if (!filter.trim()) return all;
    const f = filter.toLowerCase();
    return all.filter(
      (r) =>
        r.display_name.toLowerCase().includes(f) ||
        r.original_filename.toLowerCase().includes(f) ||
        r.file_id.toLowerCase().includes(f),
    );
  }, [files, filter]);

  const valueSet = useMemo(() => new Set(value), [value]);

  const isCapped = maxSelected != null && value.length >= maxSelected;

  const toggle = (fid: string) => {
    if (valueSet.has(fid)) {
      onChange(value.filter((v) => v !== fid));
      return;
    }
    if (mode === "single") {
      onChange([fid]);
      shouldRestoreFocusRef.current = true;
      setOpen(false);
      return;
    }
    if (isCapped) return;
    onChange([...value, fid]);
  };

  const removeAt = (fid: string) => onChange(value.filter((v) => v !== fid));

  const selectedRows = useMemo<FileRow[]>(() => {
    const all = files ?? [];
    return value
      .map((fid) => all.find((r) => r.file_id === fid))
      .filter((r): r is FileRow => Boolean(r));
  }, [value, files]);

  return (
    <div ref={wrapRef} className="relative space-y-2">
      <button
        id={id}
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-invalid={invalid || undefined}
        aria-describedby={[describedBy, ariaDescribedBy].filter(Boolean).join(" ") || undefined}
        className={cn(
          "flex w-full items-center justify-between rounded border bg-surface-raised px-3 h-9 text-sm text-ink",
          "hover:bg-surface-sunk/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30",
          invalid ? "border-danger" : "border-ink-line",
        )}
      >
        <span className="truncate">
          {value.length === 0 ? (
            <span className="text-ink-subtle">{placeholder}</span>
          ) : mode === "single" ? (
            (selectedRows[0]?.display_name ?? value[0])
          ) : (
            <>
              已选 {value.length} 个
              {maxSelected ? <span className="text-ink-subtle"> / {maxSelected}</span> : null}
            </>
          )}
        </span>
        <ChevronDown className="h-4 w-4 text-ink-muted shrink-0" />
      </button>

      {/* 已选 chip 行（multi 才显示） */}
      {mode === "multi" && value.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {selectedRows.length > 0 ? (
            selectedRows.map((r) => (
              <span
                key={r.file_id}
                title={r.original_filename}
                className="inline-flex items-center gap-1 rounded-sm border border-primary-200 bg-primary-50 px-1.5 py-0.5 text-[12px] text-primary-800 max-w-[260px]"
              >
                <span className="truncate">{r.display_name}</span>
                <button
                  type="button"
                  onClick={() => removeAt(r.file_id)}
                  aria-label={`移除 ${r.display_name}`}
                  className="text-primary-700 hover:text-primary-900 shrink-0"
                >
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))
          ) : (
            // 已勾的 fid 不在最新 list 里（被删了？）
            <span className="text-[12px] text-ink-subtle">
              已选 {value.length} 个文件（部分可能已下线）
            </span>
          )}
        </div>
      )}

      {open && (
        <div
          // a11y：popover 容器不该是 listbox（内部还有 input + button），
          // 否则 ARIA 验证会失败 + 屏幕阅读器把整个 popover 当成"选项列表"。
          // listbox 角色只放在真正包含 option 的容器上。
          role="dialog"
          aria-label="文件选择"
          aria-modal="false"
          className="absolute z-30 mt-1 left-0 right-0 max-h-[320px] rounded-md border border-ink-line bg-surface-raised shadow-pop overflow-hidden flex flex-col"
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              e.stopPropagation();
              shouldRestoreFocusRef.current = true;
              setOpen(false);
            }
          }}
        >
          <div className="px-2 py-2 border-b border-ink-line shrink-0">
            <div className="relative">
              <Search className="absolute left-2 top-2 h-4 w-4 text-ink-muted" />
              <Input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="按名称 / 文件名 / file_id 过滤"
                className="pl-8"
                autoFocus
              />
            </div>
          </div>
          <div
            role="listbox"
            aria-multiselectable={mode === "multi"}
            className="overflow-y-auto scrollbar-thin flex-1"
          >
            {isLoading && (
              <div className="flex items-center justify-center gap-2 py-6 text-sm text-ink-muted">
                <Loader2 className="h-4 w-4 animate-spin" />
                加载文件列表…
              </div>
            )}
            {isError && (
              <div className="flex items-start gap-2 px-3 py-3 text-sm text-danger">
                <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <div>加载失败：{(error as Error)?.message ?? "未知错误"}</div>
                </div>
              </div>
            )}
            {!isLoading && !isError && filtered.length === 0 && (
              <div className="px-3 py-4 text-sm text-ink-subtle">
                {files && files.length === 0 ? "尚无已索引文件" : "无匹配项"}
              </div>
            )}
            {filtered.map((r) => {
              const checked = valueSet.has(r.file_id);
              const disabled = !checked && mode === "multi" && isCapped;
              return (
                <button
                  type="button"
                  key={r.file_id}
                  onClick={() => toggle(r.file_id)}
                  disabled={disabled}
                  role="option"
                  aria-selected={checked}
                  className={cn(
                    "flex w-full items-start gap-2 px-3 py-2 text-left text-sm transition-colors",
                    "hover:bg-surface-sunk/60",
                    checked && "bg-primary-50/70",
                    disabled && "cursor-not-allowed opacity-50",
                  )}
                >
                  <span
                    className={cn(
                      "mt-0.5 h-4 w-4 shrink-0 rounded-sm border flex items-center justify-center",
                      checked
                        ? "bg-primary-600 border-primary-600 text-surface-raised"
                        : "border-ink-line",
                    )}
                  >
                    {checked && <Check className="h-3 w-3" />}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-ink truncate">{r.display_name}</span>
                    <span className="block text-[11px] text-ink-subtle font-mono truncate">
                      {r.file_id} · {r.suffix} · {r.page_count ?? "?"}p
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
          <div className="flex items-center justify-between border-t border-ink-line px-2 py-1.5 shrink-0">
            <span className="text-[11px] text-ink-subtle font-mono">
              {value.length}/{maxSelected ?? "∞"} 已选
            </span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => {
                shouldRestoreFocusRef.current = true;
                setOpen(false);
              }}
            >
              完成
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
