import { useState, type KeyboardEvent } from "react";
import { Plus, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

/**
 * 多 chip 集合输入：用户键入文本，回车 / 点 + 加进 chips；点 chip
 * 上的 × 删除。
 *
 * 用在：Compare 的属性维度、PolicyCalc 的 calc_targets、Recommend 的
 * 健康史 / 家族史 free-text 集。比文本框输 csv 更克制 + 不容易误。
 *
 * 给 RHF 的 Controller 用：value/onChange 受控，所有逻辑都在父表单里。
 */
export function ChipsField({
  id,
  value,
  onChange,
  placeholder,
  presets,
  maxItems,
  maxChars = 80,
  invalid,
  describedBy,
  // FormField cloneElement 注入的 aria-describedby —— ChipsField 是自定
  // 义组件，React 不会自动透；这里显式接 + merge 给真正的 input。
  "aria-describedby": ariaDescribedBy,
}: {
  id?: string;
  value: string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
  /** 一键塞入的常见值。点 chip 加进 value（已存在则跳过）。 */
  presets?: readonly string[];
  maxItems?: number;
  maxChars?: number;
  invalid?: boolean;
  describedBy?: string;
  "aria-describedby"?: string;
}) {
  const [draft, setDraft] = useState("");

  const limit = maxItems ?? Infinity;
  const isFull = value.length >= limit;

  const tryAdd = (raw: string) => {
    const v = raw.trim();
    if (!v) return;
    if (v.length > maxChars) return;
    if (value.includes(v)) return;
    if (value.length >= limit) return;
    onChange([...value, v]);
    setDraft("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      tryAdd(draft);
    } else if (e.key === "Backspace" && !draft && value.length > 0) {
      // backspace 一个 empty input 删掉最后一个 chip — 友好
      onChange(value.slice(0, -1));
    }
  };

  const removeAt = (i: number) => {
    onChange(value.filter((_, idx) => idx !== i));
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-1.5 min-h-[2.25rem] rounded border border-ink-line bg-surface-raised px-2 py-1.5">
        {value.length === 0 && (
          <span className="text-[12px] text-ink-subtle">尚未添加任何项目</span>
        )}
        {value.map((chip, i) => (
          <span
            key={`${chip}-${i}`}
            className="inline-flex items-center gap-1 rounded-sm border border-primary-200 bg-primary-50 px-1.5 py-0.5 text-[12px] text-primary-800"
          >
            <span className="break-all">{chip}</span>
            <button
              type="button"
              onClick={() => removeAt(i)}
              aria-label={`移除 ${chip}`}
              className="text-primary-700 hover:text-primary-900"
            >
              <X className="h-3 w-3" />
            </button>
          </span>
        ))}
      </div>

      <div className="flex gap-1.5">
        <Input
          id={id}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={isFull ? `已达上限 ${limit}` : (placeholder ?? "输入后回车添加")}
          disabled={isFull}
          aria-invalid={invalid || undefined}
          aria-describedby={[describedBy, ariaDescribedBy].filter(Boolean).join(" ") || undefined}
          className={cn(invalid && "border-danger focus-visible:border-danger")}
          maxLength={maxChars}
        />
        <Button
          type="button"
          variant="secondary"
          size="md"
          onClick={() => tryAdd(draft)}
          disabled={!draft.trim() || isFull}
          className="whitespace-nowrap shrink-0"
        >
          <Plus className="h-3.5 w-3.5 shrink-0" />
          <span className="whitespace-nowrap">加入</span>
        </Button>
      </div>

      {presets && presets.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          <span className="text-[11px] uppercase tracking-[0.14em] text-ink-subtle font-mono mr-1 self-center">
            常用
          </span>
          {presets.map((p) => {
            const active = value.includes(p);
            const disabled = active || isFull;
            return (
              <button
                key={p}
                type="button"
                disabled={disabled}
                onClick={() => tryAdd(p)}
                className={cn(
                  "inline-flex items-center rounded-sm border px-1.5 py-0.5 text-[11px] transition-colors",
                  active
                    ? "border-primary-300 bg-primary-50 text-primary-700 cursor-default"
                    : "border-ink-line text-ink-muted hover:border-primary-300 hover:bg-primary-50 hover:text-primary-700",
                  disabled && "cursor-not-allowed opacity-60",
                )}
                aria-pressed={active}
              >
                {p}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
