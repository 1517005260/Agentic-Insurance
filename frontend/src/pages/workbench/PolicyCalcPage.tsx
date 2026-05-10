import { useMemo } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Code2, ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

import { Input } from "@/components/ui/input";
import { FormField, FormSection } from "@/components/forms/FormPrimitives";
import { ChipsField } from "@/components/forms/ChipsField";
import { FileMultiSelect } from "@/components/forms/FileMultiSelect";
import { MarkdownWithSup } from "@/components/chat/MarkdownWithSup";
import { WorkbenchScaffold } from "@/components/workbench/WorkbenchScaffold";
import { useWorkbenchStream } from "@/components/workbench/useWorkbenchStream";
import { CitationChipRow } from "@/components/workbench/CitationChipRow";
import { useCitationStore } from "@/stores/citation";
import type { CitationItem, SSEEvent } from "@/lib/sse-types";
import type { WorkbenchTurn } from "@/components/workbench/turn";
import { cn } from "@/lib/utils";

/**
 * G7c 保单精算 — BaseAgent + 强制 code_run。
 *
 * 表单字段对齐后端 PolicyParams + calc_targets (≤6, ≤300chars/each)。
 * 答案区在 markdown 之上叠一个"代码执行"折叠面板，把 stream 里的
 * code_run tool_call/tool_result 拉出来给用户看（精算计算的 stdout/
 * stderr 是说服力关键）。
 */

const PolicyParamsSchema = z.object({
  age_at_issue: z.number().int().min(0).max(100),
  gender: z.enum(["M", "F", "X"]),
  premium_mode: z.enum(["annual", "monthly", "single"]),
  premium_amount: z.number().min(0),
  term_years: z.number().int().min(1).max(80),
  sum_assured: z.number().min(0),
  currency: z.string().min(1).max(8),
  target_age: z.number().int().min(0).max(120).optional(),
  target_year: z.number().int().min(0).max(100).optional(),
});

const FormSchema = z.object({
  file_id: z.string().min(1, "请选择保单"),
  policy_params: PolicyParamsSchema,
  calc_targets: z
    .array(z.string().max(300))
    .min(1, "至少 1 个计算目标")
    .max(6, "最多 6 个"),
});
type FormValues = z.infer<typeof FormSchema>;

const TARGET_PRESETS = [
  "现金价值按年表",
  "第 20 年退保价值",
  "IRR 至 65 岁",
  "回本年份",
  "嵌入价值 EV (VIF + ANAV)",
  "NBV margin",
  "保费融资净 IRR (5% 利率)",
  "保证 vs 非保证 IRR 拆解",
] as const;

const GENDER_OPTS = [
  { value: "M", label: "男" },
  { value: "F", label: "女" },
  { value: "X", label: "其他" },
] as const;
const PREMIUM_MODES = [
  { value: "annual", label: "年缴" },
  { value: "monthly", label: "月缴" },
  { value: "single", label: "趸缴" },
] as const;
const CURRENCIES = ["HKD", "USD", "RMB", "SGD"] as const;

export default function PolicyCalcPage() {
  const { turn, busy, runStream, abort, reset } = useWorkbenchStream("PolicyCalc");

  const form = useForm<FormValues>({
    resolver: zodResolver(FormSchema),
    mode: "onChange",
    defaultValues: {
      file_id: "",
      policy_params: {
        age_at_issue: 35,
        gender: "M",
        premium_mode: "annual",
        premium_amount: 100000,
        term_years: 5,
        sum_assured: 1_000_000,
        currency: "HKD",
      },
      calc_targets: [],
    },
  });

  // 不要 useMemo([form]) 缓存 —— form 引用稳定，watch 的内部状态变化
  // 不会触发 useMemo 重算；选完文件后 fileIds 会一直停在 []，FileMultiSelect
  // 的下拉勾选 + 触发按钮 label 都拿不到当前选中。
  const fileId = form.watch("file_id");
  const fileIds = fileId ? [fileId] : [];

  const targets = form.watch("calc_targets");
  // 注意把 isValid 纳入；mode="onChange" + zod 让 number 字段空值
  // 立即标记 invalid，按钮自动禁用，避免点击后 handleSubmit 静默吞。
  const canSubmit =
    !busy && form.formState.isValid && fileId.length > 0 && targets.length >= 1;

  const onSubmit = form.handleSubmit((v) => {
    const pp: Record<string, unknown> = {
      age_at_issue: v.policy_params.age_at_issue,
      gender: v.policy_params.gender,
      premium_mode: v.policy_params.premium_mode,
      premium_amount: v.policy_params.premium_amount,
      term_years: v.policy_params.term_years,
      sum_assured: v.policy_params.sum_assured,
      currency: v.policy_params.currency,
    };
    if (v.policy_params.target_age != null) pp.target_age = v.policy_params.target_age;
    if (v.policy_params.target_year != null) pp.target_year = v.policy_params.target_year;
    runStream("/insurance/policy-calc/stream", {
      file_id: v.file_id,
      policy_params: pp,
      calc_targets: v.calc_targets,
    });
  });

  const onReset = () => {
    reset();
    form.reset();
  };

  const gender = form.watch("policy_params.gender");
  const mode = form.watch("policy_params.premium_mode");
  const currency = form.watch("policy_params.currency");

  return (
    <WorkbenchScaffold
      title="保单精算"
      description="选 1 份保单 + 输入保单参数 + 计算目标 chips → BaseAgent 强制走 code_run（numpy_financial / scipy）。"
      modeLabel="PolicyCalc"
      turn={turn}
      busy={busy}
      canSubmit={!!canSubmit}
      onSubmit={() => void onSubmit()}
      onAbort={abort}
      onReset={onReset}
      hideCitationChips
      renderAnswer={(t) => <PolicyCalcAnswer turn={t} />}
      renderForm={() => (
        <>
          <FormSection title="保单 + 货币">
            <FormField
              label="保单文档"
              required
              error={form.formState.errors.file_id?.message}
            >
              <FileMultiSelect
                mode="single"
                value={fileIds}
                onChange={(v) =>
                  form.setValue("file_id", v[0] ?? "", { shouldValidate: true })
                }
                placeholder="选择 1 份保单"
                invalid={!!form.formState.errors.file_id}
              />
            </FormField>
            <FormField label="货币" required>
              <div className="flex gap-1.5">
                {CURRENCIES.map((c) => {
                  const active = currency === c;
                  return (
                    <button
                      key={c}
                      type="button"
                      onClick={() =>
                        form.setValue("policy_params.currency", c, { shouldValidate: true })
                      }
                      className={cn(
                        "rounded-sm border px-3 py-1 text-sm transition-colors font-mono",
                        active
                          ? "border-primary-300 bg-primary-50 text-primary-800"
                          : "border-ink-line text-ink-muted hover:border-primary-300 hover:bg-primary-50 hover:text-primary-700",
                      )}
                    >
                      {c}
                    </button>
                  );
                })}
              </div>
            </FormField>
          </FormSection>

          <FormSection title="投保人">
            <div className="grid grid-cols-2 gap-3">
              <FormField label="投保年龄" htmlFor="age" required>
                <Input
                  id="age"
                  type="number"
                  min={0}
                  max={100}
                  inputMode="numeric"
                  {...form.register("policy_params.age_at_issue", {
                    setValueAs: (v) => (v === "" || v == null ? undefined : Number(v)),
                  })}
                />
              </FormField>
              <FormField label="性别" required>
                <div className="flex gap-1.5">
                  {GENDER_OPTS.map((g) => {
                    const active = gender === g.value;
                    return (
                      <button
                        key={g.value}
                        type="button"
                        onClick={() =>
                          form.setValue("policy_params.gender", g.value, {
                            shouldValidate: true,
                          })
                        }
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
              </FormField>
            </div>
          </FormSection>

          <FormSection title="保费 / 保额">
            <FormField label="缴费方式" required>
              <div className="flex gap-1.5">
                {PREMIUM_MODES.map((m) => {
                  const active = mode === m.value;
                  return (
                    <button
                      key={m.value}
                      type="button"
                      onClick={() =>
                        form.setValue("policy_params.premium_mode", m.value, {
                          shouldValidate: true,
                        })
                      }
                      className={cn(
                        "rounded-sm border px-3 py-1 text-sm transition-colors",
                        active
                          ? "border-primary-300 bg-primary-50 text-primary-800"
                          : "border-ink-line text-ink-muted hover:border-primary-300 hover:bg-primary-50 hover:text-primary-700",
                      )}
                    >
                      {m.label}
                    </button>
                  );
                })}
              </div>
            </FormField>
            <div className="grid grid-cols-2 gap-3">
              <FormField label="单期保费" htmlFor="premium" required>
                <Input
                  id="premium"
                  type="number"
                  min={0}
                  step="0.01"
                  inputMode="decimal"
                  {...form.register("policy_params.premium_amount", {
                    setValueAs: (v) => (v === "" || v == null ? undefined : Number(v)),
                  })}
                />
              </FormField>
              <FormField label="缴费年期" htmlFor="term" required>
                <Input
                  id="term"
                  type="number"
                  min={1}
                  max={80}
                  inputMode="numeric"
                  {...form.register("policy_params.term_years", {
                    setValueAs: (v) => (v === "" || v == null ? undefined : Number(v)),
                  })}
                />
              </FormField>
            </div>
            <FormField label="保额 sum_assured" htmlFor="sa" required>
              <Input
                id="sa"
                type="number"
                min={0}
                step="0.01"
                inputMode="decimal"
                {...form.register("policy_params.sum_assured", {
                  setValueAs: (v) => (v === "" || v == null ? undefined : Number(v)),
                })}
              />
            </FormField>
          </FormSection>

          <FormSection title="目标年龄 / 年份 (可选)">
            <div className="grid grid-cols-2 gap-3">
              <FormField label="目标年龄" htmlFor="ta">
                <Input
                  id="ta"
                  type="number"
                  min={0}
                  max={120}
                  placeholder="不限"
                  {...form.register("policy_params.target_age", {
                    setValueAs: (v) => (v === "" || v == null ? undefined : Number(v)),
                  })}
                />
              </FormField>
              <FormField label="目标年份" htmlFor="ty">
                <Input
                  id="ty"
                  type="number"
                  min={0}
                  max={100}
                  placeholder="不限"
                  {...form.register("policy_params.target_year", {
                    setValueAs: (v) => (v === "" || v == null ? undefined : Number(v)),
                  })}
                />
              </FormField>
            </div>
          </FormSection>

          <FormSection title="计算目标">
            <FormField
              label="自由维度 (1-6 项)"
              required
              error={form.formState.errors.calc_targets?.message}
              hint="按回车或点 + 加入；可点常用 chips 直接塞入"
            >
              <ChipsField
                value={targets}
                onChange={(v) =>
                  form.setValue("calc_targets", v, { shouldValidate: true })
                }
                placeholder="例：第 20 年退保价值"
                presets={TARGET_PRESETS}
                maxItems={6}
                maxChars={300}
                invalid={!!form.formState.errors.calc_targets}
              />
            </FormField>
          </FormSection>
        </>
      )}
    />
  );
}

// ----------------------------------------------------- custom answer

interface CodeRunSlot {
  loop: number;
  args?: string;
  preview?: string;
  error?: string;
  done: boolean;
}

function collectCodeRunSlots(events: SSEEvent[]): CodeRunSlot[] {
  // 同一 loop 内可以多次 code_run；按 FIFO 配对：tool_call 推一个 pending
  // slot，tool_result 找该 loop 第一个未完成 slot 写回。仅按 loop key 一份
  // 会丢失重复调用。
  const out: CodeRunSlot[] = [];
  const pendingByLoop = new Map<number, CodeRunSlot[]>();
  for (const ev of events) {
    if (ev.event === "tool_call" && ev.data.name === "code_run") {
      const slot: CodeRunSlot = {
        loop: ev.data.loop,
        args: JSON.stringify(ev.data.args, null, 2),
        done: false,
      };
      out.push(slot);
      const q = pendingByLoop.get(ev.data.loop) ?? [];
      q.push(slot);
      pendingByLoop.set(ev.data.loop, q);
    } else if (ev.event === "tool_result" && ev.data.name === "code_run") {
      const q = pendingByLoop.get(ev.data.loop);
      if (q && q.length > 0) {
        const slot = q.shift()!;
        slot.preview = ev.data.preview;
        slot.error = ev.data.error;
        slot.done = true;
      }
      // 没有匹配的 pending（孤儿 result）就丢弃 —— 避免异步乱序污染列表
    }
  }
  return out;
}

function PolicyCalcAnswer({ turn }: { turn: WorkbenchTurn }) {
  const open_ = useCitationStore((s) => s.open_);
  const slots = useMemo(() => collectCodeRunSlots(turn.progressEvents), [
    turn.progressEvents,
  ]);

  return (
    <div className="space-y-4">
      <MarkdownWithSup
        content={turn.answer}
        citations={turn.citations}
        parseSup
      />

      {slots.length > 0 && (
        <CodeRunPanel slots={slots} />
      )}

      {turn.citations && turn.citations.length > 0 && (
        <CitationChipRow
          items={turn.citations}
          onOpen={(target) => open_(turn.citations as CitationItem[], target)}
        />
      )}
    </div>
  );
}

function CodeRunPanel({ slots }: { slots: CodeRunSlot[] }) {
  const [open, setOpen] = useState(false);
  const successCount = slots.filter((s) => s.done && !s.error).length;
  const errorCount = slots.filter((s) => s.done && s.error).length;
  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
      className="rounded-md border border-ink-line/70 bg-surface-raised/60"
    >
      <summary className="flex items-center gap-2 px-3 py-2 cursor-pointer text-sm select-none rounded-md hover:bg-surface-sunk/60 list-none">
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 text-ink-subtle" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-ink-subtle" />
        )}
        <Code2 className="h-3.5 w-3.5 text-primary-700" />
        <span className="font-medium text-ink">code_run</span>
        <span className="text-ink-muted">
          · 共 {slots.length} 次
          {successCount > 0 && ` · ${successCount} 成功`}
          {errorCount > 0 && (
            <span className="text-danger"> · {errorCount} 失败</span>
          )}
        </span>
      </summary>
      <div className="border-t border-ink-line/50 px-3 py-2 space-y-3">
        {slots.map((s, i) => (
          // 同 loop 内可能多次 code_run；用 idx 把 key 唯一化（仅用 s.loop
          // 会撞 React duplicate key warning + 复用错 DOM）。
          <div key={`${s.loop}:${i}`} className="space-y-1">
            <div className="text-[11px] font-mono text-ink-subtle">
              loop {s.loop}
              {s.error ? <span className="text-danger"> · 失败</span> : s.done ? " · 完成" : " · 进行中"}
            </div>
            {s.args && (
              <pre className="bg-surface-sunk rounded p-2 text-[12px] leading-5 overflow-x-auto scrollbar-thin font-mono whitespace-pre">
                {s.args}
              </pre>
            )}
            {s.preview && (
              <pre className="bg-surface-sunk rounded p-2 text-[12px] leading-5 overflow-x-auto scrollbar-thin font-mono whitespace-pre">
                {s.preview}
              </pre>
            )}
            {s.error && (
              <pre className="bg-danger-soft border border-danger/20 text-danger rounded p-2 text-[12px] leading-5 overflow-x-auto scrollbar-thin font-mono whitespace-pre">
                {s.error}
              </pre>
            )}
          </div>
        ))}
      </div>
    </details>
  );
}
