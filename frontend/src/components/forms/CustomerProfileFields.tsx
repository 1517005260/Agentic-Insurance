import type { UseFormReturn } from "react-hook-form";

import { Input } from "@/components/ui/input";
import { FormField, FormSection } from "@/components/forms/FormPrimitives";
import { ChipsField } from "@/components/forms/ChipsField";
import { cn } from "@/lib/utils";

/**
 * 共享 customer profile 字段组合 — Exclusion / Recommend 都用。
 *
 * 字段映射后端 `CustomerProfile` schema (api/schemas/insurance.py)：
 *   age / gender / occupation / occupation_risk? / health_history[]
 *   / family_history[] / budget_annual? / goal? / notes?
 *
 * 用法：
 *   <CustomerProfileFields form={form} prefix="customer" />
 *
 * `prefix` 让我们把整个 profile nested 在某 key 下（exclusion / recommend
 * body shape 都是 `{customer: {...}}`），form 的 fieldName 是 "customer.age" 等。
 */

const GENDER_OPTS = [
  { value: "M", label: "男" },
  { value: "F", label: "女" },
  { value: "X", label: "其他" },
] as const;

const RISK_OPTS = [
  { value: "low", label: "低" },
  { value: "med", label: "中" },
  { value: "high", label: "高" },
] as const;

const HEALTH_PRESETS = [
  "高血压",
  "糖尿病",
  "高血脂",
  "心脏病",
  "甲状腺",
  "癌症",
  "慢性肾炎",
  "肝炎",
] as const;

const FAMILY_PRESETS = [
  "心脏病",
  "癌症",
  "糖尿病",
  "中风",
  "高血压",
] as const;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyForm = UseFormReturn<any>;

/**
 * Caller 应传 `form as unknown as AnyForm` 以避免 RHF 的不变量
 * (UseFormReturn invariant in TForm) 让具体表单类型无法 widen 成 any。
 */
export function CustomerProfileFields({
  form,
  prefix,
}: {
  form: AnyForm;
  prefix: string;
}) {
  // 用 string 拼字段名 —— RHF 跨页复用 nested 字段最干净的办法是把
  // form 的泛型 narrow 成 any（具体类型已在调用页 zod schema 校验）。
  const k = (name: string) => `${prefix}.${name}`;
  const errAt = (name: string): string | undefined => {
    const e = (form.formState.errors as Record<string, unknown>)[prefix] as
      | Record<string, { message?: string }>
      | undefined;
    return e?.[name]?.message;
  };

  const gender = form.watch(k("gender")) as string | undefined;
  const risk = form.watch(k("occupation_risk")) as string | undefined;
  const health = (form.watch(k("health_history")) as string[] | undefined) ?? [];
  const family = (form.watch(k("family_history")) as string[] | undefined) ?? [];

  return (
    <>
      <FormSection title="基本信息">
        <div className="grid grid-cols-2 gap-3">
          <FormField label="年龄" htmlFor={`${prefix}-age`} required error={errAt("age")}>
            <Input
              id={`${prefix}-age`}
              type="number"
              min={0}
              max={120}
              inputMode="numeric"
              {...form.register(k("age"), {
                setValueAs: (v: unknown) => (v === "" || v == null ? undefined : Number(v)),
              })}
            />
          </FormField>
          <FormField label="性别" required error={errAt("gender")}>
            <div role="radiogroup" className="flex gap-1.5">
              {GENDER_OPTS.map((g) => {
                const active = gender === g.value;
                return (
                  <button
                    key={g.value}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    onClick={() =>
                      form.setValue(k("gender"), g.value, { shouldValidate: true })
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
        <FormField
          label="职业"
          htmlFor={`${prefix}-occupation`}
          required
          error={errAt("occupation")}
        >
          <Input
            id={`${prefix}-occupation`}
            placeholder="例：消防员、建筑工程师、教师"
            {...form.register(k("occupation"))}
          />
        </FormField>
        <FormField label="职业风险" hint="可空 — 由 underwriter 自行评估">
          <div role="radiogroup" className="flex gap-1.5">
            <button
              type="button"
              role="radio"
              aria-checked={risk == null || risk === ""}
              onClick={() =>
                form.setValue(k("occupation_risk"), undefined, { shouldValidate: true })
              }
              className={cn(
                "rounded-sm border px-3 py-1 text-sm transition-colors",
                (risk == null || risk === "")
                  ? "border-primary-300 bg-primary-50 text-primary-800"
                  : "border-ink-line text-ink-muted hover:border-primary-300 hover:bg-primary-50 hover:text-primary-700",
              )}
            >
              不指定
            </button>
            {RISK_OPTS.map((r) => {
              const active = risk === r.value;
              return (
                <button
                  key={r.value}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  onClick={() =>
                    form.setValue(k("occupation_risk"), r.value, { shouldValidate: true })
                  }
                  className={cn(
                    "rounded-sm border px-3 py-1 text-sm transition-colors",
                    active
                      ? "border-primary-300 bg-primary-50 text-primary-800"
                      : "border-ink-line text-ink-muted hover:border-primary-300 hover:bg-primary-50 hover:text-primary-700",
                  )}
                >
                  {r.label}
                </button>
              );
            })}
          </div>
        </FormField>
      </FormSection>

      <FormSection title="健康史">
        <FormField label="既往病史 (≤ 20)" hint="ICD-10 名词或常见说法均可">
          <ChipsField
            value={health}
            onChange={(v) =>
              form.setValue(k("health_history"), v, { shouldValidate: true })
            }
            placeholder="例：高血压"
            presets={HEALTH_PRESETS}
            maxItems={20}
          />
        </FormField>
        <FormField label="家族史 (≤ 20)" hint="一/二级亲属相关疾病">
          <ChipsField
            value={family}
            onChange={(v) =>
              form.setValue(k("family_history"), v, { shouldValidate: true })
            }
            placeholder="例：心脏病"
            presets={FAMILY_PRESETS}
            maxItems={20}
          />
        </FormField>
      </FormSection>

      <FormSection title="偏好 / 备注">
        <div className="grid grid-cols-2 gap-3">
          <FormField label="年预算 (HKD/RMB)" htmlFor={`${prefix}-budget`}>
            <Input
              id={`${prefix}-budget`}
              type="number"
              min={0}
              inputMode="numeric"
              placeholder="可选"
              {...form.register(k("budget_annual"), {
                setValueAs: (v: unknown) => (v === "" || v == null ? undefined : Number(v)),
              })}
            />
          </FormField>
          <FormField label="主诉求" htmlFor={`${prefix}-goal`}>
            <Input
              id={`${prefix}-goal`}
              placeholder="例：保障家庭"
              {...form.register(k("goal"))}
            />
          </FormField>
        </div>
        <FormField label="补充说明 (≤ 500)" htmlFor={`${prefix}-notes`}>
          <textarea
            id={`${prefix}-notes`}
            rows={3}
            maxLength={500}
            {...form.register(k("notes"))}
            className={cn(
              "w-full rounded border border-ink-line bg-surface-raised px-3 py-2 text-sm text-ink",
              "placeholder:text-ink-subtle",
              "focus-visible:border-primary-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30",
            )}
            placeholder="任何对核保 / 推荐有用的额外信息…"
          />
        </FormField>
      </FormSection>
    </>
  );
}

// Schema + normaliser 已搬到 ./customerProfileSchema.ts —— Vite Fast
// Refresh 不允许同一文件既导出 React 组件又导出非组件值。Caller
// 需要 schema/normalizeProfile 时直接 import from
// "@/components/forms/customerProfileSchema"。本文件只导出组件。
