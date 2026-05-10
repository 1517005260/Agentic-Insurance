import { useMemo, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { GitCompare, Sparkles } from "lucide-react";

import { FormField, FormSection } from "@/components/forms/FormPrimitives";
import { ChipsField } from "@/components/forms/ChipsField";
import { FileMultiSelect } from "@/components/forms/FileMultiSelect";
import { CustomerProfileFields } from "@/components/forms/CustomerProfileFields";
import {
  CustomerProfileSchema,
  normalizeProfile,
} from "@/components/forms/customerProfileSchema";
import { WorkbenchScaffold } from "@/components/workbench/WorkbenchScaffold";
import { useWorkbenchStream } from "@/components/workbench/useWorkbenchStream";
import { cn } from "@/lib/utils";

/**
 * 产品对比与推荐工作台 —— 把"多产品对比"和"客户保障规划"合并到同一
 * 个 page 下，用顶部 tab 切换。两个 tab 各自维护 useWorkbenchStream，
 * 切 tab 时旧 panel 直接卸载、流通过 cleanup 抢占式中止。
 *
 * 后端 endpoint 不变（/insurance/compare/stream + /insurance/recommend/stream），
 * 只是前端入口收敛到 sidebar 上的"产品对比推荐"一项。
 */

type Tab = "compare" | "needs";

const TABS: {
  id: Tab;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  hint: string;
}[] = [
  { id: "compare", label: "产品对比", icon: GitCompare, hint: "BaseAgent N×M 矩阵" },
  { id: "needs", label: "客户保障规划", icon: Sparkles, hint: "BaseAgent 推荐 / 缺口分析" },
];

export default function ProductsPage() {
  const [tab, setTab] = useState<Tab>("compare");

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-ink-line bg-surface-raised flex shrink-0">
        {TABS.map((t) => {
          const active = t.id === tab;
          const Icon = t.icon;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              title={t.hint}
              className={cn(
                "flex items-center gap-1.5 px-4 py-2.5 text-[13px] transition-colors border-b-2 -mb-px",
                active
                  ? "border-primary-600 text-primary-700 font-medium bg-primary-50/40"
                  : "border-transparent text-ink-muted hover:text-ink hover:bg-surface-sunk/40",
              )}
              aria-pressed={active}
            >
              <Icon className="h-4 w-4" />
              {t.label}
            </button>
          );
        })}
      </div>

      <div className="flex-1 min-h-0">
        {tab === "compare" && <CompareTab />}
        {tab === "needs" && <NeedsAnalysisTab />}
      </div>
    </div>
  );
}

// ============================================================ Compare tab

const CompareSchema = z.object({
  file_ids: z.array(z.string()).min(2, "至少选 2 个产品").max(8, "最多 8 个产品"),
  properties: z.array(z.string()).min(1, "至少 1 个对比维度").max(12, "最多 12 个维度"),
});
type CompareValues = z.infer<typeof CompareSchema>;

const PRESET_DIMS = [
  "等待期",
  "免责条款",
  "责任范围",
  "保费回赠",
  "现金价值",
  "退保价值",
  "宽限期",
  "复效",
  "豁免保费",
  "理赔时效",
] as const;

function CompareTab() {
  const { turn, busy, runStream, abort, reset } = useWorkbenchStream("Compare");

  const form = useForm<CompareValues>({
    resolver: zodResolver(CompareSchema),
    mode: "onChange",
    defaultValues: { file_ids: [], properties: [] },
  });

  const file_ids = form.watch("file_ids");
  const properties = form.watch("properties");
  const canSubmit = useMemo(
    () =>
      !busy &&
      file_ids.length >= 2 &&
      file_ids.length <= 8 &&
      properties.length >= 1 &&
      properties.length <= 12,
    [busy, file_ids, properties],
  );

  const onSubmit = form.handleSubmit((v) => {
    runStream("/insurance/compare/stream", {
      file_ids: v.file_ids,
      properties: v.properties,
    });
  });

  const onReset = () => {
    reset();
    form.reset();
  };

  return (
    <WorkbenchScaffold
      title="多产品对比"
      description="选 2-8 个保单文档 + 至多 12 个对比维度，BaseAgent 输出 N×M 矩阵；缺证据 cell 写 `待查`。"
      modeLabel="Compare"
      turn={turn}
      busy={busy}
      canSubmit={canSubmit}
      onSubmit={() => void onSubmit()}
      onAbort={abort}
      onReset={onReset}
      parseSup
      renderForm={() => (
        <>
          <FormSection title="对比对象">
            <FormField
              label="保单文档 (2-8 个)"
              required
              error={form.formState.errors.file_ids?.message}
            >
              <FileMultiSelect
                mode="multi"
                value={form.watch("file_ids")}
                onChange={(v) => form.setValue("file_ids", v, { shouldValidate: true })}
                maxSelected={8}
                invalid={!!form.formState.errors.file_ids}
              />
            </FormField>
          </FormSection>

          <FormSection title="对比维度">
            <FormField
              label="自由维度 (1-12 个)"
              required
              error={form.formState.errors.properties?.message}
              hint="按回车或点 + 加入"
            >
              <ChipsField
                value={form.watch("properties")}
                onChange={(v) => form.setValue("properties", v, { shouldValidate: true })}
                placeholder="例：等待期、免责、责任范围"
                presets={PRESET_DIMS}
                maxItems={12}
                invalid={!!form.formState.errors.properties}
              />
            </FormField>
          </FormSection>
        </>
      )}
    />
  );
}

// ============================================================ Needs analysis tab

const NeedsSchema = z.object({
  customer: CustomerProfileSchema,
  held_policies_file_ids: z.array(z.string()).max(20, "最多 20 份"),
});
type NeedsValues = z.infer<typeof NeedsSchema>;

function NeedsAnalysisTab() {
  const { turn, busy, runStream, abort, reset } = useWorkbenchStream("NeedsAnalysis");

  const form = useForm<NeedsValues>({
    resolver: zodResolver(NeedsSchema),
    mode: "onChange",
    defaultValues: {
      held_policies_file_ids: [],
      customer: {
        age: 35,
        gender: "M",
        occupation: "",
        health_history: [],
        family_history: [],
      },
    },
  });

  const occupation = form.watch("customer.occupation");
  const age = form.watch("customer.age");
  const heldPolicies = form.watch("held_policies_file_ids");
  const canSubmit =
    !busy && occupation && occupation.trim().length > 0 && age >= 0;

  const onSubmit = form.handleSubmit((v) => {
    const body: Record<string, unknown> = {
      customer: normalizeProfile(v.customer),
    };
    if (v.held_policies_file_ids.length > 0) {
      body.held_policies_file_ids = v.held_policies_file_ids;
    }
    runStream("/insurance/recommend/stream", body);
  });

  const onReset = () => {
    reset();
    form.reset();
  };

  const heldHint =
    heldPolicies.length === 0
      ? "未填 → BaseAgent 在全库挑 top-3 适配产品"
      : `已选 ${heldPolicies.length} 份 → 运行缺口分析 + 推荐互补补足产品`;

  return (
    <WorkbenchScaffold
      title="客户保障规划"
      description="客户档案 (+ 可选已持有保单) → BaseAgent 推荐适配产品；填了已持有保单则改走缺口分析 + 互补推荐。"
      modeLabel="NeedsAnalysis"
      turn={turn}
      busy={busy}
      canSubmit={!!canSubmit}
      onSubmit={() => void onSubmit()}
      onAbort={abort}
      onReset={onReset}
      parseSup
      renderForm={() => (
        <>
          <FormSection title="已持有保单（可选）">
            <FormField
              label="已持有保单 (0-20 份)"
              hint={heldHint}
              error={form.formState.errors.held_policies_file_ids?.message}
            >
              <FileMultiSelect
                mode="multi"
                value={heldPolicies}
                onChange={(v) =>
                  form.setValue("held_policies_file_ids", v, { shouldValidate: true })
                }
                maxSelected={20}
                invalid={!!form.formState.errors.held_policies_file_ids}
              />
            </FormField>
          </FormSection>

          <CustomerProfileFields
            form={form as unknown as Parameters<typeof CustomerProfileFields>[0]["form"]}
            prefix="customer"
          />
        </>
      )}
    />
  );
}
