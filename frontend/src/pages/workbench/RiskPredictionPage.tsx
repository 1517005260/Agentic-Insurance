import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  ShieldAlert,
  ShieldCheck,
  FileSearch,
  Network,
} from "lucide-react";

import { Input } from "@/components/ui/input";
import { FormField, FormSection } from "@/components/forms/FormPrimitives";
import { FileMultiSelect } from "@/components/forms/FileMultiSelect";
import { CustomerProfileFields } from "@/components/forms/CustomerProfileFields";
import {
  CustomerProfileSchema,
  normalizeProfile,
} from "@/components/forms/customerProfileSchema";
import { WorkbenchScaffold } from "@/components/workbench/WorkbenchScaffold";
import { useWorkbenchStream } from "@/components/workbench/useWorkbenchStream";
import { ProofBoard } from "@/components/workbench/ProofBoard";
import { RiskExploreCanvas } from "@/components/workbench/RiskExploreCanvas";
import { RiskSankeyCanvas } from "@/components/workbench/RiskSankeyCanvas";
import { MarkdownWithSup } from "@/components/chat/MarkdownWithSup";
import { cn } from "@/lib/utils";
import type { RiskSubgraph } from "@/lib/sse-types";

/**
 * 理赔风险预测工作台 —— 与毕设标题对齐的核心模块。四个互不依赖的
 * 检查方法收编在同一个 page 下，用顶部 tab 切换。每个 tab 装一个独立的
 * `useWorkbenchStream` 实例，切 tab 时旧 panel 直接卸载、流通过 `reset`
 * cleanup 抢占式中止，不会出现 cross-talk。
 *
 * Tab 列表（论文章节锚点）：
 *  - 投保前预测   → GraphAgent + PPR-anchored Sankey （proactive 主线）
 *  - 除外触发审查 → ProofAgent forall(exclusions)
 *  - 已发事件理赔 → BaseAgent 三栏 schema
 *  - 图谱风险发现 → 单次 PPR + LLM (URL = /insurance/fraud-ppr/stream)
 */

type Tab = "predict" | "exclusion" | "claim" | "hidden_risk";

const TABS: {
  id: Tab;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  hint: string;
}[] = [
  { id: "predict", label: "投保前风险预测", icon: ShieldAlert, hint: "GraphAgent + Sankey — 客户档案 × 候选保单" },
  { id: "exclusion", label: "除外触发审查", icon: ShieldCheck, hint: "ProofAgent forall — 单产品 × 客户档案" },
  { id: "claim", label: "已发事件理赔", icon: FileSearch, hint: "BaseAgent 三栏 — 多产品 × 事件描述" },
  { id: "hidden_risk", label: "图谱风险发现", icon: Network, hint: "PPR + LLM — 找语义邻近的隐藏条款" },
];

export default function RiskPredictionPage() {
  const [tab, setTab] = useState<Tab>("predict");

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
        {tab === "predict" && <RiskPredictTab />}
        {tab === "exclusion" && <ExclusionTab />}
        {tab === "claim" && <ClaimCheckTab />}
        {tab === "hidden_risk" && <HiddenRiskTab />}
      </div>
    </div>
  );
}

// ============================================================ Risk predict tab

const RiskPredictSchema = z.object({
  file_id: z.string().min(1, "请选择候选保单"),
  customer: CustomerProfileSchema,
  scenario: z.string().max(500).optional(),
});
type RiskPredictValues = z.infer<typeof RiskPredictSchema>;

function RiskPredictTab() {
  const { turn, busy, runStream, abort, reset } = useWorkbenchStream("RiskPredict");
  const form = useForm<RiskPredictValues>({
    resolver: zodResolver(RiskPredictSchema),
    mode: "onChange",
    defaultValues: {
      file_id: "",
      customer: {
        age: 35,
        gender: "M",
        occupation: "",
        health_history: [],
        family_history: [],
      },
      scenario: "",
    },
  });

  const fileId = form.watch("file_id");
  const fileIds = fileId ? [fileId] : [];
  const occupation = form.watch("customer.occupation");
  const age = form.watch("customer.age");

  const canSubmit =
    !busy &&
    fileId.length > 0 &&
    occupation &&
    occupation.trim().length > 0 &&
    age >= 0;

  const onSubmit = form.handleSubmit((v) => {
    const body: Record<string, unknown> = {
      file_id: v.file_id,
      customer: normalizeProfile(v.customer),
    };
    if (v.scenario && v.scenario.trim()) body.scenario = v.scenario.trim();
    runStream("/insurance/risk-predict/stream", body);
  });

  const onReset = () => {
    reset();
    form.reset();
  };

  return (
    <WorkbenchScaffold
      title="投保前风险预测"
      description="客户档案 + 候选保单 → GraphAgent 顺 PPR → neighbors → read 流水线推理；下方 Sankey 展示客户字段 → 风险因子 → 触发条款的传导链。"
      modeLabel="RiskPredict"
      turn={turn}
      busy={busy}
      canSubmit={!!canSubmit}
      onSubmit={() => void onSubmit()}
      onAbort={abort}
      onReset={onReset}
      parseSup
      renderExtras={(t) => (
        <div className="space-y-3">
          <RiskExploreCanvas turn={t} />
          <RiskSankeyCanvas
            data={(t.finalSummary?.risk_subgraph as RiskSubgraph | undefined) ?? undefined}
            citations={t.citations}
          />
        </div>
      )}
      renderAnswer={(t) => (
        <div className="space-y-2">
          {t.finalSummary?.risk_subgraph ? null : null}
          <MarkdownWithSup
            content={t.answer}
            citations={t.citations}
            parseSup
          />
        </div>
      )}
      renderForm={() => (
        <>
          <FormSection title="候选保单">
            <FormField
              label="保单文档"
              required
              error={form.formState.errors.file_id?.message}
              hint="GraphAgent 在该产品的子图上做 PPR 探索"
            >
              <FileMultiSelect
                mode="single"
                value={fileIds}
                onChange={(v) =>
                  form.setValue("file_id", v[0] ?? "", { shouldValidate: true })
                }
                placeholder="选择 1 份候选保单"
                invalid={!!form.formState.errors.file_id}
              />
            </FormField>
          </FormSection>

          <CustomerProfileFields
            form={form as unknown as Parameters<typeof CustomerProfileFields>[0]["form"]}
            prefix="customer"
          />

          <FormSection title="假设场景（可选）">
            <FormField
              label="场景描述"
              htmlFor="risk-scenario"
              hint="给 PPR 和 GraphAgent 一条明确的角度，例如「投保后半年内出境长期旅游」"
            >
              <textarea
                id="risk-scenario"
                rows={3}
                maxLength={500}
                {...form.register("scenario")}
                className={cn(
                  "w-full rounded border bg-surface-raised px-3 py-2 text-sm text-ink",
                  "placeholder:text-ink-subtle",
                  "focus-visible:border-primary-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30",
                  "border-ink-line",
                )}
                placeholder="例：客户考虑在投保 6 个月内出境长期旅游"
              />
            </FormField>
          </FormSection>
        </>
      )}
    />
  );
}

// ============================================================ Exclusion tab

const ExclusionSchema = z.object({
  file_id: z.string().min(1, "请选择保单"),
  customer: CustomerProfileSchema,
});
type ExclusionValues = z.infer<typeof ExclusionSchema>;

function ExclusionTab() {
  const { turn, busy, runStream, abort, reset } = useWorkbenchStream("Exclusion");
  const form = useForm<ExclusionValues>({
    resolver: zodResolver(ExclusionSchema),
    mode: "onChange",
    defaultValues: {
      file_id: "",
      customer: {
        age: 35,
        gender: "M",
        occupation: "",
        health_history: [],
        family_history: [],
      },
    },
  });

  const fileId = form.watch("file_id");
  const fileIds = fileId ? [fileId] : [];
  const occupation = form.watch("customer.occupation");
  const age = form.watch("customer.age");

  const canSubmit =
    !busy && fileId.length > 0 && occupation && occupation.trim().length > 0 && age >= 0;

  const onSubmit = form.handleSubmit((v) => {
    runStream("/insurance/exclusion-audit/stream", {
      file_id: v.file_id,
      customer: normalizeProfile(v.customer),
    });
  });

  const onReset = () => {
    reset();
    form.reset();
  };

  return (
    <WorkbenchScaffold
      title="除外触发审查"
      description="单产品 × 客户档案 → ProofAgent forall(exclusions)，逐条免责条款匹配 + 触发判定。"
      modeLabel="Exclusion"
      turn={turn}
      busy={busy}
      canSubmit={!!canSubmit}
      onSubmit={() => void onSubmit()}
      onAbort={abort}
      onReset={onReset}
      parseSup
      renderExtras={(t) => <ProofBoard events={t.progressEvents} />}
      renderForm={() => (
        <>
          <FormSection title="审查目标">
            <FormField
              label="保单文档"
              required
              error={form.formState.errors.file_id?.message}
              hint="ProofAgent 仅审查这一份产品的免责条款"
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

// ============================================================ Claim check tab

const ClaimSchema = z.object({
  file_ids: z.array(z.string()).min(1, "至少选 1 份保单").max(8, "最多 8 份"),
  event: z.object({
    type: z.string().min(1, "请选择事件类型").max(80),
    date: z.string().min(1, "请填写发生日期").max(40),
    location: z.string().max(120).optional(),
    description: z.string().min(1, "请描述事件详情").max(2000),
    amount: z.number().min(0).optional(),
  }),
});
type ClaimValues = z.infer<typeof ClaimSchema>;

const EVENT_TYPES = [
  "意外身故",
  "意外伤残",
  "意外医疗",
  "重疾确诊",
  "住院治疗",
  "门诊报销",
  "财产损失",
  "其他",
] as const;

function ClaimCheckTab() {
  const { turn, busy, runStream, abort, reset } = useWorkbenchStream("Claim");
  const form = useForm<ClaimValues>({
    resolver: zodResolver(ClaimSchema),
    mode: "onChange",
    defaultValues: {
      file_ids: [],
      event: {
        type: "意外医疗",
        date: new Date().toISOString().slice(0, 10),
        description: "",
      },
    },
  });

  const fileIds = form.watch("file_ids");
  const evType = form.watch("event.type");

  const canSubmit =
    !busy &&
    fileIds.length >= 1 &&
    fileIds.length <= 8 &&
    !!form.watch("event.date") &&
    !!form.watch("event.description");

  const onSubmit = form.handleSubmit((v) => {
    const ev: Record<string, unknown> = {
      type: v.event.type,
      date: v.event.date,
      description: v.event.description,
    };
    if (v.event.location && v.event.location.trim()) ev.location = v.event.location.trim();
    if (v.event.amount != null) ev.amount = v.event.amount;
    runStream("/insurance/claim-check/stream", {
      file_ids: v.file_ids,
      event: ev,
    });
  });

  const onReset = () => {
    reset();
    form.reset();
  };

  return (
    <WorkbenchScaffold
      title="已发事件理赔"
      description="多保单 × 事件描述 → BaseAgent 三栏判定（覆盖判定 / 适用条款 / 所需材料）。"
      modeLabel="Claim"
      turn={turn}
      busy={busy}
      canSubmit={canSubmit}
      onSubmit={() => void onSubmit()}
      onAbort={abort}
      onReset={onReset}
      parseSup
      renderForm={() => (
        <>
          <FormSection title="涉及保单">
            <FormField
              label="保单文档 (1-8 份)"
              required
              error={form.formState.errors.file_ids?.message}
            >
              <FileMultiSelect
                mode="multi"
                value={fileIds}
                onChange={(v) => form.setValue("file_ids", v, { shouldValidate: true })}
                maxSelected={8}
                invalid={!!form.formState.errors.file_ids}
              />
            </FormField>
          </FormSection>

          <FormSection title="事件详情">
            <FormField label="事件类型" required>
              <div className="flex flex-wrap gap-1.5">
                {EVENT_TYPES.map((t) => {
                  const active = evType === t;
                  return (
                    <button
                      key={t}
                      type="button"
                      onClick={() => form.setValue("event.type", t, { shouldValidate: true })}
                      className={cn(
                        "rounded-sm border px-2.5 py-1 text-sm transition-colors",
                        active
                          ? "border-primary-300 bg-primary-50 text-primary-800"
                          : "border-ink-line text-ink-muted hover:border-primary-300 hover:bg-primary-50 hover:text-primary-700",
                      )}
                      aria-pressed={active}
                    >
                      {t}
                    </button>
                  );
                })}
              </div>
            </FormField>
            <div className="grid grid-cols-2 gap-3">
              <FormField
                label="发生日期"
                htmlFor="event-date"
                required
                error={form.formState.errors.event?.date?.message}
              >
                <Input id="event-date" type="date" {...form.register("event.date")} />
              </FormField>
              <FormField label="涉及金额 (可选)" htmlFor="event-amount">
                <Input
                  id="event-amount"
                  type="number"
                  min={0}
                  step="0.01"
                  inputMode="decimal"
                  placeholder="可选"
                  {...form.register("event.amount", {
                    setValueAs: (v: unknown) => (v === "" || v == null ? undefined : Number(v)),
                  })}
                />
              </FormField>
            </div>
            <FormField label="地点 (可选)" htmlFor="event-location">
              <Input
                id="event-location"
                placeholder="例：香港、深圳福田"
                {...form.register("event.location")}
              />
            </FormField>
            <FormField
              label="详细描述"
              htmlFor="event-desc"
              required
              error={form.formState.errors.event?.description?.message}
              hint="时间地点经过 / 受伤部位 / 已发生的处理"
            >
              <textarea
                id="event-desc"
                rows={5}
                maxLength={2000}
                {...form.register("event.description")}
                className={cn(
                  "w-full rounded border bg-surface-raised px-3 py-2 text-sm text-ink",
                  "placeholder:text-ink-subtle",
                  "focus-visible:border-primary-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30",
                  form.formState.errors.event?.description ? "border-danger" : "border-ink-line",
                )}
                placeholder="例：2024-12-15 在香港中环办公楼内午餐时滑倒，左踝扭伤；当日急诊照 X 光，自费 HKD 1,800…"
              />
            </FormField>
          </FormSection>
        </>
      )}
    />
  );
}

// ============================================================ Hidden risk tab

const HiddenRiskSchema = z.object({
  file_ids: z.array(z.string()).max(8, "最多 8 份"),
  query: z.string().min(1, "请输入要排查的风险点 / 客户场景").max(2000),
});
type HiddenRiskValues = z.infer<typeof HiddenRiskSchema>;

function HiddenRiskTab() {
  const { turn, busy, runStream, abort, reset } = useWorkbenchStream("HiddenRisk");
  const form = useForm<HiddenRiskValues>({
    resolver: zodResolver(HiddenRiskSchema),
    mode: "onChange",
    defaultValues: { file_ids: [], query: "" },
  });

  const fileIds = form.watch("file_ids");
  const query = form.watch("query");
  const canSubmit = !busy && query.trim().length > 0 && fileIds.length <= 8;

  const onSubmit = form.handleSubmit((v) => {
    const body: Record<string, unknown> = { query: v.query.trim() };
    if (v.file_ids.length > 0) body.file_ids = v.file_ids;
    runStream("/insurance/fraud-ppr/stream", body);
  });

  const onReset = () => {
    reset();
    form.reset();
  };

  return (
    <WorkbenchScaffold
      title="图谱风险发现"
      description="PPR 检索语义邻域 → LLM 单次分析。展示与问题相邻、用户未必知道要去查的相关条款。"
      modeLabel="HiddenRisk"
      turn={turn}
      busy={busy}
      canSubmit={canSubmit}
      onSubmit={() => void onSubmit()}
      onAbort={abort}
      onReset={onReset}
      parseSup
      renderForm={() => (
        <>
          <FormSection title="检索范围">
            <FormField
              label="保单文档 (可选 0-8 份)"
              hint="留空 = 在全库范围内找语义邻近条款"
              error={form.formState.errors.file_ids?.message}
            >
              <FileMultiSelect
                mode="multi"
                value={fileIds}
                onChange={(v) => form.setValue("file_ids", v, { shouldValidate: true })}
                maxSelected={8}
                invalid={!!form.formState.errors.file_ids}
              />
            </FormField>
          </FormSection>

          <FormSection title="排查目标">
            <FormField
              label="风险点 / 客户场景"
              htmlFor="hr-query"
              required
              error={form.formState.errors.query?.message}
              hint="用一句话描述要从图谱里找邻近条款的场景"
            >
              <textarea
                id="hr-query"
                rows={5}
                maxLength={2000}
                {...form.register("query")}
                className={cn(
                  "w-full rounded border bg-surface-raised px-3 py-2 text-sm text-ink",
                  "placeholder:text-ink-subtle",
                  "focus-visible:border-primary-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30",
                  form.formState.errors.query ? "border-danger" : "border-ink-line",
                )}
                placeholder="例：客户在职业列表里登记为高空作业，想知道意外险有哪些与该职业相关、但条款里没有显式列出的限制？"
              />
            </FormField>
          </FormSection>
        </>
      )}
    />
  );
}
