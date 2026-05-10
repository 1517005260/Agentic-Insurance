/**
 * Shared Zod schema + body normaliser for the customer profile.
 *
 * 单独成文件以满足 Vite Fast Refresh 规则（`react-refresh/only-export-components`）：
 * 同一文件不能既导出组件又导出非组件值，否则 HMR 会卡。
 */
import { z } from "zod";

/**
 * Zod 校验 — Exclusion / Recommend 共享。
 *
 * 不用 `.default([])` —— zod 4 在 `default` 字段上 input/output 类型分裂
 * (input 可空 / output 必填) 会让 RHF 的 Resolver 类型对不齐。固定要求
 * 表单层把 health/family 初始化为 [] 即可。
 */
export const CustomerProfileSchema = z.object({
  age: z.number().int().min(0).max(120),
  gender: z.enum(["M", "F", "X"]),
  occupation: z.string().min(1, "请填写职业").max(80),
  occupation_risk: z.enum(["low", "med", "high"]).optional(),
  health_history: z.array(z.string()).max(20),
  family_history: z.array(z.string()).max(20),
  budget_annual: z.number().int().min(0).max(10_000_000).optional(),
  goal: z.string().max(80).optional(),
  notes: z.string().max(500).optional(),
});
export type CustomerProfileForm = z.infer<typeof CustomerProfileSchema>;

/**
 * 把表单的 raw object 整成"清掉空字符串/undefined"的 body 片段；后端
 * Pydantic Optional 字段空字符串会校验失败，这里统一去空。
 */
export function normalizeProfile(p: CustomerProfileForm): CustomerProfileForm {
  const out: Record<string, unknown> = {
    age: p.age,
    gender: p.gender,
    occupation: p.occupation.trim(),
    health_history: p.health_history ?? [],
    family_history: p.family_history ?? [],
  };
  if (p.occupation_risk) out.occupation_risk = p.occupation_risk;
  if (p.budget_annual != null) out.budget_annual = p.budget_annual;
  if (p.goal && p.goal.trim()) out.goal = p.goal.trim();
  if (p.notes && p.notes.trim()) out.notes = p.notes.trim();
  return out as CustomerProfileForm;
}
