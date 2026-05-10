import {
  cloneElement,
  isValidElement,
  type ReactElement,
  type ReactNode,
} from "react";

import { cn } from "@/lib/utils";
import { FieldHint, Label } from "@/components/ui/input";

/**
 * 表单原子件 — 标 label + 错误提示 + a11y 关联。
 *
 * 关键约束：error / hint 文案的 id 必须能让屏幕阅读器关联回 input。
 * RHF 的 register 不会自动注入 aria-describedby，所以我们在 FormField
 * 层做 cloneElement，把 `aria-describedby={errorId|hintId}` 和
 * `aria-invalid={!!error}` 注入到子元素上。
 *
 * 子元素若已自管 aria-describedby（如 ChipsField / FileMultiSelect 自接
 * describedBy prop），就 merge；否则直接注入。
 *
 * 注意：cloneElement 只对单一 React 元素生效；如果 children 是 fragment
 * 或多个元素，自动注入会跳过，调用页要自己显式传 describedBy。
 */
export function FormField({
  label,
  required,
  hint,
  error,
  htmlFor,
  children,
  className,
}: {
  label: string;
  required?: boolean;
  hint?: ReactNode;
  error?: string;
  htmlFor?: string;
  children: ReactNode;
  className?: string;
}) {
  const errorId = htmlFor ? `${htmlFor}-error` : undefined;
  const hintId = htmlFor && hint ? `${htmlFor}-hint` : undefined;
  const describedBy = error ? errorId : hintId;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let injected: ReactNode = children;
  if (isValidElement(children) && describedBy) {
    const child = children as ReactElement<{
      "aria-describedby"?: string;
      "aria-invalid"?: boolean;
    }>;
    const existing = child.props["aria-describedby"];
    const merged = existing
      ? `${existing} ${describedBy}`.trim()
      : describedBy;
    injected = cloneElement(child, {
      "aria-describedby": merged,
      "aria-invalid": error ? true : (child.props["aria-invalid"] ?? undefined),
    });
  }

  return (
    <div className={cn("space-y-1", className)}>
      <Label htmlFor={htmlFor}>
        {label}
        {required && <span className="ml-0.5 text-danger">*</span>}
      </Label>
      {injected}
      {error ? (
        <FieldHint tone="danger">
          <span id={errorId}>{error}</span>
        </FieldHint>
      ) : hint ? (
        <FieldHint tone="muted">
          <span id={hintId}>{hint}</span>
        </FieldHint>
      ) : null}
    </div>
  );
}

/**
 * 一组相关字段的视觉分组；在工作台左侧表单里区分"基本信息 / 健康史 / 偏好"等。
 */
export function FormSection({
  title,
  description,
  children,
  className,
}: {
  title?: string;
  description?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("space-y-3", className)}>
      {(title || description) && (
        <div className="space-y-0.5">
          {title && (
            <h3 className="text-[13px] font-medium uppercase tracking-[0.16em] text-ink-subtle">
              {title}
            </h3>
          )}
          {description && (
            <p className="text-xs text-ink-muted">{description}</p>
          )}
        </div>
      )}
      <div className="space-y-3">{children}</div>
    </section>
  );
}
