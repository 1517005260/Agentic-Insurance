import * as React from "react";
import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "link";
type Size = "sm" | "md" | "lg";

const VARIANT: Record<Variant, string> = {
  primary:
    "bg-primary-600 text-surface-raised hover:bg-primary-700 active:bg-primary-800 disabled:bg-primary-300 shadow-sm",
  secondary:
    "bg-surface-raised text-ink border border-ink-line hover:bg-surface-sunk active:bg-ink-line/40 disabled:opacity-60",
  ghost:
    "bg-transparent text-ink hover:bg-surface-sunk active:bg-ink-line/40 disabled:opacity-50",
  danger:
    "bg-danger text-surface-raised hover:bg-danger/90 active:bg-danger/80 disabled:opacity-60",
  link:
    "bg-transparent text-accent-700 underline-offset-2 hover:underline px-0",
};

const SIZE: Record<Size, string> = {
  sm: "h-8 px-3 text-sm",
  md: "h-9 px-4 text-sm",
  lg: "h-10 px-5 text-base",
};

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", loading, disabled, children, ...rest }, ref) => (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded font-medium transition-colors",
        // focus ring 用 primary-600 实色，offset 跟 surface-raised 等卡
        // 片白配合时 ring 仍清晰可见 (≥3:1)。
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-600 focus-visible:ring-offset-2 focus-visible:ring-offset-surface-raised",
        "disabled:cursor-not-allowed",
        VARIANT[variant],
        SIZE[size],
        className,
      )}
      {...rest}
    >
      {loading && (
        <span
          aria-hidden
          className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent"
        />
      )}
      {children}
    </button>
  ),
);
Button.displayName = "Button";
