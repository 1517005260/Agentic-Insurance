import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...rest }, ref) => (
  <input
    ref={ref}
    className={cn(
      "h-9 w-full rounded border border-ink-line bg-surface-raised px-3 text-sm text-ink",
      "placeholder:text-ink-subtle",
      "focus-visible:border-primary-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30",
      "disabled:cursor-not-allowed disabled:opacity-60",
      className,
    )}
    {...rest}
  />
));
Input.displayName = "Input";

export const Label = React.forwardRef<
  HTMLLabelElement,
  React.LabelHTMLAttributes<HTMLLabelElement>
>(({ className, ...rest }, ref) => (
  <label
    ref={ref}
    className={cn("text-sm font-medium text-ink", className)}
    {...rest}
  />
));
Label.displayName = "Label";

export const FieldHint = ({
  children,
  tone = "muted",
}: {
  children: React.ReactNode;
  tone?: "muted" | "danger";
}) => (
  <p
    className={cn(
      "text-xs",
      tone === "muted" ? "text-ink-subtle" : "text-danger",
    )}
  >
    {children}
  </p>
);
