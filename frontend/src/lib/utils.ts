import clsx, { type ClassValue } from "clsx";

/**
 * Tailwind className 拼接 helper.
 *
 * 我们暂未引入 tailwind-merge —— 项目内不存在大量 conditional
 * 覆盖同类 token 的需求；如果后续出现 `cn("p-2", "p-4")` 想取后者
 * 的场景，再加 `tailwind-merge` 这一层。
 */
export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs);
}
