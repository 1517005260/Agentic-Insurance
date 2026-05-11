import { QueryClient } from "@tanstack/react-query";

/**
 * Singleton react-query client.
 *
 * 单独成模块的关键原因：auth 状态变化（logout / 401 / 切换账号）
 * 必须 `queryClient.clear()` 否则上一个用户已经 cached 的列表 /
 * 详情会被下一个登录者看到。本文件被 App.tsx 与 stores/auth.ts
 * 同时 import，所以两边操作的是同一个实例。
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // 后端中转站偶发慢；不要乐观地频繁 refetch。
      staleTime: 30_000,
      // GC: 默认 5min 在频繁切 chat session / 浏览文件 / 切图谱视角
      // 时会让多份未引用的 cache 同时驻留几分钟，长会话内存压力可
      // 见。1min 已足够覆盖"快速回切上一个 query"的常见 UX，更长
      // 的稳定性场景（如 schema 列表）单独在调用点声明 gcTime。
      gcTime: 60_000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});
