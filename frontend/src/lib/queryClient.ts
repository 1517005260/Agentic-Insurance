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
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});
