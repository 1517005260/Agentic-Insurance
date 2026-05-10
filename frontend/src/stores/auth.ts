import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

import { queryClient } from "@/lib/queryClient";
import type { User } from "@/api/types";

interface AuthState {
  token: string | null;
  user: User | null;

  /** localStorage rehydrate 完成（同步本地态）。 */
  hydrated: boolean;
  /**
   * token 已经被后端验证过（/auth/me 成功）。冷启动持久化的 token
   * 可能已过期，守卫必须等 `verified` 才放行 protected route 否则
   * 会出现"看起来已登录但所有接口 401"的鬼影状态。
   */
  verified: boolean;

  /** 登录成功后调用 — token + 已验证的 user 一并写入。 */
  setSession: (token: string, user: User) => void;
  /** 401 / 登出 — 清空本地 + react-query cache，下次 selector 触发跳 /login。 */
  clear: () => void;
  /** 应用启动时同步 markHydrated；不代表 token 有效。 */
  markHydrated: () => void;
  /** /auth/me 成功后调用，标记当前 token 已验证。 */
  markVerified: (user: User) => void;

  isAuthenticated: () => boolean;
  isAdmin: () => boolean;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      hydrated: false,
      verified: false,

      setSession: (token, user) => {
        // 切换不同 token = 切换账号；上一个用户的 react-query 缓
        // 存必须清掉，否则文件列表 / 配置 / 审计等 cached query 会
        // 串号。同 token 的二次 setSession（rare：同一账号刷新）
        // 不需要清。
        const prev = get().token;
        if (prev && prev !== token) {
          queryClient.clear();
        }
        set({ token, user, verified: true });
      },
      clear: () => {
        // 切换 / 退出账号必须清掉 react-query 缓存，否则上个用户的
        // chat sessions / files / config 会被新用户看到。
        queryClient.clear();
        set({ token: null, user: null, verified: false });
      },
      markHydrated: () => set({ hydrated: true }),
      markVerified: (user) => set({ user, verified: true }),

      isAuthenticated: () => !!get().token,
      isAdmin: () => get().user?.role === "admin",
    }),
    {
      name: "agentic.auth",
      storage: createJSONStorage(() => localStorage),
      // 只持久化 token + user；hydrated / verified 都是运行时态，
      // 每次冷启动都要重新跑一遍 verification。
      partialize: (s) => ({ token: s.token, user: s.user }),
      onRehydrateStorage: () => (state) => {
        state?.markHydrated();
      },
    },
  ),
);
