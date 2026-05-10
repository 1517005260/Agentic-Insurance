import { useEffect } from "react";
import { Navigate, useLocation } from "react-router-dom";

import { authApi } from "@/api/auth";
import { useAuthStore } from "@/stores/auth";

/**
 * 必须登录才能进入。
 *
 * 三态：
 *   - 未 hydrate / 已 hydrate 但 token 未 verify → render null（不放行 children，避免子组件先发请求）
 *   - 无 token                                    → 跳 /login，state.from 带 path+search+hash
 *   - 有 token + verified                          → 渲染 children
 *
 * 冷启动时 localStorage 可能存着已过期 token；不主动 verify 会让
 * 用户看似登录然后所有接口 401，体验差。在 `useEffect` 里同步触发
 * 一次 `/auth/me`，但要 token-bind：发请求前抓当前 token，回调时
 * 比对，避免 token A 的 verify 回写到 token B 会话。
 */
export function RequireAuth({ children }: { children: React.ReactNode }) {
  const token = useAuthStore((s) => s.token);
  const hydrated = useAuthStore((s) => s.hydrated);
  const verified = useAuthStore((s) => s.verified);
  const markVerified = useAuthStore((s) => s.markVerified);
  const clear = useAuthStore((s) => s.clear);
  const location = useLocation();

  useEffect(() => {
    if (!hydrated || !token || verified) return;
    const startedWith = token;
    const controller = new AbortController();

    authApi
      .me({ signal: controller.signal })
      .then((u) => {
        if (controller.signal.aborted) return;
        // store 里的 token 已经被换掉（其它 tab logout / 重登），
        // 这次 verify 结果与当前会话无关，丢弃。
        if (useAuthStore.getState().token !== startedWith) return;
        markVerified(u);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        if (useAuthStore.getState().token !== startedWith) return;
        // 任意失败（401 / 网络 / 10s 超时）都走 clear()：让用户回
        // 登录页比卡在 verifying 永远转圈要明确得多。
        // eslint-disable-next-line no-console
        console.warn("[auth] /auth/me failed; clearing session", err);
        clear();
      });

    // unmount / token 变更 → abort in-flight /auth/me。早先只用
    // ``cancelled`` flag 把回调吞掉，HTTP 请求仍在飞，慢的后端会
    // 持续占连接池。abort 真正释放。
    return () => {
      controller.abort();
    };
  }, [hydrated, token, verified, markVerified, clear]);

  // 还没 rehydrate — 渲染最小 loading（早先 return null 让用户看到
  // "只剩底色"的白屏，无法区分应用挂了 vs 鉴权握手中）。
  if (!hydrated) return <AuthLoading hint="正在恢复登录状态…" />;
  if (!token) {
    const from = location.pathname + location.search + location.hash;
    return <Navigate to="/login" replace state={{ from }} />;
  }
  // 有 token 但还没 verify（/auth/me 在飞）— 不放行 children 防抢跑，
  // 但要给视觉反馈：连接池满或后端慢时这里能挂数百毫秒到数秒。
  if (!verified) return <AuthLoading hint="正在校验登录…" />;
  return <>{children}</>;
}

function AuthLoading({ hint }: { hint: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: 12,
        color: "#475569",
        fontFamily: "ui-sans-serif, system-ui, sans-serif",
        fontSize: 13,
      }}
    >
      <div
        aria-hidden
        style={{
          width: 24,
          height: 24,
          borderRadius: "50%",
          border: "2px solid #cbd5e1",
          borderTopColor: "#0f172a",
          animation: "spin 0.8s linear infinite",
        }}
      />
      <div>{hint}</div>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

/**
 * 必须是 admin。analyst 进入会被踢回 /chat。
 *
 * 进到这里说明 RequireAuth 已经放行，user 一定已 verify。
 */
export function RequireAdmin({ children }: { children: React.ReactNode }) {
  const role = useAuthStore((s) => s.user?.role);
  if (role !== "admin") {
    return <Navigate to="/chat" replace />;
  }
  return <>{children}</>;
}
