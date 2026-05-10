import axios, {
  AxiosError,
  type AxiosInstance,
  type InternalAxiosRequestConfig,
} from "axios";

import { useAuthStore } from "@/stores/auth";

/**
 * 单例 axios 实例。
 *
 * baseURL 顺序：
 *   import.meta.env.VITE_API_BASE > "/api"
 *
 * 本地开发：vite proxy 把 `/api/*` 转给 :8000（vite.config.ts 里
 * rewrite `^/api` → `''`）。生产部署有两条路：
 *   (a) 在反向代理（nginx / caddy / traefik）上保留 `/api` 前缀
 *       并 strip 给 FastAPI ——前端不用动；
 *   (b) 部署时设 `VITE_API_BASE=https://api.example.com`，前端直
 *       连后端独立域名（生产仍要后端 CORS 放行该域）。
 */
const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

export const api: AxiosInstance = axios.create({
  baseURL: API_BASE,
  headers: { Accept: "application/json" },
});

api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  // 只在 caller 没显式传 Authorization 时塞 store 里的 token。
  // 关键场景：登录刚拿到新 token B，但旧 token A 还在 store；如果
  // 这里无条件覆盖，会让 meWithToken(B) 实际带着 A 走，导致拿错
  // 用户身份。`headers.has` 在 axios v1 的 AxiosHeaders 上可用。
  const hasExplicit =
    typeof config.headers?.has === "function"
      ? config.headers.has("Authorization")
      : !!(config.headers as Record<string, unknown>)?.Authorization;
  if (!hasExplicit) {
    const token = useAuthStore.getState().token;
    if (token) {
      config.headers.set("Authorization", `Bearer ${token}`);
    }
  }
  return config;
});

/**
 * 防抖锁：只在真正即将 location.replace 前置位，避免在已经身处
 * /login 的场景把 flag 永久卡住，导致后续真 401 静默被吞。
 */
let isRedirectingToLogin = false;

api.interceptors.response.use(
  (resp) => resp,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      const reqUrl = error.config?.url ?? "";
      // /auth/login 自身 401 不要触发跳转 —— login form 自己显示
      // 错误，避免 "登录失败 → 跳 /login" 循环。
      if (!reqUrl.endsWith("/auth/login")) {
        // Stale-session 守卫：如果这个 401 来自一个使用旧 token 的
        // 飞行中请求（例如 token A 的 /auth/me 还没回，store 已切
        // 到 token B），盲目 clear() 会把当前有效 session B 清掉。
        // 比较请求里的 Bearer 跟当前 store 的 token，不一致直接吞。
        const sent = (
          error.config?.headers as Record<string, unknown> | undefined
        )?.Authorization;
        const sentToken =
          typeof sent === "string" ? sent.replace(/^Bearer\s+/i, "") : null;
        const currentToken = useAuthStore.getState().token;
        const isStale =
          !!sentToken && !!currentToken && sentToken !== currentToken;

        if (!isStale) {
          // 任何 protected 401（且属于当前会话）都先把 store 清掉。
          useAuthStore.getState().clear();

          // 真正可能 replace 时才置 flag；当前已经在 /login 时不
          // 设，防止 flag 永久卡死。
          if (
            !isRedirectingToLogin &&
            window.location.pathname !== "/login"
          ) {
            isRedirectingToLogin = true;
            // location.replace 而不是 assign：浏览器 back 不会回到
            // 已失效的 protected page。
            window.location.replace("/login");
          }
        }
      }
    }
    return Promise.reject(error);
  },
);

/** Pydantic validation error → human-readable single string. */
export function explainAxiosError(err: unknown): string {
  if (!(err instanceof AxiosError)) return String(err);
  const data = err.response?.data as { detail?: unknown } | undefined;
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const loc = Array.isArray(d?.loc) ? d.loc.join(".") : "";
        return loc ? `${loc}: ${d.msg}` : d.msg;
      })
      .join("; ");
  }
  return err.message;
}
