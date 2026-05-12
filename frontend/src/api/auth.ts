import { api } from "@/api/client";
import type { LoginRequest, RegisterRequest, TokenOut, User } from "@/api/types";

export const authApi = {
  /**
   * 后端 ``POST /auth/login`` 走 OAuth2PasswordRequestForm —
   * 必须 `application/x-www-form-urlencoded`，字段 `username` /
   * `password`（grant_type 等可选字段省略）。
   */
  async login(body: LoginRequest): Promise<TokenOut> {
    const form = new URLSearchParams();
    form.set("username", body.username);
    form.set("password", body.password);
    const { data } = await api.post<TokenOut>("/auth/login", form, {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });
    return data;
  },

  /**
   * 后端 ``POST /auth/register`` 接 JSON。响应同 login —
   * 一个可直接用的 access_token，前端可以一步登录、不用再调 /login。
   */
  async register(body: RegisterRequest): Promise<TokenOut> {
    const { data } = await api.post<TokenOut>("/auth/register", body);
    return data;
  },

  async me(opts?: { signal?: AbortSignal; timeout?: number }): Promise<User> {
    const { data } = await api.get<User>("/auth/me", {
      signal: opts?.signal,
      // 默认 10s；后端 /auth/me 只做一次 token decode + 一行 SELECT，
      // 慢于此值通常意味着进程整体被卡住（例如 ingest 占满 GIL），
      // 此时 hang 让 RequireAuth 永远 verifying 反而是最差体验 ——
      // 让请求超时 + clear() 让用户被踢回登录页是更明确的失败信号。
      timeout: opts?.timeout ?? 10_000,
    });
    return data;
  },

  /**
   * 用显式 token 调 /auth/me — 用于登录刚拿到 access_token 但还没
   * setSession 的瞬间，避免给 store 写一个未验证的 user 占位。
   */
  async meWithToken(token: string): Promise<User> {
    const { data } = await api.get<User>("/auth/me", {
      headers: { Authorization: `Bearer ${token}` },
    });
    return data;
  },
};
