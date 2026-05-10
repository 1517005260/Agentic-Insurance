import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { authApi } from "@/api/auth";
import { explainAxiosError } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader, CardSubtitle, CardTitle } from "@/components/ui/card";
import { FieldHint, Input, Label } from "@/components/ui/input";
import { useAuthStore } from "@/stores/auth";

export default function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const setSession = useAuthStore((s) => s.setSession);
  const isVerified = useAuthStore((s) => s.verified && !!s.token);

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // 已登录访问 /login → 直接跳主页（避免重复登录覆盖 token）。
  // 注意：必须等 verified 后才跳，否则冷启动还在 verify 时 token 已
  // 经 hydrate 出来会一帧误跳到 /chat。
  useEffect(() => {
    if (isVerified) navigate("/chat", { replace: true });
  }, [isVerified, navigate]);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setError(null);
    setBusy(true);
    try {
      // 1) 拿 token (flat TokenOut)
      const tok = await authApi.login({ username, password });
      // 2) 用显式 Authorization header 调 /me（不污染全局 store），
      //    成功后一次性 setSession。这样 RequireAuth 的 verified
      //    标记和 user 总是一致 — 不会出现 "token 已设但 user.id=
      //    -1" 的中间帧。
      const me = await authApi.meWithToken(tok.access_token);
      setSession(tok.access_token, me);
      const from = (location.state as { from?: string } | null)?.from;
      navigate(from && from !== "/login" ? from : "/chat", { replace: true });
    } catch (err) {
      setError(explainAxiosError(err) || "登录失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-surface flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <div className="font-serif text-3xl text-primary-700 tracking-tight">
            Agentic Insurance
          </div>
          <div className="mt-2 text-sm text-ink-muted">
            香港 + 内地 保险业咨询智能终端
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>登录</CardTitle>
            <CardSubtitle>
              请使用管理员或分析师账号登录。默认管理员 admin / admin123
            </CardSubtitle>
          </CardHeader>

          <CardBody>
            <form className="space-y-4" onSubmit={onSubmit}>
              <div className="space-y-1.5">
                <Label htmlFor="username">用户名</Label>
                <Input
                  id="username"
                  autoComplete="username"
                  autoFocus
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="admin"
                  required
                />
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="password">密码</Label>
                <Input
                  id="password"
                  type="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>

              {error && <FieldHint tone="danger">{error}</FieldHint>}

              <Button
                type="submit"
                size="lg"
                loading={busy}
                className="w-full"
              >
                登录
              </Button>
            </form>
          </CardBody>
        </Card>

        <p className="mt-6 text-center text-xs text-ink-subtle">
          GLK · 本科毕业设计 · 2026
        </p>
      </div>
    </div>
  );
}
