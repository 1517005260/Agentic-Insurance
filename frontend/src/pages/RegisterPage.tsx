import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";

import { authApi } from "@/api/auth";
import { explainAxiosError } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader, CardSubtitle, CardTitle } from "@/components/ui/card";
import { FieldHint, Input, Label } from "@/components/ui/input";
import { useAuthStore } from "@/stores/auth";

/**
 * 自助注册页。后端 ``POST /auth/register`` 创建 ``analyst`` 账号并
 * 直接回 token —— 注册成功 = 已登录，不再走 /login 二次回环。
 *
 * 两次密码校验只在前端做：用户体验上的"打字错了"防错；后端不接
 * confirm 字段，避免把同样的明文重复发一次。
 */
export default function RegisterPage() {
  const navigate = useNavigate();
  const setSession = useAuthStore((s) => s.setSession);
  const isVerified = useAuthStore((s) => s.verified && !!s.token);

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // 已登录访问 /register 同 /login 处理 —— 直接跳主页。
  useEffect(() => {
    if (isVerified) navigate("/chat", { replace: true });
  }, [isVerified, navigate]);

  // 客户端的纯展示性校验。真正的"够不够安全"判定在后端
  // ``enforce_password_policy`` —— 这里只是把常见错误（两次不
  // 一致、长度不足）拦在请求前，少跑一次网络。
  const mismatch = confirm.length > 0 && password !== confirm;
  const tooShort = password.length > 0 && password.length < 8;
  const canSubmit =
    username.length >= 2 &&
    password.length >= 8 &&
    password === confirm &&
    !busy;

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setError(null);
    setBusy(true);
    try {
      const tok = await authApi.register({ username, password });
      // 拿到 token 后再调一次 /me 以保证 store 里写入的 user
      // 字段和 RequireAuth 看到的一致 —— 同 LoginPage 的设计。
      const me = await authApi.meWithToken(tok.access_token);
      setSession(tok.access_token, me);
      navigate("/chat", { replace: true });
    } catch (err) {
      setError(explainAxiosError(err) || "注册失败");
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
            注册新账号 · 默认 analyst 角色
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>注册</CardTitle>
            <CardSubtitle>
              用户名 2–64 位，仅限字母 / 数字 / <code>._-</code>；密码 ≥ 8 位且至少 1 个字母 + 1 个数字。
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
                  placeholder="2–64 位"
                  required
                  minLength={2}
                  maxLength={64}
                  pattern="[A-Za-z0-9._\-]+"
                />
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="password">密码</Label>
                <Input
                  id="password"
                  type="password"
                  autoComplete="new-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={8}
                  maxLength={128}
                />
                {tooShort && <FieldHint tone="danger">密码至少 8 位</FieldHint>}
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="confirm">确认密码</Label>
                <Input
                  id="confirm"
                  type="password"
                  autoComplete="new-password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  required
                  minLength={8}
                  maxLength={128}
                />
                {mismatch && <FieldHint tone="danger">两次输入的密码不一致</FieldHint>}
              </div>

              {error && <FieldHint tone="danger">{error}</FieldHint>}

              <Button
                type="submit"
                size="lg"
                loading={busy}
                disabled={!canSubmit}
                className="w-full"
              >
                注册并登录
              </Button>

              <p className="text-center text-sm text-ink-muted">
                已有账号？
                <Link
                  to="/login"
                  className="ml-1 text-accent-700 underline underline-offset-2 decoration-accent-300 hover:decoration-accent-600"
                >
                  返回登录
                </Link>
              </p>
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
