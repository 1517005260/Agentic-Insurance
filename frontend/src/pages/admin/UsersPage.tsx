/**
 * /admin/users —— 用户管理
 *
 * 设计取舍：
 *  - 表格风格而非卡片；admin 关心"一次扫一行"。
 *  - role / is_active 是行内 toggle —— 即时 PATCH，省去 N 次 dialog；
 *    自防护（不能改自己）显示 disabled + tooltip。
 *  - 创建 / 重置密码 / 删除走 dialog（操作不可撤销，先确认）。
 *  - 删除是 soft-delete（is_active=0）；UI 上行还在，role 也还在，
 *    跟"激活/禁用"区分开 —— 后端是同一动作，前端把它清晰地分成两列：
 *    "状态"列 toggle = 激活 / 禁用，独立的"删除"按钮 = soft-delete。
 *  - 末位 admin 防护后端兜底 422，前端做 best-effort 提示（disable
 *    最后一个 admin 的 demote / 删除按钮）。
 */
import { useMemo, useState } from "react";
import { isAxiosError } from "axios";
import {
  AlertTriangle,
  KeyRound,
  Loader2,
  Plus,
  ShieldCheck,
  Trash2,
  UserCog,
} from "lucide-react";
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { api } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useAuthStore } from "@/stores/auth";
import { cn } from "@/lib/utils";

type Role = "admin" | "analyst";

interface UserRow {
  id: number;
  username: string;
  role: Role;
  is_active: boolean;
  created_at: string;
}

const usersKey = ["admin", "users"] as const;

function fmtAxios(e: unknown): string {
  if (isAxiosError(e)) {
    const detail = e.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (detail) return JSON.stringify(detail);
    return e.message;
  }
  return e instanceof Error ? e.message : String(e);
}

export default function UsersPage() {
  const me = useAuthStore((s) => s.user);
  const qc = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);
  const [resetFor, setResetFor] = useState<UserRow | null>(null);

  const { data, isLoading, isError, error } = useQuery<UserRow[]>({
    queryKey: usersKey,
    queryFn: async () => {
      // include_inactive=true 关键：后端默认只返活跃用户，不带这个参
      // 数会让 soft-delete 后的行直接消失 —— 本页要在表内显示并允许
      // 复活（toggle is_active=true），所以必须拿全量。
      const { data } = await api.get<UserRow[]>("/admin/users", {
        params: { include_inactive: true },
      });
      return data;
    },
  });

  const activeAdminCount = useMemo(
    () => (data ?? []).filter((u) => u.role === "admin" && u.is_active).length,
    [data],
  );

  const patchMu = useMutation<
    UserRow,
    unknown,
    { id: number; patch: Partial<Pick<UserRow, "role" | "is_active">> }
  >({
    mutationFn: async ({ id, patch }) => {
      const { data } = await api.patch<UserRow>(`/admin/users/${id}`, patch);
      return data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: usersKey }),
    onError: (e) => alert(`修改失败：${fmtAxios(e)}`),
  });

  const deleteMu = useMutation<void, unknown, number>({
    mutationFn: async (id) => {
      await api.delete(`/admin/users/${id}`);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: usersKey }),
    onError: (e) => alert(`删除失败：${fmtAxios(e)}`),
  });

  const onToggleActive = (u: UserRow) => {
    if (me?.id === u.id) return; // 自防护
    patchMu.mutate({ id: u.id, patch: { is_active: !u.is_active } });
  };

  const onDemote = (u: UserRow) => {
    if (me?.id === u.id) return;
    if (u.role === "admin" && activeAdminCount <= 1) return;
    if (!confirm(`将 ${u.username} 角色改为 analyst？`)) return;
    patchMu.mutate({ id: u.id, patch: { role: "analyst" } });
  };

  const onPromote = (u: UserRow) => {
    if (!confirm(`将 ${u.username} 角色改为 admin？`)) return;
    patchMu.mutate({ id: u.id, patch: { role: "admin" } });
  };

  const onDelete = (u: UserRow) => {
    if (me?.id === u.id) return;
    if (u.role === "admin" && u.is_active && activeAdminCount <= 1) return;
    if (!confirm(`删除（停用）${u.username}？该操作可由 PATCH is_active=true 复活。`)) return;
    deleteMu.mutate(u.id);
  };

  return (
    <div className="flex flex-col gap-4 p-6">
      <header className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-medium text-ink">用户管理</h1>
          <p className="text-[12px] text-ink-subtle mt-0.5">
            共 {data?.length ?? 0} 个账号 · 活跃 admin {activeAdminCount} 人。RBAC：
            <span className="font-mono">admin</span> /
            <span className="font-mono">analyst</span>。末位 admin 不可降级或删除。
          </p>
        </div>
        <Button variant="primary" size="md" onClick={() => setCreateOpen(true)}>
          <Plus className="h-4 w-4" /> 新增用户
        </Button>
      </header>

      {isLoading && (
        <div className="flex items-center justify-center py-16 gap-2 text-ink-muted">
          <Loader2 className="h-4 w-4 animate-spin" /> 加载用户列表…
        </div>
      )}

      {isError && (
        <div className="flex items-start gap-2 px-3 py-3 text-sm text-danger">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <div>加载失败：{(error as Error)?.message ?? "未知错误"}</div>
        </div>
      )}

      {data && data.length > 0 && (
        <div className="overflow-hidden rounded border border-ink-line">
          <table className="w-full text-sm">
            <thead className="bg-surface-sunk text-ink-muted text-[12px]">
              <tr>
                <th className="text-left px-3 py-2 font-medium">用户名</th>
                <th className="text-left px-3 py-2 font-medium">角色</th>
                <th className="text-left px-3 py-2 font-medium">状态</th>
                <th className="text-left px-3 py-2 font-medium">创建时间</th>
                <th className="text-right px-3 py-2 font-medium">操作</th>
              </tr>
            </thead>
            <tbody>
              {data.map((u) => {
                const isMe = me?.id === u.id;
                const lastAdmin =
                  u.role === "admin" && u.is_active && activeAdminCount <= 1;
                return (
                  <tr
                    key={u.id}
                    className="border-t border-ink-line hover:bg-surface-sunk/30"
                  >
                    <td className="px-3 py-2 font-mono">
                      {u.username}
                      {isMe && (
                        <span className="ml-2 text-[10px] uppercase tracking-[0.16em] text-accent-700">
                          you
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2">
                      <RoleBadge
                        role={u.role}
                        disabled={isMe || lastAdmin || patchMu.isPending}
                        onChangeRole={() =>
                          u.role === "admin" ? onDemote(u) : onPromote(u)
                        }
                      />
                    </td>
                    <td className="px-3 py-2">
                      <button
                        type="button"
                        onClick={() => onToggleActive(u)}
                        disabled={isMe || lastAdmin || patchMu.isPending}
                        className={cn(
                          "inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-[12px]",
                          u.is_active
                            ? "bg-primary-100 text-primary-800"
                            : "bg-ink-line/40 text-ink-muted",
                          (isMe || lastAdmin) && "opacity-50 cursor-not-allowed",
                          !isMe && !lastAdmin && "hover:opacity-80 cursor-pointer",
                        )}
                        title={
                          isMe
                            ? "不能停用自己"
                            : lastAdmin
                              ? "末位活跃 admin 不可停用"
                              : u.is_active
                                ? "点击停用"
                                : "点击激活"
                        }
                      >
                        <span
                          className={cn(
                            "h-1.5 w-1.5 rounded-full",
                            u.is_active ? "bg-primary-600" : "bg-ink-muted",
                          )}
                        />
                        {u.is_active ? "活跃" : "停用"}
                      </button>
                    </td>
                    <td className="px-3 py-2 text-[12px] text-ink-muted font-mono">
                      {new Date(u.created_at).toLocaleString("zh-CN")}
                    </td>
                    <td className="px-3 py-2 text-right">
                      <div className="inline-flex items-center gap-1">
                        <button
                          type="button"
                          onClick={() => setResetFor(u)}
                          className="h-7 w-7 inline-flex items-center justify-center rounded text-ink-muted hover:text-ink hover:bg-surface-sunk"
                          title="重置密码"
                        >
                          <KeyRound className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => onDelete(u)}
                          disabled={isMe || lastAdmin || deleteMu.isPending}
                          className={cn(
                            "h-7 w-7 inline-flex items-center justify-center rounded",
                            isMe || lastAdmin
                              ? "text-ink-line cursor-not-allowed"
                              : "text-danger hover:bg-danger/5",
                          )}
                          title={
                            isMe
                              ? "不能删除自己"
                              : lastAdmin
                                ? "末位 admin 不可删除"
                                : "删除（停用）"
                          }
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {createOpen && (
        <CreateUserDialog onClose={() => setCreateOpen(false)} />
      )}
      {resetFor && (
        <ResetPasswordDialog user={resetFor} onClose={() => setResetFor(null)} />
      )}
    </div>
  );
}

// --------------------------------------------------------------- role badge

function RoleBadge({
  role,
  disabled,
  onChangeRole,
}: {
  role: Role;
  disabled: boolean;
  onChangeRole: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onChangeRole}
      disabled={disabled}
      className={cn(
        "inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-[12px] font-medium",
        role === "admin"
          ? "bg-accent-100 text-accent-800"
          : "bg-ink-line/40 text-ink",
        disabled && "opacity-50 cursor-not-allowed",
        !disabled && "hover:opacity-80 cursor-pointer",
      )}
      title={disabled ? "不可改" : `点击切换为 ${role === "admin" ? "analyst" : "admin"}`}
    >
      {role === "admin" ? (
        <ShieldCheck className="h-3 w-3" />
      ) : (
        <UserCog className="h-3 w-3" />
      )}
      {role}
    </button>
  );
}

// --------------------------------------------------------------- create dialog

function CreateUserDialog({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>("analyst");
  const [error, setError] = useState<string | null>(null);

  const mu = useMutation<UserRow, unknown, void>({
    mutationFn: async () => {
      const { data } = await api.post<UserRow>("/admin/users", {
        username,
        password,
        role,
      });
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: usersKey });
      onClose();
    },
    onError: (e) => setError(fmtAxios(e)),
  });

  const submitDisabled =
    mu.isPending || username.length < 2 || password.length < 8;

  return (
    <DialogShell title="新增用户" onClose={onClose}>
      <div className="space-y-3">
        <Field label="用户名">
          <Input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="2-64 字符 · A-Z a-z 0-9 . _ -"
            autoComplete="off"
          />
        </Field>
        <Field label="初始密码">
          <Input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="≥8 字符 · 至少一个数字 + 一个字母"
            autoComplete="new-password"
          />
        </Field>
        <Field label="角色">
          <div className="flex gap-2">
            {(["analyst", "admin"] as Role[]).map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setRole(r)}
                className={cn(
                  "px-3 py-1 rounded text-[13px] border",
                  role === r
                    ? "bg-primary-600 text-surface-raised border-primary-700"
                    : "border-ink-line text-ink hover:bg-surface-sunk",
                )}
              >
                {r}
              </button>
            ))}
          </div>
        </Field>
        {error && (
          <div role="alert" className="rounded border border-danger/30 bg-danger/5 px-3 py-2 text-[12px] text-danger">
            {error}
          </div>
        )}
      </div>
      <DialogFooter>
        <Button variant="ghost" onClick={onClose}>
          取消
        </Button>
        <Button
          variant="primary"
          onClick={() => mu.mutate()}
          disabled={submitDisabled}
          loading={mu.isPending}
        >
          创建
        </Button>
      </DialogFooter>
    </DialogShell>
  );
}

// --------------------------------------------------------------- reset password

function ResetPasswordDialog({
  user,
  onClose,
}: {
  user: UserRow;
  onClose: () => void;
}) {
  const [pw, setPw] = useState("");
  const [error, setError] = useState<string | null>(null);

  const mu = useMutation<void, unknown, void>({
    mutationFn: async () => {
      await api.post(`/admin/users/${user.id}/reset-password`, {
        new_password: pw,
      });
    },
    onSuccess: onClose,
    onError: (e) => setError(fmtAxios(e)),
  });

  return (
    <DialogShell title={`重置 ${user.username} 的密码`} onClose={onClose}>
      <Field label="新密码">
        <Input
          type="password"
          value={pw}
          onChange={(e) => setPw(e.target.value)}
          placeholder="≥8 字符 · 至少一个数字 + 一个字母"
          autoComplete="new-password"
        />
      </Field>
      {error && (
        <div role="alert" className="mt-3 rounded border border-danger/30 bg-danger/5 px-3 py-2 text-[12px] text-danger">
          {error}
        </div>
      )}
      <DialogFooter>
        <Button variant="ghost" onClick={onClose}>
          取消
        </Button>
        <Button
          variant="primary"
          onClick={() => mu.mutate()}
          disabled={mu.isPending || pw.length < 8}
          loading={mu.isPending}
        >
          重置
        </Button>
      </DialogFooter>
    </DialogShell>
  );
}

// --------------------------------------------------------------- shells

function DialogShell({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  return (
    <>
      <div
        className="fixed inset-0 z-40 bg-ink/20 animate-fade-in"
        onClick={onClose}
        aria-hidden
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={cn(
          "fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2",
          "w-[min(440px,calc(100vw-32px))]",
          "bg-surface-raised border border-ink-line rounded-md shadow-pop",
          "flex flex-col animate-fade-in",
        )}
      >
        <header className="px-4 py-3 border-b border-ink-line">
          <h2 className="text-sm font-medium text-ink">{title}</h2>
        </header>
        <div className="px-4 py-4">{children}</div>
      </div>
    </>
  );
}

function DialogFooter({ children }: { children: React.ReactNode }) {
  return (
    <div className="-mx-4 -mb-4 mt-4 flex items-center justify-end gap-2 px-4 py-3 border-t border-ink-line">
      {children}
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="block text-[12px] text-ink-muted mb-1">{label}</span>
      {children}
    </label>
  );
}
