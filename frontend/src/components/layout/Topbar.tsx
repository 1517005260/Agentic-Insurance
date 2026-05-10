import { useNavigate } from "react-router-dom";
import { LogOut, ShieldCheck, UserCircle2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useAuthStore } from "@/stores/auth";

export function Topbar() {
  const user = useAuthStore((s) => s.user);
  const clear = useAuthStore((s) => s.clear);
  const navigate = useNavigate();

  const onLogout = () => {
    clear();
    navigate("/login", { replace: true });
  };

  return (
    <header className="h-12 border-b border-ink-line bg-surface-raised flex items-center justify-between px-5">
      <div className="text-sm text-ink-muted">
        欢迎回来
        <span className="mx-1.5 text-ink-subtle">·</span>
        <span className="text-ink">{user?.username}</span>
      </div>

      <div className="flex items-center gap-3">
        <span className="inline-flex items-center gap-1 text-xs text-ink-muted">
          {user?.role === "admin" ? (
            <>
              <ShieldCheck className="h-3.5 w-3.5 text-primary-600" />
              管理员
            </>
          ) : (
            <>
              <UserCircle2 className="h-3.5 w-3.5 text-ink-subtle" />
              分析师
            </>
          )}
        </span>
        <Button variant="ghost" size="sm" onClick={onLogout}>
          <LogOut className="h-3.5 w-3.5" />
          退出
        </Button>
      </div>
    </header>
  );
}
