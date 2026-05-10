import { NavLink } from "react-router-dom";
import {
  MessageSquare,
  Search,
  GitCompare,
  ShieldAlert,
  Network,
  FolderOpen,
  Users,
  Settings,
  ScrollText,
  Calculator,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { useAuthStore } from "@/stores/auth";

interface NavItem {
  to: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const BUSINESS_NAV: NavItem[] = [
  { to: "/chat", label: "智能问答", icon: MessageSquare },
  { to: "/search", label: "条款检索", icon: Search },
  { to: "/products", label: "产品对比推荐", icon: GitCompare },
  { to: "/risk", label: "理赔风险预测", icon: ShieldAlert },
  { to: "/policy-calc", label: "保单精算", icon: Calculator },
  { to: "/graph", label: "知识图谱", icon: Network },
  { to: "/files", label: "文件", icon: FolderOpen },
];

const ADMIN_NAV: NavItem[] = [
  { to: "/admin/users", label: "用户管理", icon: Users },
  { to: "/admin/config", label: "系统配置", icon: Settings },
  { to: "/admin/audit", label: "审计日志", icon: ScrollText },
];

function NavRow({ item }: { item: NavItem }) {
  const Icon = item.icon;
  return (
    <NavLink
      to={item.to}
      className={({ isActive }) =>
        cn(
          "relative flex items-center gap-2.5 px-3 h-9 rounded text-sm transition-colors",
          "text-ink-muted hover:text-ink hover:bg-surface-sunk",
          isActive &&
          "bg-primary-50 text-primary-700 font-medium hover:bg-primary-50 hover:text-primary-700",
        )
      }
    >
      {({ isActive }) => (
        <>
          {/* active 左侧 2px 主色竖条，比单靠浅底色更醒目（且色弱友好） */}
          {isActive && (
            <span
              aria-hidden
              className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-full bg-primary-600"
            />
          )}
          <Icon className="h-4 w-4 shrink-0" />
          <span className="truncate">{item.label}</span>
        </>
      )}
    </NavLink>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="mt-3 mb-1 px-3 text-[10px] font-medium uppercase tracking-[0.18em] text-ink-subtle">
      {children}
    </div>
  );
}

export function Sidebar() {
  const isAdmin = useAuthStore((s) => s.user?.role === "admin");

  return (
    <aside className="hidden md:flex w-56 shrink-0 flex-col border-r border-ink-line bg-surface-raised">
      <div className="px-5 py-5 border-b border-ink-line">
        <div className="font-serif text-lg text-primary-700 leading-tight">
          Agentic
        </div>
        <div className="text-[11px] uppercase tracking-[0.18em] text-ink-subtle mt-0.5">
          Insurance Console
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto scrollbar-thin px-2 py-2 space-y-0.5">
        <SectionLabel>业务</SectionLabel>
        {BUSINESS_NAV.map((it) => (
          <NavRow key={it.to} item={it} />
        ))}

        {isAdmin && (
          <>
            <div className="my-2 mx-3 h-px bg-ink-line/60" />
            <SectionLabel>管理</SectionLabel>
            {ADMIN_NAV.map((it) => (
              <NavRow key={it.to} item={it} />
            ))}
          </>
        )}
      </nav>

      <div className="px-5 py-3 border-t border-ink-line text-[11px] text-ink-subtle">
        v0.1
      </div>
    </aside>
  );
}
