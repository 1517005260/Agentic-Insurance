/**
 * SessionSidebar — chat session 列表 + 新建 + 切换 + 删除。
 *
 * 设计：
 *  - 折叠式左侧栏（240px），ChatPage 自己决定何时显示。
 *  - 列表按 updated_at desc 直接吃后端给的顺序（后端已 ORDER BY desc）。
 *  - "新对话" = setCurrent(null)；composer 留着 mode 选择，下次 send 时
 *    才创 session（这样 mode 不固定时不会建空 session）。
 *  - 删除按 ⋯ 菜单（重用 FilesPage 同款）。删 currentId 后切到 null。
 *  - 列表 entry 显示 mode badge：[RAG] [Web] [Agent base/proof/graph] [Web Agent]
 *  - 标题 line-clamp-2，避免长 query 撑爆侧栏。
 */
import { useEffect, useRef, useState } from "react";
import {
  MessageSquarePlus,
  MoreVertical,
  PenLine,
  Trash2,
} from "lucide-react";

import { useSessionStore } from "@/stores/session";
import {
  useDeleteSession,
  useRenameSession,
  useSessions,
  type SessionRow,
} from "@/hooks/useSessions";
import { cn } from "@/lib/utils";

function modeLabel(s: SessionRow): string {
  if (s.mode === "rag") return s.web ? "Web RAG" : "RAG";
  // agent. ChatComposer 现在只允许创建 base / graph 类型的 session；
  // proof 后端仍接受（agent_kind="proof"），但 UI 上不再出现该选项。
  // 老 session 若 agent_kind="proof" 会落到 default 分支显示 "Agent"。
  if (s.web && s.agent_kind === "base") return "Web Agent";
  switch (s.agent_kind) {
    case "base":
      return "Base Agent";
    case "graph":
      return "Graph Agent";
    default:
      return "Agent";
  }
}

function modePalette(s: SessionRow): string {
  // 配色对齐 ChatComposer / AssistantTurn 的 mode pill
  if (s.mode === "rag" && !s.web) return "bg-primary-50 text-primary-700";
  if (s.mode === "rag" && s.web) return "bg-accent-50 text-accent-700";
  if (s.mode === "agent" && !s.web) return "bg-primary-100 text-primary-800";
  return "bg-accent-100 text-accent-800";
}

export function SessionSidebar() {
  const currentId = useSessionStore((s) => s.currentId);
  const setCurrent = useSessionStore((s) => s.setCurrent);
  const reset = useSessionStore((s) => s.reset);
  const { data, isLoading } = useSessions();

  return (
    <aside
      className={cn(
        "shrink-0 w-60 border-r border-ink-line bg-surface-base",
        "flex flex-col overflow-hidden",
      )}
    >
      <div className="px-3 py-3 border-b border-ink-line">
        <button
          type="button"
          onClick={reset}
          className={cn(
            "w-full inline-flex items-center justify-center gap-2 rounded",
            "border border-ink-line bg-surface-raised text-ink",
            "px-3 py-2 text-[13px] font-medium",
            "hover:border-primary-300 hover:bg-primary-50/30",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30",
          )}
        >
          <MessageSquarePlus className="h-4 w-4" />
          新对话
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        {isLoading && (
          <div className="px-2 py-3 text-[12px] text-ink-muted">加载中…</div>
        )}
        {!isLoading && (data?.length ?? 0) === 0 && (
          <div className="px-2 py-3 text-[12px] text-ink-subtle">
            尚无对话；选 mode、提问后自动建会话。
          </div>
        )}
        {data?.map((s) => (
          <SessionRowItem
            key={s.id}
            session={s}
            active={s.id === currentId}
            onSelect={() => setCurrent(s.id)}
          />
        ))}
      </div>
    </aside>
  );
}

// --------------------------------------------------------------- row

function SessionRowItem({
  session,
  active,
  onSelect,
}: {
  session: SessionRow;
  active: boolean;
  onSelect: () => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const renameMu = useRenameSession();
  const deleteMu = useDeleteSession();
  const setCurrent = useSessionStore((s) => s.setCurrent);
  const currentId = useSessionStore((s) => s.currentId);

  useEffect(() => {
    if (!menuOpen) return;
    const onClick = (e: MouseEvent) => {
      const t = e.target as Node;
      if (wrapperRef.current && !wrapperRef.current.contains(t)) setMenuOpen(false);
    };
    window.addEventListener("mousedown", onClick);
    return () => window.removeEventListener("mousedown", onClick);
  }, [menuOpen]);

  const onRename = () => {
    const next = prompt("新标题", session.title);
    if (next == null) return;
    const trimmed = next.trim();
    if (!trimmed || trimmed === session.title) return;
    renameMu.mutate({ id: session.id, title: trimmed });
  };

  const onDelete = () => {
    if (!confirm(`删除 "${session.title}"？\n该会话的消息将一并清除。`)) return;
    deleteMu.mutate(session.id, {
      onSuccess: () => {
        if (session.id === currentId) setCurrent(null);
      },
    });
  };

  return (
    <div
      ref={wrapperRef}
      className={cn(
        "group relative rounded-md",
        active && "bg-primary-50/60",
        !active && "hover:bg-surface-sunk/40",
      )}
    >
      <button
        type="button"
        onClick={onSelect}
        className={cn(
          "block w-full text-left px-2 py-2",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30 rounded-md",
        )}
        title={session.title}
      >
        <div className="flex items-center gap-1.5 mb-0.5">
          <span
            className={cn(
              "inline-flex items-center rounded px-1.5 py-0 text-[10px] font-medium",
              modePalette(session),
            )}
          >
            {modeLabel(session)}
          </span>
        </div>
        <div className="text-[12.5px] text-ink leading-snug line-clamp-2">
          {session.title}
        </div>
      </button>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setMenuOpen((v) => !v);
        }}
        aria-label="会话操作"
        className={cn(
          "absolute top-1 right-1 h-6 w-6 inline-flex items-center justify-center rounded",
          "bg-surface-raised/80 backdrop-blur border border-ink-line/60",
          "text-ink-muted hover:text-ink hover:bg-surface-raised",
          "opacity-0 group-hover:opacity-100",
          menuOpen && "opacity-100",
        )}
      >
        <MoreVertical className="h-3.5 w-3.5" />
      </button>
      {menuOpen && (
        <div
          role="menu"
          className={cn(
            "absolute right-1 top-7 z-10 w-32 rounded-md border border-ink-line",
            "bg-surface-raised shadow-pop py-1 text-[13px]",
          )}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setMenuOpen(false);
              onRename();
            }}
            className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-ink hover:bg-surface-sunk"
          >
            <PenLine className="h-3.5 w-3.5" />
            重命名
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setMenuOpen(false);
              onDelete();
            }}
            className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-danger hover:bg-danger/5"
          >
            <Trash2 className="h-3.5 w-3.5" />
            删除
          </button>
        </div>
      )}
    </div>
  );
}
