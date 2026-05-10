/**
 * /admin/audit —— 审计日志
 *
 * 设计取舍：
 *  - filter 走后端 query string（action / target 精确匹配）；前端不
 *    做模糊匹配（小语料 demo 用不上 + 后端不支持 LIKE）。
 *  - 分页 LIMIT/OFFSET，每页 50 条，按 id desc
 *  - payload_json 是字符串（后端把 dict json.dumps 后存的）；行默认折
 *    叠，点开展示 pretty JSON
 *  - 不显示 user_id 数字 —— 改用 user_id → username 反查（拉一次
 *    /admin/users 顺手 join；数据量小 demo 没必要后端 join）。删除用
 *    户后 audit row.user_id 是 SET NULL，显示 "(已删)"。
 */
import { useMemo, useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Loader2,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { api } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface AuditEntry {
  id: number;
  user_id: number | null;
  action: string;
  target: string | null;
  payload_json: string | null;
  at: string;
}

interface UserBrief {
  id: number;
  username: string;
}

const PAGE_SIZE = 50;

export default function AuditPage() {
  const [actionFilter, setActionFilter] = useState("");
  const [targetFilter, setTargetFilter] = useState("");
  const [page, setPage] = useState(0);

  // commit 后才发请求 —— 避免每键一发
  const [committed, setCommitted] = useState({ action: "", target: "" });

  const { data, isLoading, isError, error, refetch, isFetching } = useQuery<
    AuditEntry[]
  >({
    queryKey: ["admin", "audit", committed.action, committed.target, page],
    queryFn: async () => {
      const params: Record<string, string | number> = {
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      };
      if (committed.action) params.action = committed.action;
      if (committed.target) params.target = committed.target;
      const { data } = await api.get<AuditEntry[]>("/audit", { params });
      return data;
    },
  });

  // user 列表用来 join 显示 username（id → username）。include_inactive
  // 必须拿，否则 soft-delete 用户的 audit row 会显示为 "#42" 而不是
  // 真实用户名（FK on delete set null 处理的是真删，soft-delete 的 row
  // 还在 users 表里）。
  const usersQ = useQuery<UserBrief[]>({
    queryKey: ["admin", "users-brief"],
    queryFn: async () => {
      const { data } = await api.get<UserBrief[]>("/admin/users", {
        params: { include_inactive: true },
      });
      return data;
    },
    staleTime: 60_000,
  });

  const userIdToName = useMemo(() => {
    const m = new Map<number, string>();
    for (const u of usersQ.data ?? []) m.set(u.id, u.username);
    return m;
  }, [usersQ.data]);

  const onSubmitFilter = () => {
    setCommitted({ action: actionFilter.trim(), target: targetFilter.trim() });
    setPage(0);
  };

  const onClearFilter = () => {
    setActionFilter("");
    setTargetFilter("");
    setCommitted({ action: "", target: "" });
    setPage(0);
  };

  return (
    <div className="flex flex-col gap-4 p-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-medium text-ink">审计日志</h1>
          <p className="text-[12px] text-ink-subtle mt-0.5">
            管理动作 / 文件生命周期 / 配置改动；按 id 倒序。每页 {PAGE_SIZE} 条。
          </p>
        </div>
        <Button
          variant="ghost"
          size="md"
          onClick={() => refetch()}
          disabled={isFetching}
        >
          <RefreshCw className={cn("h-4 w-4", isFetching && "animate-spin")} />
          刷新
        </Button>
      </header>

      <div className="flex items-end gap-2">
        <div className="flex-1 max-w-xs">
          <label className="block text-[12px] text-ink-muted mb-1">
            action 精确匹配
          </label>
          <Input
            value={actionFilter}
            onChange={(e) => setActionFilter(e.target.value)}
            placeholder="例如 file.delete.complete"
            onKeyDown={(e) => e.key === "Enter" && onSubmitFilter()}
          />
        </div>
        <div className="flex-1 max-w-md">
          <label className="block text-[12px] text-ink-muted mb-1">
            target 精确匹配（一般是 file_id）
          </label>
          <Input
            value={targetFilter}
            onChange={(e) => setTargetFilter(e.target.value)}
            placeholder="可留空"
            onKeyDown={(e) => e.key === "Enter" && onSubmitFilter()}
          />
        </div>
        <Button variant="primary" size="md" onClick={onSubmitFilter}>
          <Search className="h-4 w-4" /> 查询
        </Button>
        {(committed.action || committed.target) && (
          <Button variant="ghost" size="md" onClick={onClearFilter}>
            <X className="h-4 w-4" /> 清空
          </Button>
        )}
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-16 gap-2 text-ink-muted">
          <Loader2 className="h-4 w-4 animate-spin" /> 加载…
        </div>
      )}
      {isError && (
        <div className="flex items-start gap-2 px-3 py-3 text-sm text-danger">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <div>加载失败：{(error as Error)?.message ?? "未知错误"}</div>
        </div>
      )}

      {data && data.length === 0 && !isLoading && (
        <div className="rounded border border-dashed border-ink-line bg-surface-raised py-16 text-center text-ink-subtle">
          {committed.action || committed.target ? "无匹配记录" : "审计日志为空"}
        </div>
      )}

      {data && data.length > 0 && (
        <div className="overflow-hidden rounded border border-ink-line">
          <table className="w-full text-sm">
            <thead className="bg-surface-sunk text-ink-muted text-[12px]">
              <tr>
                <th className="text-left px-2 py-2 font-medium w-10"></th>
                <th className="text-left px-2 py-2 font-medium w-16">id</th>
                <th className="text-left px-2 py-2 font-medium">action</th>
                <th className="text-left px-2 py-2 font-medium">user</th>
                <th className="text-left px-2 py-2 font-medium">target</th>
                <th className="text-left px-2 py-2 font-medium">at</th>
              </tr>
            </thead>
            <tbody>
              {data.map((row) => (
                <AuditRow
                  key={row.id}
                  row={row}
                  username={
                    row.user_id != null
                      ? userIdToName.get(row.user_id) ?? `#${row.user_id}`
                      : "(已删)"
                  }
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data && data.length > 0 && (
        <div className="flex items-center justify-between text-[12px] text-ink-muted">
          <span>
            第 {page + 1} 页 · 本页 {data.length} 条
            {data.length < PAGE_SIZE && " · 已到末页"}
          </span>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
            >
              上一页
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setPage((p) => p + 1)}
              disabled={data.length < PAGE_SIZE}
            >
              下一页
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------------------- row

function AuditRow({
  row,
  username,
}: {
  row: AuditEntry;
  username: string;
}) {
  const [open, setOpen] = useState(false);
  const hasPayload = !!row.payload_json;

  let prettyPayload: string | null = null;
  if (hasPayload) {
    try {
      prettyPayload = JSON.stringify(JSON.parse(row.payload_json!), null, 2);
    } catch {
      prettyPayload = row.payload_json!;
    }
  }

  return (
    <>
      <tr
        className={cn(
          "border-t border-ink-line",
          hasPayload ? "hover:bg-surface-sunk/30 cursor-pointer" : "",
        )}
        onClick={() => hasPayload && setOpen((v) => !v)}
      >
        <td className="px-2 py-2">
          {hasPayload ? (
            open ? (
              <ChevronDown className="h-3.5 w-3.5 text-ink-muted" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-ink-muted" />
            )
          ) : (
            <span className="inline-block w-3.5" />
          )}
        </td>
        <td className="px-2 py-2 text-[12px] font-mono text-ink-muted">
          {row.id}
        </td>
        <td className="px-2 py-2 text-[13px] font-mono text-ink">
          {row.action}
        </td>
        <td className="px-2 py-2 text-[12px] text-ink-muted">{username}</td>
        <td className="px-2 py-2 text-[12px] font-mono text-ink-muted">
          <span className="block max-w-[280px] truncate" title={row.target ?? ""}>
            {row.target ?? "—"}
          </span>
        </td>
        <td className="px-2 py-2 text-[11px] text-ink-muted font-mono whitespace-nowrap">
          {new Date(row.at).toLocaleString("zh-CN")}
        </td>
      </tr>
      {open && prettyPayload && (
        <tr className="bg-surface-sunk/30">
          <td colSpan={6} className="px-3 py-2">
            <pre className="text-[11px] font-mono text-ink whitespace-pre-wrap break-words max-h-64 overflow-auto">
              {prettyPayload}
            </pre>
          </td>
        </tr>
      )}
    </>
  );
}
