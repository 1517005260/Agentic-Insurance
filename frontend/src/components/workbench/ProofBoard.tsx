import { useMemo } from "react";
import { CheckCircle2, Circle, AlertCircle, X as XIcon, ScanLine } from "lucide-react";

import type { SSEEvent } from "@/lib/sse-types";
import { cn } from "@/lib/utils";

type ObligationRow = {
  id: string;
  kind: string;
  status: "OPEN" | "CLOSED" | "REMOVED";
  required: boolean;
  failure_kind?: string | null;
};
type ClaimRow = {
  id: string;
  kind: string;
  status: "live" | "REMOVED";
  by: string[];
};
type GapRow = {
  id: string;
  kind: string;
  status: "ACTIVE" | "REMOVED";
};

/**
 * Proof obligation/claim/gap 实时看板 —— Exclusion 工作台专用。
 *
 * ProgressTimeline 是按时间线全量塞 events；这里再做一道 group-by-id
 * 让用户一眼看到当前活跃的 obligation 数 / claim 数 / 未关 gap。
 *
 * 状态合并规则：
 *   - obligation: 取最后一帧 status；CLOSED/REMOVED 显示对应图标
 *   - claim:      REMOVED 状态把整条标灰
 *   - gap:        REMOVED 隐藏（不再 actionable）
 */
export function ProofBoard({ events }: { events: SSEEvent[] }) {
  const { obligations, claims, gaps } = useMemo(
    () => collectProofRows(events),
    [events],
  );

  if (obligations.size === 0 && claims.size === 0 && gaps.size === 0) {
    return null;
  }

  return (
    <section className="rounded-md border border-ink-line/70 bg-surface-raised/60 px-3 py-2 space-y-2">
      <h3 className="text-[11px] uppercase tracking-[0.16em] text-ink-subtle font-mono">
        Proof Loop
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <Column
          label={`Obligation (${obligations.size})`}
          rows={[...obligations.values()]}
          render={(o) => <ObligationItem o={o as ObligationRow} />}
        />
        <Column
          label={`Claim (${claims.size})`}
          rows={[...claims.values()]}
          render={(c) => <ClaimItem c={c as ClaimRow} />}
        />
        <Column
          label={`Gap (${gaps.size})`}
          rows={[...gaps.values()].filter((g) => (g as GapRow).status !== "REMOVED")}
          render={(g) => <GapItem g={g as GapRow} />}
        />
      </div>
    </section>
  );
}

function Column({
  label,
  rows,
  render,
}: {
  label: string;
  rows: object[];
  render: (r: object) => React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <div className="text-[11px] font-mono text-ink-subtle">{label}</div>
      {rows.length === 0 ? (
        <div className="text-[11px] text-ink-subtle">—</div>
      ) : (
        <ul className="space-y-1">
          {rows.map((r, i) => (
            <li key={i} className="text-[12px] leading-tight">
              {render(r)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ObligationItem({ o }: { o: ObligationRow }) {
  const icon =
    o.status === "CLOSED" ? (
      <CheckCircle2 className="h-3.5 w-3.5 text-success" />
    ) : o.status === "REMOVED" ? (
      <XIcon className="h-3.5 w-3.5 text-ink-subtle" />
    ) : (
      <Circle className="h-3.5 w-3.5 text-warning" />
    );
  return (
    <div className="flex items-center gap-1.5">
      {icon}
      <span
        className={cn(
          "font-mono text-[11px] truncate max-w-[140px]",
          o.status === "REMOVED" && "line-through text-ink-subtle",
        )}
        title={`${o.kind} · ${o.status}${o.failure_kind ? ` · ${o.failure_kind}` : ""}`}
      >
        {o.id}
      </span>
      {o.required && (
        <span className="text-[10px] text-danger/80 font-mono">req</span>
      )}
    </div>
  );
}

function ClaimItem({ c }: { c: ClaimRow }) {
  return (
    <div
      className={cn(
        "flex items-center gap-1.5",
        c.status === "REMOVED" && "opacity-60",
      )}
    >
      <ScanLine className="h-3.5 w-3.5 text-primary-700" />
      <span
        className={cn(
          "font-mono text-[11px] truncate max-w-[140px]",
          c.status === "REMOVED" && "line-through",
        )}
        title={`${c.kind}${c.by.length ? ` · by ${c.by.join(",")}` : ""}`}
      >
        {c.id}
      </span>
    </div>
  );
}

function GapItem({ g }: { g: GapRow }) {
  return (
    <div className="flex items-center gap-1.5">
      <AlertCircle className="h-3.5 w-3.5 text-danger" />
      <span
        className="font-mono text-[11px] truncate max-w-[140px]"
        title={g.kind}
      >
        {g.id}
      </span>
    </div>
  );
}

// ------------------------------------------------------------------ collect

function collectProofRows(events: SSEEvent[]) {
  const obligations = new Map<string, ObligationRow>();
  const claims = new Map<string, ClaimRow>();
  const gaps = new Map<string, GapRow>();

  for (const ev of events) {
    if (ev.event === "obligation") {
      const d = ev.data;
      obligations.set(d.id, {
        id: d.id,
        kind: d.kind,
        status: d.status,
        required: d.required,
        failure_kind: d.failure_kind,
      });
    } else if (ev.event === "claim") {
      const d = ev.data;
      const prev = claims.get(d.id);
      claims.set(d.id, {
        id: d.id,
        kind: d.kind ?? prev?.kind ?? "Claim",
        status: d.status === "REMOVED" ? "REMOVED" : (prev?.status ?? "live"),
        by: d.by ?? prev?.by ?? [],
      });
    } else if (ev.event === "gap") {
      const d = ev.data;
      gaps.set(d.id, { id: d.id, kind: d.kind, status: d.status });
    }
  }

  return { obligations, claims, gaps };
}
