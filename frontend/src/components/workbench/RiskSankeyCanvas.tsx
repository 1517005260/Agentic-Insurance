import { useEffect, useMemo, useRef, useState } from "react";
import { sankey, sankeyLinkHorizontal } from "d3-sankey";

import { useCitationStore } from "@/stores/citation";
import { cn } from "@/lib/utils";
import type { CitationItem, RiskSubgraph } from "@/lib/sse-types";

/**
 * 风险传导桑基图 —— Tab 1 下半部论文配图门面。
 *
 * 三层布局（左→右）：
 *   1. 客户档案字段（蓝）
 *   2. PPR 命中风险因子（橙）
 *   3. 触发条款（红，可点开 CitationDrawer 看 verbatim）
 *
 * d3-sankey 接收 nodes / links（source/target/value）→ 给出每个 rect 的
 * (x0,y0,x1,y1) + 每条 link 的 path。我们自己渲染 SVG（rect + path），
 * 比起再上一层包装器更可控。
 */

const LAYER_FILL: Record<number, string> = {
  0: "#3b82f6", // blue-500 — 客户档案
  1: "#f97316", // orange-500 — 风险因子
  2: "#ef4444", // red-500 — 触发条款
};

const LAYER_LABEL: Record<number, string> = {
  0: "客户档案字段",
  1: "PPR 命中风险因子",
  2: "触发条款",
};

interface RiskSankeyCanvasProps {
  data?: RiskSubgraph;
  citations?: CitationItem[];
  className?: string;
}

interface InternalNode {
  id: string;
  name: string;
  layer: 0 | 1 | 2;
  /** layer 2 only — used by click handler to find matching citation by
   *  (fileId, pageId)；不复用 agent-side citations[].sup 编号。 */
  fileId?: string;
  pageId?: string;
}

export function RiskSankeyCanvas({ data, citations, className }: RiskSankeyCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 360 });
  const open_ = useCitationStore((s) => s.open_);

  // 跟随容器宽度。SVG 自适应，避免父容器变窄（如打开 ProgressTimeline
  // 抽屉）时 sankey 撑出滚动条。
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const w = Math.max(320, Math.floor(e.contentRect.width));
        setSize((prev) => (prev.width === w ? prev : { ...prev, width: w }));
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const layout = useMemo(() => {
    if (!data || size.width === 0) return null;
    return computeLayout(data, size.width, size.height);
  }, [data, size.width, size.height]);

  const isEmpty =
    !data ||
    data.mode !== "ppr" ||
    data.risk_factors.length === 0 ||
    data.triggered_clauses.length === 0;

  return (
    <div
      className={cn(
        "rounded border border-ink-line bg-surface-raised overflow-hidden",
        className,
      )}
    >
      <div className="px-3 py-1.5 text-[11px] uppercase tracking-[0.16em] text-ink-subtle font-mono border-b border-ink-line bg-surface-sunk flex items-center justify-between">
        <span>风险传导 Sankey · 客户字段 → 风险因子 → 触发条款</span>
        {data?.mode && data.mode !== "ppr" && (
          <span className="text-warning normal-case tracking-normal font-sans text-[11px]">
            mode: {data.mode}
          </span>
        )}
      </div>
      <div ref={containerRef} className="w-full">
        {isEmpty || !layout ? (
          <div className="flex h-40 items-center justify-center text-[12px] text-ink-subtle px-6 text-center">
            {data?.mode === "no_seeds" || data?.mode === "no_graph"
              ? "PPR 未在该保单图谱中命中客户档案的相关风险触发点。"
              : "等待 final 事件携带 risk_subgraph…"}
          </div>
        ) : (
          <SankeySvg
            layout={layout}
            width={size.width}
            height={size.height}
            citations={citations}
            onClauseClick={(node) => {
              if (!citations || !node.fileId || !node.pageId) return;
              // 按 (file_id, page_id) 在 citations 里查匹配条目；agent
              // 真的 read 过对应页才会有 citation，否则点击仅高亮节点。
              const focus = citations.find(
                (c) =>
                  c.kind !== "web" &&
                  (c as Extract<typeof c, { file_id: string }>).file_id === node.fileId &&
                  (c as Extract<typeof c, { page_id: string }>).page_id === node.pageId,
              );
              if (focus) open_(citations, focus);
            }}
          />
        )}
      </div>
      <SankeyLegend />
    </div>
  );
}

// ============================================================ d3 layout

interface ComputedLayout {
  nodes: (InternalNode & { x0: number; x1: number; y0: number; y1: number })[];
  links: {
    source: { id: string };
    target: { id: string };
    value: number;
    width: number;
    path: string;
  }[];
}

function computeLayout(
  data: RiskSubgraph,
  width: number,
  height: number,
): ComputedLayout | null {
  // 不是显式声明 layer 的话 d3-sankey 会按 source/target 拓扑推。但
  // 我们手工指定 layer 才能保证三列严格对齐（即使某 risk_factor 没有
  // L2→L3 出边时也不会被推到中间或最右）。
  const nodes: InternalNode[] = [
    ...data.customer_fields.map((n) => ({ id: n.id, name: n.label, layer: 0 as const })),
    ...data.risk_factors.map((n) => ({ id: n.id, name: n.label, layer: 1 as const })),
    ...data.triggered_clauses.map((n) => ({
      id: n.id,
      name: `${n.file_id} p${parsePageNum(n.page_id) ?? "?"}`,
      layer: 2 as const,
      fileId: n.file_id,
      pageId: n.page_id,
    })),
  ];
  // 已配置 .nodeId(n => n.id)，links.source/target 必须用 id 字符串；用
  // numeric index d3-sankey 会拿 0 当 id 去查 map → 抛 "missing: 0"。
  const idSet = new Set(nodes.map((n) => n.id));
  const links = data.edges
    .map((e) => {
      if (!idSet.has(e.source) || !idSet.has(e.target)) return null;
      // 极小权重会被 sankey 当噪音 collapsed；做一个最低值兜底，让所有
      // 边都至少占 1 像素厚度，避免视觉上"消失的链路"。
      const value = Math.max(e.weight, 0.001);
      return { source: e.source, target: e.target, value };
    })
    .filter((x): x is { source: string; target: string; value: number } => x !== null);

  if (links.length === 0) return null;

  // d3-sankey generics require index-signatured extra props; declare a
  // wider extra-props type that satisfies SankeyExtraProperties without
  // losing the layer/id/sup fields we read back below.
  interface NodeExtra {
    [key: string]: unknown;
    id: string;
    name: string;
    layer: 0 | 1 | 2;
    fileId?: string;
    pageId?: string;
  }
  interface LinkExtra {
    [key: string]: unknown;
  }

  const sankeyGen = sankey<NodeExtra, LinkExtra>()
    .nodeId((n) => n.id)
    .nodeAlign((n) => (n as unknown as { layer: number }).layer)
    .nodeWidth(14)
    .nodePadding(12)
    .extent([
      [12, 12],
      [width - 12, height - 12],
    ]);

  const graph = sankeyGen({
    nodes: nodes.map((n) => ({ ...n })) as NodeExtra[],
    // d3-sankey expects SankeyLink[] (source/target/value required); we
    // provide the same shape but its TS surface insists on the wider
    // generic type. Cast through unknown — runtime shape is valid.
    links: links.map((l) => ({ ...l })) as unknown as Parameters<typeof sankeyGen>[0]["links"],
  });

  const linkPath = sankeyLinkHorizontal<NodeExtra, LinkExtra>();

  return {
    nodes: graph.nodes
      .filter(
        (n): n is typeof n & { x0: number; x1: number; y0: number; y1: number } =>
          n.x0 != null && n.x1 != null && n.y0 != null && n.y1 != null,
      )
      .map((n) => ({
        id: n.id,
        name: n.name,
        layer: n.layer,
        fileId: n.fileId,
        pageId: n.pageId,
        x0: n.x0,
        x1: n.x1,
        y0: n.y0,
        y1: n.y1,
      })),
    links: graph.links.map((l) => ({
      source: { id: (l.source as { id: string }).id },
      target: { id: (l.target as { id: string }).id },
      value: l.value ?? 0,
      width: l.width ?? 1,
      path: linkPath(l) ?? "",
    })),
  };
}

function parsePageNum(pageId: string): number | null {
  if (pageId.startsWith("p_")) {
    const n = parseInt(pageId.slice(2), 10);
    return Number.isNaN(n) ? null : n;
  }
  return null;
}

// ============================================================ SVG

interface SankeySvgProps {
  layout: ComputedLayout;
  width: number;
  height: number;
  citations?: CitationItem[];
  onClauseClick?: (node: InternalNode) => void;
}

function SankeySvg({ layout, width, height, citations, onClauseClick }: SankeySvgProps) {
  const citationLookup = useMemo(() => {
    // (file_id, page_id) → CitationItem。agent 实际 read 过对应页才会
    // 进 citations，没 read 的 PPR 命中段落点击只能高亮节点。
    const m = new Map<string, CitationItem>();
    for (const c of citations ?? []) {
      if (c.kind === "web") continue;
      m.set(`${c.file_id}::${c.page_id}`, c);
    }
    return m;
  }, [citations]);

  return (
    <svg width={width} height={height} className="block">
      <g>
        {layout.links.map((l, i) => (
          <path
            key={`l-${i}`}
            d={l.path}
            fill="none"
            stroke="#94a3b8"
            strokeOpacity={0.35}
            strokeWidth={l.width}
          >
            <title>
              {l.source.id} → {l.target.id} (w={l.value.toFixed(3)})
            </title>
          </path>
        ))}
      </g>
      <g>
        {layout.nodes.map((n) => {
          const isClause = n.layer === 2;
          const fill = LAYER_FILL[n.layer];
          const labelText = truncate(n.name, 22);
          const labelX = n.layer === 0 ? n.x1 + 6 : n.x0 - 6;
          const labelAnchor = n.layer === 0 ? "start" : "end";
          return (
            <g
              key={n.id}
              className={isClause ? "cursor-pointer" : undefined}
              onClick={isClause ? () => onClauseClick?.(n) : undefined}
            >
              <rect
                x={n.x0}
                y={n.y0}
                width={n.x1 - n.x0}
                height={Math.max(2, n.y1 - n.y0)}
                fill={fill}
                opacity={0.85}
              >
                <title>
                  {n.name}
                  {isClause &&
                  n.fileId &&
                  n.pageId &&
                  citationLookup.get(`${n.fileId}::${n.pageId}`)
                    ? "（点开查看条款原文）"
                    : ""}
                </title>
              </rect>
              <text
                x={labelX}
                y={(n.y0 + n.y1) / 2}
                dy="0.32em"
                fontSize={11}
                fill="#1f2937"
                textAnchor={labelAnchor}
                pointerEvents="none"
              >
                {labelText}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n - 1) + "…";
}

// ============================================================ Legend

function SankeyLegend() {
  return (
    <div className="flex flex-wrap items-center gap-3 px-3 py-1.5 text-[11px] text-ink-muted border-t border-ink-line bg-surface-raised">
      {[0, 1, 2].map((layer) => (
        <div key={layer} className="flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-3 rounded-sm"
            style={{ backgroundColor: LAYER_FILL[layer] }}
          />
          <span>{LAYER_LABEL[layer]}</span>
        </div>
      ))}
      <span className="ml-auto text-ink-subtle">点击右侧条款列可查看原文</span>
    </div>
  );
}
