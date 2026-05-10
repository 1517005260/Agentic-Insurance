import { useEffect, useRef, useState } from "react";

import { GraphCanvas, type GraphEdge, type GraphNode } from "@/components/graph/GraphCanvas";
import { api } from "@/api/client";
import { cn } from "@/lib/utils";
import type { GraphSubgraphEvent, SSEEvent } from "@/lib/sse-types";
import type { WorkbenchTurn } from "./turn";

/**
 * 风险探索画布 —— Tab 1（投保前风险预测）上半部的实时力导图。
 *
 * 跟 GraphPage 的 AgentModePanel 同源思路：消费 turn.progressEvents 中的
 * graph_subgraph 帧，累积命中节点 ids，调 /graph/expand 拉子图给 GraphCanvas
 * 渲染；最新帧的第一个节点 id 作为 focusId 触发 pan 视角。
 *
 * 跟 GraphPage 不同点：
 *  - 数据源是工作台 turn-based 的事件流（无需 useSSE）；
 *  - 没有 manual mode；只渲染 agent 探索过程；
 *  - 高度固定（h-72）让它跟 Sankey 横向并存。
 */

interface SubgraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

const EMPTY_DATA: SubgraphData = { nodes: [], edges: [] };

interface RiskExploreCanvasProps {
  turn: WorkbenchTurn;
  className?: string;
}

export function RiskExploreCanvas({ turn, className }: RiskExploreCanvasProps) {
  const [overlay, setOverlay] = useState<SubgraphData | null>(null);
  const [highlight, setHighlight] = useState<string[]>([]);
  const [focusId, setFocusId] = useState<string | null>(null);

  // 累积所有命中节点 + 已处理过的帧数。turn.id 变 → 重置（新一轮探索）。
  // accumulated 用 state 暴露给 UI 做"已发现 N 个节点"的计数；其它两个
  // 仅 effect 内读写所以保持 ref。
  const accumulatedRef = useRef<Set<string>>(new Set());
  const [accumulatedCount, setAccumulatedCount] = useState(0);
  const processedCountRef = useRef(0);
  const turnIdRef = useRef<string | null>(null);
  // 同 nodeId 不重复 expand；agent 多个 graph_subgraph 帧可能命中同一焦点。
  const expandedRef = useRef<Set<string>>(new Set());

  // turn 切换 → 全部重置，防止上一轮的子图残留到新轮上。
  useEffect(() => {
    if (turnIdRef.current !== turn.id) {
      turnIdRef.current = turn.id;
      accumulatedRef.current = new Set();
      processedCountRef.current = 0;
      expandedRef.current = new Set();
      setAccumulatedCount(0);
      setOverlay(null);
      setHighlight([]);
      setFocusId(null);
    }
  }, [turn.id]);

  useEffect(() => {
    const subgraphEvts = turn.progressEvents.filter(
      (e: SSEEvent): e is GraphSubgraphEvent => e.event === "graph_subgraph",
    );
    if (subgraphEvts.length <= processedCountRef.current) return;

    const newEvts = subgraphEvts.slice(processedCountRef.current);
    processedCountRef.current = subgraphEvts.length;

    let mutated = false;
    let nextFocus: string | undefined;
    for (const ev of newEvts) {
      const data = ev.data;
      const ids = [
        ...(data.seed_ids ?? []),
        ...(data.entity_ids ?? []),
        ...(data.candidate_ids ?? []),
      ];
      for (const id of ids) {
        if (!accumulatedRef.current.has(id)) {
          accumulatedRef.current.add(id);
          mutated = true;
        }
      }
      const candidate =
        data.seed_ids?.[0] ?? data.candidate_ids?.[0] ?? data.entity_ids?.[0];
      if (candidate) nextFocus = candidate;
    }
    if (mutated) {
      setHighlight([...accumulatedRef.current]);
      setAccumulatedCount(accumulatedRef.current.size);
    }

    if (nextFocus) {
      setFocusId(nextFocus);
      if (!expandedRef.current.has(nextFocus)) {
        expandedRef.current.add(nextFocus);
        const myTurnId = turnIdRef.current;
        api
          .get<SubgraphData>(
            `/graph/expand?node_id=${encodeURIComponent(nextFocus)}&hops=1&top_k=20`,
          )
          .then(({ data: subgraph }) => {
            if (turnIdRef.current !== myTurnId) return;
            if (subgraph.nodes.length === 0) return;
            // Merge into existing overlay so each agent step adds nodes
            // rather than replacing — the user sees the探索范围 grow.
            setOverlay((prev) => mergeSubgraph(prev, subgraph));
          })
          .catch(() => {
            /* expand 失败：保持现状，等下一帧 */
          });
      }
    }
  }, [turn.progressEvents]);

  const data = overlay ?? EMPTY_DATA;
  const empty = data.nodes.length === 0;

  return (
    <div className={cn("rounded border border-ink-line bg-surface-sunk overflow-hidden", className)}>
      <div className="px-3 py-1.5 text-[11px] uppercase tracking-[0.16em] text-ink-subtle font-mono border-b border-ink-line bg-surface-raised flex items-center justify-between">
        <span>Agent 探索过程 · 视角跟随 LLM</span>
        <span className="text-ink-muted normal-case tracking-normal font-sans text-[11px]">
          {accumulatedCount > 0 && `已发现 ${accumulatedCount} 个节点`}
        </span>
      </div>
      <div className="relative h-72">
        {empty ? (
          <div className="absolute inset-0 flex items-center justify-center text-[12px] text-ink-subtle">
            等待 GraphAgent 调用 graph_explore…
          </div>
        ) : (
          <GraphCanvas
            data={data}
            highlightIds={highlight}
            focusId={focusId}
            // 关掉 fit-to-bounds，由 focusId 单点 pan 维持镜头跟随观感。
            autoFitHighlight={false}
          />
        )}
      </div>
    </div>
  );
}

/**
 * 把新一跳的子图并入既有 overlay：节点/边按 id 去重。让画布随 agent 探索
 * 单调扩张，而不是被最新一跳完全覆盖。
 */
function mergeSubgraph(prev: SubgraphData | null, next: SubgraphData): SubgraphData {
  if (!prev) return next;
  const nodeMap = new Map<string, GraphNode>();
  for (const n of prev.nodes) nodeMap.set(n.id, n);
  for (const n of next.nodes) if (!nodeMap.has(n.id)) nodeMap.set(n.id, n);

  const edgeKey = (e: GraphEdge) => `${e.source}->${e.target}`;
  const edgeMap = new Map<string, GraphEdge>();
  for (const e of prev.edges) edgeMap.set(edgeKey(e), e);
  for (const e of next.edges) if (!edgeMap.has(edgeKey(e))) edgeMap.set(edgeKey(e), e);

  return {
    nodes: [...nodeMap.values()],
    edges: [...edgeMap.values()],
  };
}
