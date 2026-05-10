import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Compass, Loader2, Network, Sparkles, FolderOpen } from "lucide-react";
import { isAxiosError } from "axios";
import { Link } from "react-router-dom";

import { api } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  DEFAULT_GRAPH_CONFIG,
  GraphCanvas,
  type GraphConfig,
  type GraphEdge,
  type GraphNode,
} from "@/components/graph/GraphCanvas";
import { MarkdownWithSup } from "@/components/chat/MarkdownWithSup";
import { ProgressTimeline } from "@/components/chat/ProgressTimeline";
import { useSSE } from "@/lib/sse";
import type { CitationItem } from "@/lib/sse-types";
import { cn } from "@/lib/utils";

// 两个 mode 共用一个 GraphCanvas，靠 overlay (mode 自己的子图) +
// highlight (跟随 agent 高亮) 驱动；不重建画布。模式切换不重抽
// sample 底图 —— 否则切 mode 时画布抖动。
//
// 反诈 PPR 模式已搬到 /risk 的"图谱风险发现"tab；本页只保留
// "自由探索 (manual)"+"Agent 联动"两 mode。

type GraphMode = "manual" | "agent";

interface SubgraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface HoverEdge {
  source: string;
  target: string;
  weight?: number;
  type?: string;
}

const EMPTY_SUBGRAPH: SubgraphData = { nodes: [], edges: [] };

export default function GraphPage() {
  // ------------------------------------------- 全局状态 ----
  const [mode, setMode] = useState<GraphMode>("manual");
  const [config, setConfig] = useState<GraphConfig>(DEFAULT_GRAPH_CONFIG);
  const [highlightIds, setHighlightIds] = useState<string[]>([]);
  const [hoverNodeId, setHoverNodeId] = useState<string | null>(null);
  const [hoverEdge, setHoverEdge] = useState<HoverEdge | null>(null);
  // mode-specific 子图叠加：自由探索的 expand 结果 / agent 模式下
  // 增量并入的节点。空 → 用 sample 底图。
  const [overlayData, setOverlayData] = useState<SubgraphData | null>(null);
  // agent 模式专用：每帧 graph_subgraph 选一个最新焦点 id 给画布
  // pan 视角；manual 模式保持 null = 不动镜头。
  const [focusId, setFocusId] = useState<string | null>(null);

  // ------------------------------------------- 底图 sample ----
  const sampleQuery = useQuery<SubgraphData>({
    queryKey: ["graph", "sample"],
    queryFn: async () => {
      const { data } = await api.get<SubgraphData>("/graph/sample?n=100");
      return data;
    },
    staleTime: Infinity, // 后端有 per-process cache，前端不必反复刷
  });

  const canvasData = overlayData ?? sampleQuery.data ?? EMPTY_SUBGRAPH;

  // 模式切换：清 overlay + highlight，回到底图。配置面板的 nodeSize
  // / showLabels 等不变（用户审美延续到下一个模式）。
  const switchMode = (next: GraphMode) => {
    if (next === mode) return;
    setMode(next);
    setOverlayData(null);
    setHighlightIds([]);
    setFocusId(null);
  };

  // ------------------------------------------- 节点 hover 详情 ----
  // 鼠标快速扫过节点不该触发 HTTP 风暴；debounce 200ms（vis-network
  // tooltipDelay 一致）后才让 useNodeDetail 拿到稳定 id 真正发请求。
  const debouncedHoverId = useDebounced(hoverNodeId, 200);
  const hoverDetail = useNodeDetail(debouncedHoverId);

  // ------------------------------------------- 节点 dblclick 扩邻居 ----
  // 静默 .catch 是早期的偷懒——用户实际遇到 /graph/expand 失败时 UI
  // 一动不动，根本无从排查。改成把错误打到 console.warn + 通过
  // ``setExpandError`` 在 hover card 区域显示一段红字。错误类型涵盖：
  //   (a) /graph/expand 返回 404（node_id 不在 graph._name_to_vidx）
  //   (b) 503（lifespan 还没初始化）
  //   (c) 200 但返回 nodes 为空（节点确实没邻居）
  const [expandError, setExpandError] = useState<string | null>(null);
  const onDoubleClickNode = useCallback(async (nodeId: string) => {
    setExpandError(null);
    try {
      const { data } = await api.get<SubgraphData>(
        `/graph/expand?node_id=${encodeURIComponent(nodeId)}&hops=1&top_k=30`,
      );
      if (!data.nodes.length) {
        setExpandError(`节点 ${nodeId.slice(-8)} 在图谱里没有邻居（可能是孤立节点）`);
        return;
      }
      setOverlayData(data);
      setHighlightIds([nodeId]);
    } catch (e) {
      const msg = (e as Error)?.message ?? String(e);
      // eslint-disable-next-line no-console
      console.warn("[graph] /graph/expand failed", { nodeId, error: e });
      setExpandError(`展开失败: ${msg}`);
    }
  }, []);

  // ------------------------------------------- 整体布局 ----
  return (
    <div className="flex h-[calc(100vh-3.5rem)]">
      {/* Canvas 70% */}
      <div className="flex-1 relative border-r border-ink-line">
        {sampleQuery.isLoading && (
          <div className="absolute inset-0 flex items-center justify-center text-ink-muted">
            <Loader2 className="h-4 w-4 animate-spin mr-2" /> 加载底图…
          </div>
        )}
        {sampleQuery.isError &&
          // 后端 /sample 现在对"未构建图谱"返回 200 + empty，所以这条
          // 分支只剩真正的网络/服务错误。503 兜底逻辑保留，万一管理员
          // 把 graph 服务关了仍然给友好提示。
          (() => {
            const err = sampleQuery.error;
            const status = isAxiosError(err) ? err.response?.status : undefined;
            if (status === 503) {
              return <EmptyGraphPlaceholder />;
            }
            return (
              <div className="absolute inset-0 flex items-center justify-center text-danger px-4 text-center">
                加载失败：{(err as Error)?.message ?? "未知错误"}
              </div>
            );
          })()}
        {/*
         * 200 + empty 的"还没 ingest"场景：sampleQuery 成功，但 nodes
         * 数组为空。这条占位放在 canvas 之上，引导用户去文件页。
         */}
        {!sampleQuery.isLoading &&
          !sampleQuery.isError &&
          (sampleQuery.data?.nodes?.length ?? 0) === 0 &&
          overlayData === null && <EmptyGraphPlaceholder />}
        <GraphCanvas
          data={canvasData}
          config={config}
          highlightIds={highlightIds}
          // agent 模式下关掉 fit-to-bounds（焦点已经由 focusId pan）；
          // manual 模式保持自动 fit，方便看清单点扩出的子图全貌。
          autoFitHighlight={mode !== "agent"}
          focusId={mode === "agent" ? focusId : null}
          onHoverNode={setHoverNodeId}
          onHoverEdge={setHoverEdge}
          onDoubleClickNode={onDoubleClickNode}
        />
        <Legend />
      </div>

      {/* 右侧 30% */}
      <aside className="w-[420px] shrink-0 flex flex-col bg-surface-raised">
        <ModeSwitcher mode={mode} onChange={switchMode} />
        <div className="flex-1 overflow-y-auto scrollbar-thin">
          {mode === "manual" && (
            <ManualModePanel
              setOverlay={setOverlayData}
              setHighlight={setHighlightIds}
            />
          )}
          {mode === "agent" && (
            <AgentModePanel
              setOverlay={setOverlayData}
              setHighlight={setHighlightIds}
              setFocus={setFocusId}
            />
          )}
        </div>
        {expandError && (
          <div
            role="alert"
            className="mx-3 mb-2 rounded border border-danger/30 bg-danger/5 px-2 py-1.5 text-[12px] text-danger"
          >
            {expandError}
          </div>
        )}
        <HoverCard hoverId={hoverNodeId} detail={hoverDetail} hoverEdge={hoverEdge} />
        <ConfigPanel config={config} onChange={setConfig} />
      </aside>
    </div>
  );
}

// ============================================================ ModeSwitcher

const MODES: { id: GraphMode; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: "manual", label: "自由探索", icon: Compass },
  { id: "agent", label: "Agent 联动", icon: Sparkles },
];

function ModeSwitcher({
  mode,
  onChange,
}: {
  mode: GraphMode;
  onChange: (m: GraphMode) => void;
}) {
  return (
    <div className="flex border-b border-ink-line">
      {MODES.map((m) => {
        const active = m.id === mode;
        const Icon = m.icon;
        return (
          <button
            key={m.id}
            type="button"
            onClick={() => onChange(m.id)}
            className={cn(
              "flex-1 flex items-center justify-center gap-1.5 py-2 text-[12px] transition-colors",
              active
                ? "bg-primary-50 text-primary-700 font-medium border-b-2 border-primary-600 -mb-px"
                : "text-ink-muted hover:text-ink hover:bg-surface-sunk/40",
            )}
          >
            <Icon className="h-3.5 w-3.5" />
            <span>{m.label}</span>
          </button>
        );
      })}
    </div>
  );
}

// ============================================================ Mode A: Manual

interface SeedHit {
  hash_id: string;
  surface: string;
  similarity: number;
}

function EmptyGraphPlaceholder() {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center text-ink-muted px-6 text-center gap-3 pointer-events-auto bg-surface/60 backdrop-blur-sm">
      <Network className="h-8 w-8 text-ink-line" />
      <div className="text-[14px] text-ink">尚未构建知识图谱</div>
      <div className="text-[12px] text-ink-subtle max-w-xs">
        需要至少上传并解析一份 PDF 后，图谱底图才会就绪。
      </div>
      <Link
        to="/files"
        className="inline-flex items-center gap-1.5 mt-2 rounded-md bg-primary-600 text-white px-3 py-1.5 text-[12px] hover:bg-primary-700"
      >
        <FolderOpen className="h-3.5 w-3.5" /> 去上传文件
      </Link>
    </div>
  );
}

function ManualModePanel({
  setOverlay,
  setHighlight,
}: {
  setOverlay: (d: SubgraphData | null) => void;
  setHighlight: (ids: string[]) => void;
}) {
  const [q, setQ] = useState("");
  const [hops, setHops] = useState(1);
  const [seeds, setSeeds] = useState<SeedHit[] | null>(null);

  const seedMutation = useMutation({
    mutationFn: async (query: string) => {
      const { data } = await api.get<SeedHit[]>(
        `/graph/seed?q=${encodeURIComponent(query)}&top_k=10`,
      );
      return data;
    },
    onSuccess: (data) => setSeeds(data),
  });

  const expandMutation = useMutation({
    mutationFn: async (nodeId: string) => {
      // top_k=10 matches the "返回 top10 的联通" intuition the user
      // expressed; 50 was buried in the previous default and produced
      // a noisy fan-out the user couldn't visually parse. The hops
      // selector still tunes BFS depth.
      const { data } = await api.get<SubgraphData>(
        `/graph/expand?node_id=${encodeURIComponent(nodeId)}&hops=${hops}&top_k=10`,
      );
      return { nodeId, data };
    },
    onSuccess: ({ nodeId, data }) => {
      if (data.nodes.length === 0) {
        // Surface as console + visible: previously a silent no-op
        // and the user thought "图就是不动".
        // eslint-disable-next-line no-console
        console.warn("[graph-manual] /graph/expand returned empty", { nodeId });
        return;
      }
      setOverlay(data);
      setHighlight([nodeId]);
    },
    onError: (e) => {
      // eslint-disable-next-line no-console
      console.warn("[graph-manual] /graph/expand failed", e);
    },
  });

  return (
    <div className="p-3 space-y-3">
      <div>
        <div className="text-[11px] uppercase tracking-[0.14em] text-ink-subtle font-mono mb-1">
          实体搜索
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (q.trim()) seedMutation.mutate(q.trim());
          }}
          className="flex gap-1.5"
        >
          <Input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="输入实体名称（如 AXA、保费回赠）"
          />
          <Button
            type="submit"
            size="md"
            disabled={!q.trim() || seedMutation.isPending}
          >
            {seedMutation.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "搜索"}
          </Button>
        </form>
        {seedMutation.isError && (
          <div className="text-[11px] text-danger mt-1">
            搜索失败：{(seedMutation.error as Error)?.message}
          </div>
        )}
      </div>

      {seeds && seeds.length === 0 && (
        <div className="text-[12px] text-ink-subtle">
          未匹配到任何实体；尝试用 surface form（出现在原文中的写法）。
        </div>
      )}

      {seeds && seeds.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="text-[11px] uppercase tracking-[0.14em] text-ink-subtle font-mono">
              点击展开（hops=
              <select
                className="bg-transparent border-none text-[11px] mx-0.5"
                value={hops}
                onChange={(e) => setHops(Number(e.target.value))}
              >
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={3}>3</option>
              </select>
              ）
            </div>
            <button
              type="button"
              className="text-[11px] text-ink-subtle hover:text-ink"
              onClick={() => {
                setOverlay(null);
                setHighlight([]);
              }}
            >
              还原底图
            </button>
          </div>
          <ul className="space-y-1">
            {seeds.map((s) => (
              <li key={s.hash_id}>
                <button
                  type="button"
                  onClick={() => expandMutation.mutate(s.hash_id)}
                  className="w-full text-left px-2 py-1.5 rounded border border-ink-line bg-surface-raised hover:border-primary-300 hover:bg-primary-50/40 transition-colors"
                >
                  <div className="text-[13px] text-ink truncate">{s.surface}</div>
                  <div className="text-[11px] text-ink-subtle font-mono">
                    sim={s.similarity.toFixed(3)}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ============================================================ Mode B: Agent

function AgentModePanel({
  setOverlay,
  setHighlight,
  setFocus,
}: {
  setOverlay: (d: SubgraphData | null) => void;
  setHighlight: (ids: string[]) => void;
  setFocus: (id: string | null) => void;
}) {
  const [query, setQuery] = useState("");
  // 重置 turnKey 让 ProgressTimeline 折叠状态跟着新一轮探索归零
  const [turnKey, setTurnKey] = useState(() => `t-${Date.now()}`);
  // Token 帧不再进 sse.events（dropTokenFrames=true）；自行用 ref +
  // rAF flush 维护 streaming answer，避免长答案下 useMemo over
  // sse.events 的 O(N²) 退化（每来一个 token 都把整数组 filter+join 一遍）。
  const answerRef = useRef("");
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const answerFlushPendingRef = useRef(false);
  // 保存 in-flight rAF id，submit/reset/unmount 时 cancel —— 否则
  // 一个晚到的 flush 会把已经清空的 answerRef("") 写进 state，看起来
  // 像 UI 把答案"吞掉"了；组件 unmount 后写入还会触发 React 警告。
  const answerRafIdRef = useRef<number | null>(null);

  const sse = useSSE({
    dropTokenFrames: true,
    onEvent: (ev) => {
      if (ev.event !== "token") return;
      const delta = (ev.data as { delta?: string })?.delta;
      if (!delta) return;
      answerRef.current += delta;
      if (answerFlushPendingRef.current) return;
      answerFlushPendingRef.current = true;
      // rAF flush + 微批合并：token 速率高于 60Hz 时多次 delta 合到一帧 setState。
      answerRafIdRef.current = requestAnimationFrame(() => {
        answerRafIdRef.current = null;
        answerFlushPendingRef.current = false;
        setStreamingAnswer(answerRef.current);
      });
    },
  });

  // 卸载时取消 pending rAF，防止"已 unmount 还 setState" 警告。
  useEffect(() => {
    return () => {
      if (answerRafIdRef.current !== null) {
        cancelAnimationFrame(answerRafIdRef.current);
        answerRafIdRef.current = null;
        answerFlushPendingRef.current = false;
      }
    };
  }, []);
  const busy = sse.status === "connecting" || sse.status === "streaming";

  // 累积所有 highlight ids（agent 每次 graph_explore 调用都加一组）
  const accumulatedRef = useRef<Set<string>>(new Set());
  // 每个新 turn 单调递增；用来给 in-flight /graph/expand 验证响应
  // 是否还属于当前 turn —— 不属于就丢弃，避免 reset / 切 mode 后
  // 一个迟到的 subgraph 把画布盖回去。
  const turnSeqRef = useRef(0);
  // 已处理的 graph_subgraph 帧数；之前 token 帧会污染 sse.events，
  // 现在已用 dropTokenFrames 拦截，但保留计数器避免重复 expand。
  const processedSubgraphCountRef = useRef(0);

  useEffect(() => {
    const subgraphEvts = sse.events.filter(
      (e) => e.event === "graph_subgraph",
    );
    if (subgraphEvts.length <= processedSubgraphCountRef.current) return;
    const newEvts = subgraphEvts.slice(processedSubgraphCountRef.current);
    processedSubgraphCountRef.current = subgraphEvts.length;

    let mutated = false;
    let focusId: string | undefined;
    for (const ev of newEvts) {
      const data = ev.data as {
        mode: string;
        seed_ids?: string[];
        entity_ids?: string[];
        candidate_ids?: string[];
      };
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
      // 视角焦点取每帧第一个可用 id；最后一帧覆盖前面的 —— agent
      // 越往后走的 expand 越具体，应该聚焦在最新一跳。
      const next = data.seed_ids?.[0] ?? data.candidate_ids?.[0] ?? data.entity_ids?.[0];
      if (next) focusId = next;
    }
    if (mutated) setHighlight([...accumulatedRef.current]);

    if (focusId) {
      // 把最新焦点同步给画布，触发 pan 视角（保持当前 zoom）。
      setFocus(focusId);
      const myTurn = turnSeqRef.current;
      api
        .get<SubgraphData>(
          `/graph/expand?node_id=${encodeURIComponent(focusId)}&hops=1&top_k=30`,
        )
        .then(({ data: subgraph }) => {
          if (turnSeqRef.current !== myTurn) return;
          if (subgraph.nodes.length === 0) {
            // 静默失败让用户以为"图就是不动"——至少打印
            // eslint-disable-next-line no-console
            console.warn(
              "[graph-agent] /graph/expand returned empty",
              { focusId },
            );
            return;
          }
          setOverlay(subgraph);
        })
        .catch((e) => {
          // eslint-disable-next-line no-console
          console.warn("[graph-agent] /graph/expand failed", { focusId, error: e });
        });
    }
  }, [sse.events, setHighlight, setOverlay, setFocus]);

  const finalAnswer = (() => {
    const f = sse.events.find((e) => e.event === "final");
    if (!f) return streamingAnswer;
    const a = (f.data as { answer?: string }).answer;
    return a || streamingAnswer;
  })();
  const citations = useMemo(() => {
    const ev = sse.events.find((e) => e.event === "citations");
    return (ev?.data as { items?: CitationItem[] } | undefined)?.items ?? [];
  }, [sse.events]);

  // 清掉 in-flight rAF + answer ref/state；submit + reset 共用，避免
  // 旧轮的尾 flush 把上一次答案末尾或"清空状态"写到新轮里。
  const cancelPendingFlush = () => {
    if (answerRafIdRef.current !== null) {
      cancelAnimationFrame(answerRafIdRef.current);
      answerRafIdRef.current = null;
    }
    answerFlushPendingRef.current = false;
    answerRef.current = "";
    setStreamingAnswer("");
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    accumulatedRef.current = new Set();
    processedSubgraphCountRef.current = 0;
    turnSeqRef.current += 1;
    cancelPendingFlush();
    setFocus(null);
    setTurnKey(`t-${Date.now()}`);
    sse.start("/agent/stream", { query: q, kind: "graph" });
  };

  const reset = () => {
    sse.reset();
    accumulatedRef.current = new Set();
    processedSubgraphCountRef.current = 0;
    turnSeqRef.current += 1;
    cancelPendingFlush();
    setOverlay(null);
    setHighlight([]);
    setFocus(null);
    setTurnKey(`t-${Date.now()}`);
  };

  return (
    <div className="p-3 space-y-3">
      <form onSubmit={submit} className="space-y-2">
        <textarea
          rows={3}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="提一个需要顺着图谱走 2-3 跳才能答的问题。"
          className="w-full text-sm rounded border border-ink-line bg-surface-raised px-2 py-1.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30"
        />
        <div className="flex gap-1.5">
          <Button type="submit" size="md" disabled={!query.trim() || busy}>
            {busy ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> 探索中…
              </>
            ) : (
              "开始探索"
            )}
          </Button>
          {busy && (
            <Button type="button" variant="ghost" size="md" onClick={() => sse.abort()}>
              中止
            </Button>
          )}
          <Button type="button" variant="ghost" size="md" onClick={reset}>
            重置
          </Button>
        </div>
      </form>

      <ProgressTimeline
        events={sse.events}
        status={sse.status === "idle" ? "done" : sse.status}
        autoCollapsed={!!finalAnswer}
        turnKey={turnKey}
      />

      {finalAnswer && (
        <div className="rounded border border-ink-line bg-surface-raised p-2 prose prose-sm max-w-none">
          <MarkdownWithSup content={finalAnswer} citations={citations} />
        </div>
      )}
    </div>
  );
}

// ============================================================ Side cards

interface NodeDetail {
  surface: string;
  vertex_type: string;
  degree: number;
  mention_count?: number;
  /** Each entry now carries display_name (file table 真实标题), not raw file_id. */
  neighboring_files?: { file_id: string; display_name: string }[];
  file_id?: string;
  display_name?: string;
  page_number?: number;
}

function useNodeDetail(nodeId: string | null) {
  return useQuery({
    queryKey: ["graph", "node", nodeId],
    queryFn: async (): Promise<NodeDetail> => {
      const { data } = await api.get<NodeDetail>(
        `/graph/nodes/${encodeURIComponent(nodeId ?? "")}`,
      );
      return data;
    },
    enabled: !!nodeId,
    staleTime: 60_000,
  });
}

/** 把 value 延迟 ms 毫秒返回；中途变化会被合并为最后一个稳定值。
 *  典型用途：hover 节点详情 fetch debounce，避免鼠标快速扫过触发
 *  逐节点 HTTP。 */
function useDebounced<T>(value: T, ms: number): T {
  const [out, setOut] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setOut(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return out;
}

function HoverCard({
  hoverId,
  detail,
  hoverEdge,
}: {
  hoverId: string | null;
  detail: ReturnType<typeof useNodeDetail>;
  hoverEdge: HoverEdge | null;
}) {
  // 优先级：节点 hover 在边 hover 之上 —— G6 v5 在 hover node 时不会同时
  // emit edge:pointerenter（除非 cursor 真的离开节点），所以二者基本互
  // 斥，但同时有值时优先展示节点（用户主要关注节点）。
  const showEdge = !hoverId && hoverEdge;
  return (
    <div className="border-t border-ink-line p-3 min-h-[88px]">
      <div className="text-[11px] uppercase tracking-[0.14em] text-ink-subtle font-mono mb-1">
        {showEdge ? "边详情" : "节点详情"}
      </div>
      {!hoverId && !hoverEdge && (
        <div className="text-[12px] text-ink-subtle">
          鼠标悬停节点 / 边查看详情；双击节点展开 1 跳邻居。
        </div>
      )}
      {hoverId && detail.isLoading && (
        <div className="text-[12px] text-ink-muted">加载中…</div>
      )}
      {hoverId && detail.data && (
        <div className="space-y-1 text-[12px]">
          <div className="text-ink truncate">{detail.data.surface}</div>
          <div className="text-ink-subtle font-mono">
            type={detail.data.vertex_type} · degree={detail.data.degree}
            {detail.data.mention_count != null && ` · mentions=${detail.data.mention_count}`}
          </div>
          {detail.data.neighboring_files && detail.data.neighboring_files.length > 0 && (
            <div className="text-ink-subtle truncate" title={detail.data.neighboring_files.map((f) => f.display_name).join(" / ")}>
              出现在: {detail.data.neighboring_files.slice(0, 3).map((f) => f.display_name).join(", ")}
            </div>
          )}
          {detail.data.file_id && (
            <div className="text-ink-subtle truncate" title={detail.data.display_name ?? detail.data.file_id}>
              {detail.data.display_name ?? detail.data.file_id}
              {detail.data.page_number != null && ` · p${detail.data.page_number}`}
            </div>
          )}
        </div>
      )}
      {showEdge && (
        <div className="space-y-1 text-[12px]">
          <div className="text-ink-subtle font-mono break-all">
            {/* hash_id 通常长且无意义；只显示后 10 位作为视觉锚点。 */}
            {shortenHashTail(hoverEdge.source)}
            <span className="mx-1 text-ink-muted">→</span>
            {shortenHashTail(hoverEdge.target)}
          </div>
          <div className="text-ink-subtle font-mono">
            {hoverEdge.type ? `type=${hoverEdge.type}` : "type=(未标注)"}
            {hoverEdge.weight != null && ` · weight=${hoverEdge.weight.toFixed(3)}`}
          </div>
        </div>
      )}
    </div>
  );
}

function shortenHashTail(s: string): string {
  // entity-1a28… / passage-… 都是 hash_id；尾 10 位一般足以让用户在多
  // 节点 hover 时快速分辨是不是同一节点；超过 28 字符再截断。
  if (s.length <= 28) return s;
  return `…${s.slice(-10)}`;
}

// ============================================================ Config panel

function ConfigPanel({
  config,
  onChange,
}: {
  config: GraphConfig;
  onChange: (c: GraphConfig) => void;
}) {
  return (
    <div className="border-t border-ink-line p-3 space-y-2 bg-surface-sunk/40">
      <div className="text-[11px] uppercase tracking-[0.14em] text-ink-subtle font-mono">
        画布配置
      </div>
      <Slider
        label="节点大小"
        value={config.nodeSize}
        min={8}
        max={32}
        onChange={(v) => onChange({ ...config, nodeSize: v })}
      />
      <Slider
        label="斥力 (越负越散)"
        value={config.layoutForce}
        min={-150}
        max={-30}
        onChange={(v) => onChange({ ...config, layoutForce: v })}
      />
      <label className="flex items-center gap-2 text-[12px] text-ink">
        <input
          type="checkbox"
          checked={config.showLabels}
          onChange={(e) => onChange({ ...config, showLabels: e.target.checked })}
        />
        显示标签
      </label>
      <div className="flex gap-2 text-[12px] text-ink">
        {(["entity", "passage", "sentence"] as const).map((t) => (
          <label key={t} className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={!config.hideTypes.includes(t)}
              onChange={(e) => {
                const next = e.target.checked
                  ? config.hideTypes.filter((x) => x !== t)
                  : [...config.hideTypes, t];
                onChange({ ...config, hideTypes: next });
              }}
            />
            {t}
          </label>
        ))}
      </div>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between text-[12px] text-ink">
        <span>{label}</span>
        <span className="font-mono text-ink-subtle">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  );
}

// ============================================================ Legend

function Legend() {
  return (
    <div className="absolute top-3 left-3 rounded-md border border-ink-line bg-surface-raised/95 backdrop-blur px-3 py-2 text-[11px] flex items-center gap-3 shadow-sm">
      <Network className="h-3 w-3 text-ink-muted" />
      <LegendDot color="#1d4ed8" label="entity" />
      <LegendDot color="#0d9488" label="passage" />
      <LegendDot color="#a16207" label="sentence" />
      <LegendDot color="#dc2626" label="active" />
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1 text-ink">
      <span
        aria-hidden
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      {label}
    </span>
  );
}
