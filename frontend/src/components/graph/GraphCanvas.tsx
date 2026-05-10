import { useEffect, useMemo, useRef } from "react";
import { Graph, type GraphData } from "@antv/g6";

import { cn } from "@/lib/utils";

/**
 * GraphCanvas —— G6 v5 的极薄包装。
 *
 * 设计原则：
 *  - 只负责画布生命周期（mount / setData / unmount）+ 把外部回调
 *    挂到 G6 的 pointerenter / pointerleave / dblclick 事件上。
 *  - 配置（节点大小 / 标签 / 类型 filter / 力导布局参数）都通过 prop
 *    传入 —— 让父组件（GraphPage 的右侧面板）拥有所有 UI 状态。
 *  - 支持 highlight（agent 模式跟踪用）：传入一组 nodeId，画布会
 *    把它们标 active 状态、并自动 fit 视角。
 *
 * 节点视觉：按 vertex_type 着色（entity / passage / sentence），半径
 * 按 prop 传入的 nodeSize（暴露给配置面板调）。
 *
 * **交互正确性 (vis-network 参考)**：
 *  - drag-canvas 显式 `enable: targetType==="canvas"`，不抢节点 mousedown，
 *    避免 drag-canvas 默认 `enable:true` 吞掉 drag-element 的 pointer
 *    capture（这是 G6 v5 的默认坑）。
 *  - hover 不修改任何 element data；高亮通过 `node.state.hover` 声明式
 *    样式 + `setElementState(id, 'hover')` 实现，永不调用 updateNodeData。
 *  - 力导布局首次稳定后 stopLayout，drag 期间不再全局 re-tick。
 *  - data effect 用 (nodes_ids, edge_ids) 浅 hash 短路，相同 topology 的
 *    re-ref 不再 setData → 不再触发 layout 重启 → drag 不被打断。
 *  - highlight effect 不再依赖 data，避免 data 变更连带触发 highlight
 *    全量 setElementState（之前 O(N) per render，明显卡顿）。
 */

export interface GraphNode {
  id: string;
  label: string;
  vertex_type: string;          // "entity" | "passage" | "sentence" | "unknown"
  hop?: number;
  score?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight?: number;
  type?: string;
}

export interface GraphConfig {
  nodeSize: number;             // 8..32
  showLabels: boolean;
  hideTypes: string[];          // 隐藏的 vertex_type
  layoutForce: number;          // -150..-30 (越负斥力越大)
}

// 配置常量与组件同文件 — 移到 sibling 文件没价值（只一个 caller GraphPage）。
// Vite Fast Refresh 在 dev 模式下会因这条多导出而 invalidate 整个模块；
// HMR 退化但不影响功能，比拆文件成本低。
// eslint-disable-next-line react-refresh/only-export-components
export const DEFAULT_GRAPH_CONFIG: GraphConfig = {
  nodeSize: 16,
  showLabels: true,
  hideTypes: [],
  layoutForce: -60,
};

const TYPE_COLOR: Record<string, string> = {
  entity: "#1d4ed8",            // primary-700 — 实体最显眼
  passage: "#0d9488",           // teal-600 — 段落
  sentence: "#a16207",          // amber-700 — 句子
  unknown: "#6b7280",           // gray-500
};

const HIGHLIGHT_COLOR = "#dc2626"; // danger-600 — agent 联动节点
const HOVER_COLOR = "#0ea5e9";    // sky-500 — 悬停反馈，比红色高亮温和

interface GraphCanvasProps {
  data: { nodes: GraphNode[]; edges: GraphEdge[] };
  config?: Partial<GraphConfig>;
  /** 高亮的 nodeId 集合 —— agent / PPR 模式下跟随用。 */
  highlightIds?: string[];
  /** 高亮变化时是否自动把视角 fit 到这些节点上。默认 true。 */
  autoFitHighlight?: boolean;
  /**
   * 单点视角跟随。每次新值会把镜头平移到该节点（保持当前缩放），
   * 实现 "agent 走到哪、视角跟到哪" 的效果。与 highlightIds 解耦：
   * highlight 是已发现的全部节点，focusId 是最新一帧的关注点。
   * 若节点尚未挂到画布，悄悄跳过，下一轮 setData 时会自然命中。
   */
  focusId?: string | null;
  onHoverNode?: (nodeId: string | null) => void;
  /** edge 悬停。null 表示离开。payload 取自 G6 element id（``edge:src->tgt``）
   *  + 当前 data.edges 里的 raw edge（source/target/weight/type）。 */
  onHoverEdge?: (edge: { source: string; target: string; weight?: number; type?: string } | null) => void;
  onDoubleClickNode?: (nodeId: string) => void;
  className?: string;
}

export function GraphCanvas({
  data,
  config: configOverride,
  highlightIds,
  autoFitHighlight = true,
  focusId,
  onHoverNode,
  onHoverEdge,
  onDoubleClickNode,
  className,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<Graph | null>(null);
  // graph.render() 是 async。三个 effect（init / data / config）都要
  // render —— 不串行化的话 G6 在元素半挂载时被再次 render，会抛
  // "can't access property 'draw'". 共享一个 promise chain，下一次 render
  // 等上一次完成。
  //
  // ``stopLayout`` 跟在 render 之后：d3-force 默认会持续 simulation，
  // alphaDecay 慢，drag-element 把节点 pin 在拖到的位置后 simulation 仍在
  // tick → 节点视觉位置被持续重算，体验上"拖不动 / 拖完跳回"。每次
  // setData 后调一次 stopLayout 让模拟器停在当前快照，drag 后拖到哪就
  // 留在哪。下次 setData 重新启动 simulation 重排（这是预期：新数据需要
  // 重新布局）。
  const renderChainRef = useRef<Promise<void>>(Promise.resolve());
  const queueRender = (g: Graph): Promise<void> => {
    const next = renderChainRef.current
      .then(() => g.render())
      .then(() => {
        try {
          // G6 v5 公开 ``stopLayout`` 来终止当前 layout instance。typed
          // 为可选 thunk 因为不同小版本签名细节不一；空 catch 兜底。
          const stop = (g as unknown as { stopLayout?: () => void }).stopLayout;
          if (typeof stop === "function") stop.call(g);
        } catch {
          /* layout already stopped or stopLayout absent — non-fatal */
        }
      })
      .catch(() => {
        /* destroyed mid-render — destroy effect 兜底 */
      });
    renderChainRef.current = next;
    return next;
  };

  const config = useMemo<GraphConfig>(
    () => ({ ...DEFAULT_GRAPH_CONFIG, ...configOverride }),
    [configOverride],
  );

  // 把 props 里的 callback 用 ref 缓存，避免每次回调引用变更触发整个图重建
  const callbacksRef = useRef({ onHoverNode, onHoverEdge, onDoubleClickNode });
  useEffect(() => {
    callbacksRef.current = { onHoverNode, onHoverEdge, onDoubleClickNode };
  });
  // 把当前 data 也存 ref，edge hover 需要查 source/target/weight/type
  // 原始字段（G6 element id 只携带 `edge:src->tgt` 字符串）。
  const dataRef = useRef(data);
  useEffect(() => {
    dataRef.current = data;
  });
  // 当前 hover 节点 id —— 用 ref 而非 state，避免每次 hover 触发重渲染。
  // 只在 pointerenter/leave 时通过 setElementState 改 G6 内部状态。
  const hoveredRef = useRef<string | null>(null);

  // 初始化 / 销毁。仅当 container 出现 / 卸载时跑；data + config 通过
  // 单独的 effect 用 setData 更新，避免每次切换都 destroy 再造一次。
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const graph = new Graph({
      container,
      autoFit: "view",
      // animation:false 关掉 layout-tick 的逐帧动画。d3-force 配
      // animation:true 时即便 stopLayout 也会先把所有 tick 动画排队播完，
      // drag 期间体感"延迟回弹"。关掉后 layout 计算完直接落位，drag
      // 体验干净。
      animation: false,
      layout: {
        type: "d3-force",
        link: { distance: 60 },
        manyBody: { strength: config.layoutForce },
        collide: { radius: config.nodeSize + 4 },
      },
      // **关键修复**：G6 v5 的 drag-canvas 默认 `enable: true`，会吞掉
      // 节点的 mousedown 事件，drag-element 拿不到 pointer capture →
      // "鼠标在节点上按住拖不动"。显式给 drag-canvas 一个 target 过滤器，
      // 只在画布空白处生效。drag-element 用默认 enable（node + combo）。
      behaviors: [
        "zoom-canvas",
        {
          type: "drag-canvas",
          key: "drag-canvas",
          enable: (event: { targetType?: string }) => event.targetType === "canvas",
        },
        {
          type: "drag-element",
          key: "drag-element",
          // 拖动结束后保留位置（不让 d3-force 把它弹回原点）。dropEffect
          // 'move' 是默认值，显式声明留给读者参考。
          dropEffect: "move",
          // animation:false → drag 时无 ghost 动画延迟，所见即所得。
          animation: false,
        },
      ],
      node: {
        style: {
          // 节点尺寸 / 颜色按 data 字段动态映射
          size: config.nodeSize,
          fill: (d: { data?: { vertex_type?: string } }) =>
            TYPE_COLOR[d.data?.vertex_type ?? "unknown"] ?? TYPE_COLOR.unknown,
          stroke: "rgba(15, 23, 42, 0.18)",
          lineWidth: 1,
          labelText: config.showLabels
            ? (d: { data?: { label?: string } }) => truncate(d.data?.label ?? "", 18)
            : "",
          labelFontSize: 10,
          labelFill: "#1f2937",
          labelPlacement: "bottom",
          // hover 时鼠标变手型，提示可拖
          cursor: "grab",
        },
        state: {
          // hover：声明式样式，G6 在 setElementState('hover') 时直接重绘，
          // 不需要 React 介入。和 highlight 解耦 —— hover 永远是临时态。
          hover: {
            stroke: HOVER_COLOR,
            lineWidth: 2,
          },
          highlight: {
            fill: HIGHLIGHT_COLOR,
            stroke: HIGHLIGHT_COLOR,
            lineWidth: 2,
            haloLineWidth: 6,
            haloStroke: HIGHLIGHT_COLOR,
            haloStrokeOpacity: 0.25,
          },
          dim: {
            fill: "#cbd5e1",
            labelFill: "#94a3b8",
          },
        },
      },
      edge: {
        style: {
          stroke: "#cbd5e1",
          lineWidth: 1,
          endArrow: false,
        },
        state: {
          highlight: {
            stroke: HIGHLIGHT_COLOR,
            lineWidth: 1.6,
          },
        },
      },
    });

    // G6 v5 emits a generic IEvent base for both lifecycle and DOM
    // events; node-scoped events carry `target` with an `id`. Cast via
    // unknown to access the field without dragging in the full IEvent
    // hierarchy here (we only ever read .target.id).
    type NodeMouseEvent = { target?: { id?: string | number } };

    // pointerenter：声明式高亮 + 通知父组件（节点详情卡用）。setElement
    // State 在 G6 内部 mutation，不触发 React re-render，drag 安全。
    graph.on("node:pointerenter", (evt) => {
      const raw = (evt as unknown as NodeMouseEvent).target?.id;
      const id = raw != null ? String(raw) : null;
      if (!id) return;
      hoveredRef.current = id;
      try {
        graph.setElementState(id, "hover", true);
      } catch {
        /* element not on stage — ignore */
      }
      callbacksRef.current.onHoverNode?.(id);
    });
    graph.on("node:pointerleave", (evt) => {
      const raw = (evt as unknown as NodeMouseEvent).target?.id;
      const id = raw != null ? String(raw) : null;
      // 离开当前 hover 节点时清掉它的 hover state。注意 setElementState
      // 第二参可以是 falsy 字符串列表 → 这里用空数组覆盖到无 state。
      // 用 hoveredRef 兜底防止 leave 事件晚于 enter 到达另一节点。
      const target = id ?? hoveredRef.current;
      if (target) {
        try {
          graph.setElementState(target, [], false);
        } catch {
          /* element gone — ignore */
        }
      }
      hoveredRef.current = null;
      callbacksRef.current.onHoverNode?.(null);
    });
    graph.on("node:dblclick", (evt) => {
      const id = (evt as unknown as NodeMouseEvent).target?.id;
      if (id != null) callbacksRef.current.onDoubleClickNode?.(String(id));
    });

    // edge hover：bubble (source, target, weight, type) 给父组件渲染
    // tooltip。G6 element id 是 `edge:src->tgt`，从 dataRef 反查得到原始
    // edge 字段；找不到就回退仅给 source/target。
    graph.on("edge:pointerenter", (evt) => {
      const raw = (evt as unknown as NodeMouseEvent).target?.id;
      const id = raw != null ? String(raw) : null;
      if (!id || !id.startsWith("edge:")) return;
      const arrow = id.slice("edge:".length);
      const sepIdx = arrow.indexOf("->");
      if (sepIdx < 0) return;
      const source = arrow.slice(0, sepIdx);
      const target = arrow.slice(sepIdx + 2);
      const match = dataRef.current.edges.find(
        (e) => e.source === source && e.target === target,
      );
      callbacksRef.current.onHoverEdge?.({
        source,
        target,
        weight: match?.weight,
        type: match?.type,
      });
    });
    graph.on("edge:pointerleave", () => {
      callbacksRef.current.onHoverEdge?.(null);
    });

    graphRef.current = graph;

    return () => {
      // 等 in-flight render settle 再 destroy，避免在元素半挂载时拆毁
      // 触发 "context.element is undefined".
      const chain = renderChainRef.current;
      graphRef.current = null;
      void chain.finally(() => graph.destroy());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // **关键性能修复**：data prop 的 reference 经常因父组件 re-render 而
  // 变化（即便内容不变），如果直接用 [data] 作 dep 会每次都 setData →
  // re-layout → 用户 drag 中的节点跳回原位 + 整个图卡顿。这里用浅 hash
  // (节点 id 拼成字符串)，topology 真变化才执行 setData。隐藏类型 filter
  // 也参与 hash，因为它影响 rendered set。
  const dataKey = useMemo(() => {
    const hidden = config.hideTypes.length
      ? `|h=${config.hideTypes.slice().sort().join(",")}`
      : "";
    return (
      data.nodes.map((n) => n.id).join(",") +
      "||" +
      data.edges.map((e) => `${e.source}->${e.target}`).join(",") +
      hidden
    );
  }, [data, config.hideTypes]);

  useEffect(() => {
    const graph = graphRef.current;
    if (!graph) return;
    const filteredNodes = data.nodes.filter(
      (n) => !config.hideTypes.includes(n.vertex_type),
    );
    const visibleIds = new Set(filteredNodes.map((n) => n.id));
    const filteredEdges = data.edges.filter(
      (e) => visibleIds.has(e.source) && visibleIds.has(e.target),
    );

    const gd: GraphData = {
      nodes: filteredNodes.map((n) => ({
        id: n.id,
        data: {
          label: n.label,
          vertex_type: n.vertex_type,
          hop: n.hop,
          score: n.score,
        },
      })),
      edges: filteredEdges.map((e) => ({
        // 显式 id：highlight effect 用同一个 helper 拼 edge id 给
        // setElementState；G6 隐式 id 是 `source-target`，跟我们用的
        // `source->target` 不一致，会导致 edge highlight 静默失效。
        id: edgeId(e),
        source: e.source,
        target: e.target,
        data: { weight: e.weight, type: e.type },
      })),
    };
    graph.setData(gd);
    void queueRender(graph);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataKey]);

  // node 视觉配置（大小 / 标签 / 力导）变了 → 不重建，只 setOptions
  useEffect(() => {
    const graph = graphRef.current;
    if (!graph) return;
    graph.setOptions({
      node: {
        style: {
          size: config.nodeSize,
          labelText: config.showLabels
            ? (d: { data?: { label?: string } }) => truncate(d.data?.label ?? "", 18)
            : "",
        },
      },
      layout: {
        type: "d3-force",
        link: { distance: 60 },
        manyBody: { strength: config.layoutForce },
        collide: { radius: config.nodeSize + 4 },
      },
    });
    void queueRender(graph);
  }, [config.nodeSize, config.showLabels, config.layoutForce]);

  // 高亮联动：清掉之前的状态，把指定 ids 标 highlight，其余标 dim。
  // ids 为空（agent 模式 reset / PPR 失败）→ 还原默认。
  // **不依赖 data**：data 重新挂载时 highlight 会通过下一帧再次 apply
  // （G6 setElementState 调用对未存在节点是 no-op 安全的）。把 data 从
  // dep 拿掉避免 hover/缩放等导致 data ref 抖动时反复 O(N) 写状态。
  const highlightKey = useMemo(() => (highlightIds ?? []).join(","), [highlightIds]);
  useEffect(() => {
    const graph = graphRef.current;
    if (!graph) return;
    const ids = highlightIds ?? [];
    // 等 render 链 settle 再操作 element state —— G6 在 setData 还没把
    // element 挂上 stage 时调 setElementState 会找不到 element node。
    void renderChainRef.current.then(() => {
      if (graphRef.current !== graph) return; // destroyed
      if (ids.length === 0) {
        graph.setElementState({}, false);
        return;
      }
      const idSet = new Set(ids);
      const states: Record<string, string[]> = {};
      for (const node of data.nodes) {
        states[node.id] = idSet.has(node.id) ? ["highlight"] : ["dim"];
      }
      for (const edge of data.edges) {
        const bothIn = idSet.has(edge.source) && idSet.has(edge.target);
        states[edgeId(edge)] = bothIn ? ["highlight"] : ["dim"];
      }
      graph.setElementState(states, false);
      if (autoFitHighlight) {
        graph.focusElement(ids).catch(() => {
          graph.fitView().catch(() => {});
        });
      }
    });
    // 故意不放 data：见上方注释。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [highlightKey, autoFitHighlight]);

  // 视角跟随单点：用 translateTo 仅平移、不改 zoom，避免 focusElement
  // 那种 fit-to-bounds 行为把镜头拉远又拉近。空 focusId / 节点未上
  // stage 时跳过。依赖项里同时包含 dataKey：focusId 帧可能在节点 setData
  // 之前到（agent 流：先收到 graph_subgraph，再 expand 拉子图覆盖
  // overlay），第一次 getElementPosition 会 miss；data 更新触发本
  // effect 重跑就能命中刚挂上 stage 的节点。
  useEffect(() => {
    const graph = graphRef.current;
    if (!graph || !focusId) return;
    void renderChainRef.current.then(() => {
      if (graphRef.current !== graph) return; // destroyed
      try {
        const getPos = (graph as unknown as {
          getElementPosition?: (id: string) => [number, number] | undefined;
        }).getElementPosition;
        const pos = getPos?.call(graph, focusId);
        if (!pos) return;
        const translateTo = (graph as unknown as {
          translateTo?: (point: { x: number; y: number }, animate?: boolean) => void;
        }).translateTo;
        translateTo?.call(graph, { x: pos[0], y: pos[1] }, true);
      } catch {
        /* node still not on stage; next data update will retry */
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusId, dataKey]);

  return (
    <div
      ref={containerRef}
      className={cn("relative w-full h-full bg-surface-sunk", className)}
    />
  );
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n - 1) + "…";
}

function edgeId(e: { source: string; target: string }): string {
  return `edge:${e.source}->${e.target}`;
}
