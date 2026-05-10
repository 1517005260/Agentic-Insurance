import { useQuery } from "@tanstack/react-query";

import { api } from "@/api/client";
import { GraphCanvas, type GraphEdge, type GraphNode } from "@/components/graph/GraphCanvas";

/**
 * /graph/_sandbox —— G6 v5 wiring 自检页。
 *
 * 单一职责：拉 /graph/sample 把 100 个节点画出来。任何画布层 bug
 * （类型错误 / 布局崩溃 / 缩放失灵）都会先在这里复现，避免跟
 * GraphPage 的双模式逻辑搅在一起。
 */
export default function GraphSandboxPage() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["graph", "sample"],
    queryFn: async () => {
      const { data } = await api.get<{ nodes: GraphNode[]; edges: GraphEdge[] }>(
        "/graph/sample?n=100",
      );
      return data;
    },
  });

  return (
    <div className="flex flex-col h-[calc(100vh-3.5rem)]">
      <header className="px-4 py-3 border-b border-ink-line bg-surface-raised">
        <h1 className="text-sm font-medium text-ink">Graph sandbox</h1>
        <p className="text-[12px] text-ink-subtle">
          /graph/sample 100 节点 + 诱导边 —— 验证 G6 v5 wiring。
        </p>
      </header>
      <div className="flex-1 relative">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center text-ink-muted">
            加载子图…
          </div>
        )}
        {isError && (
          <div className="absolute inset-0 flex items-center justify-center text-danger">
            加载失败：{(error as Error)?.message ?? "未知错误"}
          </div>
        )}
        {data && <GraphCanvas data={data} />}
      </div>
    </div>
  );
}
