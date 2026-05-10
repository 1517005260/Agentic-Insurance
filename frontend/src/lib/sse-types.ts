/**
 * SSE event payloads — typed mirror of `docs/webapp.md §5` and
 * `src/api/runners/events.py:EventType`.
 *
 * 设计选择：用 discriminated union（按 `event` 字段分支）而不是
 * 每个事件单独一个 callback。理由：
 *   1) 5 个 workbench + chat + 反欺诈共用同一个 `useSSE` hook；
 *      callback-per-event 会让 hook 签名爆炸。
 *   2) 我们要把所有事件按时间顺序压栈给 AgentTimeline / SSEDebugger
 *      做线性渲染，array<SSEEvent> 比 N 个 ref 更自然。
 *   3) discriminated union 让 TS 在 switch(event.event) 里自动收窄，
 *      下游消费者写 narrowing 不丢类型。
 *
 * 失败模式约定（docs §5）：
 *   - `done` 永远是最后一帧；client 收到 done 即可关流。
 *   - `error` 之后必有 `done`，所以 error 不是终止信号 —— 只是把
 *     reason 暴露出来，关闭由 done 兜底。
 *   - heartbeat 是 SSE comment frame `: keepalive`，**不** 触发事件。
 */

// ============================================================
// 通用
// ============================================================

export type SSEPhase =
  | "preprocess"
  | "retrieve"
  | "rerank"
  | "answering"
  | "thinking"
  | "force_final"
  | "tool";

export interface StatusEvent {
  event: "status";
  data: { phase: SSEPhase | string };
}

export interface ErrorEvent {
  event: "error";
  data: { message: string; type?: string };
}

export interface DoneEvent {
  event: "done";
  data: Record<string, never>;
}

// ============================================================
// RAG
// ============================================================

export type PreprocessStep = "hyde" | "rewrite";
export type PreprocessPhase = "start" | "done";
export type LocalRetrievalChannel =
  | "semantic"
  | "bm25"
  | "graph_ppr"
  | "regex";
/** Web RAG 的 retrieval 帧用 channel="web"，shape 跟本地 4 通道不同。 */
export type RetrievalChannel = LocalRetrievalChannel | "web";
export type RewriteLang = "zh" | "en" | "mixed";

export interface PreprocessEvent {
  event: "preprocess";
  data: {
    step: PreprocessStep;
    phase: PreprocessPhase;
    elapsed_ms?: number;
    // step=hyde, phase=done
    hyde_preview?: string;
    hyde_chars?: number;
    // step=rewrite, phase=done
    lang?: RewriteLang;
    rewrite?: string;
    regexes?: Array<{ pattern: string; weight: number; rationale?: string }>;
  };
}

/** 本地 4 通道（semantic / bm25 / graph_ppr / regex）。 */
export interface LocalRetrievalEvent {
  event: "retrieval";
  data: {
    channel: LocalRetrievalChannel;
    elapsed_ms: number;
    hits: Array<{ file_id: string; page_id: string; score: number }>;
  };
}

/** Web RAG 单通道 — Tavily 召回。后端 src/api/services/web_rag.py 发。 */
export interface WebRetrievalEvent {
  event: "retrieval";
  data: {
    channel: "web";
    n_results: number;
    sources: Array<{
      title?: string;
      url: string;
      content_preview?: string;
      score?: number;
    }>;
  };
}

export type RetrievalEvent = LocalRetrievalEvent | WebRetrievalEvent;

export interface RerankedEvent {
  event: "reranked";
  data: {
    elapsed_ms: number;
    pages: Array<{
      rank: number;
      file_id: string;
      page_id: string;
      page_number: number;
      score: number;
    }>;
  };
}

export interface TokenEvent {
  event: "token";
  data: { delta: string };
}

// ============================================================
// Agent (base / proof / graph / web shared)
// ============================================================

/**
 * Agent 的中间 reasoning content（LLM 在工具调用前/没调用工具时输出的
 * 文本）。BaseAgent / ProofAgent 在每次 LLM completion 后若 content
 * 非空就 emit 一次。前端在 ProgressTimeline 渲染成"思考"块。
 */
export interface ThoughtEvent {
  event: "thought";
  data: {
    loop: number;
    text: string;
  };
}

export interface ToolCallEvent {
  event: "tool_call";
  data: {
    loop: number;
    name: string;
    args: Record<string, unknown>;
  };
}

export interface ToolResultEvent {
  event: "tool_result";
  data: {
    loop: number;
    name: string;
    preview?: string;
    retrieved_tokens?: number;
    error?: string;
    /** Runner 标记：read / proof_scan = true，其它工具 = false。
     *  前端按 true 把它从一般 explore 步骤升级成证据 chip。
     *  后端见 `api/runners/agent_runner.py:_is_evidence`。 */
    is_evidence?: boolean;
    /** proof only */
    observation_id?: string;
    /** proof only */
    must_finalize_next?: boolean;
  };
}

// ============================================================
// Proof-only incremental events
// ============================================================

export interface ObligationEvent {
  event: "obligation";
  data: {
    id: string;
    kind: string;
    status: "OPEN" | "CLOSED" | "REMOVED";
    required: boolean;
    failure_kind?: string | null;
  };
}

export interface ClaimEvent {
  event: "claim";
  data: {
    id: string;
    kind: string; // ScanClaim / WitnessClaim / ...
    by?: string[]; // observation_ids
    status?: "REMOVED";
  };
}

export interface GapEvent {
  event: "gap";
  data: {
    id: string;
    kind: string;
    status: "ACTIVE" | "REMOVED";
  };
}

// ============================================================
// Graph-agent live replay
// ============================================================

/**
 * Graph 模式专用：runner 把 graph_explore 工具的 envelope 投影成
 * canvas 友好的形状。GraphPage 用它驱动 highlight / camera fit /
 * 增量 expand。后端见 `api/runners/agent_runner._project_graph_explore`。
 */
export interface GraphSubgraphEvent {
  event: "graph_subgraph";
  data: {
    loop: number | null;
    mode: "neighbors" | "ppr" | "entity_lookup";
    /** mode == "neighbors" only */
    seed_ids?: string[];
    entity_ids?: string[];
    page_refs?: { file_id: string; page_id: string }[];
    hops?: number;
    /** mode == "ppr" / "entity_lookup" */
    question?: string;
    /** mode == "ppr" only — seed surfaces, no hash_ids */
    seed_surfaces?: string[];
    /** mode == "entity_lookup" only */
    candidate_ids?: string[];
  };
}

// ============================================================
// 共有收尾（顺序固定 citations → final → done）
// ============================================================

/** 本地保单文档的引用 — RAG / Agent / Proof 共用。 */
export interface LocalCitation {
  sup: number;
  /** kind 字段缺省视为 "local"。 */
  kind?: "local";
  file_id: string;
  page_id: string;
  /** 后端 dataclass 是 Optional[int]；某些 Agent 反推出来的 evidence chip
   * 在 args 仅给 file_ids 时也无法填上，前端必须容忍 undefined。 */
  page_number?: number;
  page_preview?: string;
  /** proof only — 引用回 observation 而不是 page */
  observation_id?: string;
}

/** Web RAG / Web Agent 的网页引用，shape 跟本地完全不同。 */
export interface WebCitation {
  kind: "web";
  sup: number;
  title: string;
  url: string;
  snippet?: string;
  score?: number;
  published_date?: string | null;
}

export type CitationItem = LocalCitation | WebCitation;

export interface CitationsEvent {
  event: "citations";
  data: { items: CitationItem[] };
}

/**
 * 风险预测专属：后端 risk_predict_runner 把 PPR pre-pass 的 3 层
 * Sankey 邻接结构搬到 final.risk_subgraph 上。前端 RiskSankeyCanvas
 * 直接消费这个 shape。后端见
 * `api/runners/risk_predict_runner._build_risk_subgraph`.
 */
export interface RiskSubgraph {
  customer_fields: { id: string; label: string }[];
  risk_factors: { id: string; label: string; ppr_score: number }[];
  /** triggered_clauses 不带 sup — agent 实际 read 顺序（citations[].sup）
   *  与 PPR rank 是不同编号空间。前端按 (file_id, page_id) 在 citations
   *  中查匹配的 LocalCitation 来打开 CitationDrawer。 */
  triggered_clauses: {
    id: string;
    file_id: string;
    page_id: string;
    ppr_score?: number;
  }[];
  edges: { source: string; target: string; weight: number }[];
  mode: "ppr" | "no_seeds" | "no_graph" | string;
}

/**
 * `final` 是 "终态摘要"；payload shape 取决于 runner 类型。我们用
 * 宽松 union — Phase 2 SSE 联调阶段先收下整包 dict 给下游消费，
 * 等 ChatPage / workbench 真正消费时再按 runner 类型 narrow。
 */
export interface FinalEvent {
  event: "final";
  data: Record<string, unknown> & {
    // RAG
    answer_chars?: number;
    reranked_count?: number;
    channels_hit_counts?: Record<string, number>;
    timings_ms?: Record<string, number>;
    // Agent (base / graph / web)
    answer?: string;
    exit_reason?: string;
    loops?: number;
    total_cost?: number;
    // Agent (proof)
    decision?: "CERTIFIED" | "ABSTAIN";
    // Risk-predict (graph + PPR-anchored Sankey wrapper)
    flavor?: string;
    risk_subgraph?: RiskSubgraph;
    // Ingestion (GET /files/{id}/jobs/stream)
    file_id?: string;
    status?: "ready" | "failed" | string;
    page_count?: number | null;
    error?: string | null;
    job_status?: string;
    log_tail?: string | null;
    stages?: Array<{
      name: string;
      items: number;
      skipped_reason?: string | null;
    }>;
  };
}

// ============================================================
// Ingestion progress (GET /files/{id}/jobs/stream)
// ============================================================

/**
 * 一段 ingest 阶段（parse / page_assets / text_dense / vision_dense /
 * bm25 / graph，reingest 路径多一段 purge）。phase=start 没 elapsed_ms；
 * phase=done 必有 elapsed_ms（成功），失败时 error 字段非空。
 *
 * 后端 emit 点：src/pipeline/parse_and_index.py（parse / page_assets /
 * 4 builders）+ src/api/services/files.py（purge — reingest 专属）。
 */
export type IngestStageName =
  | "parse"
  | "page_assets"
  | "text_dense"
  | "vision_dense"
  | "bm25"
  | "graph"
  | "purge";

export interface StageEvent {
  event: "stage";
  data: {
    stage: IngestStageName | string;
    phase: "start" | "done";
    elapsed_ms?: number;
    items?: number;
    skipped_reason?: string | null;
    error?: string;
  };
}

// ============================================================
// Discriminated union
// ============================================================

export type SSEEvent =
  | StatusEvent
  | ErrorEvent
  | DoneEvent
  | PreprocessEvent
  | RetrievalEvent
  | RerankedEvent
  | TokenEvent
  | ThoughtEvent
  | ToolCallEvent
  | ToolResultEvent
  | ObligationEvent
  | ClaimEvent
  | GapEvent
  | GraphSubgraphEvent
  | CitationsEvent
  | FinalEvent
  | StageEvent;

export type SSEEventName = SSEEvent["event"];

/** Stream lifecycle state — surfaced by `useSSE` to drive UI. */
export type SSEStatus =
  | "idle"
  | "connecting"
  | "streaming"
  | "done"
  | "error"
  | "aborted";
