# Agentic Insurance Console — 前端

Vite + React 19 + TypeScript + Tailwind 3 单页应用。挂在后端 `:8000` 上，反代 `/api/*` 走 `vite.config.ts`。

## 启动

### 一次性

```bash
pnpm install                     # Node 22+ / pnpm 10
```

### 开发模式

```bash
pnpm dev                         # http://localhost:5173，HMR + proxy → :8000
```

后端必须在跑（`PYTHONPATH=src uv run uvicorn api.main:app --port 8000`），否则前端能起但任何 API 请求 502。

### 类型检查 / lint / 构建

```bash
pnpm exec tsc -p tsconfig.app.json     # 类型检查（CI 用）
pnpm exec eslint src/                   # lint（含 react-hooks rules）
pnpm build                              # tsc -b && vite build → dist/
pnpm preview                            # 预览 build 产物
```

### OpenAPI 类型 codegen

后端跑着时：

```bash
pnpm gen:api
# → src/api/types.gen.ts （3000+ 行，paths + components）
# 用法：
#   import type { components } from "@/api/types.gen";
#   type FileOut = components["schemas"]["FileOut"];
```

`src/api/types.ts` 是手写 thin surface 给 auth bootstrap 用，新组件优先从 `types.gen.ts` 拉。

## 模块布局

```
frontend/src/
├── App.tsx                  # react-router 路由 + RequireAuth/RequireAdmin 守卫
├── main.tsx                 # ReactDOM.createRoot
├── index.css                # Tailwind 入口 + 全局 token
│
├── api/                     # axios client + 类型
│   ├── client.ts            # axios 实例 + JWT interceptor + 401 stale-session 守卫
│   ├── auth.ts              # /auth/login + /auth/me
│   ├── types.ts             # 手写 thin DTO（auth bootstrap 用）
│   └── types.gen.ts         # openapi-typescript 自动生成
│
├── lib/                     # 框架级公共
│   ├── queryClient.ts       # @tanstack/react-query QueryClient
│   ├── sse.ts               # fetch + ReadableStream 解析；useSSE hook（POST 流）+ fetchSSE async generator（任意方法）
│   ├── sse-types.ts         # SSEEvent 一坨 discriminated union
│   └── utils.ts             # cn（clsx + tailwind-merge）
│
├── stores/                  # zustand
│   ├── auth.ts              # token + user + verified flag；persist localStorage
│   ├── citation.ts          # CitationDrawer 全站通道
│   ├── session.ts           # 当前 chat session id（Phase 6 多轮）
│   └── config.ts            # /admin/config react-query 钩子
│
├── hooks/
│   ├── useFiles.ts          # GET /files；useReadyFiles 过滤 status='ready'
│   └── useSessions.ts       # /chat/sessions list/detail/create/rename/delete
│
├── components/
│   ├── ui/                  # button / input / card 极简 primitives
│   ├── auth/guards.tsx      # RequireAuth / RequireAdmin
│   ├── layout/              # Sidebar + Topbar + LayoutShell
│   ├── chat/
│   │   ├── ChatComposer.tsx       # 输入框 + web/agent toggle
│   │   ├── MessageList.tsx        # auto-scroll + sticky
│   │   ├── UserBubble.tsx
│   │   ├── AssistantTurn.tsx      # 答案 + Markdown sup 引用 + evidence chip
│   │   ├── ProgressTimeline.tsx   # tool_call/tool_result FIFO + thought 折叠
│   │   ├── MarkdownWithSup.tsx    # mdast plugin 把 [^k] 转成可点 sup
│   │   ├── evidence.ts            # is_evidence=true 反推 chip
│   │   ├── SessionSidebar.tsx     # 多轮会话列表
│   │   └── types.ts               # Turn / UserTurn / AssistantTurn
│   ├── citation/
│   │   ├── CitationDrawer.tsx     # 右侧抽屉 + react-pdf 跳页 + ←→/Esc
│   │   └── PdfPageViewer.tsx      # axios+JWT → blob → react-pdf
│   ├── graph/GraphCanvas.tsx      # G6 v5 wrapper
│   ├── files/
│   │   ├── UploadDialog.tsx       # admin 上传（select → uploading → progress → done）
│   │   └── IngestProgress.tsx     # 5 stage 时间线（订阅 SSE）
│   ├── workbench/                 # 6 工作台共享脚手架
│   └── forms/                     # FormPrimitives + ChipsField + FileMultiSelect
│
└── pages/
    ├── LoginPage.tsx
    ├── ChatPage.tsx                 # 4 端点 + 侧栏（多轮）
    ├── FilesPage.tsx                # 文件浏览（admin 多上传 / ⋯ 菜单 / IngestProgress 抽屉）
    ├── GraphPage.tsx                # 知识图谱（手动 / 反欺诈 PPR / Agent 联动）
    ├── GraphSandboxPage.tsx         # G6 v5 调试沙盒
    ├── workbench/
    │   ├── SearchPage.tsx           # 条款检索（3 粒度 × 4 通道 + RRF + rerank）
    │   ├── ComparePage.tsx          # 多产品对比 N×M
    │   ├── ExclusionPage.tsx        # 免责审查（ProofAgent forall）
    │   ├── ClaimCheckPage.tsx       # 理赔判定（三栏）
    │   ├── RecommendPage.tsx        # 产品推荐 top-3
    │   ├── PolicyCalcPage.tsx       # 保单精算（code_run）
    │   └── RegulationPage.tsx       # 法规速查（Tavily + LLM）
    └── admin/
        ├── UsersPage.tsx            # 用户管理 + 自防护 + 末位 admin 防护
        ├── ConfigPage.tsx           # 36 项分组（含 chat.history_turns / linear_rag.* / graph_explore.*）
        └── AuditPage.tsx            # 审计日志 + payload pretty-JSON
```
