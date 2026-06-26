# Agentic Insurance Console — Web API 层

FastAPI + SQLAlchemy 2 async + aiosqlite + Alembic。挂在 `:8000`，给前端 (`:5173` 反代 `/api/*`) 提供：登录 / 文件 / 聊天 / 工作台 / 图谱 / 管理。

## 启动

### 一次性

```bash
uv sync                          # Python 3.11，依赖锁定在 pyproject.toml
cd [项目根目录]
cp .env.example .env # 然后填 CHAT/EMBEDDING/JWT_SECRET 等
```

### 开发模式

```bash
PYTHONPATH=src ALLOW_INSECURE_JWT=1 uv run uvicorn api.main:app --port 8000
```

加 `--reload` 开 hot-reload；生产前必须把 `JWT_SECRET` 换成真值（默认 `change-me-in-prod` 启动会拒）。

### Lifespan 启动顺序（看 log 能确认）

1. **`_validate_jwt_secret`** — 占位符 + 没 `ALLOW_INSECURE_JWT=1` 直接拒启
2. **`init_db`** — Alembic stamp / upgrade head（已有 dev DB stamp，全新 DB upgrade，已 head no-op）
3. **`_seed_admin`** — `users` 空时建 `admin / admin123`（强警告）
4. **`reconcile_after_restart`** — 把上次崩在 `parsing/indexing/deleting` 的 files / `pending/running` 的 jobs 翻 `failed`
5. **`sweep_orphan_uploads`** — 删 `uploads/<id>.<suffix>` 但 `files` 表没行的崩残 blob（跳 `.part`）
6. **`ConfigStore.from_app_db`** — 读 `app_config` 表 + 合并 schema 默认（33 项）
7. **shared singletons**：`LLMClient` / `EmbeddingClient` / `VisualEmbeddingClient` / `PageStore` / `InventoryStore` / `GraphPPRChannel`（共享 `linear_config`）
8. **spaCy NER 预热** — `_ensure_spacy()` 一次跑，~8 s + 1.6 GB anon heap，避免首次 graph_explore / fraud-ppr 冷加载 OOM
9. **RAGPipeline** + **3 agent 单例** (`base / proof / graph`) + **Tavily** + **WebAgent** + **GraphService**
10. `Application startup complete.` → ready

`GET /health` → `{"status":"ok"}` 即可使。

### Lifespan 结束

`finally` 关 `requests.Session`：tavily / LLM / embedding / visual / rerank / web_fetch tool 各一份，避免 reload cycle 漏 connection pool。


## 模块布局

```
src/api/
├── main.py                  # FastAPI 实例 + lifespan + CORS + 路由注册
├── auth.py                  # JWT (HS256) 签发/校验 + bcrypt(sha256-pre) 密码 hash
├── db.py                    # async engine + Alembic init_db + session_scope（bg task 用）
├── deps.py                  # get_session / get_current_user / require_admin / require_role(...)
├── models.py                # 7 张表的 SQLAlchemy 模型（User / FileRecord / IngestJob / ChatSession / ChatMessage / AppConfig / AuditLog）
├── sse.py                   # SSE 编码（heartbeat 15s，单行 JSON ensure_ascii=False）
├── prompts/                 # API 层独家 system prompt（rag_business 等）
│
├── routes/                  # FastAPI 路由（每个文件一组端点）
│   ├── auth.py              # POST /auth/login + GET /auth/me
│   ├── files.py             # CRUD + /jobs/stream SSE + /preview 缩略图 + /download
│   ├── chat.py              # 6 个 session 端点 + POST /chat/sessions/{id}/messages（多轮）+ /rag/stream + /agent/stream + /web-rag/stream
│   ├── trace.py             # GET /chat/messages/{id}/trace（owner / admin）
│   ├── search.py            # POST /search 高级条款检索（3 粒度 × 任选通道 + RRF + rerank）
│   ├── graph.py             # /graph/overview /seed /expand /nodes/{hash} /ppr-subgraph /sample
│   ├── insurance.py         # 6 SSE 端点：compare / exclusion-audit / claim-check / recommend / policy-calc / fraud-ppr（前端把后 3 个收编为"保单审查"3 tab）
│   ├── audit.py             # GET /audit?action=&target= + GET /audit/{id}（admin）
│   ├── admin.py             # GET/PATCH/DELETE /admin/config（33 项 batch all-or-422）
│   └── admin_users.py       # /admin/users CRUD + reset-password + 自防护 + 末位 admin 防护
│
├── runners/                 # 每个 SSE 流的"runner"：起 thread 跑算法 + EventBus 桥到 async generator
│   ├── events.py            # EventBus（asyncio.Queue + call_soon_threadsafe）+ EventType 词表
│   ├── _tracing.py          # CapturingTracer（包 Tracer 抓 last_run_dir）
│   ├── _workbench.py        # 5 workbench 共享脚手架（read 累计 → CitationItem dedup → sup → final 之前 emit）
│   ├── rag_runner.py        # /chat/sessions/.. mode=rag → RAGPipeline.run + token 流 + CitationBuilder
│   ├── agent_runner.py      # /agent/stream + /chat/sessions/.. mode=agent (base/proof/graph/web) — _compose_query_with_history 拼多轮
│   ├── web_rag_runner.py    # /web-rag/stream → web_rag_svc.stream_chat
│   ├── ingestion_runner.py  # /files/{id}/jobs/stream 用的 EventBus 注册表 + claim_stream 单消费者守卫
│   ├── compare_runner.py    # 多产品对比矩阵
│   ├── exclusion_runner.py  # 免责审查（ProofAgent forall）
│   ├── claim_runner.py      # 理赔判定三栏
│   ├── recommend_runner.py  # 客户保障规划：开放语料 top-3 / 已持有保单缺口分析 + 互补推荐（按 held_policies_file_ids 切换）
│   ├── policy_calc_runner.py # 保单精算（BaseAgent + code_run）
│   └── fraud_ppr_runner.py  # 单次 PPR + 流式 LLM 分析（不走 agent loop）— 现在驱动"隐藏风险关联"，URL/flavor 名字保留
│
├── services/                # 业务逻辑（路由调它，单元测试就测它）
│   ├── files.py             # ingest 状态机 + INGEST_LOCK + bg task（run_parse_index / run_reingest / run_delete）+ orphan sweep
│   ├── chat.py              # session/message CRUD + metadata 编码 + 1500-char 标题截断
│   ├── citation.py          # CitationBuilder：sup 编号 + 页头内联 [^k] + parse_response 抽 [^k] 列表
│   ├── history.py           # 多轮 load_recent_turns：从 chat_messages.user.content + final.json.answer 拼 (q, a) pair
│   ├── graph_service.py     # /graph/* 端点的 service（overview/seed_search/expand/node_detail/ppr_subgraph/sample）
│   ├── search.py            # /search 端点的 service（任选通道 + post-filter overfetch + n_pre_filter 遥测）
│   └── web_rag.py           # Tavily 检索 + LLM 总结 + sup 标号；驱动 chat web mode SSE
│
├── schemas/                 # Pydantic v2 DTO（OpenAPI 自动生成）
│   ├── auth.py              # LoginIn / TokenOut / MeOut
│   ├── chat.py              # SessionCreate / SessionUpdate / SessionOut / MessagePost / MessageOut + RagStreamRequest / AgentStreamRequest / WebRagStreamRequest
│   ├── admin.py             # ConfigEntrySchema / ConfigSnapshotResponse / ConfigPatchRequest / ConfigPatchResponse
│   ├── users.py             # UserOut / UserCreate / UserUpdate / PasswordReset
│   ├── search.py            # SearchRequest / SearchResponse
│   ├── graph.py             # NodeOut / EdgeOut / ExpandResponse / PPRSubgraphRequest / NodeDetail
│   └── insurance.py         # 6 工作台 input / output schema
│
└── prompts/                 # API 层独占的 system prompt（不落算法层默认）
    └── rag_business.py      # 业务 RAG prompt（强制 cite + abstain 例外 + 数值禁外推）
```

## 数据库迁移（Alembic）

要点：

- baseline migration 在 `alembic/versions/v0_1.py`（revision id `v0_1`）
- 改 ORM 后跑 `ALEMBIC_URL_OVERRIDE=sqlite:///$(mktemp -d)/empty.db uv run alembic revision --autogenerate -m "..."`
- 启动时 `init_db` 自动 stamp 老 dev DB / upgrade 全新 DB
