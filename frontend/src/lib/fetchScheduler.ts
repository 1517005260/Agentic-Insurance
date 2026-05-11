/**
 * 命名队列 + 并发上限的轻量调度器。
 *
 * 动机：HTTP/1.1 浏览器对同一 host 默认 6 路并发。前端同时跑 4 个
 * ingest SSE + 上传批次 + 缩略图列表 + /graph/expand，会把连接池干光，
 * 让 /auth/me / chat 列表等关键请求被 Queued 几秒。本调度器按"用途"
 * 分队列，每队设独立 cap，避免单一类型的请求把连接池抢光。
 *
 * 两套使用模式共享同一份 cap：
 *  - ``schedule(queue, task)`` —— 一次性 promise；slot 自动释放。
 *  - ``acquireSlot(queue)`` —— 长持有连接（SSE / WebSocket）；返回的
 *    release 函数必须在断流时调用一次（多次调用幂等）。
 *
 * 队列名是 typed union（``FetchQueue``），打错名字直接 TS 报错——
 * 上一版用 raw string + DEFAULT_CAPS 兜底为 Infinity 时，typo 会让
 * cap 静默失效，等于把整套限流退化成"什么都不限"。
 *
 * 取消：调度本身不接管 AbortSignal（caller 用 axios signal 即可）；
 * 排队中的任务被起跑后看到 signal.aborted → 立即返回 → slot 释放，
 * 不会泄漏。caller 也可以在 task body 顶部做一次 ``signal.aborted``
 * 早断短路，避免空转一次微任务。
 */

export type FetchQueue = "upload" | "thumbnail" | "graph_expand" | "ingest_sse";

const DEFAULT_CAPS: Record<FetchQueue, number> = {
  // 上传：multipart POST 体大且要全程占连接；8GB WSL 后端 PARSE_SEM=4，
  // 前端不超过 2 in-flight，留 4 路给其它流量。
  upload: 2,
  // 缩略图：blob fetch 短小但 N 张同屏；cap=4 够铺一行而不抢光池。
  thumbnail: 4,
  // /graph/expand：agent 多步长查询时一帧 graph_subgraph → 一次 expand，
  // cap=2 让"前一步还没回 + 当前步发请求"两条同时跑，但不至于堆积。
  graph_expand: 2,
  // IngestProgress 的 GET SSE 长连接：8 文件同时上传时，每行挂一个
  // 流就是 8 路 SSE 同开，等于把整个 6/host 连接池干光（root cause）。
  // cap=2 让排在后面的进度面板等待，前面 done 后接力开流；UI 体感是
  // 进度条比预期晚几秒出现。
  ingest_sse: 2,
};

interface QueueState {
  cap: number;
  active: number;
  pending: Array<() => void>;
}

const QUEUES = new Map<FetchQueue, QueueState>();

function ensureQueue(name: FetchQueue): QueueState {
  let q = QUEUES.get(name);
  if (!q) {
    q = { cap: DEFAULT_CAPS[name], active: 0, pending: [] };
    QUEUES.set(name, q);
  }
  return q;
}

/** 在 active < cap 范围内尽量放 pending 任务跑。 */
function drain(q: QueueState): void {
  while (q.active < q.cap) {
    const next = q.pending.shift();
    if (!next) break;
    next();
  }
}

function acquireInternal(name: FetchQueue): Promise<void> {
  const q = ensureQueue(name);
  if (q.active < q.cap) {
    q.active += 1;
    return Promise.resolve();
  }
  return new Promise<void>((resolve) => {
    q.pending.push(() => {
      q.active += 1;
      resolve();
    });
  });
}

function release(name: FetchQueue): void {
  const q = ensureQueue(name);
  q.active = Math.max(0, q.active - 1);
  drain(q);
}

/**
 * 在命名队列下排队执行 `task`，遵守该队列的 concurrency cap。
 * 不管 task resolve / reject / throw，slot 都会 release —— caller 不需
 * 要做任何 finally 清理。
 */
export async function schedule<T>(queue: FetchQueue, task: () => Promise<T>): Promise<T> {
  await acquireInternal(queue);
  try {
    return await task();
  } finally {
    release(queue);
  }
}

/**
 * 长持有 slot —— 适用于 SSE / WebSocket 这类长连接。返回的 release
 * 由 caller 在持有者断流时调用一次；多次调用幂等，可以安全放在
 * useEffect cleanup 里。
 */
export async function acquireSlot(queue: FetchQueue): Promise<() => void> {
  await acquireInternal(queue);
  let released = false;
  return () => {
    if (released) return;
    released = true;
    release(queue);
  };
}

/**
 * 动态调整队列上限。用于 admin 面板 / dev tool 即时调参。
 *  - 调大上限：立刻 drain pending。
 *  - 调小上限：active 暂时高于新 cap 是允许的（不会强行 abort 已 inflight
 *    的任务），下一次 release 会被 drain 自然控制在新 cap 内。
 */
export function setQueueCap(queue: FetchQueue, cap: number): void {
  if (!(cap > 0)) {
    throw new Error(`queue cap must be > 0 (got ${cap})`);
  }
  const q = ensureQueue(queue);
  q.cap = cap;
  drain(q);
}

/** 当前所有队列的快照，用于调试 / admin 面板观察。 */
export function getQueueStats(): Record<
  FetchQueue,
  { cap: number; active: number; pending: number }
> {
  const out = {} as Record<
    FetchQueue,
    { cap: number; active: number; pending: number }
  >;
  for (const name of Object.keys(DEFAULT_CAPS) as FetchQueue[]) {
    const q = ensureQueue(name);
    out[name] = { cap: q.cap, active: q.active, pending: q.pending.length };
  }
  return out;
}
