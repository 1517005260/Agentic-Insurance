import type { CitationItem, LocalCitation, SSEEvent } from "@/lib/sse-types";

/**
 * 从 agent SSE 事件流抽 evidence 引用列表。
 *
 * 适用：Base / Proof / Graph Agent chat 模式 —— 这些路径不发 `citations` 事件、
 * 也不在最终答案里写 `[^k]`，引用全部来自 `read` / `proof_scan` 工具调用。
 *
 * 反推优先级：
 *   1. `tool_result.preview` 里正则解析 `(file_id, page_number)` 对儿 —— 后端
 *      preview 是 JSON 截前 300 字符，一般足以切到第一个 unit 的 file/page。
 *   2. 配对的 `tool_call.args.file_ids[0]` + `args.page_range[0]` 兜底 ——
 *      如果 LLM 用 page_range 而不是 unit_ids，前端只能给个起始页指针。
 *   3. 仅 `args.file_ids[0]`，page_number 留 undefined（抽屉显示"未提供页码"）。
 *
 * **proof_scan limitation**：proof_scan 的 args 没有 file_ids/page_range，
 * tool_result preview 也是 counts/status 不是 (file_id, page_number) 三元组；
 * 因此 proof_scan-only 的 evidence 当前反推不出 chip。等后端在 tool_result
 * 加结构化 `units: [{file_id, page_number}]` 字段后再补。
 *
 * **同 loop 多 tool_call**：BaseAgent.run 单 loop 允许多个 tool_calls
 * （`base.py:290+`，逐个串行 execute → emit），且 SSE event 没有
 * `tool_call_index` / `tool_call_id`。前端按事件出现顺序维护 `(loop, name)`
 * 队列：tool_call enqueue，tool_result shift，避免后者覆盖前者的 args。
 *
 * 后续后端补结构化 `tool_result.units` 字段后，preview 解析路径可整体下线。
 */
export function extractEvidenceCitations(events: SSEEvent[]): LocalCitation[] {
  const out: LocalCitation[] = [];
  const seen = new Set<string>();

  // (loop, name) → FIFO queue。同一对在事件流里按顺序成对（同步 emit）；
  // 每个 tool_call enqueue，tool_result 按到达顺序 shift。
  const callQueues = new Map<string, Array<Record<string, unknown>>>();

  for (const ev of events) {
    if (ev.event === "tool_call" && _isEvidenceTool(ev.data.name)) {
      const k = _keyOf(ev.data.loop, ev.data.name);
      const q = callQueues.get(k);
      if (q) q.push(ev.data.args ?? {});
      else callQueues.set(k, [ev.data.args ?? {}]);
      continue;
    }
    if (ev.event !== "tool_result") continue;
    const data = ev.data as ToolResultData;
    if (!data.is_evidence) continue;
    if (data.error) continue;

    const q = callQueues.get(_keyOf(data.loop, data.name));
    const args = q?.shift() ?? {};
    const fromPreview = _parsePreview(data.preview ?? "");

    // 去重 key 包含 observation_id —— page_number 缺失时仅 (file_id, ?) 会
    // 把同一文件不同 read 全折叠成一条；observation_id（proof 路径）可区分。
    if (fromPreview.length > 0) {
      for (const unit of fromPreview) {
        // 同一 (file, page) 在不同 loop 重复 read 时也保留两条 chip ——
        // agent 二次回读说明它在新 evidence 链路里又用了一次，不应折叠。
        const key = `${unit.file_id}|${unit.page_number ?? "?"}|${data.observation_id ?? ""}|${data.loop}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({
          sup: out.length + 1,
          kind: "local",
          file_id: unit.file_id,
          page_id: unit.page_id ?? `${unit.file_id}/p?`,
          page_number: unit.page_number,
          page_preview: unit.snippet,
          observation_id: data.observation_id,
        });
      }
      continue;
    }

    // preview 解析不出来 → args 兜底
    const fileIds = _asStringArray(args.file_ids);
    const pageRange = _asNumPair(args.page_range);
    const unitIds = _asStringArray(args.unit_ids);
    const fallbackFileId =
      fileIds[0] ?? unitIds[0]?.split("/")[0] ?? null;
    if (!fallbackFileId) continue;
    const key = `${fallbackFileId}|${pageRange?.[0] ?? "?"}|${data.observation_id ?? ""}|${data.loop}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({
      sup: out.length + 1,
      kind: "local",
      file_id: fallbackFileId,
      page_id: `${fallbackFileId}/p?`,
      page_number: pageRange?.[0],
      page_preview:
        data.preview && data.preview.length > 0
          ? data.preview.slice(0, 240)
          : undefined,
      observation_id: data.observation_id,
    });
  }

  return out;
}

/** 是否需要把 citations 当作 LocalCitation chip 渲染。 */
export function isLocalCitation(c: CitationItem): c is LocalCitation {
  return !("kind" in c) || c.kind !== "web";
}

// ---------------------------------------------------------- internals

interface ToolResultData {
  loop: number;
  name: string;
  preview?: string;
  retrieved_tokens?: number;
  error?: string;
  is_evidence?: boolean;
  observation_id?: string;
}

const EVIDENCE_TOOLS = new Set(["read", "proof_scan"]);

function _isEvidenceTool(name: string | undefined): boolean {
  return name != null && EVIDENCE_TOOLS.has(name);
}

function _keyOf(loop: number, name: string): string {
  return `${loop}:${name}`;
}

function _asStringArray(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}

function _asNumPair(v: unknown): [number, number] | null {
  if (!Array.isArray(v) || v.length < 2) return null;
  const a = Number(v[0]);
  const b = Number(v[1]);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return [a, b];
}

interface PreviewUnit {
  file_id: string;
  page_id?: string;
  page_number?: number;
  snippet?: string;
}

/**
 * 用正则在 300 字符的 JSON-ish preview 里抠 (file_id, page_id, page_number) 三元组。
 *
 * read 工具实际 preview 形如：
 *   {"observation_type":"PageReadObservation","unit_type":"page","units":[
 *     {"unit_id":"f.../p_0001","file_id":"f...","page_id":"p_0001","page_number":12,"text":"...
 *
 * 正则按出现顺序扫匹配；同一文件多个 page_number 都收走（用户能在抽屉里翻页）。
 */
function _parsePreview(preview: string): PreviewUnit[] {
  if (!preview) return [];
  const results: PreviewUnit[] = [];
  // 把每段 "file_id":"...","page_id":"...","page_number":N" 视为一个 unit
  const re = /"file_id"\s*:\s*"([^"]+)"\s*,\s*"page_id"\s*:\s*"([^"]+)"\s*,\s*"page_number"\s*:\s*(\d+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(preview)) !== null) {
    results.push({
      file_id: m[1],
      page_id: m[2],
      page_number: Number(m[3]),
    });
  }
  // 退而求其次：单独的 file_id + page_number（顺序不一定贴得这么近）
  if (results.length === 0) {
    const re2 = /"file_id"\s*:\s*"([^"]+)"[^}]{0,80}?"page_number"\s*:\s*(\d+)/g;
    while ((m = re2.exec(preview)) !== null) {
      results.push({ file_id: m[1], page_number: Number(m[2]) });
    }
  }
  return results;
}
