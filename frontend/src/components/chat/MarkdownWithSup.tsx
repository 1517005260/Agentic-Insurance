import {
  Fragment,
  cloneElement,
  isValidElement,
  useDeferredValue,
  useMemo,
  type ReactElement,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
// KaTeX stylesheet — required for the rendered math to look right.
// Imported once here so any consumer of MarkdownWithSup gets it
// without having to remember to import in their own bundle.
import "katex/dist/katex.min.css";

import type { CitationItem } from "@/lib/sse-types";
import { useCitationStore } from "@/stores/citation";
import { cn } from "@/lib/utils";

/**
 * Pre-process the LLM's math output so KaTeX can read it. Three concerns:
 *
 *   1. Upstream models often emit OpenAI-style delimiters — ``\(...\)``
 *      for inline math and ``\[...\]`` for block math — which
 *      ``remark-math`` does not recognise. The result is a raw ``$``-free
 *      string of ``\(``/``\[`` markers rendered as literal text.
 *
 *   2. ``remark-math`` refuses ``$ x $`` (whitespace adjacent to the
 *      delimiter). LLMs sometimes pad math with spaces, and the
 *      delimiters then survive as literal ``$`` characters in the
 *      output.
 *
 *   3. Inside an ``align`` / ``cases`` block the model writes
 *      ``\[4pt]`` (LaTeX line-break with vertical skip). KaTeX would
 *      otherwise parse the single backslash as a begin-math ``\[`` and
 *      choke; we rewrite to the canonical ``\\[4pt]`` form.
 *
 * Code fences and inline code are pulled out first via placeholders so
 * a backticked snippet containing ``$`` or ``\[`` is never rewritten.
 */
function normalizeMathDelimiters(src: string): string {
  if (!src) return src;

  // 0) Pull every code region out before any math rewrite. The
  //    placeholder uses the private-use code point U+E000 (written as
  //    a ``\uE000`` escape in source so the .tsx file stays ASCII-
  //    clean — git keeps it text, code review sees a normal diff).
  //    Private-use code points never appear in normal LLM output and
  //    contain no regex-special characters, so a placeholder cannot
  //    collide with real prose. Backtick (```` ``` ````) and tilde
  //    (``~~~``) fenced blocks are both protected, plus single-
  //    backtick inline code.
  const SENTINEL = "\uE000";
  const placeholders: string[] = [];
  const protect = (chunk: string): string => {
    const i = placeholders.length;
    placeholders.push(chunk);
    return `${SENTINEL}C${i}${SENTINEL}`;
  };
  let s = src
    .replace(/```[\s\S]*?```/g, protect)
    .replace(/~~~[\s\S]*?~~~/g, protect)
    .replace(/`[^`\n]+`/g, protect);

  // 1) OpenAI-style delimiters → dollar-sign forms. Bodies are bounded
  //    so an unclosed opener cannot drive O(N^2) backtracking on
  //    adversarial input — each ``\(`` or ``\[`` scans at most
  //    ``CAP`` characters before the regex engine gives up and moves
  //    on. The negative lookbehind on ``\\`` rules out ``\\(`` /
  //    ``\\[`` (a LaTeX line break inside an already-delimited math
  //    region) and the closing lookbehind likewise protects ``\\)``
  //    / ``\\]``.
  const INLINE_CAP = 400;
  const BLOCK_CAP = 4000;
  const inlineOpenAI = new RegExp(
    `(?<!\\\\)\\\\\\(([\\s\\S]{1,${INLINE_CAP}}?)(?<!\\\\)\\\\\\)`,
    "g",
  );
  const blockOpenAI = new RegExp(
    `(?<!\\\\)\\\\\\[([\\s\\S]{1,${BLOCK_CAP}}?)(?<!\\\\)\\\\\\]`,
    "g",
  );
  s = s.replace(inlineOpenAI, (_m, b) => `$${b.trim()}$`);
  s = s.replace(blockOpenAI, (_m, b) => `\n\n$$\n${b.trim()}\n$$\n\n`);

  // 2) Tighten ``$ x $`` → ``$x$`` so remark-math will accept it. The
  //    opener lookbehind excludes both ``$`` (so the block-math
  //    ``$$`` delimiter isn't shredded) and ``\\`` (so a deliberately
  //    escaped ``\$`` literal — common in prose about money — stays a
  //    literal dollar). Currency-style ``$50`` has no whitespace
  //    adjacent to the dollar and is therefore untouched.
  s = s.replace(
    /(?<![$\\])\$[ \t]+([^\n$]{1,300}?)[ \t]+\$(?!\$)/g,
    (_m, b) => `$${b.trim()}$`,
  );

  // 3) ``\[4pt]`` → ``\\[4pt]`` inside an existing math region, and
  //    a bare ``\[`` at the end of a math line → ``\\``. Only
  //    applied within math regions so BibTeX-ish prose containing
  //    ``\[`` outside math is left alone.
  const fixLineBreak = (body: string): string =>
    body
      .replace(/(?<!\\)\\(\[\s*\d+(?:\.\d+)?\s*[a-zA-Z]+\s*\])/g, "\\\\$1")
      .replace(/(?<!\\)\\\[(?=\s*$)/gm, "\\\\");
  const blockRe = /(\${2})([\s\S]*?)\1/g;
  const inlineRe = /(?<!\$)(\$)(?!\$)([\s\S]*?)(?<!\$)\1(?!\$)/g;
  s = s
    .replace(blockRe, (_m, d, b) => `${d}${fixLineBreak(b)}${d}`)
    .replace(inlineRe, (_m, d, b) => `${d}${fixLineBreak(b)}${d}`);

  // Restore protected code chunks. If a sentinel index points at a
  // hole (should not happen, but cheap to defend), keep the literal
  // marker rather than throwing.
  return s.replace(
    new RegExp(`${SENTINEL}C(\\d+)${SENTINEL}`, "g"),
    (_m, k) => placeholders[Number(k)] ?? _m,
  );
}


interface Props {
  /** LLM 输出的原始 markdown 文本（可能含 [^k] 角标）。 */
  content: string;
  /** 当前 turn 的 citations（来自 SSE `citations` 事件）。 */
  citations?: CitationItem[];
  /**
   * 是否解析 [^k] 上标。仅用于 RAG / Web RAG / Web Agent / 业务 workbench；
   * Base/Proof/Graph Agent chat 模式传 false（这些路径不发 sup）。
   */
  parseSup?: boolean;
}

/**
 * 答案 markdown 渲染。
 *
 * remark-gfm 默认把 `[^k]` 当 footnote ref，但它要求文档底部有 `[^k]: ...`
 * 的定义；LLM 不会写定义块，于是 remark-gfm 解析出错或保留字面。
 *
 * 我们的处理：在交给 ReactMarkdown 前用正则把 `[^k]` 替换成自定义 token
 * `<sup data-cite="k">[^k]</sup>`；ReactMarkdown 把它当 raw HTML 透出来后由
 * 自定义 renderer 接管，点击 dispatch citation store。这样既不依赖 remark-gfm
 * 的 footnote 行为，也避免 markdown 语法干扰。
 *
 * 关于安全：LLM 输出本身不可信，但我们这里只允许一个有限的 sup 形式
 * （`<sup data-cite="\d+">[^N]</sup>`，由我们自己合成）；真正的 user-controlled
 * raw HTML 不打开 — 不传 rehype-raw，所以 LLM 写的 `<script>` 等仍会被
 * react-markdown 当字面文本，不执行。
 */
export function MarkdownWithSup({ content, citations, parseSup = true }: Props) {
  const open_ = useCitationStore((s) => s.open_);

  // 不再做"先全局 replace 再 markdown parse"——那样代码块里的字面 [^1]
  // 也会被换成 token，bypass code 的 React 层 transform 也救不回来字面值。
  // 改在 mdast 阶段走 remark plugin 替换 text 节点的 [^k]，遇到 inlineCode /
  // code 整树跳过；plugin 仅在 parseSup=true 时挂载。
  //
  // 不再做"先全局 replace"。改在 mdast 阶段：
  //   1) 普通 text 节点里的 `[^k]` 切成多段 + 插 token (`«CITE:k»`)
  //   2) remark-gfm 已经把 `[^k]` 解析成 `footnoteReference` 节点 ——
  //      把它直接转成 text 节点 `«CITE:k»`，让下游和 plain 路径完全一致
  //   3) 把孤立的 `footnoteDefinition` 整块从 root 移除（LLM 真写定义
  //      时不让它在文末多渲染一段 footnote 列表）
  // plugin 顺序无关：remark-gfm 是解析期 micromark 扩展，在 plugin 之前
  // 就把语法识别成 footnoteRef 节点了；本 plugin 是转换期 transformer，
  // 看到的 mdast 已经带 footnoteReference / footnoteDefinition。
  // remark-math identifies ``$...$`` / ``$$...$$`` regions; rehype-katex
  // does the actual rendering at the hast stage. Order matters:
  // remark-math first (mdast → hast pass needs math nodes), then our
  // sup-cite transformer (which only walks text nodes), then rehype.
  const remarkPlugins = parseSup
    ? [remarkGfm, remarkMath, remarkCiteTokens]
    : [remarkGfm, remarkMath];
  const rehypePlugins = [rehypeKatex];

  // ``useDeferredValue`` lets React skip intermediate token-by-token
  // renders during streaming: the raw ``content`` updates immediately
  // but ``deferred`` lags by one or more reconciliation passes when the
  // main thread is busy. Without this, a ~5000-token web-RAG answer
  // forces remark + rehype to re-parse the entire growing string on
  // every token (~5000 × 5000 / 2 = 12.5M chars of work per turn) —
  // two consecutive turns are enough to OOM Chrome's tab heap from
  // garbage allocation pressure.
  const deferred = useDeferredValue(content);
  const normalized = useMemo(() => normalizeMathDelimiters(deferred), [deferred]);

  const onSupClick = (sup: number) => {
    if (!citations || citations.length === 0) return;
    const target = citations.find((c) => c.sup === sup);
    if (!target) return;
    open_(citations, target);
  };

  // 不依赖 @tailwindcss/typography，用任意子选择器把 markdown 默认元素
  // 调到符合 Theme A 的克制风格。
  //
  // sup 解析走 deep-recursive transformer：在所有可能含 inline 文本的标签上
  // override，遍历 children 把 string 里的 «CITE:N» token 切成 React.Fragment
  // + SupCite。会跳过 `<code>` / `<pre>`（inline / 块代码内 [^k] 是字面，
  // 不能被替换；安全也避免代码内容里的 token 误杀）。
  //
  // 对每个有 inline 内容的 components 注入同一个 deep transformer：strong /
  // em / a / blockquote / span / h1..6 / p / li / td / th。code 故意不传 deep。
  //
  // wrapper 必须 spread 全部 props（react-markdown 会注入 className /
  // align / id 等给原生元素，丢就破坏 GFM table align、heading id 锚点等）。
  // 用 any 接 react-markdown 给每个 tag 的不同 props（HTMLAttributes 类型
  // 互不兼容、Index signature 缺失），把 children 拆出来 deepTransform，
  // 其余原样 spread —— 避免丢 GFM table align / heading id / link title 等。
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wrap = (Tag: keyof React.JSX.IntrinsicElements): any => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const Wrapped = (props: any) => {
      const { children, node: _node, ...rest } = props ?? {};
      void _node;
      const TagComp = Tag as unknown as React.ElementType;
      return (
        <TagComp {...rest}>
          {deepTransform(children, onSupClick, citations)}
        </TagComp>
      );
    };
    Wrapped.displayName = `MdSup(${Tag})`;
    return Wrapped;
  };

  return (
    <div className={MD_CLASS}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        components={{
          p: wrap("p"),
          li: wrap("li"),
          td: wrap("td"),
          th: wrap("th"),
          strong: wrap("strong"),
          em: wrap("em"),
          a: wrap("a"),
          blockquote: wrap("blockquote"),
          h1: wrap("h1"),
          h2: wrap("h2"),
          h3: wrap("h3"),
          h4: wrap("h4"),
          h5: wrap("h5"),
          h6: wrap("h6"),
          // code / pre 不传 deep —— 配合 remarkCiteTokens 跳过它们的 text，
          // 代码块内 [^k] 会以字面渲染，不可点也不被 token 替换。
        }}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  );
}

// ----------------------------------------------------- mdast plugin

/**
 * Remark plugin：递归 mdast，把 text 节点里的 `[^k]` 切成多段，匹配到的位置
 * 替成形如 `«CITE:k»` 的标记字符串。React 渲染时 deepTransform 再把这个标记
 * 切成 SupCite 元素。
 *
 * 跳过 inlineCode / code / html 节点 —— 代码块 / 内联代码里的字面 `[^k]` 不应
 * 被替换。手写 walker 替代 unist-util-visit，避免新增依赖。
 */
function remarkCiteTokens() {
  return (tree: MdastNode) => {
    walkMdast(tree, null);
    dropFootnoteDefs(tree);
  };
}

interface MdastNode {
  type: string;
  value?: string;
  identifier?: string;
  label?: string;
  children?: MdastNode[];
}

/**
 * 递归 mdast：text 节点切 `[^k]` → 插 token；遇到 inlineCode/code/html 整树跳过。
 * 同时把 remark-gfm 解析出的 `footnoteReference` 节点 in-place 改成 text token。
 */
function walkMdast(node: MdastNode | null | undefined, parent: MdastNode | null): void {
  if (!node) return;
  const t = node.type;
  if (t === "inlineCode" || t === "code" || t === "html") return;

  // remark-gfm 的 [^k] 已识别 → 直接退化成纯文本 token
  if (t === "footnoteReference" && parent && Array.isArray(parent.children)) {
    const k = String(node.identifier ?? node.label ?? "").trim();
    if (/^\d+$/.test(k)) {
      const idx = parent.children.indexOf(node);
      if (idx >= 0) {
        parent.children.splice(idx, 1, { type: "text", value: `«CITE:${k}»` });
      }
      return;
    }
    // 非数字 identifier 不是我们关心的引用，留给默认 footnote 渲染
    return;
  }

  if (t === "text" && typeof node.value === "string" && node.value.includes("[^")) {
    if (!parent || !Array.isArray(parent.children)) return;
    const value = node.value;
    const re = /\[\^(\d+)\]/g;
    const parts: MdastNode[] = [];
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(value)) !== null) {
      if (m.index > last) parts.push({ type: "text", value: value.slice(last, m.index) });
      parts.push({ type: "text", value: `«CITE:${m[1]}»` });
      last = m.index + m[0].length;
    }
    if (parts.length === 0) return;
    if (last < value.length) parts.push({ type: "text", value: value.slice(last) });
    const idx = parent.children.indexOf(node);
    if (idx >= 0) parent.children.splice(idx, 1, ...parts);
    return;
  }

  if (Array.isArray(node.children)) {
    // snapshot 防止 splice 影响迭代
    const kids = [...node.children];
    for (const c of kids) walkMdast(c, node);
  }
}

/**
 * LLM 偶尔会真写 GFM footnote 定义块（`[^1]: foo`）。它已被识别成
 * `footnoteDefinition` —— 在 root.children 里就地移除，避免页底渲染一段
 * 跟 sup 系统并存的 footnote 列表。
 */
function dropFootnoteDefs(tree: MdastNode): void {
  if (!Array.isArray(tree.children)) return;
  tree.children = tree.children.filter((c) => c.type !== "footnoteDefinition");
}

const MD_CLASS = [
  "text-[15px] leading-7 text-ink max-w-none",
  // 段落 / 标题
  "[&>p]:my-2 [&>p:first-child]:mt-0 [&>p:last-child]:mb-0",
  "[&>h1]:text-xl [&>h1]:font-semibold [&>h1]:mt-4 [&>h1]:mb-2 [&>h1]:text-ink",
  "[&>h2]:text-lg [&>h2]:font-semibold [&>h2]:mt-4 [&>h2]:mb-2 [&>h2]:text-ink",
  "[&>h3]:text-base [&>h3]:font-semibold [&>h3]:mt-3 [&>h3]:mb-1.5 [&>h3]:text-ink",
  "[&>h4]:text-[15px] [&>h4]:font-semibold [&>h4]:mt-3 [&>h4]:mb-1 [&>h4]:text-ink",
  // 列表
  "[&>ul]:list-disc [&>ul]:pl-6 [&>ul]:my-2 [&>ul>li]:my-1",
  "[&>ol]:list-decimal [&>ol]:pl-6 [&>ol]:my-2 [&>ol>li]:my-1",
  // 行内代码
  "[&_code:not(pre_code)]:bg-surface-sunk [&_code:not(pre_code)]:text-accent-700 [&_code:not(pre_code)]:px-1 [&_code:not(pre_code)]:py-0.5 [&_code:not(pre_code)]:rounded [&_code:not(pre_code)]:text-[13px] [&_code:not(pre_code)]:font-mono",
  // 代码块
  "[&>pre]:bg-surface-sunk [&>pre]:rounded-md [&>pre]:p-3 [&>pre]:my-3 [&>pre]:text-[13px] [&>pre]:overflow-x-auto [&>pre]:scrollbar-thin",
  // 表格
  "[&>table]:my-3 [&>table]:w-full [&>table]:text-[14px] [&>table]:border-collapse",
  "[&>table_th]:bg-surface-sunk [&>table_th]:font-medium [&>table_th]:text-left [&>table_th]:px-2.5 [&>table_th]:py-1.5 [&>table_th]:border [&>table_th]:border-ink-line/70",
  "[&>table_td]:px-2.5 [&>table_td]:py-1.5 [&>table_td]:border [&>table_td]:border-ink-line/70 [&>table_td]:align-top",
  // 链接
  "[&_a]:text-accent-700 [&_a]:underline [&_a]:underline-offset-2 [&_a]:decoration-accent-300 hover:[&_a]:decoration-accent-600",
  // 引用块
  "[&>blockquote]:border-l-2 [&>blockquote]:border-primary-200 [&>blockquote]:pl-3 [&>blockquote]:my-2 [&>blockquote]:text-ink-muted",
  // 分割线
  "[&>hr]:my-4 [&>hr]:border-ink-line",
].join(" ");

// ----------------------------------------------------- token replacement

const CITE_RE = /«CITE:(\d+)»/g;

/** 标签集合：遇到这些就停（字面保留 [^k]）。 */
const SKIP_TAGS = new Set(["code", "pre", "kbd", "samp"]);

/**
 * 深度遍历 ReactNode 树，把所有 string leaf 里的 «CITE:N» token 替成
 * <SupCite>。穿过任意嵌套（**bold[^1]**、heading、<a>[^2]</a> 等都覆盖），
 * 但遇到 <code>/<pre> 整体 bypass —— 代码块里的字面 [^k] 不应被吃掉。
 */
function deepTransform(
  node: ReactNode,
  onClick: (sup: number) => void,
  citations: CitationItem[] | undefined,
): ReactNode {
  if (node == null || typeof node === "boolean") return node;
  if (typeof node === "string") {
    return splitCiteString(node, onClick, citations);
  }
  if (typeof node === "number") return node;
  if (Array.isArray(node)) {
    return node.map((c, i) => (
      <Fragment key={i}>{deepTransform(c, onClick, citations)}</Fragment>
    ));
  }
  if (isValidElement(node)) {
    const el = node as ReactElement<{ children?: ReactNode }>;
    const tag = typeof el.type === "string" ? el.type : "";
    if (tag && SKIP_TAGS.has(tag)) return el;
    const kids = el.props?.children;
    if (kids === undefined) return el;
    return cloneElement(el, undefined, deepTransform(kids, onClick, citations));
  }
  return node;
}


function splitCiteString(
  s: string,
  onClick: (sup: number) => void,
  citations: CitationItem[] | undefined,
): ReactNode {
  if (typeof s !== "string" || !s.includes("«CITE:")) return s;
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  s.replace(CITE_RE, (match, kStr: string, offset: number) => {
    if (offset > lastIndex) parts.push(s.slice(lastIndex, offset));
    const k = Number(kStr);
    const known = citations?.some((c) => c.sup === k) ?? false;
    parts.push(
      <SupCite
        key={`${offset}-${k}`}
        sup={k}
        known={known}
        onClick={() => onClick(k)}
      />,
    );
    lastIndex = offset + match.length;
    return match;
  });
  if (lastIndex < s.length) parts.push(s.slice(lastIndex));
  return parts;
}

function SupCite({
  sup,
  known,
  onClick,
}: {
  sup: number;
  known: boolean;
  onClick: () => void;
}) {
  return (
    <sup
      role="button"
      tabIndex={known ? 0 : -1}
      aria-label={`引用 ${sup}`}
      onClick={known ? onClick : undefined}
      onKeyDown={(e) => {
        if (!known) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick();
        }
      }}
      className={cn(
        "mx-0.5 inline-block align-super text-[10px] font-mono leading-none px-1 py-0.5 rounded-sm transition-colors select-none",
        known
          ? "bg-accent-50 text-accent-700 hover:bg-accent-100 hover:text-accent-800 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/40"
          : "bg-ink-line/50 text-ink-subtle cursor-not-allowed",
      )}
      title={known ? `查看引用 [^${sup}]` : `[^${sup}] 缺失映射`}
    >
      {sup}
    </sup>
  );
}
