import { useRef, useEffect } from "react";
import { Globe, Sparkles, ArrowUp, Square } from "lucide-react";

import { cn } from "@/lib/utils";

export interface ComposerMode {
  /** 联网增强：开启走 web_search / web_rag 路径 */
  web: boolean;
  /** Agent 模式：开启走 agent loop（tool calling） */
  agent: boolean;
}

interface Props {
  value: string;
  onChange: (v: string) => void;
  mode: ComposerMode;
  onModeChange: (m: ComposerMode) => void;
  onSend: () => void;
  onAbort: () => void;
  busy: boolean;
  /** mode → endpoint 文本，给小提示。 */
  modeLabel: string;
}

/**
 * 类似 ChatGPT 主页的中央输入栏。
 *
 * 两个 toggle 图标：
 *   🌐 web   — 关 = 本地文档；开 = 联网（Tavily）
 *   ✨ agent — 关 = 一次性 RAG；开 = agent loop
 *
 * 4 种组合对应 4 个后端端点（详见 ChatPage 的 specFor()）：
 *   none         → /rag/stream         (RAG)
 *   web          → /web-rag/stream     (Web RAG)
 *   agent        → /agent/stream base  (Base Agent)
 *   web + agent  → /agent/stream base+web (Web Agent)
 */
export function ChatComposer({
  value,
  onChange,
  mode,
  onModeChange,
  onSend,
  onAbort,
  busy,
  modeLabel,
}: Props) {
  const taRef = useRef<HTMLTextAreaElement>(null);

  // 自适应高度：单行起步，最多 6 行高（再多内部滚动）。
  //
  // 关键：只有删字这种"内容可能变小"的场景才需要 height='auto'
  // 重新量；加字时 scrollHeight 已经比 clientHeight 大，直接写
  // 新高度即可。这样在常态打字时完全不写 'auto'，避免冗余写。
  //
  // useEffect 已经只在 value 变化时跑；rAF 把 read-then-write 收
  // 进同一个 layout pass。
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    const max = 6 * 24 + 16;
    const raf = requestAnimationFrame(() => {
      const cur = ta.clientHeight;
      const sh = ta.scrollHeight;
      if (sh > cur) {
        // 内容比当前高度高 → 应当长高（或封顶 max）。
        const next = `${Math.min(sh, max)}px`;
        if (ta.style.height !== next) ta.style.height = next;
      } else if (cur > 0) {
        // 当前高度可能大于实际所需（删字 / 粘贴变短）→ 'auto' 后
        // 再读 scrollHeight 才是真实尺寸；写回精确 px。
        ta.style.height = "auto";
        const next = `${Math.min(ta.scrollHeight, max)}px`;
        if (ta.style.height !== next) ta.style.height = next;
      }
    });
    return () => cancelAnimationFrame(raf);
  }, [value]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      if (!busy && value.trim()) onSend();
    }
  };

  const canSend = !busy && value.trim().length > 0;

  return (
    <div className="w-full">
      <div
        className={cn(
          "relative rounded-2xl border bg-surface-raised shadow-card",
          "border-ink-line focus-within:border-primary-500 focus-within:ring-2 focus-within:ring-primary-500/20",
          "transition-colors",
        )}
      >
        <textarea
          ref={taRef}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="向 Agentic 提问…  (Shift+Enter 换行)"
          disabled={busy}
          className={cn(
            "w-full resize-none bg-transparent px-4 pt-3.5 pb-12 text-[15px] leading-6 text-ink",
            "placeholder:text-ink-subtle outline-none",
            "scrollbar-thin",
            busy && "opacity-70",
          )}
        />

        <div className="absolute bottom-2 left-2 flex items-center gap-1">
          <ToggleIconButton
            label="联网"
            tip={mode.web ? "联网检索 已开启" : "本地文档"}
            active={mode.web}
            disabled={busy}
            onClick={() => onModeChange({ ...mode, web: !mode.web })}
            icon={<Globe className="h-4 w-4" />}
          />
          <ToggleIconButton
            label="Agent"
            tip={mode.agent ? "Agent 模式 已开启" : "一次性 RAG"}
            active={mode.agent}
            disabled={busy}
            onClick={() => onModeChange({ ...mode, agent: !mode.agent })}
            icon={<Sparkles className="h-4 w-4" />}
          />
          <span className="ml-1.5 hidden sm:inline-flex items-center text-[11px] uppercase tracking-[0.16em] text-ink-subtle">
            {modeLabel}
          </span>
        </div>

        <div className="absolute bottom-2 right-2">
          {busy ? (
            <button
              type="button"
              onClick={onAbort}
              aria-label="中止"
              className={cn(
                "h-8 w-8 inline-flex items-center justify-center rounded-md",
                "bg-ink text-surface-raised hover:bg-ink/90",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink/40",
              )}
            >
              <Square className="h-3.5 w-3.5 fill-current" />
            </button>
          ) : (
            <button
              type="button"
              onClick={onSend}
              disabled={!canSend}
              aria-label="发送"
              className={cn(
                "h-8 w-8 inline-flex items-center justify-center rounded-md transition-colors",
                canSend
                  ? "bg-primary-600 text-surface-raised hover:bg-primary-700"
                  : "bg-surface-sunk text-ink-subtle cursor-not-allowed",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/40",
              )}
            >
              <ArrowUp className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      <p className="mt-2 text-center text-[11px] text-ink-subtle">
        生成内容仅作辅助，请以原始保单条款为准。
      </p>
    </div>
  );
}

interface IconBtnProps {
  label: string;
  tip: string;
  active: boolean;
  disabled: boolean;
  onClick: () => void;
  icon: React.ReactNode;
}

function ToggleIconButton({ label, tip, active, disabled, onClick, icon }: IconBtnProps) {
  return (
    <button
      type="button"
      aria-label={label}
      aria-pressed={active}
      title={tip}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "h-8 w-8 inline-flex items-center justify-center rounded-md transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30",
        active
          ? "bg-primary-600 text-surface-raised hover:bg-primary-700"
          : "bg-transparent text-ink-muted hover:bg-surface-sunk hover:text-ink",
        disabled && "opacity-50 cursor-not-allowed",
      )}
    >
      {icon}
    </button>
  );
}
