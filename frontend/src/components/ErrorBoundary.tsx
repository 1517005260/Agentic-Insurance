/**
 * 全局兜底 ErrorBoundary。
 *
 * React 18 StrictMode 下任何 render-time 抛出都会让 ReactDOM 卸载整棵
 * 子树 → 表现就是"页面只剩底色"。我们用一层 boundary 接住、显式渲染
 * 红屏 + reload 按钮，避免用户面对未知白屏只能 F12。
 *
 * 同时挂 ``window.onerror`` / ``unhandledrejection`` 把异步路径的异常
 * 也打到 console (boundary 看不到这些)，方便后续接 Sentry。
 */
import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
  info: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, info: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[ErrorBoundary] render crashed", error, info);
    this.setState({ info });
  }

  private handleReload = () => {
    window.location.reload();
  };

  private handleHome = () => {
    window.location.href = "/chat";
  };

  render() {
    const { error, info } = this.state;
    if (!error) return this.props.children;
    return (
      <div
        role="alert"
        style={{
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: "24px",
          background: "#fff8f8",
          color: "#7f1d1d",
          fontFamily: "ui-sans-serif, system-ui, sans-serif",
          gap: "16px",
        }}
      >
        <div
          style={{
            maxWidth: 720,
            width: "100%",
            background: "#fff",
            border: "1px solid #fecaca",
            borderRadius: 8,
            padding: 24,
            boxShadow: "0 1px 3px rgba(0,0,0,.08)",
          }}
        >
          <h1 style={{ margin: 0, fontSize: 18, fontWeight: 600 }}>
            页面渲染异常
          </h1>
          <p style={{ marginTop: 8, color: "#475569", fontSize: 13 }}>
            前端组件抛出未捕获错误。可点击下方按钮恢复，或把以下堆栈贴给开发。
          </p>
          <pre
            style={{
              marginTop: 12,
              maxHeight: 280,
              overflow: "auto",
              fontSize: 12,
              background: "#0f172a",
              color: "#e2e8f0",
              padding: 12,
              borderRadius: 6,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {error.name}: {error.message}
            {error.stack ? `\n\n${error.stack}` : ""}
            {info?.componentStack ? `\n\nComponent stack:${info.componentStack}` : ""}
          </pre>
          <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
            <button
              type="button"
              onClick={this.handleReload}
              style={btnPrimary}
            >
              重新加载
            </button>
            <button
              type="button"
              onClick={this.handleHome}
              style={btnGhost}
            >
              回到首页
            </button>
          </div>
        </div>
      </div>
    );
  }
}

const btnPrimary: React.CSSProperties = {
  background: "#0f172a",
  color: "#fff",
  border: "1px solid #0f172a",
  padding: "8px 14px",
  borderRadius: 6,
  fontSize: 13,
  cursor: "pointer",
};

const btnGhost: React.CSSProperties = {
  background: "transparent",
  color: "#0f172a",
  border: "1px solid #cbd5e1",
  padding: "8px 14px",
  borderRadius: 6,
  fontSize: 13,
  cursor: "pointer",
};

/**
 * Install once on app boot — captures async-path errors that
 * ErrorBoundary cannot see (rejected promises, setTimeout throws, ...).
 * Just logs; future hookup point for Sentry.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function installGlobalErrorReporters(): void {
  window.addEventListener("error", (event) => {
    console.error("[window.onerror]", event.message, event.error);
  });
  window.addEventListener("unhandledrejection", (event) => {
    console.error("[unhandledrejection]", event.reason);
  });
}
