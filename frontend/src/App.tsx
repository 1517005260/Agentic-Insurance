import { Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";

import { RequireAdmin, RequireAuth } from "@/components/auth/guards";
import LayoutShell from "@/components/layout/LayoutShell";
import { queryClient } from "@/lib/queryClient";
import { lazyWithRetry } from "@/lib/lazyWithRetry";
// LoginPage / NotFoundPage 保持 eager：前者是 auth gate 入口（任何
// 未登录访问都 fall through 到这里，懒加载会闪一帧 fallback），后者是
// 兜底 404，体积本来就小。其它业务页全部 React.lazy 拆 chunk —— G6
// (~11 MB)、pdfjs (~36 MB)、katex (~4 MB) 这些大依赖只在用户真访问
// 对应路由时才拉，首包从 ~10 MB 量级降到几 MB 量级。
//
// 走 lazyWithRetry 而不是裸 lazy：部署后旧 HTML 拉不到旧 chunk URL
// 时 reload 一次，避免把"chunk 404"暴露成全屏 ErrorBoundary。
import LoginPage from "@/pages/LoginPage";
import NotFoundPage from "@/pages/NotFoundPage";
import RegisterPage from "@/pages/RegisterPage";

const ChatPage = lazyWithRetry(() => import("@/pages/ChatPage"), "ChatPage");
const FilesPage = lazyWithRetry(() => import("@/pages/FilesPage"), "FilesPage");
const GraphPage = lazyWithRetry(() => import("@/pages/GraphPage"), "GraphPage");
const GraphSandboxPage = lazyWithRetry(
  () => import("@/pages/GraphSandboxPage"),
  "GraphSandboxPage",
);
const SearchPage = lazyWithRetry(() => import("@/pages/workbench/SearchPage"), "SearchPage");
const ProductsPage = lazyWithRetry(
  () => import("@/pages/workbench/ProductsPage"),
  "ProductsPage",
);
const RiskPredictionPage = lazyWithRetry(
  () => import("@/pages/workbench/RiskPredictionPage"),
  "RiskPredictionPage",
);
const PolicyCalcPage = lazyWithRetry(
  () => import("@/pages/workbench/PolicyCalcPage"),
  "PolicyCalcPage",
);
const UsersPage = lazyWithRetry(() => import("@/pages/admin/UsersPage"), "UsersPage");
const ConfigPage = lazyWithRetry(() => import("@/pages/admin/ConfigPage"), "ConfigPage");
const AuditPage = lazyWithRetry(() => import("@/pages/admin/AuditPage"), "AuditPage");

/**
 * 路由切换 fallback：占位仅在主区域显示，sidebar / topbar 由
 * LayoutShell 保持挂载，所以视觉抖动只在内容区。chunk fetch 命中
 * 缓存时几乎察觉不到；首次访问大页（GraphPage）能看见 spinner。
 */
function RouteFallback() {
  return (
    <div className="flex flex-1 items-center justify-center text-ink-muted">
      <Loader2 className="h-4 w-4 animate-spin" aria-label="加载中" />
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter
        future={{
          // 提前开 v7 行为，避免后续 v7 升级再统一改一遍。当前路
          // 由没有复杂 splat / suspense，迁移成本接近零。
          v7_startTransition: true,
          v7_relativeSplatPath: true,
        }}
      >
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          <Route
            element={
              <RequireAuth>
                <LayoutShell />
              </RequireAuth>
            }
          >
            <Route index element={<Navigate to="/chat" replace />} />
            {/*
             * Suspense 包在每个 lazy 路由元素上，比把整个 <Routes>
             * 包一层颗粒度更细：route 切换时只有 outlet 区域闪
             * fallback，sidebar / topbar 不受影响。下面 element 都
             * 复用同一个 RouteFallback 占位。
             */}
            <Route
              path="/chat"
              element={<Suspense fallback={<RouteFallback />}><ChatPage /></Suspense>}
            />
            <Route
              path="/search"
              element={<Suspense fallback={<RouteFallback />}><SearchPage /></Suspense>}
            />
            <Route
              path="/products"
              element={<Suspense fallback={<RouteFallback />}><ProductsPage /></Suspense>}
            />
            <Route
              path="/risk"
              element={<Suspense fallback={<RouteFallback />}><RiskPredictionPage /></Suspense>}
            />
            <Route
              path="/policy-calc"
              element={<Suspense fallback={<RouteFallback />}><PolicyCalcPage /></Suspense>}
            />
            <Route
              path="/graph"
              element={<Suspense fallback={<RouteFallback />}><GraphPage /></Suspense>}
            />
            <Route
              path="/graph/_sandbox"
              element={<Suspense fallback={<RouteFallback />}><GraphSandboxPage /></Suspense>}
            />
            <Route
              path="/files"
              element={<Suspense fallback={<RouteFallback />}><FilesPage /></Suspense>}
            />

            <Route
              path="/admin/users"
              element={
                <RequireAdmin>
                  <Suspense fallback={<RouteFallback />}>
                    <UsersPage />
                  </Suspense>
                </RequireAdmin>
              }
            />
            <Route
              path="/admin/config"
              element={
                <RequireAdmin>
                  <Suspense fallback={<RouteFallback />}>
                    <ConfigPage />
                  </Suspense>
                </RequireAdmin>
              }
            />
            <Route
              path="/admin/audit"
              element={
                <RequireAdmin>
                  <Suspense fallback={<RouteFallback />}>
                    <AuditPage />
                  </Suspense>
                </RequireAdmin>
              }
            />
          </Route>

          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
