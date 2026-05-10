import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";

import { RequireAdmin, RequireAuth } from "@/components/auth/guards";
import LayoutShell from "@/components/layout/LayoutShell";
import { queryClient } from "@/lib/queryClient";
import ChatPage from "@/pages/ChatPage";
import LoginPage from "@/pages/LoginPage";
import NotFoundPage from "@/pages/NotFoundPage";
import FilesPage from "@/pages/FilesPage";
import GraphPage from "@/pages/GraphPage";
import GraphSandboxPage from "@/pages/GraphSandboxPage";
import SearchPage from "@/pages/workbench/SearchPage";
import ProductsPage from "@/pages/workbench/ProductsPage";
import RiskPredictionPage from "@/pages/workbench/RiskPredictionPage";
import PolicyCalcPage from "@/pages/workbench/PolicyCalcPage";
import UsersPage from "@/pages/admin/UsersPage";
import ConfigPage from "@/pages/admin/ConfigPage";
import AuditPage from "@/pages/admin/AuditPage";

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

          <Route
            element={
              <RequireAuth>
                <LayoutShell />
              </RequireAuth>
            }
          >
            <Route index element={<Navigate to="/chat" replace />} />
            <Route path="/chat"          element={<ChatPage />} />
            <Route path="/search"        element={<SearchPage />} />
            <Route path="/products"      element={<ProductsPage />} />
            <Route path="/risk"          element={<RiskPredictionPage />} />
            <Route path="/policy-calc"   element={<PolicyCalcPage />} />
            <Route path="/graph"           element={<GraphPage />} />
            <Route path="/graph/_sandbox"  element={<GraphSandboxPage />} />
            <Route path="/files"       element={<FilesPage />} />

            <Route
              path="/admin/users"
              element={
                <RequireAdmin>
                  <UsersPage />
                </RequireAdmin>
              }
            />
            <Route
              path="/admin/config"
              element={
                <RequireAdmin>
                  <ConfigPage />
                </RequireAdmin>
              }
            />
            <Route
              path="/admin/audit"
              element={
                <RequireAdmin>
                  <AuditPage />
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
