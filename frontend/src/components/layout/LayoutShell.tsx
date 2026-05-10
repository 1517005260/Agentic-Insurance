import { Outlet } from "react-router-dom";

import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";
import { CitationDrawer } from "@/components/citation/CitationDrawer";

export default function LayoutShell() {
  return (
    <div className="h-screen w-screen flex bg-surface">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <Topbar />
        {/*
         * main 不直接滚动 —— 让子页面自己决定滚动容器。这样 ChatPage
         * 这种 "MessageList 自己管 overflow + Composer 固定底部" 的
         * 布局不会撞双滚动条。
         *
         * 子页面如果不需要自管滚动（如 PlaceholderPage），自己包一层
         * `overflow-y-auto`。
         */}
        <main className="flex-1 min-h-0 flex flex-col">
          <Outlet />
        </main>
      </div>

      {/*
       * 引用抽屉是全站全局通道：Chat 答案里 [^k] sup、agent evidence chip、
       * 工作台矩阵 cell、图谱 passage 节点浮卡里点击都汇到这里。挂在
       * LayoutShell 最外层而不是 ChatPage，是为了让 SearchPage / Compare
       * / GraphPage 等共用一份 zustand store + DOM 节点。
       */}
      <CitationDrawer />
    </div>
  );
}
