import { useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { Loader2, AlertTriangle } from "lucide-react";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

// pdfjs worker — vite 会把 ?url import 打到 dist/assets/，开发期相对路径也 OK
import pdfWorkerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";
pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

import type { AxiosResponse } from "axios";

import { api } from "@/api/client";

interface Props {
  fileId: string;
  pageNumber: number;
  /** 容器宽度（px）。Drawer 内传死宽避免 ResizeObserver 风暴。 */
  width: number;
}

/**
 * 单页 PDF 渲染。
 *
 * 关键点：
 *   1. 缓存当前 file_id 的 ArrayBuffer，pageNumber 切换时不重新拉文件 —
 *      只是 react-pdf <Page> 改 pageNumber，PDFDocumentProxy 复用。
 *   2. 走 axios apiClient（带 JWT）拉 `/files/{id}/download`，response 类型
 *      `arraybuffer`；不能让 react-pdf 自己 fetch（缺 Authorization 头）。
 *   3. ArrayBuffer 是 transferable —— 第一次给 Document 后 react-pdf 会
 *      detach 它，第二次同 file 切 page 不会真的重新 parse PDF（react-pdf
 *      内部按 `data` 引用稳定决定是否重建）。我们把 buffer 用 `useRef` 锁
 *      在同一个 file_id 上。
 */
export default function PdfPageViewer({ fileId, pageNumber, width }: Props) {
  const [data, setData] = useState<{ fileId: string; buf: ArrayBuffer } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const reqIdRef = useRef(0);

  useEffect(() => {
    if (data?.fileId === fileId) return;

    const reqId = ++reqIdRef.current;
    const ctrl = new AbortController();
    // setLoading/setError sync 在 effect 是 fetch UI 的标准模式：
    // fileId 一变就要把上一份 PDF 切到 loading 态。React 19 推荐用
    // Suspense + use() 解掉这个 lint，但那是更大的改造，且会卷入
    // CitationDrawer 全树。当前两条状态翻转是必要的。
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setError(null);

    api
      .get(`/files/${encodeURIComponent(fileId)}/download`, {
        responseType: "arraybuffer",
        signal: ctrl.signal,
      })
      .then((resp: AxiosResponse<ArrayBuffer>) => {
        if (reqId !== reqIdRef.current) return; // 已被新请求覆盖
        setData({ fileId, buf: resp.data });
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (reqId !== reqIdRef.current) return;
        // axios v1 把 abort 包成 CanceledError，不展示给用户
        const e = err as { code?: string; name?: string; message?: string };
        if (e?.code === "ERR_CANCELED" || e?.name === "CanceledError") return;
        setError(e?.message ?? "PDF 加载失败");
        setLoading(false);
      });

    // 关闭抽屉 / 切 fileId 时取消未完成的下载，省带宽 + 防 unmount setState
    return () => ctrl.abort();
  }, [fileId, data?.fileId]);

  // pdf.js 把 ArrayBuffer 当 transferable 转给 worker —— 一旦 transfer，state
  // 里这份 buffer 就 detached（再访问抛 TypeError）。对每个 (fileId, buf)
  // 组合 memoize 一份独立 slice，让 react-pdf 的 `file` prop 引用稳定（避免
  // resize / pageNumber 切换 / parent re-render 触发 PDF 重 load），同时不会
  // 把我们 state 里的原 buf 也 detach 掉。
  const docFile = useMemo(
    () => (data ? data.buf.slice(0) : null),
    [data],
  );

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-ink-muted">
        <Loader2 className="h-5 w-5 animate-spin" />
        <span className="mt-3 text-sm">加载 PDF…</span>
      </div>
    );
  }
  if (error) {
    return (
      <div className="flex items-start gap-2 rounded-md bg-danger-soft border border-danger/20 px-3 py-2 mx-4 my-6 text-sm text-danger">
        <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
        <span className="break-words">{error}</span>
      </div>
    );
  }
  if (!data) return null;

  return (
    <Document
      file={docFile}
      loading={
        <div className="flex items-center justify-center py-16 text-ink-muted">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      }
      onLoadError={(err) => setError(err?.message ?? "PDF 解析失败")}
      className="flex flex-col items-center"
    >
      <Page
        pageNumber={pageNumber}
        width={width}
        renderAnnotationLayer={false}
        renderTextLayer
        loading={
          <div className="flex items-center justify-center py-16 text-ink-muted">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        }
        className="shadow-card border border-ink-line/60 rounded-sm bg-white"
      />
    </Document>
  );
}
