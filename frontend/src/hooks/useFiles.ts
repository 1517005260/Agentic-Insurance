import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";

import { api } from "@/api/client";
import { useIngestQueue } from "@/stores/ingestQueue";

/**
 * 已索引的保单 / 文档列表（来自 /files）。
 *
 * 工作台多文件多选用同一个数据源；放 react-query 共享缓存避免每个
 * 工作台各拉一次。
 *
 * 轮询策略：
 *  - 默认 staleTime 5 分钟（demo 期没人主动刷文件列表也不会过陈）。
 *  - 当 ingestQueue 非空（有 minimized 索引任务），把 staleTime 拉到
 *    0 + refetchInterval 5s。理由：minimized chip 靠 syncWithFiles 在
 *    /files 拿到新数据时清掉过期条目；如果不主动短轮询，已 ready 的
 *    file 会一直被 chip 假装"索引中"直到下次手动刷新。空了恢复默认。
 */
export interface FileRow {
  file_id: string;
  display_name: string;
  original_filename: string;
  suffix: string;
  byte_size: number;
  page_count: number | null;
  status: string;
}

export function useFiles() {
  const ingestActive = useIngestQueue(
    (s) => Object.keys(s.entries).length > 0,
  );
  const syncIngestQueue = useIngestQueue((s) => s.syncWithFiles);
  const query = useQuery<FileRow[]>({
    queryKey: ["files"],
    queryFn: async () => {
      const { data } = await api.get<FileRow[]>("/files");
      return data;
    },
    staleTime: ingestActive ? 0 : 5 * 60 * 1000,
    refetchInterval: ingestActive ? 5_000 : false,
    refetchIntervalInBackground: false,
  });

  // Sync the ingest queue from EVERY useFiles consumer (workbench
  // FileMultiSelect, FilesPage, sidebar) — we used to wire this only
  // in FilesPage, which meant a user who minimized an ingest and then
  // navigated to a workbench would keep paying the 5s polling cost
  // forever (chip never drained because syncWithFiles was never
  // called). Hooking it here makes any /files refetch — wherever it
  // happens — also drain the queue.
  useEffect(() => {
    if (query.data) syncIngestQueue(query.data);
  }, [query.data, syncIngestQueue]);

  return query;
}

/**
 * 给文件下拉只显示 status==='ready' 的（pending / indexing 没建好索引，
 * 选了也跑不通；failed 同理）。
 */
export function useReadyFiles() {
  const q = useFiles();
  return {
    ...q,
    data: q.data?.filter((f) => f.status === "ready") ?? undefined,
  };
}
