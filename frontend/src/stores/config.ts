/**
 * stores/config.ts — admin 配置中心 react-query 接入。
 *
 * 设计要点：
 *  - 一次拉 GET /admin/config 同时拿 schema + snapshot（一个 round-trip）
 *  - PATCH 走 batch all-or-422，前端只在用户点"保存"时刷一次
 *  - DELETE /admin/config/{key} 复位单 key，按 ConfigPage 行的"重置"按钮触发
 *  - 不持久化到 localStorage —— admin 配置是服务器端权威；本地缓存只
 *    服务 staleTime，避免下方 admin 页面的频繁重新拉
 */
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { api } from "@/api/client";

export type ConfigType = "int" | "str" | "float" | "bool";
export type ConfigValue = number | string | boolean;

export interface ConfigEntry {
  key: string;
  type: ConfigType;
  default: ConfigValue;
  description: string;
  group: string;
  min: number | null;
  max: number | null;
  min_length: number | null;
  max_length: number | null;
}

export interface ConfigSnapshotResponse {
  snapshot: Record<string, ConfigValue>;
  /** 后端用 alias `schema` 序列化 schema_ 字段，前端按 alias 接 */
  schema: ConfigEntry[];
}

export interface ConfigPatchResponse {
  diffs: Record<string, { old: unknown; new: unknown }>;
  snapshot: Record<string, ConfigValue>;
}

export const adminConfigKey = ["admin", "config"] as const;

export function useAdminConfig() {
  return useQuery<ConfigSnapshotResponse>({
    queryKey: adminConfigKey,
    queryFn: async () => {
      const { data } = await api.get<ConfigSnapshotResponse>("/admin/config");
      return data;
    },
    staleTime: 30_000,
  });
}

export function useAdminConfigPatch() {
  const qc = useQueryClient();
  return useMutation<ConfigPatchResponse, Error, Record<string, ConfigValue>>({
    mutationFn: async (updates) => {
      const { data } = await api.patch<ConfigPatchResponse>("/admin/config", {
        updates,
      });
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: adminConfigKey });
    },
  });
}

export function useAdminConfigReset() {
  const qc = useQueryClient();
  return useMutation<void, Error, string>({
    mutationFn: async (key) => {
      await api.delete(`/admin/config/${encodeURIComponent(key)}`);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: adminConfigKey });
    },
  });
}
