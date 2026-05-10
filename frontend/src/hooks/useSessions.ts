/**
 * react-query 接 chat session 端点。
 *
 * 设计：
 *  - list（GET /chat/sessions）:用户拥有的 session 列表，倒序按 updated_at
 *  - detail（GET /chat/sessions/{id}）：session header + messages list
 *  - create（POST /chat/sessions）+ patch（PATCH .../{id}）+ delete
 *  - 所有 mutation 成功后 invalidate ['chat-sessions']
 *
 * staleTime 30s：admin 单人 demo 不会频繁改，列表也不需要实时刷
 */
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { api } from "@/api/client";

export type SessionMode = "rag" | "agent";
export type AgentKind = "base" | "proof" | "graph";

export interface SessionRow {
  id: number;
  title: string;
  mode: SessionMode;
  agent_kind: AgentKind | null;
  web: boolean;
  created_at: string;
  updated_at: string;
}

export interface MessageRow {
  id: number;
  role: "user" | "assistant" | "tool" | "system";
  content: string;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface SessionDetail {
  session: SessionRow;
  messages: MessageRow[];
}

export interface SessionCreate {
  mode: SessionMode;
  agent_kind?: AgentKind | null;
  web?: boolean;
  title?: string;
}

export const sessionsKey = ["chat-sessions"] as const;
export const sessionDetailKey = (id: number) => ["chat-sessions", id] as const;

export function useSessions() {
  return useQuery<SessionRow[]>({
    queryKey: sessionsKey,
    queryFn: async () => {
      const { data } = await api.get<SessionRow[]>("/chat/sessions");
      return data;
    },
    staleTime: 30_000,
  });
}

export function useSessionDetail(id: number | null) {
  return useQuery<SessionDetail>({
    queryKey: id != null ? sessionDetailKey(id) : ["chat-sessions", "none"],
    queryFn: async () => {
      const { data } = await api.get<SessionDetail>(`/chat/sessions/${id}`);
      return data;
    },
    enabled: id != null,
    staleTime: 0,
  });
}

export function useCreateSession() {
  const qc = useQueryClient();
  return useMutation<SessionRow, Error, SessionCreate>({
    mutationFn: async (body) => {
      const { data } = await api.post<SessionRow>("/chat/sessions", body);
      return data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: sessionsKey }),
  });
}

export function useRenameSession() {
  const qc = useQueryClient();
  return useMutation<SessionRow, Error, { id: number; title: string }>({
    mutationFn: async ({ id, title }) => {
      const { data } = await api.patch<SessionRow>(`/chat/sessions/${id}`, {
        title,
      });
      return data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: sessionsKey }),
  });
}

export function useDeleteSession() {
  const qc = useQueryClient();
  return useMutation<void, Error, number>({
    mutationFn: async (id) => {
      await api.delete(`/chat/sessions/${id}`);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: sessionsKey }),
  });
}
