/**
 * stores/ingestQueue.ts — track ingest jobs whose progress dialog is hidden.
 *
 * Why: the upload + reingest progress UIs were modal-blocking — once a
 * user kicked off ingest they couldn't dismiss the dialog without
 * losing the SSE stream. The bg task keeps running on the backend
 * regardless of whether anyone is watching, but the user had no way
 * to "minimize" the progress view and come back later.
 *
 * This store records the file_ids that have an active (non-terminal)
 * ingest whose dialog the user dismissed. The FilesPage reads it to
 * render a "索引中 (n)" chip; clicking the chip re-opens
 * ProgressDrawer for the chosen file (a fresh GET /files/{id}/jobs/stream
 * subscription). The backend EventBus is single-consumer, so reopening
 * works only after the previous client connection has been GC'd; in
 * practice this happens in the same animation frame as dialog unmount,
 * so a manual "minimize → restore" round-trip is reliable.
 *
 * Entries auto-evict via ``syncWithFiles`` on every /files refetch:
 * any tracked file whose status reached ready / failed / deleting (or
 * whose row vanished) is dropped, so the chip count stays honest even
 * when the user never re-opens the drawer.
 */
import { create } from "zustand";

import type { FileRow } from "@/hooks/useFiles";

export interface IngestEntry {
  fileId: string;
  displayName: string;
  variant: "fresh" | "reingest";
  /** epoch ms, for the chip's "started 2m ago" label */
  startedAt: number;
}

interface IngestQueueState {
  entries: Record<string, IngestEntry>;
  enqueue: (entry: IngestEntry) => void;
  dequeue: (fileId: string) => void;
  /**
   * Drop entries whose backing file has reached a terminal status
   * (ready / failed / deleting / not-listed). Called by FilesPage on
   * every /files refetch so the chip count never lies.
   *
   * Treats ``deleting`` as terminal because the user can't open a
   * progress view for a delete (no stage timeline) and we want the
   * chip to stop tracking it as "indexing".
   */
  syncWithFiles: (rows: FileRow[]) => void;
}

const TERMINAL_STATUSES: ReadonlySet<string> = new Set([
  "ready",
  "failed",
  "deleting",
]);

export const useIngestQueue = create<IngestQueueState>((set) => ({
  entries: {},
  enqueue: (entry) =>
    set((s) => ({ entries: { ...s.entries, [entry.fileId]: entry } })),
  dequeue: (fileId) =>
    set((s) => {
      if (!(fileId in s.entries)) return s;
      const { [fileId]: _drop, ...rest } = s.entries;
      void _drop;
      return { entries: rest };
    }),
  syncWithFiles: (rows) =>
    set((s) => {
      const byId = new Map(rows.map((r) => [r.file_id, r] as const));
      let mutated = false;
      const next: Record<string, IngestEntry> = {};
      for (const [fileId, entry] of Object.entries(s.entries)) {
        const row = byId.get(fileId);
        if (!row || TERMINAL_STATUSES.has(row.status)) {
          mutated = true;
          continue;
        }
        next[fileId] = entry;
      }
      return mutated ? { entries: next } : s;
    }),
}));
