/**
 * Hand-written API surface — kept thin to avoid drift from the
 * backend Pydantic schemas.
 *
 * Phase 6.D added a codegen escape hatch: ``pnpm gen:api`` regenerates
 * ``src/api/types.gen.ts`` (3000+ lines: full ``paths`` + ``components``
 * map) straight from ``http://localhost:8000/openapi.json``. Use it
 * for any new component:
 *
 *   import type { components } from "@/api/types.gen";
 *   type FileOut = components["schemas"]["FileOut"];
 *
 * The hand-written types below stay because they're load-bearing for
 * the auth bootstrap path (``stores/auth.ts``, ``api/auth.ts``);
 * migrate to the generated forms lazily as consumers change.
 *
 * Naming convention: same as backend Pydantic (MeOut → User) for easy swap.
 */

export type UserRole = "admin" | "analyst";

/** GET /auth/me — also reused as the user object stashed in auth store. */
export interface User {
  id: number;
  username: string;
  role: UserRole;
  is_active: boolean;
}

export interface LoginRequest {
  username: string;
  password: string;
}

/**
 * Self-registration payload. `role` is fixed server-side (`analyst`)
 * so the frontend doesn't send one — the backend ignores anything
 * the client would put here.
 */
export interface RegisterRequest {
  username: string;
  password: string;
}

/**
 * 后端 `TokenOut` 是 flat shape；登录成功后我们再调一次 /auth/me 把
 * 完整 user 拿回来，避免在前端解 JWT。
 */
export interface TokenOut {
  access_token: string;
  token_type: string; // "bearer"
  role: UserRole;
  username: string;
}

/** /files endpoints — Phase 1 only needs a list shape. */
export interface FileRecord {
  file_id: string;
  filename: string;
  suffix: string;
  size_bytes: number;
  status: "uploading" | "queued" | "ingesting" | "ready" | "failed";
  error_msg?: string | null;
  created_at: string;
  updated_at: string;
}

/** Generic pydantic-style validation error. */
export interface APIError {
  detail:
    | string
    | Array<{ loc: (string | number)[]; msg: string; type: string }>;
}
