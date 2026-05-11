import { lazy, type ComponentType, type LazyExoticComponent } from "react";

/**
 * `React.lazy` 包一层 reload-once 兜底，应付"部署后旧 HTML 拉不到旧
 * chunk URL"的常见 race。
 *
 * 场景：用户开着浏览器，后端发了新版本，CDN 上旧 chunk 已被替换 →
 * 用户切路由触发 lazy import → 404 / module-script-load failed →
 * `React.lazy` throw → 全局 ErrorBoundary 接管成"出错了"提示。这种
 * 情况下其实 reload 一次就好（拿到新 HTML + 新 chunk URL）。
 *
 * 策略：
 *  - 第一次失败：sessionStorage 写入 timestamp，调用
 *    `window.location.reload()` 触发重新加载；返回一个永不 resolve
 *    的 promise，让 React.lazy 不会同步拿到 reject 而抢在 reload
 *    之前 unwind。``RELOAD_TIMEOUT_MS`` 兜底：如果 reload 真的没
 *    发生（极端环境），超时后 reject 给 ErrorBoundary。
 *  - 第二次失败（``FLAG_TTL_MS`` 内已记录过 timestamp）：清掉标记，
 *    throw 让 ErrorBoundary 接管 —— 这次 reload 也救不了。
 *  - 失败但 timestamp 已过期：当作"第一次"再 reload 一次。flag 用
 *    timestamp 而非 boolean，避免"用户手动 reload 后 flag 残留" →
 *    把无关的下次真失败误判为"第二次"。
 *  - 加载成功：清掉标记，下次再失败仍享受 reload-once。
 *
 * `key` 必须每个 lazy 调用唯一（用页面名即可），否则两个 lazy 共用
 * 一份 timestamp 会互相挡。
 */

/** 视为同一波 deploy-race 的窗口；窗口外的失败重新算"第一次"。 */
const FLAG_TTL_MS = 60_000;
/** reload 触发后多久仍未换页就视为 reload 自身失败，throw 兜底。 */
const RELOAD_TIMEOUT_MS = 10_000;

export function lazyWithRetry<T extends ComponentType<unknown>>(
  importFn: () => Promise<{ default: T }>,
  key: string,
): LazyExoticComponent<T> {
  const flagKey = `lazy_retry_${key}`;
  return lazy(() =>
    importFn()
      .then((mod) => {
        if (typeof sessionStorage !== "undefined") {
          sessionStorage.removeItem(flagKey);
        }
        return mod;
      })
      .catch((err: unknown) => {
        const isWithinFlagWindow = (): boolean => {
          if (typeof sessionStorage === "undefined") return false;
          const raw = sessionStorage.getItem(flagKey);
          if (!raw) return false;
          const ts = Number(raw);
          if (!Number.isFinite(ts)) return false;
          return Date.now() - ts < FLAG_TTL_MS;
        };

        if (typeof window !== "undefined" && !isWithinFlagWindow()) {
          if (typeof sessionStorage !== "undefined") {
            sessionStorage.setItem(flagKey, String(Date.now()));
          }
          window.location.reload();
          // reload 是异步换页；在被换页之前不能 reject —— 否则
          // React.lazy 会 surface 错误到 ErrorBoundary 闪一下"出错"
          // 才 reload。返回一个永不 resolve 的 promise 让 Suspense
          // 持续显 fallback，直到 reload 真正发生。RELOAD_TIMEOUT_MS
          // 兜底：reload 因任何原因没换页（CSP block / iframe 限制
          // / extension 拦截），超时后把控制权交回 ErrorBoundary，
          // 至少不会一直转圈。
          return new Promise<{ default: T }>((_, reject) => {
            setTimeout(() => {
              reject(
                err instanceof Error
                  ? err
                  : new Error(`lazy chunk reload did not navigate within ${RELOAD_TIMEOUT_MS}ms`),
              );
            }, RELOAD_TIMEOUT_MS);
          });
        }
        // window 缺失（SSR / 测试环境）或已经在 TTL 窗口内重试过：不再
        // 救，把原始错误抛给上层 ErrorBoundary。同时清掉 flag，让下
        // 一波 deploy 还能享受 reload-once。
        if (typeof sessionStorage !== "undefined") {
          sessionStorage.removeItem(flagKey);
        }
        throw err;
      }),
  );
}
