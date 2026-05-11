import { useEffect, useState } from "react";

/**
 * IntersectionObserver 的 React 包装：观察 ref 指向的元素是否进入视口。
 *
 * 设计：
 *  - `rootMargin` 字符串走 IntersectionObserverInit 的标准语法，例
 *    如 "200px" 让元素在距视口 200px 时就算"已可见"，便于做 lazy
 *    fetch 的提前量。
 *  - `triggerOnce=true` 时一旦命中就 disconnect —— 适合"载入即留存"
 *    的资源（缩略图、图片预览）；`false` 走双向 toggle，适合纯渲染
 *    控制（虚拟列表 / 暂停动画）。
 *
 * 不支持 `root` / `threshold` 自定义：当前调用方都是默认 viewport +
 * 默认 0 阈值；YAGNI，等真有需求再加。
 */
export function useInView<T extends Element>(
  ref: React.RefObject<T | null>,
  rootMargin = "0px",
  triggerOnce = false,
): boolean {
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          if (triggerOnce) obs.disconnect();
        } else if (!triggerOnce) {
          setInView(false);
        }
      },
      { rootMargin },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [ref, rootMargin, triggerOnce]);

  return inView;
}
