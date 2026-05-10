import { Card, CardBody, CardHeader, CardSubtitle, CardTitle } from "@/components/ui/card";

interface Props {
  title: string;
  hint?: string;
}

export default function PlaceholderPage({ title, hint }: Props) {
  return (
    <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin px-8 py-10 max-w-3xl">
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardSubtitle>
            {hint ?? "本页面将在 Phase 2-5 实现。"}
          </CardSubtitle>
        </CardHeader>
        <CardBody>
          <p className="text-sm text-ink-muted leading-relaxed">
            后端接口已就绪。当前为 Phase 1 骨架验证用占位页 ——
            真实实现将在对应 Phase 落地，详见{" "}
            <code className="px-1.5 py-0.5 bg-surface-sunk rounded text-[12px] text-ink">
              docs/webapp.md §15A
            </code>
            。
          </p>
        </CardBody>
      </Card>
    </div>
  );
}
