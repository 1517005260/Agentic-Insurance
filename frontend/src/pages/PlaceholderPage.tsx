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
            {hint ?? "本页面尚未实现。"}
          </CardSubtitle>
        </CardHeader>
        <CardBody>
          <p className="text-sm text-ink-muted leading-relaxed">
            后端接口已就绪，前端界面待实现。
          </p>
        </CardBody>
      </Card>
    </div>
  );
}
