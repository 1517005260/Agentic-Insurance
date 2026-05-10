import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";

export default function NotFoundPage() {
  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-surface px-4">
      <div className="max-w-md text-center">
        <div className="font-serif text-6xl text-primary-700">404</div>
        <p className="mt-3 text-ink-muted">
          路径不存在或已被删除。
        </p>
        <div className="mt-6">
          <Link to="/chat">
            <Button variant="secondary">返回主页</Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
