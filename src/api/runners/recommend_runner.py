"""Product recommendation — BaseAgent over the open corpus."""
from typing import AsyncIterator

from agentic.agent.base import BaseAgent
from api.runners._workbench import stream_workbench_agent
from api.runners.exclusion_runner import _format_profile
from api.schemas.insurance import CustomerProfile
from config.config_store import ConfigStore


def _build_user_prompt(customer: CustomerProfile) -> str:
    """No file_ids here — the agent self-discovers via list_files."""
    return (
        "请根据客户档案，从已索引的所有产品中推荐 top-3 最匹配的产品。\n\n"
        f"## 客户档案\n{_format_profile(customer)}\n\n"
        "工作流：\n"
        "1. `list_files` 列出语料中所有产品。\n"
        "2. 针对客户主诉求 (goal) 做 `semantic_search` / `bm25_search`，"
        "聚合候选产品。\n"
        "3. 对候选产品的核心条款做 `read`，确认年龄、保费、保障范围、"
        "等待期是否匹配客户。\n"
        "4. 排出 top-3，每个产品给出 `适配理由 / 关键条款 / 年保费估算 / "
        "注意事项` 四项，关键事实必须 [^k] 引用。\n"
        "5. 末尾 `## 风险提醒` 段，列出客户档案触发的潜在 exclusion 项。\n"
        "若候选不足 3 个，明示 + 列出可选的次优产品。"
    )


def stream_recommend(
    *,
    customer: CustomerProfile,
    agent: BaseAgent,
    config: ConfigStore,
    tracer=None,
    result_future=None,
) -> AsyncIterator[bytes]:
    user_prompt = _build_user_prompt(customer)
    return stream_workbench_agent(
        user_prompt=user_prompt,
        agent=agent,
        kind="base",
        config=config,
        prompt_key="prompt.recommend",
        flavor="recommend",
        final_extras={
            "customer_age": customer.age,
            "customer_goal": customer.goal,
        },
        tracer=tracer,
        result_future=result_future,
    )
