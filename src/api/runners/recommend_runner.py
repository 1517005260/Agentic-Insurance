"""Needs-analysis runner — BaseAgent over the open corpus.

When the analyst supplies ``held_policies_file_ids``, the runner
prepends a "held coverage" block so the agent does gap analysis (read
the existing policies → summarize covered risks → recommend
complementary products only) instead of generic top-3 selection. With
no held policies the prompt collapses to the original open-corpus
wording so existing behaviour is preserved.
"""
from typing import AsyncIterator, List, Optional

from agentic.agent.base import BaseAgent
from api.runners._workbench import stream_workbench_agent
from api.schemas.insurance import CustomerProfile
from config.config_store import ConfigStore


def _format_profile(p: CustomerProfile) -> str:
    """Pretty-print the customer profile as a labeled block."""
    parts: List[str] = [
        f"- 年龄: {p.age}",
        f"- 性别: {p.gender}",
        f"- 职业: {p.occupation}",
    ]
    if p.occupation_risk:
        parts.append(f"- 职业风险: {p.occupation_risk}")
    if p.health_history:
        parts.append(f"- 病史: {', '.join(p.health_history)}")
    if p.family_history:
        parts.append(f"- 家族史: {', '.join(p.family_history)}")
    if p.budget_annual is not None:
        parts.append(f"- 年预算: {p.budget_annual}")
    if p.goal:
        parts.append(f"- 主诉求: {p.goal}")
    if p.notes:
        parts.append(f"- 补充说明: {p.notes}")
    return "\n".join(parts)


def _build_user_prompt(
    customer: CustomerProfile,
    held_policies_file_ids: Optional[List[str]],
) -> str:
    held = [fid for fid in (held_policies_file_ids or []) if fid]
    if not held:
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

    held_block = "\n".join(f"- {fid}" for fid in held)
    return (
        "请基于客户档案 + 已持有的保单，做缺口分析并推荐补足产品。\n\n"
        f"## 客户档案\n{_format_profile(customer)}\n\n"
        f"## 已持有保单 (file_ids)\n{held_block}\n\n"
        "工作流：\n"
        "1. 对上面每个 file_id 做 `read`，提取险种 / 主要保障范围 / 保额 / "
        "等待期 / 关键除外。把这些汇总为 `## 现有保障` 表格（一个 file_id "
        "一行），关键数字 [^k] 引用。\n"
        "2. 对照客户档案 + 主诉求，列出 `## 保障缺口` —— 哪些风险类别还没"
        "覆盖，或现有保额明显低于客户预算 / 家族病史风险。\n"
        "3. `list_files` 看看语料里还有什么候选产品；用 `semantic_search` "
        "/ `bm25_search` 围绕缺口找补充候选。\n"
        "4. `read` 候选产品的核心条款，确认与现有保单**功能互补、不重叠**。\n"
        "5. 推荐 top-2~3 个补足产品，每个给出 `补足缺口 / 关键条款 / 年保费"
        "估算 / 与现有保单的协同方式` 四项，关键事实 [^k] 引用。\n"
        "6. 末尾 `## 注意事项` 段，提醒可能与已有保单产生冲突或重复的条款。\n"
        "若语料中没有合适的补足产品，明示 + 给出"
        "客户应该向哪类产品询价的方向。"
    )


def stream_recommend(
    *,
    customer: CustomerProfile,
    agent: BaseAgent,
    config: ConfigStore,
    held_policies_file_ids: Optional[List[str]] = None,
    tracer=None,
    result_future=None,
) -> AsyncIterator[bytes]:
    user_prompt = _build_user_prompt(customer, held_policies_file_ids)
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
            "held_policies_count": len(held_policies_file_ids or []),
        },
        tracer=tracer,
        result_future=result_future,
    )
