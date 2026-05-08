"""Exclusion / underwriting audit — ProofAgent + customer profile."""
from typing import AsyncIterator, List

from agentic.agent.proof_agent import ProofAgent
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


def _build_user_prompt(file_id: str, customer: CustomerProfile) -> str:
    return (
        f"请对产品 file_id=`{file_id}` 进行核保审计。\n\n"
        f"## 客户档案\n{_format_profile(customer)}\n\n"
        "工作流（严格按 proof loop）：\n"
        "1. `proof_plan_init`，定义一个 forall(exclusion clauses) 的 obligation，"
        "scope 限定该 file_id。\n"
        "2. `proof_scan` / `read` 列出该产品所有除外条款。\n"
        "3. 对每条 exclusion，匹配客户档案字段，ingest 对应 claim "
        "(triggered: yes/no)，cite 条款 verbatim。\n"
        "4. `proof_finalize` 出最终结论。\n"
        "5. 输出: 每条 exclusion 一行的 markdown 表 + `## 核保结论` 段。"
    )


def stream_exclusion_audit(
    *,
    file_id: str,
    customer: CustomerProfile,
    agent: ProofAgent,
    config: ConfigStore,
    tracer=None,
    result_future=None,
) -> AsyncIterator[bytes]:
    user_prompt = _build_user_prompt(file_id, customer)
    return stream_workbench_agent(
        user_prompt=user_prompt,
        agent=agent,
        kind="proof",
        config=config,
        prompt_key="prompt.exclusion_audit",
        flavor="exclusion",
        final_extras={
            "file_id": file_id,
            "customer_age": customer.age,
            "customer_occupation": customer.occupation,
        },
        tracer=tracer,
        result_future=result_future,
    )
