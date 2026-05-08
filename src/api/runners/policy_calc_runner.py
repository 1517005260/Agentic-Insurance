"""Policy actuarial calculation — BaseAgent + code_run.

Workbench-specific user prompt + system prompt heavily emphasise
that arithmetic MUST go through ``code_run`` (sandbox has
``numpy_financial`` for IRR / NPV / PMT / PV / FV / RATE, plus
``decimal``, ``scipy``, ``sympy``). The agent is expected to
``read`` the policy's cash-value / illustration table once, hand
the values into ``code_run`` as ``INPUTS``, and surface the
``OUTPUT`` dict as a markdown section.
"""
from typing import AsyncIterator, List

from agentic.agent.base import BaseAgent
from api.runners._workbench import stream_workbench_agent
from api.schemas.insurance import PolicyParams
from config.config_store import ConfigStore


def _format_params(p: PolicyParams) -> str:
    parts: List[str] = [
        f"- 投保年龄 (age_at_issue): {p.age_at_issue}",
        f"- 性别: {p.gender}",
        f"- 缴费方式 (premium_mode): {p.premium_mode}",
        f"- 保费金额 (premium_amount): {p.premium_amount} {p.currency}",
        f"- 缴费年期 (term_years): {p.term_years}",
        f"- 保额 (sum_assured): {p.sum_assured} {p.currency}",
        f"- 货币 (currency): {p.currency}",
    ]
    if p.target_age is not None:
        parts.append(f"- 目标年龄 (target_age): {p.target_age}")
    if p.target_year is not None:
        parts.append(f"- 目标年份 (target_year): {p.target_year}")
    return "\n".join(parts)


def _build_user_prompt(
    file_id: str,
    policy_params: PolicyParams,
    calc_targets: List[str],
) -> str:
    targets = "\n".join(f"- `{t}`" for t in calc_targets)
    return (
        f"请对保单 file_id=`{file_id}` 做精算计算。\n\n"
        f"## 保单参数\n{_format_params(policy_params)}\n\n"
        f"## 计算目标 ({len(calc_targets)} 项)\n{targets}\n\n"
        "硬性要求：\n"
        "1. 先 `read` 保单内的现金价值表 / 利率示例 / 分红率页面。\n"
        "2. 所有数学运算 (IRR / NPV / 复利 / 退保价值 / 插值) 必须走 "
        "`code_run`，禁止心算。\n"
        "3. 沙箱可用包包括 `numpy`, `numpy_financial`, `scipy`, "
        "`sympy`, `decimal`。优先用 `numpy_financial.irr/npv/pmt/pv/fv` "
        "做现金流计算。\n"
        "4. 表格里的关键数 (现金价值、退保系数、分红率) cite [^k]。\n"
        "5. 每个 calc_target 输出一节: 假设 / 公式 / 结果 (含币种和单位)，"
        "末尾 `## 关键假设与免责` 段标明哪些数来自 PDF、哪些来自用户输入。"
    )


def stream_policy_calc(
    *,
    file_id: str,
    policy_params: PolicyParams,
    calc_targets: List[str],
    agent: BaseAgent,
    config: ConfigStore,
    tracer=None,
    result_future=None,
) -> AsyncIterator[bytes]:
    user_prompt = _build_user_prompt(file_id, policy_params, calc_targets)
    return stream_workbench_agent(
        user_prompt=user_prompt,
        agent=agent,
        kind="base",
        config=config,
        prompt_key="prompt.policy_calc",
        flavor="policy_calc",
        final_extras={
            "file_id": file_id,
            "calc_targets": list(calc_targets),
            "currency": policy_params.currency,
        },
        tracer=tracer,
        result_future=result_future,
    )
