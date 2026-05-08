"""Claim coverage check — BaseAgent + 三栏 schema prompt."""
from typing import AsyncIterator, List

from agentic.agent.base import BaseAgent
from api.runners._workbench import stream_workbench_agent
from api.schemas.insurance import ClaimEvent
from config.config_store import ConfigStore


def _format_event(e: ClaimEvent) -> str:
    parts: List[str] = [
        f"- 事件类型: {e.type}",
        f"- 发生日期: {e.date}",
    ]
    if e.location:
        parts.append(f"- 地点: {e.location}")
    if e.amount is not None:
        parts.append(f"- 涉及金额: {e.amount}")
    parts.append(f"- 描述: {e.description}")
    return "\n".join(parts)


def _build_user_prompt(file_ids: List[str], event: ClaimEvent) -> str:
    products = "\n".join(f"- {fid}" for fid in file_ids)
    return (
        "请对以下保险事件做理赔覆盖判定。\n\n"
        f"## 涉及保单 ({len(file_ids)} 份)\n{products}\n\n"
        f"## 事件详情\n{_format_event(event)}\n\n"
        "工作流：\n"
        "1. 对每份 file_id，定位与事件类型相关的条款 (semantic + read)。\n"
        "2. 交叉核对等待期、除外、地理范围、保障定义。\n"
        "3. 输出三段:\n"
        "   `## 1. 覆盖判定` (整体 + 每个 file_id 一行)\n"
        "   `## 2. 适用条款` (markdown 表: file_id / 条款 / 摘要 / [^k])\n"
        "   `## 3. 所需材料` (bullet 列表 + [^k])\n"
        "信息不足时标 `INSUFFICIENT_DATA`，不要拍脑袋。"
    )


def stream_claim_check(
    *,
    file_ids: List[str],
    event: ClaimEvent,
    agent: BaseAgent,
    config: ConfigStore,
    tracer=None,
    result_future=None,
) -> AsyncIterator[bytes]:
    user_prompt = _build_user_prompt(file_ids, event)
    return stream_workbench_agent(
        user_prompt=user_prompt,
        agent=agent,
        kind="base",
        config=config,
        prompt_key="prompt.claim_check",
        flavor="claim",
        final_extras={
            "file_ids": file_ids,
            "event_type": event.type,
        },
        tracer=tracer,
        result_future=result_future,
    )
