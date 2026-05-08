"""Multi-product comparison — BaseAgent + structured prompt."""
from typing import AsyncIterator, List, Optional

from agentic.agent.base import BaseAgent
from api.runners._workbench import stream_workbench_agent
from config.config_store import ConfigStore


def _build_user_prompt(file_ids: List[str], properties: List[str]) -> str:
    """Compose the matrix instruction.

    Lay out the request as: list of products → list of dimensions →
    explicit instruction to call ``list_files`` first if the agent
    isn't sure which file_id is which product. The system prompt
    (admin key ``prompt.compare``) already covers the output format
    and abstain rules; the user prompt only carries the inputs.
    """
    products = "\n".join(f"- {fid}" for fid in file_ids)
    dims = "\n".join(f"- {p}" for p in properties)
    return (
        "请对比以下保险产品在指定维度上的差异。\n\n"
        f"## 产品 file_ids ({len(file_ids)} 个)\n{products}\n\n"
        f"## 对比维度 ({len(properties)} 个)\n{dims}\n\n"
        "工作流：\n"
        "1. 调 `list_files` 确认每个 file_id 对应的产品名 (filename)。\n"
        "2. 对每个 (产品, 维度) 单元格做检索 + read，提取 verbatim 表述。\n"
        "3. 输出 markdown 矩阵，每个 cell 一句话 + [^k] 引用。\n"
        "4. 找不到的 cell 写 `待查`，不要外推。\n"
        "5. 最后给一段 `## 关键差异` 总结 (2-4 个 bullet)。"
    )


def stream_compare(
    *,
    file_ids: List[str],
    properties: List[str],
    agent: BaseAgent,
    config: ConfigStore,
    tracer=None,
    result_future=None,
) -> AsyncIterator[bytes]:
    user_prompt = _build_user_prompt(file_ids, properties)
    return stream_workbench_agent(
        user_prompt=user_prompt,
        agent=agent,
        kind="base",
        config=config,
        prompt_key="prompt.compare",
        flavor="compare",
        final_extras={
            "file_ids": file_ids,
            "properties": properties,
            "matrix_dims": (len(file_ids), len(properties)),
        },
        tracer=tracer,
        result_future=result_future,
    )
