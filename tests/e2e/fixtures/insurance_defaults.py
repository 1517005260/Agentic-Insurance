"""Real-data demo defaults for the live insurance / web e2e suite.

These values are NOT mocks. The e2e tests below hit:
  * the real Tavily API  (TAVILY_API_KEY in .env)
  * the real LLM provider (CHAT_API_KEY in .env)
  * the real local index (3 AXA PDFs already in local_storage/page_assets)

Override any of them per-test with a kwarg if a particular suite
needs a different shape; this file just centralizes the "what we
demo with" answer so the fixtures don't drift across tests.

The constants below are deliberately keyed by what they represent
(not by which test uses them) so adding a new workbench test is
"import the relevant default" rather than copy-paste.
"""
from typing import List

# Three AXA products already indexed in local_storage/page_assets.
# Names are file_ids (page_assets/<file_id>.json). They are the demo
# corpus for every workbench in the live e2e suite.
DEFAULT_FILE_IDS: List[str] = [
    "AXA安盛「盛利II-至尊」保费回赠及预缴利率_截止至12月31日（英文版）_4a5deaa25d7dda9a",
    "盛利2-至尊_宣传彩页（英文版）_3f375b5635323f41",
    "盛利2-至尊_小册子（英文版）_d2fe8002daf75382",
]

# A primary single-product file_id used by single-product workbenches
# (exclusion audit, policy calc).
DEFAULT_PRIMARY_FILE_ID: str = DEFAULT_FILE_IDS[2]


# ---------------------------------------------------------- compare


DEFAULT_COMPARE_PROPERTIES: List[str] = [
    "保费缴付方式",
    "保单年期",
    "现金价值表",
    "退保安排",
]


# ---------------------------------------------------------- customer profile


DEFAULT_CUSTOMER = {
    "age": 35,
    "gender": "M",
    "occupation": "软件工程师",
    "occupation_risk": "low",
    "health_history": [],
    "family_history": [],
    "budget_annual": 15000,
    "goal": "储蓄 / 长期增值",
    "notes": "希望兼顾子女教育金与退休规划",
}


# ---------------------------------------------------------- claim


DEFAULT_CLAIM_EVENT = {
    "type": "退保",
    "date": "2026-04-15",
    "location": "香港",
    "description": "持有 5 年的盛利 II 保单，因急需现金考虑退保，希望了解退保价值与扣减项",
    "amount": 100000.0,
}


# ---------------------------------------------------------- policy calc


DEFAULT_POLICY_PARAMS = {
    "age_at_issue": 35,
    "gender": "M",
    "premium_mode": "annual",
    "premium_amount": 12000.0,
    "term_years": 20,
    "sum_assured": 1_000_000.0,
    "currency": "HKD",
    "target_age": 65,
    "target_year": 10,
}

DEFAULT_CALC_TARGETS: List[str] = [
    "cash_value_by_year",
    "surrender_at_year",
    "irr_to_age",
]


# ---------------------------------------------------------- hidden risk (PPR)


DEFAULT_HIDDEN_RISK_QUERY: str = (
    "客户在职业列表里登记为高空作业，想了解意外险有哪些与该职业相关的限制条款？"
)


# ---------------------------------------------------------- web RAG


DEFAULT_WEB_RAG_QUERY: str = "What is the most recent solvency framework for HK life insurers?"
DEFAULT_WEB_AGENT_QUERY: str = (
    "查找香港 IA 关于人寿保险产品披露的最新指引，并总结关键要点"
)
