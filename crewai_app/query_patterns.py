from __future__ import annotations

import re
from typing import List

from langchain_core.messages import HumanMessage


def extract_latest_human_query(messages) -> str:
    if not messages:
        return ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()
    return ""

def looks_like_sector_keyword(text: str) -> bool:
    """判断是否为行业/赛道关键词，用于行业模式检测。"""
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > 24:
        return False
    disqualifiers = ("。", "?", "？", "!", "！", ".", "，", ";", "；", ",", "\n")
    if any(ch in stripped for ch in disqualifiers):
        return False
    return True


def looks_like_company_query(text: str) -> bool:
    """判断输入是否指向公司/股票，避免误触行业模式。"""
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    company_markers = (
        "公司",
        "企业",
        "集团",
        "股份",
        "科技",
        "控股",
        "有限公司",
        "股份有限公司",
        "有限责任公司",
        "Co.",
        "Inc",
        "Corp",
        "Corporation",
        "PLC",
        "Ltd",
        "LLC",
        "AG",
        "SA",
    )
    if any(marker in t for marker in company_markers):
        return True
    if re.search(r"\b(SZ|SH|HK|NASDAQ|NYSE)\b", t, flags=re.IGNORECASE):
        return True
    if re.search(r"\b(60|00|30)\d{4}\b", t):
        return True
    return False


def is_broad_topic(text: str) -> bool:
    """判断主题是否属于“行业/赛道”级别。"""
    if not text:
        return False
    t = text.strip()
    if not t or looks_like_company_query(t):
        return False
    if looks_like_sector_keyword(t):
        return True
    broad_markers = (
        "行业",
        "赛道",
        "市场",
        "生态",
        "供应链",
        "价值链",
        "应用",
        "需求",
        "痛点",
        "产品",
        "设备",
        "材料",
        "标准",
        "规范",
        "检测",
        "监管",
        "认证",
    )
    if any(marker in t for marker in broad_markers):
        return True
    if len(t) <= 36 and not any(ch in t for ch in ("。", "?", "？", "!", "！", ";", "；", ",", "\n")):
        return True
    return False


def build_overview_queries(topic: str) -> List[str]:
    kw = (topic or "").strip()
    if not kw:
        return []
    return [
        f"{kw} 行业概览 价值链 场景应用",
        f"{kw} 市场规模 增速 预测",
        f"{kw} 产业链 上下游 核心设备",
        f"{kw} 关键参与者 客户类型",
        f"{kw} 应用 场景 价值 痛点",
        f"{kw} 标准 规范 清单 认证 要求",
        f"{kw} 监管 合规 要求",
        f"{kw} 典型应用 细分场景",
    ]


def build_lead_finder_queries(topic: str) -> List[str]:
    kw = (topic or "").strip()
    if not kw:
        return []
    return [
        f"{kw} 行业公司 客户 列表",
        f"{kw} 产线 扩产 招标 2023 2024 2025",
        f"{kw} 典型客户 需求 场景",
        f"{kw} 品牌 渠道 出海",
        f"{kw} 招聘 质量 认证 主管 工程师",
        f"{kw} 展会 行业 协会",
    ]


def build_demand_signal_queries(topic: str) -> List[str]:
    kw = (topic or "").strip()
    if not kw:
        return []
    return [
        f"{kw} 招标 采购 RFQ RFP",
        f"{kw} 法规 监管 通知 公示",
        f"{kw} 新标准 发布 实施 时间",
        f"{kw} 项目 备案 验收",
        f"{kw} 样品 要求 AQL 检测",
    ]


def build_tic_queries(topic: str) -> List[str]:
    keyword = (topic or "").strip()
    if not keyword:
        return []
    return [
        f"{keyword} 检测 认证 标准 要求",
        f"{keyword} 测试 渠道 可靠性 项目",
        f"{keyword} 环保 RoHS REACH 合规",
        f"{keyword} EMC CE UL CCC",
        f"{keyword} IPC A-600 A-610 AEC-Q200 标准",
        f"{keyword} 实验室 CNAS CMA IECQ TAT 价格",
        f"{keyword} 认证 路径 证书 费用",
    ]


def build_company_list_queries(topic: str) -> List[str]:
    kw = (topic or "").strip()
    if not kw:
        return []
    return [
        f"{kw} 行业 公司 榜单 名录",
        f"{kw} 龙头 企业",
        f"{kw} 核心 企业 清单",
        f"{kw} 典型 客户 列表",
        f"{kw} 产业链 企业 图谱",
    ]


def deprioritize_tic_providers(queries: List[str], topic: str) -> List[str]:
    """将检测机构相关的查询推后，优先安排市场/需求类关键词。"""
    providers = (
        "SGS",
        "Intertek",
        "TUV",
        "TÜV",
        "UL",
        "DEKRA",
        "BV",
        "通标",
        "天祥",
        "必维",
        "通测",
        "谱尼",
    )
    topic_l = (topic or "").lower()
    kept: List[str] = []
    provider_q: List[str] = []
    for q in queries:
        qn = (q or "").strip()
        if not qn:
            continue
        ql = qn.lower()
        if topic_l and topic_l in ql:
            kept.append(qn)
            continue
        is_provider = any(p.lower() in ql for p in providers)
        has_manufacturer_hint = any(w in qn for w in ("公司", "企业", "厂", "供应", "客户"))
        (provider_q if is_provider and not has_manufacturer_hint else kept).append(qn)
    return kept + provider_q
