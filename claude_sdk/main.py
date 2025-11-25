from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict

import dotenv

from crewai_app.configuration import Configuration

from claude_sdk.workflow import ProSearchClaudeWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行基于 Claude Agent SDK 的 TIC Pro Search 工作流。",
    )
    parser.add_argument(
        "question",
        help="调研主题或业务问题，例如“水冷板行业现状”。",
    )
    parser.add_argument(
        "--initial-queries",
        type=int,
        default=3,
        help="首轮需要生成的搜索查询数量（默认 3）。",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=2,
        help="允许的最大研究循环次数（默认 2）。",
    )
    parser.add_argument(
        "--disable-kb",
        action="store_true",
        help="禁用 Excel 知识库检索。",
    )
    parser.add_argument(
        "--disable-industry-report",
        action="store_true",
        help="关闭行业报告模板，始终输出通用回答。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印 Claude Agent SDK 的流式输出与工具调用。",
    )
    return parser.parse_args()


def build_configuration(args: argparse.Namespace) -> Configuration:
    overrides: Dict[str, Any] = {
        "number_of_initial_queries": args.initial_queries,
        "max_research_loops": args.max_loops,
        "enable_knowledge_base_search": not args.disable_kb,
        "enable_industry_report_mode": not args.disable_industry_report,
    }
    base = Configuration.from_runnable_config({"configurable": overrides})
    return base


def main() -> None:
    dotenv.load_dotenv(override=True)
    args = parse_args()
    configuration = build_configuration(args)
    workflow = ProSearchClaudeWorkflow(configuration)
    result = asyncio.run(workflow.run(args.question, verbose=args.verbose))
    print(result)


if __name__ == "__main__":
    main()
