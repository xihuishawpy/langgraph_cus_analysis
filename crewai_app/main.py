from __future__ import annotations

import argparse
from typing import Any, Dict

import dotenv

from crewai_app.configuration import Configuration
from crewai_app.crew_builder import ProSearchCrewBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行基于 CrewAI 的 TIC Pro Search 多智能体工作流。",
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
        help="允许的最大研究循环次数（含首轮，默认 2）。",
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
        help="在终端展示 CrewAI 的详细执行日志。",
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
    dotenv.load_dotenv()
    args = parse_args()
    configuration = build_configuration(args)
    builder = ProSearchCrewBuilder(configuration, verbose=args.verbose)
    crew = builder.build(args.question)
    result = crew.kickoff(inputs={"topic": args.question})
    output_text = getattr(result, "raw", str(result))
    print(output_text)


if __name__ == "__main__":
    main()
