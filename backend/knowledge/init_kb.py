"""CLI tool to pre-build the Excel knowledge base FAISS cache."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List

import dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
SRC_ROOT = BACKEND_ROOT / "src"

# Ensure `agent.*` modules can be imported when executing from repo root.
sys.path.insert(0, str(SRC_ROOT))

from agent.configuration import Configuration  # noqa: E402
from agent.knowledge_base import ExcelKnowledgeBase  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "构建/刷新本地 Excel 知识库的 FAISS 索引缓存。"
            " 默认读取配置项（可通过环境变量或 `.env` 覆盖）。"
        )
    )
    parser.add_argument(
        "--paths",
        help=(
            "逗号分隔的 Excel 路径列表；留空则使用 Configuration.knowledge_base_paths。"
            " 支持相对路径（基于仓库根目录）。"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        help="嵌入模型名称，默认取配置项 knowledge_base_embedding_model。",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["dashscope", "local"],
        help="嵌入后端，默认取配置项 knowledge_base_embedding_backend。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="嵌入批大小，默认取配置项 knowledge_base_embedding_batch_size。",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="自定义缓存目录（默认使用 `backend/.kb_cache`）。",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=REPO_ROOT / ".env",
        help="指定要加载的 .env 文件路径，默认是仓库根目录下的 .env。",
    )
    return parser.parse_args()


def _resolve_paths(raw_string: str) -> List[Path]:
    paths: List[Path] = []
    for chunk in raw_string.split(","):
        candidate = chunk.strip()
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = REPO_ROOT / candidate
        paths.append(path)
    return paths


def _summarize_files(paths: Iterable[Path]) -> str:
    summaries = []
    for path in paths:
        if path.exists():
            summaries.append(f"{path.name} ({path.stat().st_size / 1_048_576:.2f} MB)")
        else:
            summaries.append(f"{path.name} (missing)")
    return ", ".join(summaries)


def main() -> None:
    args = _parse_args()

    if args.dotenv and args.dotenv.exists():
        dotenv.load_dotenv(dotenv_path=args.dotenv)
    else:
        dotenv.load_dotenv()

    # 读取配置，CLI 参数优先。
    config = Configuration.from_runnable_config()
    paths_string = args.paths or config.knowledge_base_paths
    embedding_model = args.embedding_model or config.knowledge_base_embedding_model
    embedding_backend = (
        args.embedding_backend or config.knowledge_base_embedding_backend
    )
    batch_size = args.batch_size or config.knowledge_base_embedding_batch_size

    paths = _resolve_paths(paths_string)
    if not paths:
        raise SystemExit("未提供有效的 Excel 知识库路径。")

    print("开始构建知识库索引...")
    print(f"Excel 文件: {_summarize_files(paths)}")
    print(
        f"向量后端: {embedding_backend}, 模型: {embedding_model}, 批大小: {batch_size}"
    )

    start = time.perf_counter()
    kb = ExcelKnowledgeBase(
        paths,
        embedding_model=embedding_model,
        embedding_backend=embedding_backend,
        embedding_batch_size=batch_size,
        cache_dir=args.cache_dir,
    )
    elapsed = time.perf_counter() - start

    if kb.is_empty:
        raise SystemExit("构建失败：未生成有效的知识库索引。")

    cache_dir = args.cache_dir or kb.cache_dir
    print(
        f"构建完成，耗时 {elapsed:.2f} 秒，文档数 {kb.document_count}, 缓存目录 {cache_dir}"
    )


if __name__ == "__main__":
    main()
