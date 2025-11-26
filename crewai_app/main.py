from pathlib import Path
from typing import Optional

from crew import TicOpportunityCrew

BASE_DIR = Path(__file__).resolve().parents[1]


def _load_markdown(md_path: Path) -> str:
    if not md_path.exists():
        raise FileNotFoundError(f"未找到 Markdown 文件: {md_path}")
    return md_path.read_text(encoding="utf-8")


def _resolve_md_path(md_path: Optional[str]) -> Path:
    if md_path:
        candidate = Path(md_path).expanduser()
        if not candidate.is_absolute():
            candidate = (BASE_DIR / candidate).resolve()
    else:
        candidate = BASE_DIR / "data/三安光电.md"
    return candidate


def run(md_path: Optional[str] = None):
    path = _resolve_md_path(md_path)
    md_content = _load_markdown(path)
    inputs = {
        'md_path': str(path),
        'md_content': md_content,
    }
    return TicOpportunityCrew().crew().kickoff(inputs=inputs)


if __name__ == "__main__":
    result = run()
    print("\n======================")
    print("TIC 机会识别结果")
    print("======================\n")
    print(result)
