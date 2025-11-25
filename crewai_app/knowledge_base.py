"""Vector-store backed Excel knowledge base using FAISS with pluggable embeddings."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

CACHE_VERSION = "1"
# CrewAI 应用与 backend 位于同一仓库层级，因此此处直接回溯到仓库根目录
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".kb_cache"

try:
    import faiss
except ImportError as exc:  
    raise ImportError(
        "FAISS 未安装，请运行 `pip install faiss-cpu` 或等效命令。"
    ) from exc


@dataclass
class KnowledgeRecord:
    score: float
    text: str
    source: str
    row_index: int


class BaseEmbeddingClient:
    def embed(self, texts: List[str]) -> np.ndarray:  
        raise NotImplementedError


class QwenEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, model: str, batch_size: int = 10):
        try:
            from dashscope import TextEmbedding  
        except ImportError as exc:  
            raise ImportError(
                "dashscope SDK 未安装，请运行 `pip install dashscope`。"
            ) from exc

        self._TextEmbedding = TextEmbedding
        self.model = model
        # DashScope 向量接口单批最大 10
        self.batch_size = max(1, min(int(batch_size), 10))

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._TextEmbedding.call(model=self.model, input=batch)
            if response.status_code != HTTPStatus.OK:
                raise RuntimeError(
                    f"DashScope 向量接口调用失败: {response.message}"
                )
            embeddings = response.output.get("embeddings", [])
            embeddings.sort(key=lambda item: item.get("text_index", 0))
            vectors.extend(item["embedding"] for item in embeddings)
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True) + 1e-10
        return array / norms


class LocalEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, model: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers 未安装，请运行 `pip install sentence-transformers`。"
            ) from exc

        self._model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        array = self._model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array


class ExcelKnowledgeBase:
    """Load Excel rows into a FAISS index with configurable embeddings."""

    def __init__(
        self,
        paths: Iterable[Path],
        *,
        embedding_model: str,
        embedding_backend: str = "dashscope",
        embedding_batch_size: int = 10,
        cache_dir: Path | None = None,
    ):
        self._paths = [Path(path) for path in paths]
        backend = embedding_backend.lower()
        self._embedding_model = embedding_model
        self._embedding_backend = backend
        self._embedding_batch_size = embedding_batch_size
        self._cache_dir = (cache_dir or DEFAULT_CACHE_DIR).expanduser().resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_key = self._compute_cache_key()

        if backend == "local":
            self._embedding_client: BaseEmbeddingClient = LocalEmbeddingClient(
                embedding_model
            )
        else:
            self._embedding_client = QwenEmbeddingClient(
                embedding_model, batch_size=embedding_batch_size
            )
        self._docs: List[dict] = []
        self._index: faiss.Index | None = None
        if not self._load_from_cache():
            self._load_documents()
            if self._docs:
                self._build_index()
                self._persist_cache()

    def _load_documents(self) -> None:
        for path in self._paths:
            if not path:
                continue
            resolved = path.expanduser().resolve()
            if not resolved.exists():
                continue
            try:
                df = pd.read_excel(resolved)
            except Exception:
                continue
            df.columns = [str(col).strip() for col in df.columns]
            for row_idx, row in df.iterrows():
                fields = {
                    col: row[col]
                    for col in df.columns
                    if pd.notna(row[col]) and str(row[col]).strip()
                }
                if not fields:
                    continue
                text = " | ".join(f"{col}: {fields[col]}" for col in fields)
                self._docs.append(
                    {
                        "text": text,
                        "source": resolved.name,
                        "row_index": row_idx + 2,
                    }
                )

    def _build_index(self) -> None:
        texts = [doc["text"] for doc in self._docs]
        embeddings = self._embedding_client.embed(texts)
        if embeddings.size == 0:
            return
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self._index = index

    def _persist_cache(self) -> None:
        if self._index is None or not self._docs:
            return
        index_path = self._cache_dir / f"{self._cache_key}.faiss"
        docs_path = self._cache_dir / f"{self._cache_key}.json"
        try:
            faiss.write_index(self._index, str(index_path))
            with docs_path.open("w", encoding="utf-8") as f:
                json.dump(self._docs, f, ensure_ascii=False)
        except Exception:
            # 缓存写入失败时忽略，保持运行
            pass

    def _load_from_cache(self) -> bool:
        index_path = self._cache_dir / f"{self._cache_key}.faiss"
        docs_path = self._cache_dir / f"{self._cache_key}.json"
        if not index_path.exists() or not docs_path.exists():
            return False
        try:
            self._index = faiss.read_index(str(index_path))
            with docs_path.open("r", encoding="utf-8") as f:
                self._docs = json.load(f)
            return True
        except Exception:
            self._index = None
            self._docs = []
            return False

    def _compute_cache_key(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"cache_version:{CACHE_VERSION}".encode("utf-8"))
        hasher.update(f"backend:{self._embedding_backend}".encode("utf-8"))
        hasher.update(f"model:{self._embedding_model}".encode("utf-8"))
        for path in self._paths:
            resolved = path.expanduser().resolve()
            hasher.update(str(resolved).encode("utf-8"))
            if resolved.exists():
                stat = resolved.stat()
                hasher.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
                hasher.update(str(stat.st_size).encode("utf-8"))
            else:
                hasher.update(b"missing")
        return hasher.hexdigest()

    def search(self, query: str, top_k: int = 3) -> List[KnowledgeRecord]:
        if not query or self._index is None:
            return []
        query_vec = self._embedding_client.embed([query])
        if query_vec.size == 0:
            return []
        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []
        scores, indices = self._index.search(query_vec, k)
        results: List[KnowledgeRecord] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self._docs[idx]
            results.append(
                KnowledgeRecord(
                    score=float(score),
                    text=doc["text"],
                    source=doc["source"],
                    row_index=doc["row_index"],
                )
            )
        return results

    @property
    def is_empty(self) -> bool:
        return self._index is None or self._index.ntotal == 0

    @property
    def document_count(self) -> int:
        return len(self._docs)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir
