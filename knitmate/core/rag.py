"""RAG: 뜨개 지식 문서 → 청킹 → 임베딩 → FAISS 벡터 인덱스 → 하이브리드 검색.

- 임베딩: 임베딩 모델(Azure: AOAI_DEPLOY_EMBEDDING / 호환 API: EMBEDDING_MODEL)이 있으면 사용,
  없으면 로컬 문자 n-gram TF-IDF 벡터로 자동 대체(오프라인에서도 동작).
- 검색: 벡터 유사도 + 용어 정확 일치 보너스(짧은 한국어 용어 검색 보강).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..config import DATA_DIR, INDEX_DIR, KNOWLEDGE_DIR, embedding_id, get_embeddings

try:
    import faiss  # type: ignore
except Exception:  # faiss 미설치 환경 대비
    faiss = None


@dataclass
class Doc:
    id: str
    text: str
    meta: dict


# ---------- 1) 문서 로딩 & 청킹 ----------

def _chunk_markdown(path) -> list[Doc]:
    text = path.read_text(encoding="utf-8")
    title = ""
    docs: list[Doc] = []
    current_h2, buf = None, []

    def flush():
        if current_h2 and buf:
            body = "\n".join(buf).strip()
            if body:
                craft = "crochet" if "crochet" in path.stem else ("knit" if "knit" in path.stem else "all")
                docs.append(Doc(
                    id=f"{path.stem}#{len(docs)}",
                    text=f"[{title}] {current_h2}\n{body}",
                    meta={"source": path.name, "section": current_h2, "type": "knowledge", "craft": craft},
                ))

    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            current_h2, buf = line[3:].strip(), []
        else:
            buf.append(line)
    flush()
    return docs


def load_documents() -> list[Doc]:
    docs: list[Doc] = []
    for p in sorted(KNOWLEDGE_DIR.glob("*.md")):
        docs.extend(_chunk_markdown(p))
    for e in json.loads((DATA_DIR / "glossary.json").read_text(encoding="utf-8")):
        docs.append(Doc(
            id=f"glossary#{e['code']}",
            text=(f"[용어] {e['ko']} ({e['en']}, 약어 {e['code']}) 별칭: {', '.join(e.get('aliases', []))}\n"
                  f"{e['desc']} 코 수 규칙: 이전 단 {e['consumes']}코 사용 → {e['produces']}코 생성."),
            meta={"source": "glossary.json", "section": e["ko"], "type": "glossary", "craft": e["craft"], "code": e["code"]},
        ))
    return docs


# ---------- 2) 벡터화 ----------

def _tokens(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", text.lower())
    words = re.findall(r"[a-z0-9]+|[가-힣]+", t)
    grams = []
    for w in words:
        grams.append(w)
        if re.match(r"[가-힣]", w) and len(w) > 1:
            grams += [w[i:i + 2] for i in range(len(w) - 1)]
    return grams


class TfidfEncoder:
    """외부 API 없이 동작하는 폴백 임베딩 (단어 + 한글 2-gram TF-IDF, 해싱)."""

    dim = 2048

    def __init__(self, idf: dict[str, float] | None = None):
        self.idf = idf or {}

    def fit(self, texts: list[str]):
        df = Counter()
        for t in texts:
            df.update(set(_tokens(t)))
        n = len(texts)
        self.idf = {k: math.log((n + 1) / (v + 1)) + 1 for k, v in df.items()}
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok, cnt in Counter(_tokens(t)).items():
                out[i, hash_token(tok) % self.dim] += (1 + math.log(cnt)) * self.idf.get(tok, 1.0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


def hash_token(tok: str) -> int:
    # 파이썬 hash()는 실행마다 달라지므로 고정 해시 사용
    h = 2166136261
    for ch in tok.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


# ---------- 3) 인덱스 빌드 / 로드 ----------

def build_index(verbose: bool = True) -> dict:
    docs = load_documents()
    texts = [d.text for d in docs]
    emb = get_embeddings()
    mode = "azure"
    vectors = None
    if emb is not None:
        try:
            vectors = _normalize(np.array(emb.embed_documents(texts), dtype="float32"))
        except Exception as e:  # 임베딩 배포 오류 시 폴백
            if verbose:
                print(f"[RAG] API 임베딩 실패 → TF-IDF로 대체: {e}")
            vectors = None
    tfidf = TfidfEncoder().fit(texts)
    if vectors is None:
        mode = "tfidf"
        vectors = tfidf.encode(texts)

    np.save(INDEX_DIR / "vectors.npy", vectors)
    if faiss is not None:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        # 한글 경로(Windows)에서 faiss 파일 I/O가 실패하므로 바이트로 직렬화해 저장
        (INDEX_DIR / "faiss.index").write_bytes(faiss.serialize_index(index).tobytes())
    (INDEX_DIR / "docs.json").write_text(json.dumps([d.__dict__ for d in docs], ensure_ascii=False, indent=1), encoding="utf-8")
    (INDEX_DIR / "meta.json").write_text(json.dumps({"mode": mode, "embedding": embedding_id() if mode == "azure" else None, "dim": int(vectors.shape[1]), "count": len(docs), "idf": tfidf.idf}, ensure_ascii=False), encoding="utf-8")
    load_index.cache_clear()
    if verbose:
        print(f"[RAG] 문서 {len(docs)}개 인덱싱 완료 (임베딩: {'API' if mode == 'azure' else 'TF-IDF'}, FAISS: {'사용' if faiss else '미설치→numpy'})")
    return {"mode": mode, "count": len(docs)}


class KnowledgeIndex:
    def __init__(self):
        meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
        self.mode = meta["mode"]
        if self.mode == "azure" and meta.get("embedding") != embedding_id():
            raise RuntimeError("임베딩 제공자/모델이 바뀌어 인덱스를 다시 만들어야 합니다.")
        self.docs = [Doc(**d) for d in json.loads((INDEX_DIR / "docs.json").read_text(encoding="utf-8"))]
        self.vectors = np.load(INDEX_DIR / "vectors.npy")
        self.tfidf = TfidfEncoder(meta.get("idf"))
        self.faiss_index = None
        if faiss is not None and (INDEX_DIR / "faiss.index").exists():
            raw = np.frombuffer((INDEX_DIR / "faiss.index").read_bytes(), dtype=np.uint8)
            self.faiss_index = faiss.deserialize_index(raw)

    def _embed_query(self, q: str) -> np.ndarray:
        if self.mode == "azure":
            emb = get_embeddings()
            if emb is not None:
                try:
                    return _normalize(np.array([emb.embed_query(q)], dtype="float32"))
                except Exception:
                    pass
            raise RuntimeError("API 임베딩을 쓸 수 없어 인덱스를 다시 만들어야 합니다.")
        return self.tfidf.encode([q])

    def search(self, query: str, k: int = 4, craft: str | None = None) -> list[dict]:
        qv = self._embed_query(query)
        n = len(self.docs)
        if self.faiss_index is not None:
            scores, ids = self.faiss_index.search(qv, min(n, 40))
            pairs = list(zip(ids[0].tolist(), scores[0].tolist()))
        else:
            sims = (self.vectors @ qv[0]).tolist()
            pairs = sorted(enumerate(sims), key=lambda x: -x[1])[:40]

        q_norm = re.sub(r"\s", "", query.lower())
        ranked = []
        for i, s in pairs:
            if i < 0:
                continue
            d = self.docs[i]
            bonus = 0.0
            section = re.sub(r"\s", "", d.meta.get("section", "").split("(")[0].lower())
            if section and (section in q_norm or (len(q_norm) >= 2 and q_norm in section)):
                bonus += 0.25  # 하이브리드: 용어명 정확 일치 가산점
            code = d.meta.get("code")
            if code and re.search(rf"(?<![a-z]){re.escape(code)}(?![a-z0-9])", query.lower()):
                bonus += 0.2
            if craft in ("knit", "crochet") and d.meta.get("craft") not in (craft, "all"):
                bonus -= 0.15
            ranked.append((s + bonus, d))
        ranked.sort(key=lambda x: -x[0])
        return [{"score": round(sc, 3), "text": d.text, **d.meta} for sc, d in ranked[:k]]


@lru_cache(maxsize=1)
def load_index() -> KnowledgeIndex:
    if not (INDEX_DIR / "meta.json").exists():
        build_index(verbose=False)
    try:
        return KnowledgeIndex()
    except RuntimeError:
        build_index(verbose=False)
        return KnowledgeIndex()


def search_knowledge(query: str, k: int = 4, craft: str | None = None) -> list[dict]:
    idx = load_index()
    try:
        return idx.search(query, k=k, craft=craft)
    except RuntimeError:
        build_index(verbose=False)
        return load_index().search(query, k=k, craft=craft)


def format_hits(hits: list[dict]) -> str:
    return "\n\n".join(f"[출처: {h['source']} > {h['section']}]\n{h['text']}" for h in hits)
