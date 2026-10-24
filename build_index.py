"""RAG 지식 인덱스 생성 스크립트.

python build_index.py
- data/knowledge/*.md, data/glossary.json 을 청킹·임베딩해 storage/index/ 에 FAISS 인덱스로 저장
- .env에 임베딩 모델(AOAI_DEPLOY_EMBEDDING 또는 EMBEDDING_MODEL)이 있으면 API 임베딩, 없으면 로컬 TF-IDF 사용
- 앱 첫 실행 시 인덱스가 없으면 자동으로 만들어지므로 필수는 아님 (문서를 수정했을 때 실행)
"""
from knitmate.core.rag import build_index, search_knowledge

if __name__ == "__main__":
    build_index()
    for q in ["한길긴뜨기 어떻게 떠?", "k2tog", "게이지 재는 법"]:
        top = search_knowledge(q, k=1)[0]
        print(f"  검색 테스트 '{q}' → {top['source']} > {top['section']} ({top['score']})")
