# 🧶 뜨개메이트 (KnitMate)

뜨개 도안(PDF·이미지·텍스트)을 **한국어 도안 + 기호 차트**로 바꾸고, **코 수를 자동 검증**하며,
**기법 질문·영상 추천·진행 중 코 수 체크**까지 돕는 LangGraph 멀티 에이전트 서비스입니다.

기획 문서: [`docs/기획서.md`](docs/기획서.md)

## 빠른 실행

```bash
# 1) 가상환경 (선택)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) 패키지 설치
pip install -r requirements.txt

# 3) 환경 변수
cp .env.example .env             # Windows: copy .env.example .env
#   → AOAI_API_KEY, AOAI_ENDPOINT, AOAI_DEPLOY_GPT4O, AOAI_API_VERSION 입력
#   또는 LLM_PROVIDER=openai + LLM_API_KEY, LLM_BASE_URL, LLM_MODEL 입력
#   (Gemini·Groq·Ollama·OpenAI 등 OpenAI 호환 API, 예시는 .env.example 참고)

# 4) 지식 인덱스 생성 (생략하면 첫 실행 때 자동 생성)
python build_index.py

# 5) 실행
streamlit run main.py
#   원격 서버: streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```

- `.env`가 없어도 **샘플 도안 빠른 데모**, **진행 체크**, **코 수 검증/차트**는 동작합니다.
- 임베딩 배포(`AOAI_DEPLOY_EMBEDDING`)가 없으면 로컬 TF-IDF 검색, YouTube 키가 없으면 검색 링크로 자동 대체됩니다.

### 백엔드 API (선택)
```bash
uvicorn api:app --port 8000      # 문서: http://localhost:8000/docs
```

### 테스트
```bash
python -m pytest -q tests        # LLM 없이 실행 (가짜 LLM으로 그래프 흐름까지 검증)
```

### Docker (선택)
```bash
docker build -t knitmate . && docker run -p 8501:8501 --env-file .env knitmate
```

## 화면 구성

| 탭 | 기능 |
|---|---|
| 📄 도안 변환 | 파일/텍스트/샘플 입력 → 한국어 도안, 기호 차트, 단별 코 수 검증표, 다운로드, 프로젝트 시작 |
| 💬 뜨개 코치 | 기법 설명 + 영상, 도안 해설, 게이지·실 양 계산, 진행 기록 (대화 메모리) |
| 📏 진행 체크 | 현재 단 안내, 코 수 입력 → 비교 → 불일치 원인 진단, 차트에서 현재 단 강조, 기록 |
| ℹ️ 안내 | 에이전트 구성과 안전장치 |

## 구조

```
knitmate/
├─ main.py                 # Streamlit UI
├─ api.py                  # FastAPI 백엔드 (선택)
├─ build_index.py          # RAG 인덱스 생성
├─ knitmate/
│  ├─ config.py            # 환경 변수, LLM/임베딩 팩토리 (상대 경로 기준)
│  ├─ schemas.py           # 구조화 출력 스키마 (PatternSpec, KoreanPattern, Route, Diagnosis)
│  ├─ agents/
│  │  ├─ pattern_graph.py  # 도안 변환 그래프: 추출 → 번역 ∥ 파서 → 검증(자기 수정 루프) → 차트
│  │  ├─ coach_graph.py    # 라우터 + 전문 에이전트 5종 + 대화 메모리
│  │  ├─ react.py          # 공통 ReAct 서브그래프, 영상 URL 가드레일
│  │  ├─ tools.py          # Function Calling 도구
│  │  ├─ diagnosis.py      # 코 수 불일치 진단 에이전트
│  │  └─ prompts.py        # 역할·CoT·Few-shot 프롬프트
│  └─ core/
│     ├─ stitch_counter.py # 코 수 검증 엔진 (결정적)
│     ├─ chart.py          # 기호 차트 렌더러
│     ├─ rag.py            # 청킹·임베딩·FAISS·하이브리드 검색
│     ├─ glossary.py       # 용어집 조회
│     ├─ videos.py         # 영상 추천 3단 폴백
│     ├─ storage.py        # SQLite 프로젝트/진행 기록
│     ├─ pdf_reader.py     # PDF/이미지 읽기
│     └─ formatting.py     # 한국어 도안 마크다운
├─ data/
│  ├─ glossary.json        # 약어·한국어·코 수 규칙 (단일 출처)
│  ├─ knowledge/*.md       # RAG 지식 문서
│  ├─ videos.json          # 영상 큐레이션 (직접 확인한 영상 추가)
│  └─ sample_patterns/     # 자체 제작 샘플 도안 + 구조화 결과
├─ tests/                  # 단위·그래프 테스트
└─ docs/기획서.md
```

## 영상 큐레이션 추가 방법
`data/videos.json`에서 기법의 `videos` 배열에 직접 확인한 영상을 넣으면 그 영상이 1순위로 추천됩니다.
```json
{"title": "한길긴뜨기 기초", "channel": "채널명", "url": "https://www.youtube.com/watch?v=...", "level": "초급"}
```

## 제출(압축) 시 주의
- `.env`, `.venv/`, `storage/`, `__pycache__/`는 넣지 않습니다. (`storage/`는 실행 시 자동 생성)
- 서버에서 `unzip knitmate.zip` → `cd knitmate` → `pip install -r requirements.txt` → `.env` 작성 → `streamlit run main.py`
