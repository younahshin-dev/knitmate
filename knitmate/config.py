"""환경 설정과 LLM/임베딩 팩토리.

모든 경로는 이 파일 기준 상대 경로라서 윈도우/리눅스 어디서 압축을 풀어도 동작합니다.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
SAMPLE_DIR = DATA_DIR / "sample_patterns"
INDEX_DIR = ROOT_DIR / "storage" / "index"
DB_PATH = ROOT_DIR / "storage" / "knitmate.db"
OUTPUT_DIR = ROOT_DIR / "storage" / "outputs"

# .env는 프로젝트 루트에서 찾는다 (실행 위치와 무관)
load_dotenv(ROOT_DIR / ".env")

for d in (INDEX_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is not None:
        value = value.strip().strip('"').strip("'")
    return value or default


# LLM 제공자: azure(기본) | gemini (Google Gemini 전용) | openai (OpenAI 호환 API — OpenAI, Groq, Ollama 등)
LLM_PROVIDER = (env("LLM_PROVIDER", "azure") or "azure").lower()

REQUIRED_LLM_VARS = {
    "azure": ("AOAI_API_KEY", "AOAI_ENDPOINT", "AOAI_DEPLOY_GPT4O", "AOAI_API_VERSION"),
    "openai": ("LLM_API_KEY", "LLM_MODEL"),
    "gemini": ("LLM_API_KEY", "LLM_MODEL"),
}


def missing_llm_vars() -> list[str]:
    required = REQUIRED_LLM_VARS.get(LLM_PROVIDER)
    if required is None:
        return [f"LLM_PROVIDER (azure, gemini, openai 중 하나, 현재 값: {LLM_PROVIDER})"]
    return [k for k in required if not env(k)]


def llm_available() -> bool:
    return not missing_llm_vars()


def embedding_model() -> str | None:
    return env("AOAI_DEPLOY_EMBEDDING") if LLM_PROVIDER == "azure" else env("EMBEDDING_MODEL")


def embedding_available() -> bool:
    return llm_available() and bool(embedding_model())


def provider_label() -> str:
    """사이드바 표시용 이름."""
    if LLM_PROVIDER == "azure":
        return "Azure OpenAI"
    if LLM_PROVIDER == "gemini":
        return f"Google Gemini ({env('LLM_MODEL')})"
    return f"OpenAI 호환 API ({env('LLM_MODEL')} @ {env('LLM_BASE_URL', 'api.openai.com')})"


def embedding_id() -> str:
    """인덱스를 만든 임베딩 식별자. 제공자/모델이 바뀌면 인덱스를 다시 만든다."""
    return f"{LLM_PROVIDER}:{embedding_model()}"


def youtube_available() -> bool:
    return bool(env("YOUTUBE_API_KEY"))


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.2):
    """채팅 모델. LLM_PROVIDER=azure 이면 Azure OpenAI(GPT-4o 배포), gemini 이면 Google Gemini, openai 이면 OpenAI 호환 API."""
    missing = missing_llm_vars()
    if missing:
        raise RuntimeError(f".env에 다음 값이 없습니다: {', '.join(missing)}")
    if LLM_PROVIDER == "gemini":
        # Gemini 3 계열은 도구 호출 다음 턴에 thought_signature를 돌려줘야 해서 전용 클라이언트를 쓴다
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            api_key=env("LLM_API_KEY"),
            model=env("LLM_MODEL"),
            temperature=temperature,
            max_retries=2,
            timeout=300,  # Gemini 3 계열은 응답이 느려 120초면 504(DEADLINE_EXCEEDED)가 난다
        )
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=env("LLM_API_KEY"),
            base_url=env("LLM_BASE_URL"),  # 비우면 OpenAI 공식 API
            model=env("LLM_MODEL"),
            temperature=temperature,
            max_retries=2,
            timeout=120,
        )

    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        api_key=env("AOAI_API_KEY"),
        azure_endpoint=env("AOAI_ENDPOINT"),
        azure_deployment=env("AOAI_DEPLOY_GPT4O"),
        api_version=env("AOAI_API_VERSION"),
        temperature=temperature,
        max_retries=2,
        timeout=120,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """임베딩 모델. 모델/배포명이 없으면 None (로컬 TF-IDF로 대체)."""
    if not embedding_available():
        return None
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(api_key=env("LLM_API_KEY"), model=env("EMBEDDING_MODEL"))
    if LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=env("LLM_API_KEY"),
            base_url=env("LLM_BASE_URL"),
            model=env("EMBEDDING_MODEL"),
            check_embedding_ctx_length=False,  # 토큰 ID 대신 문자열 전송 (OpenAI 외 호환 API용)
            chunk_size=50,  # 호환 API의 요청당 입력 개수 제한 대비
        )

    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        api_key=env("AOAI_API_KEY"),
        azure_endpoint=env("AOAI_ENDPOINT"),
        azure_deployment=env("AOAI_DEPLOY_EMBEDDING"),
        api_version=env("AOAI_EMBEDDING_API_VERSION", env("AOAI_API_VERSION")),
    )


def get_callbacks() -> list:
    """Langfuse 키가 있으면 트레이싱 콜백을 붙인다 (선택)."""
    if not (env("LANGFUSE_PUBLIC_KEY") and env("LANGFUSE_SECRET_KEY")):
        return []
    try:
        from langfuse.langchain import CallbackHandler  # langfuse v3
    except Exception:
        try:
            from langfuse.callback import CallbackHandler  # langfuse v2
        except Exception:
            return []
    try:
        return [CallbackHandler()]
    except Exception:
        return []
