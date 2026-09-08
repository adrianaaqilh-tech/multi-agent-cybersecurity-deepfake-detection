"""Configuration for the Multi-Agent System."""

from langchain_openai import ChatOpenAI
import requests


LLM_BASE_URL = "http://localhost:1234/v1"


llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key="lm-studio",
    model="google/gemma-4-e4b",
    temperature=0,
    timeout=30,
    max_retries=1,
)


def is_llm_available() -> bool:
    """Check whether the local LM Studio server is available."""
    try:
        response = requests.get(
            f"{LLM_BASE_URL}/models",
            timeout=1
        )
        return response.status_code == 200
    except requests.RequestException:
        return False