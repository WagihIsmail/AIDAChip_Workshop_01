import httpx
from openai import AsyncOpenAI

from config import settings
from providers.base import Provider


def _check_ollama() -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{settings.ollama_base_url}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


def get_available_providers() -> list[Provider]:
    providers: list[Provider] = []

    if settings.groq_api_key:
        providers.append(Provider(
            name="Groq",
            model="llama-3.3-70b-versatile",
            client=AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.groq_api_key,
            ),
        ))

    if settings.gemini_api_key:
        providers.append(Provider(
            name="Gemini",
            model="gemini-2.0-flash",
            client=AsyncOpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=settings.gemini_api_key,
            ),
        ))

    if settings.openrouter_api_key:
        providers.append(Provider(
            name="OpenRouter",
            model="meta-llama/llama-3.1-8b-instruct:free",
            client=AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
                default_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "AI Agent Manager",
                },
            ),
        ))

    if _check_ollama():
        providers.append(Provider(
            name="Ollama",
            model=settings.ollama_model,
            client=AsyncOpenAI(
                base_url=f"{settings.ollama_base_url}/v1",
                api_key="ollama",
            ),
        ))

    return providers
