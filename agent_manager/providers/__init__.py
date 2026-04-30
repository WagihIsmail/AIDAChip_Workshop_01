import httpx
from openai import AsyncOpenAI

from config import settings
from providers.base import Provider

# ── Groq free models ──────────────────────────────────────────────────────────
GROQ_MODELS = [
    ("Llama-3.3-70B", "llama-3.3-70b-versatile"),
    ("Llama-3.1-8B",  "llama-3.1-8b-instant"),
]

# ── OpenRouter free models ────────────────────────────────────────────────────
# All require a single OpenRouter API key; append :free for zero-cost tier.
OPENROUTER_MODELS = [
    ("Llama-3.1-8B",  "meta-llama/llama-3.1-8b-instruct:free"),
    ("Kimi-K2",       "moonshotai/kimi-k2:free"),
    ("Nemotron-70B",  "nvidia/llama-3.1-nemotron-70b-instruct:free"),
    ("Hermes-3-405B", "nousresearch/hermes-3-llama-3.1-405b:free"),
]


def _ollama_available_models() -> list[str]:
    """Returns model names currently pulled in the local Ollama instance."""
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{settings.ollama_base_url}/api/tags")
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def get_available_providers() -> list[Provider]:
    providers: list[Provider] = []

    # ── Groq ──────────────────────────────────────────────────────────────────
    if settings.groq_api_key:
        groq_client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
        )
        for label, model in GROQ_MODELS:
            providers.append(Provider(
                name=f"Groq · {label}",
                model=model,
                client=groq_client,
            ))

    # ── Google Gemini ─────────────────────────────────────────────────────────
    if settings.gemini_api_key:
        providers.append(Provider(
            name="Gemini · Flash-2.0",
            model="gemini-2.0-flash",
            client=AsyncOpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=settings.gemini_api_key,
            ),
        ))

    # ── OpenRouter (one Provider per model) ───────────────────────────────────
    if settings.openrouter_api_key:
        or_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "AI Agent Manager",
            },
        )
        for label, model in OPENROUTER_MODELS:
            providers.append(Provider(
                name=f"OpenRouter · {label}",
                model=model,
                client=or_client,
            ))

    # ── Ollama (one Provider per locally-available model) ─────────────────────
    ollama_models = _ollama_available_models()
    if ollama_models:
        ollama_client = AsyncOpenAI(
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama",
        )
        for model in ollama_models:
            providers.append(Provider(
                name=f"Ollama · {model}",
                model=model,
                client=ollama_client,
            ))

    return providers
