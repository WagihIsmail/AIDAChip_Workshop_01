import json
import re

from pydantic import BaseModel

from providers.base import Provider

PASS_THRESHOLD = 7  # score >= this is considered passing


class AuditResult(BaseModel):
    score: int
    passed: bool
    issues: list[str]
    suggestions: list[str]
    summary: str


_AUDITOR_SYSTEM = """You are a strict quality auditor. You review an AI-generated answer against the original task and score it objectively.

Respond with ONLY a valid JSON object — no markdown, no explanation.

{
  "score": <integer 1-10>,
  "passed": <true if score >= 7, else false>,
  "issues": ["<concrete problem 1>", "<concrete problem 2>"],
  "suggestions": ["<specific actionable fix 1>", "<specific actionable fix 2>"],
  "summary": "<2-3 sentences: what is good, what is lacking, overall verdict>"
}

Scoring rubric:
  9-10  Excellent — comprehensive, accurate, well-structured, nothing missing
  7-8   Good — mostly complete, minor gaps or imprecision
  5-6   Adequate — covers the basics but missing important elements
  3-4   Poor — significant inaccuracies or major gaps
  1-2   Failing — fundamentally wrong or off-topic

Be rigorous. Do not pass mediocre answers."""


async def audit_answer(
    original_task: str,
    answer: str,
    provider: Provider,
) -> AuditResult:
    response = await provider.complete(
        messages=[
            {"role": "system", "content": _AUDITOR_SYSTEM},
            {"role": "user", "content": (
                f"Original task:\n{original_task}\n\n"
                f"Answer to audit:\n{answer}"
            )},
        ],
        max_tokens=1024,
    )

    json_str = response.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str)
    if match:
        json_str = match.group(1)

    data = json.loads(json_str)
    data["passed"] = int(data.get("score", 0)) >= PASS_THRESHOLD
    return AuditResult(**data)
