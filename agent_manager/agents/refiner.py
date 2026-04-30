from agents.auditor import AuditResult
from providers.base import Provider


_REFINER_SYSTEM = """You are a refinement AI. You receive a draft answer, an auditor's critique, and the original task.
Your job is to produce a fully revised, improved answer that:
- Fixes every issue the auditor identified
- Implements every suggestion provided
- Preserves everything that was already correct and good
- Is complete and standalone — do not reference the previous draft or the audit process."""


async def refine_answer(
    original_task: str,
    draft: str,
    audit: AuditResult,
    provider: Provider,
) -> str:
    issues_text = "\n".join(f"  • {i}" for i in audit.issues)
    suggestions_text = "\n".join(f"  • {s}" for s in audit.suggestions)

    user_message = (
        f"Original task:\n{original_task}\n\n"
        f"Current draft answer:\n{draft}\n\n"
        f"Auditor score: {audit.score}/10\n"
        f"Auditor summary: {audit.summary}\n\n"
        f"Issues to fix:\n{issues_text}\n\n"
        f"Improvements to make:\n{suggestions_text}\n\n"
        f"Produce the fully revised answer now."
    )

    return await provider.complete(
        messages=[
            {"role": "system", "content": _REFINER_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        max_tokens=4096,
    )
