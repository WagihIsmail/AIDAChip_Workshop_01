from agents.planner import SubTask
from providers.base import Provider


_ORCHESTRATOR_SYSTEM = """You are a synthesis AI. You receive results from multiple AI agents that each handled a sub-task of a larger goal.

Your job is to integrate their outputs into a single, coherent, and well-structured final response.
- Remove redundancy while preserving all important information.
- Maintain a logical flow between sections.
- Format with clear headers and structure where appropriate.
- Do NOT mention the sub-task structure or that multiple agents were used — just deliver a unified, high-quality answer."""


async def orchestrate(
    original_task: str,
    sub_tasks: list[SubTask],
    results: dict[str, str],
    provider: Provider,
) -> str:
    agent_outputs = "\n\n".join(
        f"### {st.description}\n{results.get(st.id, '[no result]')}"
        for st in sub_tasks
    )

    user_message = f"""Original task: {original_task}

Agent outputs:
{agent_outputs}

Please synthesize the above into a single comprehensive response."""

    return await provider.complete(
        messages=[
            {"role": "system", "content": _ORCHESTRATOR_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        max_tokens=4096,
    )
