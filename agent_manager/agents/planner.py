import json
import re

from pydantic import BaseModel

from providers.base import Provider


class SubTask(BaseModel):
    id: str
    description: str
    prompt: str


class Plan(BaseModel):
    complexity: int
    task_type: str
    summary: str
    sub_tasks: list[SubTask]


_PLANNER_SYSTEM = """You are a task planning AI. Analyze a user task and decompose it into sub-tasks for parallel AI agents.

Respond with ONLY a valid JSON object — no markdown fences, no explanation, just raw JSON.

JSON structure:
{
  "complexity": <integer 1-10>,
  "task_type": "<code | research | writing | general>",
  "summary": "<one sentence describing the overall task>",
  "sub_tasks": [
    {
      "id": "task_1",
      "description": "<brief sub-task label>",
      "prompt": "<complete self-contained prompt for an AI agent to execute this sub-task>"
    }
  ]
}

Decomposition rules:
- complexity 1-3  → 1 sub-task (route directly, no decomposition)
- complexity 4-6  → 2-3 sub-tasks
- complexity 7-10 → 3-5 sub-tasks

Each sub-task prompt must be fully self-contained with all necessary context."""


async def plan_task(task: str, provider: Provider) -> Plan:
    response = await provider.complete(
        messages=[
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": f"Task: {task}"},
        ],
        max_tokens=2048,
    )

    json_str = response.strip()
    # Strip markdown fences if the model wraps output anyway
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str)
    if match:
        json_str = match.group(1)

    data = json.loads(json_str)
    return Plan(**data)
