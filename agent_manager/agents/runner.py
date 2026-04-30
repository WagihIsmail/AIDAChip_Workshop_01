import asyncio
import json

from agents.planner import SubTask
from providers.base import Provider


async def collect_results(
    sub_tasks: list[SubTask],
    providers: list[Provider],
    event_queue: asyncio.Queue,
) -> dict[str, str]:
    """Runs all sub-tasks in parallel, pushes SSE events to the queue, returns results."""
    results: dict[str, str] = {}

    async def run_one(sub_task: SubTask, provider: Provider) -> None:
        await event_queue.put({
            "event": "agent_start",
            "data": json.dumps({
                "task_id": sub_task.id,
                "description": sub_task.description,
                "provider": provider.name,
                "model": provider.model,
            }),
        })
        try:
            result = await provider.complete(
                messages=[{"role": "user", "content": sub_task.prompt}]
            )
        except Exception as exc:
            result = f"[Error from {provider.name}]: {exc}"

        results[sub_task.id] = result
        await event_queue.put({
            "event": "agent_done",
            "data": json.dumps({
                "task_id": sub_task.id,
                "description": sub_task.description,
                "result": result,
                "provider": provider.name,
            }),
        })

    await asyncio.gather(*[
        asyncio.create_task(run_one(st, providers[i % len(providers)]))
        for i, st in enumerate(sub_tasks)
    ])
    return results
