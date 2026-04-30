import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agents.auditor import audit_answer
from agents.orchestrator import orchestrate
from agents.planner import plan_task
from agents.refiner import refine_answer
from agents.runner import collect_results
from config import settings
from providers import get_available_providers

app = FastAPI(title="AI Agent Manager")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class TaskRequest(BaseModel):
    task: str


@app.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/api/providers")
async def list_providers() -> dict:
    return {
        "providers": [
            {"name": p.name, "model": p.model}
            for p in get_available_providers()
        ]
    }


@app.post("/api/task")
async def run_task(request: TaskRequest) -> EventSourceResponse:
    async def generate():
        def emit(event: str, data: dict) -> dict:
            return {"event": event, "data": json.dumps(data)}

        providers = get_available_providers()
        if not providers:
            yield emit("error", {
                "message": (
                    "No LLM providers configured. "
                    "Copy .env.example to .env and add at least one API key."
                )
            })
            return

        # Assign distinct roles to different providers where possible
        orchestrator = providers[0]
        auditor      = providers[-1]                  # last — likely a different model
        refiner      = providers[len(providers) // 2] # middle — third distinct model

        # ── 1. Planning ──────────────────────────────────────────────────────
        yield emit("status", {"stage": "planning", "message": "Analyzing task..."})
        try:
            plan = await plan_task(request.task, orchestrator)
        except Exception as exc:
            yield emit("error", {"message": f"Planning failed: {exc}"})
            return

        yield emit("planning_done", plan.model_dump())

        # ── 2. Sub-agents (parallel) ─────────────────────────────────────────
        n = len(plan.sub_tasks)
        yield emit("status", {
            "stage": "running",
            "message": f"Running {n} agent{'s' if n != 1 else ''} in parallel...",
        })

        queue: asyncio.Queue[dict] = asyncio.Queue()
        runner_task = asyncio.create_task(
            collect_results(plan.sub_tasks, providers, queue)
        )

        completed = 0
        while completed < n:
            event = await queue.get()
            if event["event"] == "agent_done":
                completed += 1
            yield event

        results: dict[str, str] = await runner_task

        # ── 3. Orchestration ─────────────────────────────────────────────────
        if n > 1:
            yield emit("status", {
                "stage": "orchestrating",
                "message": f"Synthesizing results ({orchestrator.name})...",
            })
            try:
                current_answer = await orchestrate(
                    request.task, plan.sub_tasks, results, orchestrator
                )
            except Exception as exc:
                yield emit("error", {"message": f"Orchestration failed: {exc}"})
                return
        else:
            current_answer = results.get(plan.sub_tasks[0].id, "")

        # ── 4. Audit → Refine loop ────────────────────────────────────────────
        max_iter = settings.max_audit_iterations
        for iteration in range(max_iter):
            round_num = iteration + 1

            yield emit("status", {
                "stage": "auditing",
                "message": f"Auditing answer — round {round_num}/{max_iter} ({auditor.name})...",
            })
            try:
                audit = await audit_answer(request.task, current_answer, auditor)
            except Exception as exc:
                yield emit("error", {"message": f"Audit failed: {exc}"})
                return

            yield emit("audit_done", {
                "iteration": round_num,
                "score": audit.score,
                "passed": audit.passed,
                "issues": audit.issues,
                "suggestions": audit.suggestions,
                "summary": audit.summary,
                "provider": auditor.name,
            })

            if audit.passed:
                break

            # Still more iterations left — refine
            if iteration < max_iter - 1:
                yield emit("status", {
                    "stage": "refining",
                    "message": f"Refining answer — round {round_num} ({refiner.name})...",
                })
                try:
                    current_answer = await refine_answer(
                        request.task, current_answer, audit, refiner
                    )
                except Exception as exc:
                    yield emit("error", {"message": f"Refinement failed: {exc}"})
                    return

                yield emit("refine_done", {
                    "iteration": round_num,
                    "provider": refiner.name,
                })

        yield emit("done", {"final_answer": current_answer})

    return EventSourceResponse(generate())


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
