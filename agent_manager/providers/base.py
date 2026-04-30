from dataclasses import dataclass, field
from openai import AsyncOpenAI


@dataclass
class Provider:
    name: str
    model: str
    client: AsyncOpenAI

    async def complete(self, messages: list[dict], max_tokens: int = 4096) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
