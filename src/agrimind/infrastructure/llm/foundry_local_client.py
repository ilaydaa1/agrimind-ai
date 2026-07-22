"""Client for Microsoft Foundry Local's OpenAI-compatible chat endpoint."""

from openai import OpenAI

from agrimind.config import (
    FOUNDRY_API_KEY,
    FOUNDRY_BASE_URL,
    FOUNDRY_MODEL,
    GENERATION_TEMPERATURE,
)


class FoundryLocalClient:
    """Sends a prompt to the local model and returns the generated text.

    Transport only: it does not build prompts, retrieve context, or apply
    refusal logic. The composition root creates a single instance and owns its
    lifetime.
    """

    def __init__(self) -> None:
        self._client = OpenAI(base_url=FOUNDRY_BASE_URL, api_key=FOUNDRY_API_KEY)

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=FOUNDRY_MODEL,
            temperature=GENERATION_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
