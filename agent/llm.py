import functools
import os
import re
import time

from google import genai
from google.genai import errors, types

MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 1.0
MAX_ATTEMPTS = 5


@functools.lru_cache(maxsize=1)
def client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        for line in open(".env", encoding="utf-8"):
            if line.startswith("GEMINI_API_KEY"):
                key = line.split("=", 1)[1].strip().strip("\"'")
    return genai.Client(api_key=key)


def call(system: str, contents: list[dict], declarations: list[dict]) -> tuple[dict, dict]:
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=TEMPERATURE,
        tools=[types.Tool(function_declarations=declarations)] if declarations else None,
    )
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client().models.generate_content(
                model=MODEL, contents=contents, config=config)
            break
        except errors.APIError as failure:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            asked = re.search(r"retry in ([\d.]+)s", str(failure))
            time.sleep(float(asked.group(1)) + 1 if asked else 2 ** attempt)
    used = response.usage_metadata
    tokens = {"prompt_tokens": used.prompt_token_count or 0,
              "completion_tokens": (used.candidates_token_count or 0)
                                   + (used.thoughts_token_count or 0)}
    return response.candidates[0].content.model_dump(mode="json", exclude_none=True), tokens
