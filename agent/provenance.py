import json
from dataclasses import dataclass


@dataclass
class Span:
    origin: str
    content: dict


def text_span(origin: str, text: str) -> Span:
    role = "system" if origin == "system" else "user"
    return Span(origin, {"role": role, "parts": [{"text": text}]})


def tool_output_span(tool: str, output: str) -> Span:
    return Span("tool_output", {"role": "user", "parts": [
        {"function_response": {"name": tool, "response": {"output": output}}}]})


def span_text(span: Span) -> str:
    chunks = []
    for part in span.content.get("parts", []):
        if "text" in part:
            chunks.append(part["text"])
        elif "function_response" in part:
            chunks.append(part["function_response"]["response"]["output"])
        elif "function_call" in part:
            called = part["function_call"]
            chunks.append(f"{called['name']}({json.dumps(called.get('args', {}))})")
    return "\n".join(chunks)


def dump_spans(spans: list[Span]) -> list[dict]:
    return [{"origin": span.origin, "content": span.content} for span in spans]
