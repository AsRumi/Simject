import json
from dataclasses import dataclass

from agent.provenance import Span, dump_spans


@dataclass
class AgentState:
    spans: list[Span]
    action_log: list[dict]
    step: int = 0


def save(state: AgentState, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"spans": dump_spans(state.spans), "action_log": state.action_log,
                   "step": state.step}, handle, indent=2)


def load(path: str) -> AgentState:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    spans = [Span(entry["origin"], entry["content"]) for entry in raw["spans"]]
    return AgentState(spans, raw["action_log"], raw["step"])
