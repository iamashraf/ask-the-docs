"""Standalone eval: runs route() against labeled real-world-shaped inputs the unit tests' mocked LLM responses can't exercise."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from ask_the_docs.config import Settings
from ask_the_docs.router import Turn, route


@dataclass
class RoutingCase:
    input: str
    expected_intent: Literal["doc", "chitchat", "meta"]
    category: str
    history: list[Turn] = field(default_factory=list)


@dataclass
class RoutingResult:
    case: RoutingCase
    actual_intent: str
    source: str
    critical: bool  # expected doc, got non-doc


def load_routing_cases(path: Path) -> list[RoutingCase]:
    data = yaml.safe_load(path.read_text())
    return [
        RoutingCase(
            input=item["input"],
            expected_intent=item["expected_intent"],
            category=item["category"],
            history=[Turn(**turn) for turn in item.get("history", [])],
        )
        for item in data
    ]


def run_routing_eval(settings: Settings, cases_path: Path) -> list[RoutingResult]:
    results = []
    for case in load_routing_cases(cases_path):
        result = route(case.input, case.history, settings)
        results.append(
            RoutingResult(
                case=case,
                actual_intent=result.intent,
                source=result.source,
                critical=case.expected_intent == "doc" and result.intent != "doc",
            )
        )
    return results


def routing_report(results: list[RoutingResult]) -> str:
    """Per-category correct/total, every misroute, and the critical-misroute count."""
    lines = []
    for category in sorted({r.case.category for r in results}):
        in_category = [r for r in results if r.case.category == category]
        correct = sum(1 for r in in_category if r.actual_intent == r.case.expected_intent)
        lines.append(f"{category}: {correct}/{len(in_category)}")

    misroutes = [r for r in results if r.actual_intent != r.case.expected_intent]
    if misroutes:
        lines.append("\nMisroutes:")
        for r in misroutes:
            flag = " [critical]" if r.critical else ""
            lines.append(
                f"- [{r.case.category}] {r.case.input!r}: expected {r.case.expected_intent}, "
                f"got {r.actual_intent} (source: {r.source}){flag}"
            )

    critical_count = sum(1 for r in results if r.critical)
    lines.append(f"\ncritical: {critical_count}")
    return "\n".join(lines)
