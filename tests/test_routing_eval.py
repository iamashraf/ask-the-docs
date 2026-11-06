
from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.router import Route, Turn
from ask_the_docs.routing_eval import (
    RoutingCase,
    RoutingResult,
    load_routing_cases,
    routing_report,
    run_routing_eval,
)


def make_settings(**overrides) -> Settings:
    fields = dict(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
    )
    fields.update(overrides)
    return Settings(**fields)


def test_load_routing_cases_parses_yaml_including_history(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
- input: "hi"
  expected_intent: "chitchat"
  category: "chitchat"
- input: "why does that help?"
  expected_intent: "doc"
  category: "follow-up"
  history:
    - role: "user"
      content: "What is Chain-of-Thought?"
    - role: "assistant"
      content: "It's a prompting technique."
"""
    )

    cases = load_routing_cases(path)

    assert cases == [
        RoutingCase(input="hi", expected_intent="chitchat", category="chitchat"),
        RoutingCase(
            input="why does that help?",
            expected_intent="doc",
            category="follow-up",
            history=[
                Turn(role="user", content="What is Chain-of-Thought?"),
                Turn(role="assistant", content="It's a prompting technique."),
            ],
        ),
    ]


def test_run_routing_eval_passes_history_to_route(tmp_path, monkeypatch):
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
- input: "why does that help?"
  expected_intent: "doc"
  category: "follow-up"
  history:
    - role: "user"
      content: "What is Chain-of-Thought?"
"""
    )
    captured = {}

    def fake_route(query, history, settings):
        captured["query"] = query
        captured["history"] = history
        return Route("doc", query, "", "llm")

    monkeypatch.setattr("ask_the_docs.routing_eval.route", fake_route)

    run_routing_eval(make_settings(), path)

    assert captured["query"] == "why does that help?"
    assert captured["history"] == [Turn(role="user", content="What is Chain-of-Thought?")]


def test_doc_expected_misrouted_to_chitchat_is_critical(tmp_path, monkeypatch):
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
- input: "what is LoRA?"
  expected_intent: "doc"
  category: "doc"
"""
    )
    monkeypatch.setattr(
        "ask_the_docs.routing_eval.route",
        lambda query, history, settings: Route("chitchat", "", "hi", "llm"),
    )

    results = run_routing_eval(make_settings(), path)

    assert results[0].actual_intent == "chitchat"
    assert results[0].critical is True


def test_chitchat_expected_misrouted_to_doc_is_not_critical(tmp_path, monkeypatch):
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
- input: "thanks!"
  expected_intent: "chitchat"
  category: "chitchat"
"""
    )
    monkeypatch.setattr(
        "ask_the_docs.routing_eval.route",
        lambda query, history, settings: Route("doc", "thanks!", "", "llm"),
    )

    results = run_routing_eval(make_settings(), path)

    assert results[0].actual_intent == "doc"
    assert results[0].critical is False


def test_correct_doc_route_is_not_critical(tmp_path, monkeypatch):
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
- input: "what is LoRA?"
  expected_intent: "doc"
  category: "doc"
"""
    )
    monkeypatch.setattr(
        "ask_the_docs.routing_eval.route",
        lambda query, history, settings: Route("doc", "what is LoRA?", "", "llm"),
    )

    results = run_routing_eval(make_settings(), path)

    assert results[0].critical is False


def test_routing_report_shows_per_category_accuracy_and_misroutes():
    results = [
        RoutingResult(
            case=RoutingCase(input="hi", expected_intent="chitchat", category="chitchat"),
            actual_intent="chitchat",
            source="rules",
            critical=False,
        ),
        RoutingResult(
            case=RoutingCase(input="what is LoRA?", expected_intent="doc", category="doc"),
            actual_intent="chitchat",
            source="llm",
            critical=True,
        ),
    ]

    report = routing_report(results)

    assert "chitchat: 1/1" in report
    assert "doc: 0/1" in report
    assert "what is LoRA?" in report
    assert "expected doc" in report
    assert "got chitchat" in report
    assert "source: llm" in report
    assert "critical" in report.lower()


def test_routing_report_counts_criticals():
    results = [
        RoutingResult(
            case=RoutingCase(input="what is LoRA?", expected_intent="doc", category="doc"),
            actual_intent="chitchat",
            source="llm",
            critical=True,
        ),
        RoutingResult(
            case=RoutingCase(input="thanks!", expected_intent="chitchat", category="chitchat"),
            actual_intent="doc",
            source="llm",
            critical=False,
        ),
    ]

    report = routing_report(results)

    assert "critical: 1" in report.lower()
