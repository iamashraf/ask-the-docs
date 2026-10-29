from ask_the_docs.config import ProviderConfig, Settings
from ask_the_docs.tracing import configure_tracing


def make_settings() -> Settings:
    return Settings(
        generator=ProviderConfig(provider="ollama", model="llama3.1:8b"),
        judge=ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="x"),
        otel_exporter_endpoint="http://test-jaeger:4317",
    )


class FakeProvider:
    def __init__(self, resource=None):
        self.resource = resource
        self.processors = []

    def add_span_processor(self, processor):
        self.processors.append(processor)


def _patch_sdk(monkeypatch, exporter_calls, provider_calls):
    monkeypatch.setattr("ask_the_docs.tracing._configured", False)
    monkeypatch.setattr(
        "ask_the_docs.tracing.OTLPSpanExporter",
        lambda endpoint, insecure=True: exporter_calls.append(endpoint) or object(),
    )
    monkeypatch.setattr("ask_the_docs.tracing.BatchSpanProcessor", lambda exporter: object())

    def fake_provider(resource=None):
        provider_calls.append(resource)
        return FakeProvider(resource)

    monkeypatch.setattr("ask_the_docs.tracing.TracerProvider", fake_provider)
    monkeypatch.setattr("ask_the_docs.tracing.trace.set_tracer_provider", lambda provider: None)


def test_configure_tracing_sets_up_an_exporter_with_the_configured_endpoint(monkeypatch):
    exporter_calls, provider_calls = [], []
    _patch_sdk(monkeypatch, exporter_calls, provider_calls)

    configure_tracing("ask-the-docs-app", make_settings())

    assert exporter_calls == ["http://test-jaeger:4317"]


def test_configure_tracing_sets_the_service_name_resource(monkeypatch):
    exporter_calls, provider_calls = [], []
    _patch_sdk(monkeypatch, exporter_calls, provider_calls)
    captured = {}
    monkeypatch.setattr(
        "ask_the_docs.tracing.Resource.create",
        staticmethod(lambda attrs: captured.setdefault("attrs", attrs)),
    )

    configure_tracing("ask-the-docs-worker", make_settings())

    assert captured["attrs"]["service.name"] == "ask-the-docs-worker"


def test_configure_tracing_is_idempotent(monkeypatch):
    exporter_calls, provider_calls = [], []
    _patch_sdk(monkeypatch, exporter_calls, provider_calls)

    configure_tracing("ask-the-docs-app", make_settings())
    configure_tracing("ask-the-docs-app", make_settings())

    assert len(exporter_calls) == 1


def test_configure_tracing_instruments_redis_and_requests(monkeypatch):
    """Review finding: the worker only got Redis/requests spans because
    cli.py happens to import api.py (which calls instrument() at module
    level) for every subcommand -- a wiring that works by accident. Both
    instrument() calls belong here, in the one function every entrypoint
    (api.py's lifespan, worker.py's run_worker) actually calls."""
    exporter_calls, provider_calls = [], []
    _patch_sdk(monkeypatch, exporter_calls, provider_calls)
    instrumented = []
    monkeypatch.setattr(
        "ask_the_docs.tracing.RedisInstrumentor",
        lambda: type("R", (), {"instrument": lambda self: instrumented.append("redis")})(),
    )
    monkeypatch.setattr(
        "ask_the_docs.tracing.RequestsInstrumentor",
        lambda: type("Q", (), {"instrument": lambda self: instrumented.append("requests")})(),
    )

    configure_tracing("ask-the-docs-worker", make_settings())

    assert set(instrumented) == {"redis", "requests"}
