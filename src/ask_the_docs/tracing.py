"""OpenTelemetry SDK setup -- one TracerProvider per process, exporting to
Tempo over OTLP; mirrors the module-level singleton pattern jobs.py/
queue.py use for their own connection config."""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ask_the_docs.config import Settings

_configured = False


def configure_tracing(service_name: str, settings: Settings) -> None:
    """Instruments Redis/requests here, not at each entrypoint's own import
    time -- api.py and worker.py both call this function, so wiring the
    library-level instrumentation into it (instead of into api.py's module
    body, which worker.py never imports) means neither process's spans
    depend on an incidental import chain."""
    global _configured
    if _configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    RedisInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    _configured = True
