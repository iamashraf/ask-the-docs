"""OpenTelemetry SDK setup for the trace/metric/log providers, exporting to the OTel Collector over OTLP."""

import logging

from opentelemetry import _logs as logs_api
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ask_the_docs.config import Settings

_configured = False
_metrics_configured = False
_logging_configured = False


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


def configure_metrics(service_name: str, settings: Settings) -> None:
    global _metrics_configured
    if _metrics_configured:
        return
    exporter = OTLPMetricExporter(
        endpoint=settings.otel_exporter_endpoint, insecure=True
    )
    reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(
        resource=Resource.create({"service.name": service_name}),
        metric_readers=[reader],
    )
    metrics.set_meter_provider(provider)
    _metrics_configured = True


def configure_logging(service_name: str, settings: Settings) -> None:
    global _logging_configured
    if _logging_configured:
        return
    provider = LoggerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPLogExporter(endpoint=settings.otel_exporter_endpoint, insecure=True)
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    logs_api.set_logger_provider(provider)
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=provider)
    logging.getLogger("ask_the_docs").addHandler(handler)
    _logging_configured = True
