from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import REGISTRY

from ask_the_docs import metrics


def test_cache_result_total_has_result_label():
    before = REGISTRY.get_sample_value("cache_result_total", {"result": "hit"}) or 0
    metrics.cache_result_total.labels(result="hit").inc()
    after = REGISTRY.get_sample_value("cache_result_total", {"result": "hit"})
    assert after - before == 1.0


def test_long_running_histograms_cover_their_realistic_range():
    # Default buckets top out at 10s; these operations routinely run longer.
    assert max(metrics.generate_latency_seconds._upper_bounds[:-1]) >= 120
    assert max(metrics.gen_ai_client_operation_duration._upper_bounds[:-1]) >= 120
    assert max(metrics.build_index_latency_seconds._upper_bounds[:-1]) >= 600
    assert max(metrics.worker_job_latency_seconds._upper_bounds[:-1]) >= 600


def test_trace_exemplar_empty_without_active_span():
    assert metrics.trace_exemplar() == {}


def test_trace_exemplar_has_trace_id_with_active_span():
    tracer = TracerProvider().get_tracer(__name__)
    with tracer.start_as_current_span("x"):
        exemplar = metrics.trace_exemplar()
    assert "trace_id" in exemplar and len(exemplar["trace_id"]) == 32
