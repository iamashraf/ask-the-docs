from opentelemetry import metrics as metrics_api
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from ask_the_docs import metrics

reader = InMemoryMetricReader()
# set_meter_provider only takes effect once per process; later calls elsewhere
# are silently ignored, so this reader wins for the whole test session.
metrics_api.set_meter_provider(MeterProvider(metric_readers=[reader]))


def get_metrics_data():
    return reader.get_metrics_data()


def find_metric(data, name):
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == name:
                    return metric
    raise AssertionError(f"no metric named {name!r} found")


def find_data_point(metric, **attributes):
    for point in metric.data.data_points:
        if dict(point.attributes) == attributes:
            return point
    raise AssertionError(
        f"no data point with attributes {attributes!r} on {metric.name!r}"
    )


def test_cache_result_total_records_with_result_attribute():
    metrics.cache_result_total.add(1, {"result": "hit"})
    metric = find_metric(get_metrics_data(), "cache_result")
    point = find_data_point(metric, result="hit")
    assert point.value >= 1


def test_citation_guardrail_total_records_with_result_attribute():
    metrics.citation_guardrail_total.add(1, {"result": "pass"})
    metric = find_metric(get_metrics_data(), "citation_guardrail")
    point = find_data_point(metric, result="pass")
    assert point.value >= 1


def test_refusal_total_records():
    metrics.refusal_total.add(1, {})
    metric = find_metric(get_metrics_data(), "refusal")
    point = find_data_point(metric)
    assert point.value >= 1


def test_retrieve_latency_records_duration_with_default_latency_buckets():
    metrics.retrieve_latency_seconds.record(0.75, {})
    metric = find_metric(get_metrics_data(), "retrieve_latency")
    point = find_data_point(metric)
    assert point.sum >= 0.75
    assert point.count >= 1
    assert tuple(point.explicit_bounds) == metrics.DEFAULT_LATENCY_BUCKETS
    # 0.75 falls in the (0.5, 0.75] bucket, not lumped in with multi-second values.
    bucket_index = metrics.DEFAULT_LATENCY_BUCKETS.index(0.75)
    assert point.bucket_counts[bucket_index] >= 1


def test_rerank_latency_records_duration_with_default_latency_buckets():
    metrics.rerank_latency_seconds.record(0.05, {})
    metric = find_metric(get_metrics_data(), "rerank_latency")
    point = find_data_point(metric)
    assert point.sum >= 0.05
    assert point.count >= 1
    assert tuple(point.explicit_bounds) == metrics.DEFAULT_LATENCY_BUCKETS
    # 0.05 falls in the (0.025, 0.05] bucket -- only resolvable with sub-5s buckets.
    bucket_index = metrics.DEFAULT_LATENCY_BUCKETS.index(0.05)
    assert point.bucket_counts[bucket_index] >= 1


def test_generate_latency_uses_llm_latency_buckets():
    metrics.generate_latency_seconds.record(3, {})
    metric = find_metric(get_metrics_data(), "generate_latency")
    point = find_data_point(metric)
    assert point.sum >= 3
    assert tuple(point.explicit_bounds) == metrics.LLM_LATENCY_BUCKETS


def test_ingest_job_total_records_with_status_attribute():
    metrics.ingest_job_total.add(1, {"status": "success"})
    metric = find_metric(get_metrics_data(), "ingest_job")
    point = find_data_point(metric, status="success")
    assert point.value >= 1


def test_build_index_latency_records_duration_with_its_buckets():
    metrics.build_index_latency_seconds.record(12, {})
    metric = find_metric(get_metrics_data(), "build_index_latency")
    point = find_data_point(metric)
    assert point.sum >= 12
    assert tuple(point.explicit_bounds) == (1, 5, 10, 30, 60, 120, 300, 600)


def test_job_queue_depth_gauge_reads_back_last_set_value():
    metrics.job_queue_depth.set(5, {})
    metrics.job_queue_depth.set(3, {})
    metric = find_metric(get_metrics_data(), "job_queue_depth")
    point = find_data_point(metric)
    assert point.value == 3


def test_worker_job_latency_records_duration_with_its_buckets():
    metrics.worker_job_latency_seconds.record(45, {})
    metric = find_metric(get_metrics_data(), "worker_job_latency")
    point = find_data_point(metric)
    assert point.sum >= 45
    assert tuple(point.explicit_bounds) == (1, 5, 10, 30, 60, 120, 300, 600, 900, 1800)


def test_gen_ai_client_inference_duration_records_with_llm_latency_buckets():
    metrics.gen_ai_client_inference_duration.record(
        2,
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "gpt-4",
        },
    )
    metric = find_metric(get_metrics_data(), "gen_ai.client.inference.duration")
    point = find_data_point(
        metric,
        **{
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "gpt-4",
        },
    )
    assert point.sum >= 2
    assert tuple(point.explicit_bounds) == metrics.LLM_LATENCY_BUCKETS


def test_gen_ai_client_inference_usage_input_tokens_records_count():
    metrics.gen_ai_client_inference_usage_input_tokens.add(
        10,
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.token.modality": "text",
            "gen_ai.request.model": "gpt-4",
        },
    )
    metric = find_metric(
        get_metrics_data(), "gen_ai.client.inference.usage.input_tokens"
    )
    point = find_data_point(
        metric,
        **{
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.token.modality": "text",
            "gen_ai.request.model": "gpt-4",
        },
    )
    assert point.value >= 10


def test_gen_ai_client_inference_usage_output_tokens_records_count():
    metrics.gen_ai_client_inference_usage_output_tokens.add(
        7,
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.token.modality": "text",
            "gen_ai.request.model": "gpt-4",
        },
    )
    metric = find_metric(
        get_metrics_data(), "gen_ai.client.inference.usage.output_tokens"
    )
    point = find_data_point(
        metric,
        **{
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.token.modality": "text",
            "gen_ai.request.model": "gpt-4",
        },
    )
    assert point.value >= 7
