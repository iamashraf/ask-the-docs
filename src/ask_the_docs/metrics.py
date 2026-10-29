"""Prometheus metric singletons shared across api, worker, and retrieval/generation code."""

from opentelemetry import trace
from prometheus_client import Counter, Gauge, Histogram

cache_result_total = Counter(
    "cache_result",
    "Answer cache lookups by outcome",
    ["result"],
)
citation_guardrail_total = Counter(
    "citation_guardrail",
    "Citation guardrail checks by outcome",
    ["result"],
)
refusal_total = Counter("refusal", "Answers refused by the pipeline")

retrieve_latency_seconds = Histogram(
    "retrieve_latency_seconds", "Hybrid retrieval latency"
)
rerank_latency_seconds = Histogram("rerank_latency_seconds", "Reranking latency")
LLM_LATENCY_BUCKETS = (0.5, 1, 2.5, 5, 10, 20, 30, 60, 90, 120)
generate_latency_seconds = Histogram(
    "generate_latency_seconds", "LLM generation latency", buckets=LLM_LATENCY_BUCKETS
)

gen_ai_client_token_usage = Histogram(
    "gen_ai_client_token_usage",
    "Tokens used per LLM call, by token type",
    ["gen_ai_token_type", "gen_ai_request_model"],
)
gen_ai_client_operation_duration = Histogram(
    "gen_ai_client_operation_duration",
    "LLM client operation duration",
    ["gen_ai_operation_name", "gen_ai_provider_name", "gen_ai_request_model"],
    buckets=LLM_LATENCY_BUCKETS,
)

ingest_job_total = Counter(
    "ingest_job",
    "Ingestion jobs processed by the worker, by outcome",
    ["status"],
)
build_index_latency_seconds = Histogram(
    "build_index_latency_seconds",
    "Index build latency per ingestion job",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)
job_queue_depth = Gauge("job_queue_depth", "Pending jobs in the NATS JetStream consumer")
worker_job_latency_seconds = Histogram(
    "worker_job_latency_seconds",
    "Time from job enqueue to completion",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 900, 1800),
)


def trace_exemplar() -> dict[str, str]:
    """Current span's trace ID as an exemplar label; empty when no valid span is active."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return {}
    return {"trace_id": format(ctx.trace_id, "032x")}
