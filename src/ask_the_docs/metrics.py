"""OTel metric instrument singletons shared across api, worker, and retrieval/generation code."""

from opentelemetry import metrics

meter = metrics.get_meter("ask_the_docs")

LLM_LATENCY_BUCKETS = (0.5, 1, 2.5, 5, 10, 20, 30, 60, 90, 120)
# prometheus_client's own default histogram buckets -- OTel SDK's default
# (5ms..10s floor at 5) is too coarse for sub-5s retrieval/rerank latencies.
DEFAULT_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
)

cache_result_total = meter.create_counter(
    "cache_result", description="Answer cache lookups by outcome"
)
citation_guardrail_total = meter.create_counter(
    "citation_guardrail", description="Citation guardrail checks by outcome"
)
refusal_total = meter.create_counter(
    "refusal", description="Answers refused by the pipeline"
)

retrieve_latency_seconds = meter.create_histogram(
    "retrieve_latency",
    unit="s",
    description="Hybrid retrieval latency",
    explicit_bucket_boundaries_advisory=DEFAULT_LATENCY_BUCKETS,
)
rerank_latency_seconds = meter.create_histogram(
    "rerank_latency",
    unit="s",
    description="Reranking latency",
    explicit_bucket_boundaries_advisory=DEFAULT_LATENCY_BUCKETS,
)
generate_latency_seconds = meter.create_histogram(
    "generate_latency",
    unit="s",
    description="LLM generation latency",
    explicit_bucket_boundaries_advisory=LLM_LATENCY_BUCKETS,
)

gen_ai_client_inference_duration = meter.create_histogram(
    "gen_ai.client.inference.duration",
    unit="s",
    description="Duration of GenAI client inference operations",
    explicit_bucket_boundaries_advisory=LLM_LATENCY_BUCKETS,
)
gen_ai_client_inference_usage_input_tokens = meter.create_counter(
    "gen_ai.client.inference.usage.input_tokens",
    unit="{token}",
    description="Input tokens used per GenAI client inference operation",
)
gen_ai_client_inference_usage_output_tokens = meter.create_counter(
    "gen_ai.client.inference.usage.output_tokens",
    unit="{token}",
    description="Output tokens used per GenAI client inference operation",
)

ingest_job_total = meter.create_counter(
    "ingest_job", description="Ingestion jobs processed by the worker, by outcome"
)
build_index_latency_seconds = meter.create_histogram(
    "build_index_latency",
    unit="s",
    description="Index build latency per ingestion job",
    explicit_bucket_boundaries_advisory=(1, 5, 10, 30, 60, 120, 300, 600),
)
job_queue_depth = meter.create_gauge(
    "job_queue_depth", description="Pending jobs in the NATS JetStream consumer"
)
worker_job_latency_seconds = meter.create_histogram(
    "worker_job_latency",
    unit="s",
    description="Time from job enqueue to completion",
    explicit_bucket_boundaries_advisory=(1, 5, 10, 30, 60, 120, 300, 600, 900, 1800),
)
