# Generator model comparison: Ollama vs. hosted

The generator provider and model are config-only (`GENERATOR__PROVIDER`,
`GENERATOR__MODEL`, `GENERATOR__BASE_URL`; see `config.py`), so switching
between Ollama and a hosted model needs no code change. This page compares the
local Ollama models against the hosted reference on latency, answer accuracy,
and grounding, so the default rests on measurements. A fast model that invents
answers is worse than a slow one that does not.

## Method

- **Question set.** `evals/model_comparison_questions.yaml` holds 10
  questions the seeded corpus answers and 5 in-domain near-misses. A
  near-miss shares a topic with a seeded paper but asks for a specific fact or
  extrapolation the paper does not state. The file is separate from the
  routing and regression cases in `golden_questions.yaml`.
- **Judge.** Fixed at `gpt-4o` for every candidate, so scores are comparable
  across runs.
- **Isolation.** Each Ollama model runs with every other model unloaded first
  (`ollama stop`). `OLLAMA_KEEP_ALIVE` otherwise leaves a model resident after
  use. Running candidates back-to-back without unloading leaves several models
  loaded at once, which oversubscribes the host CPU and corrupts later
  candidates' latency numbers.
- **Guardrail behavior.** `answer.py` accepts a model's deliberate refusal
  (`grounded: false` on the structured path, or the exact refusal message on
  the free-text path) on the first attempt and does not retry it. A retry
  re-samples the same nondeterministic call and can turn a correct refusal
  into a wrong, confidently cited answer. Keep this in mind when you read the
  near-miss columns below.

Metrics per candidate, all over the 15-question set:

| Metric | Meaning |
|---|---|
| mean / p95 latency | End-to-end `answer()` time: routing, retrieval, and generation |
| refusals correct | Of the 5 near-miss questions, how many correctly refused |
| over-refused | Of the 10 answerable questions, how many wrongly refused |
| hallucinated | Of the 5 near-miss questions, how many got a wrong, cited answer instead of refusing |
| judge accuracy | Mean accuracy score (0–1) from the judge, over answered questions only |

## Environment

Apple M5 Pro, with Ollama running inside a podman container. It is CPU-only,
because macOS container runtimes have no path to the GPU. The podman machine
has 8 vCPU / 32 GB. These numbers apply to that setup only, not to the
models' speed in general.

## Results

| Candidate | Mean latency | p95 latency | Refusals correct | Over-refused | Hallucinated | Judge accuracy |
|---|---|---|---|---|---|---|
| openai: gpt-4o-mini (reference) | 9.2s | 10.2s | 4/5 | 0/10 | 1/5 | 0.99 |
| ollama: llama3.1:8b | 24.4s | 32.6s | 4/5 | 0/10 | 1/5 | 0.81 |
| ollama: granite3.3:2b | 20.1s | 33.4s | 3/5 | 0/10 | 2/5 | 0.86 |
| ollama: granite3.3:8b | 42.9s | 56.7s | 4/5 | 0/10 | 1/5 | 0.89 |
| ollama: qwen3:4b (thinking off) | 92.7s | 109.0s | 5/5 | 10/10 | 0/5 | N/A |
| ollama: gemma3:4b | 22.9s | 28.1s | 2/5 | 0/10 | 3/5 | 0.90 |
| ollama: llama3.2:3b | 15.7s | 21.0s | 2/5 | 1/10 | 3/5 | 0.80 |

qwen3:4b refused every answerable question. Its reasoning tokens likely
exhaust `OLLAMA_MAX_OUTPUT_TOKENS` before it produces the structured answer,
so the retry loop runs out and falls back to refusal.

## Conclusion

The generator stays on the hosted `openai`/`gpt-4o-mini` config. No tested
Ollama model matches it on latency and grounding together. Every faster local
model (`granite3.3:2b`, `gemma3:4b`, `llama3.2:3b`) grounds worse, with 2 or 3
hallucinated near-misses instead of 1. The one local model that grounds as
well as `llama3.1:8b` (`granite3.3:8b`) is slower, not faster. `llama3.1:8b`
remains the best Ollama option, at roughly 2.5x the hosted reference's
latency.

## Reproducing

```bash
GENERATOR__PROVIDER=ollama GENERATOR__MODEL=<model> GENERATOR__BASE_URL=http://ollama:11434 \
  podman compose run app evaluate --questions evals/model_comparison_questions.yaml
```

Unload any other resident model first, so it does not compete for CPU with
the run:

```bash
podman compose exec ollama ollama stop <other-model>
```
