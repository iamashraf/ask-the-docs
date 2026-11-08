# Generator model comparison: Ollama vs. hosted

The generator provider/model is config-only (`GENERATOR__PROVIDER`,
`GENERATOR__MODEL`, `GENERATOR__BASE_URL`; see `config.py`), so switching
between Ollama and a hosted model needs no code change. This compares the
local Ollama models against the hosted reference on latency, answer
accuracy, and grounding discipline, so the default is picked by
measurement rather than assumption. A fast model that invents answers is
worse than a slow one that doesn't.

## Method

- **Question set**: `evals/model_comparison_questions.yaml`: 10 questions
  the seeded corpus answers, and 5 in-domain near-misses (same topic as a
  seeded paper, but a specific fact or extrapolation the paper doesn't
  state), kept separate from `golden_questions.yaml`'s routing/regression
  cases.
- **Judge**: fixed at `gpt-4o` for every candidate, so scores are
  comparable across runs.
- **Isolation**: each Ollama model is run with every other model unloaded
  first (`ollama stop`). `OLLAMA_KEEP_ALIVE` otherwise leaves a model
  resident after use, and running candidates back-to-back without
  unloading leaves several models loaded at once, oversubscribing the
  host's CPU and corrupting later candidates' latency numbers.
- **Guardrail behavior**: a model's deliberate refusal (`grounded: false`
  on the structured path, or the exact refusal message on the free-text
  path) is accepted on the first attempt instead of being retried
  (`answer.py`). Retrying a correct refusal just re-samples the same
  nondeterministic call, and can turn it into a wrong, confidently-cited
  answer on the second attempt. This matters for reading the near-miss
  columns below honestly.

Metrics per candidate, all over the 15-question set:

| Metric | Meaning |
|---|---|
| mean / p95 latency | End-to-end `answer()` time: routing + retrieval + generation |
| refusals correct | Of the 5 near-miss questions, how many correctly refused |
| over-refused | Of the 10 answerable questions, how many wrongly refused |
| hallucinated | Of the 5 near-miss questions, how many got a wrong, cited answer instead of refusing |
| judge accuracy | Mean accuracy score (0–1) from the judge, over genuinely-answered questions only |

## Environment

Apple M5 Pro, Ollama running inside a podman container (CPU-only: macOS
container runtimes have no path to the GPU, and the podman machine itself is
allocated 8 vCPU / 32 GB). These numbers are specific to that setup, not a
general claim about any of these models' speed.

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

qwen3:4b refused every answerable question; its reasoning tokens likely
exhaust `OLLAMA_MAX_OUTPUT_TOKENS` before producing the actual structured
answer, so the retry loop exhausts and falls back to refusal.

## Conclusion

The generator stays on the hosted `openai`/`gpt-4o-mini` config. None of
the tested Ollama models match it on the combination of latency and
grounding discipline: every faster local model (`granite3.3:2b`,
`gemma3:4b`, `llama3.2:3b`) trades away grounding (2 or 3 hallucinated
near-misses instead of 1), and the one local model with grounding as good
as `llama3.1:8b` (`granite3.3:8b`) is slower, not faster. `llama3.1:8b`
remains the best Ollama option, at roughly 2.5x the hosted reference's
latency.

## Reproducing

```bash
GENERATOR__PROVIDER=ollama GENERATOR__MODEL=<model> GENERATOR__BASE_URL=http://ollama:11434 \
  podman compose run app evaluate --questions evals/model_comparison_questions.yaml
```

Unload any other resident model first so the run isn't contending for
CPU with it:

```bash
podman compose exec ollama ollama stop <other-model>
```
