# Security policy

ask-the-docs is built to run on your own machine. The API has no
authentication, so do not expose it to a network you do not trust. See
[dev-docs/GUARDRAILS.md](dev-docs/GUARDRAILS.md) for what the project checks
today and what it does not.

## Report a vulnerability

ask-the-docs is a demonstration project and has no private reporting channel
or response time. Open a GitHub issue to report a problem. Do not include
secrets, credentials, or personal data in it. Include the steps to reproduce,
the commit you tested, and what an attacker gains.

## Supported versions

Only the latest commit on `main` is supported. There are no tagged releases
yet.

## What counts as a security issue

- A way to bypass a check listed in [GUARDRAILS.md](dev-docs/GUARDRAILS.md):
  the SSRF and path checks on ingestion, the size limits, the citation check.
- A prompt or document that makes the system leak data it should not, or
  follow instructions found inside a document.
- Secrets, personal data, or credentials in the repository, logs, or default
  configuration.
- A default setting that exposes a service beyond the local machine.

## What does not count

- Missing authentication on the API. This is a known limit of the local-use
  design. Run it on localhost, or put your own access control in front of it.
- Hosted LLM providers receiving your queries and retrieved text. That is how
  a hosted provider works. Claude is the default generator and judge, so this
  applies out of the box. Use the Ollama setup in `.env.example` if it matters.
- Model mistakes, such as a wrong or incomplete answer.
