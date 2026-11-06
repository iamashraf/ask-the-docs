# Contributing

Thanks for looking. This is a learning project built in small steps, so small,
focused changes are easiest to review.

## Set up

```bash
uv sync          # installs the locked dependencies
uv run pytest    # runs the full suite; no containers or API keys needed
```

Run the whole suite before you open a pull request, not only the new tests.

## How to make a change

1. Write a failing test for the behavior you want.
2. Confirm it fails for the right reason.
3. Make it pass with the smallest change.
4. Commit. One change per commit and per pull request.

Conventions:

- Unit tests mock the network, SDKs, and model loading. The fast suite must
  stay fast and work offline.
- If you import a package directly, add it with `uv add`, even if another
  package already brings it in.
- Do not add `torch`-family packages without giving them the same CPU-only
  index override that `torch` has in `pyproject.toml`.
- Docstrings are one line and say why, not what. Skip them when the name
  already says it.

See [dev-docs/ARCHITECTURE.md](dev-docs/ARCHITECTURE.md) for how the pieces fit
together.

## Guardrails and security tests

The checks listed in [dev-docs/GUARDRAILS.md](dev-docs/GUARDRAILS.md) are
covered by ordinary tests. If your change touches one of them, keep its tests
passing, or explain in the pull request why a test had to change.

- Changes to input limits, prompt building, the citation check, or the path
  and URL checks need a test for the new behavior and for its boundary.
- If you find a way around a guardrail, add the case as a failing test with
  the fix. For anything serious, report it privately first. See
  [SECURITY.md](SECURITY.md).
- If you add a guardrail, add a row to GUARDRAILS.md. Say what it does, how to
  configure it, and whether it fails closed or open. List only what the code
  does today.

## Pull requests

Describe what changed and why, and how you checked it. Mention anything you
did not check. Do not include secrets, personal data, or private hostnames in
code, tests, docs, or logs you paste.
