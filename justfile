lint:
  uv run ruff check --fix && uv run ruff format

lint-check:
  uv run ruff check && uv run ruff format --check

type:
  uv run pyrefly check

test:
  uv run python -m pytest tests --doctest-modules src
