# Contributing

Thanks for your interest in contributing to Super Biz Agent.

## Getting Started

```bash
git clone <repo-url>
cd super-biz-agent
uv sync
cp .env.example .env
# Edit .env with your API keys
```

## Development

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 9900 --reload
```

## Running Tests

```bash
python -m pytest
```

## Code Style

This project uses Ruff for linting and formatting:

```bash
python -m ruff check .
python -m ruff format .
```

## Commit Guidelines

- Use present tense imperative ("Add feature", "Fix bug")
- Keep commits focused and atomic

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all tests pass: `python -m pytest`
4. Run lint: `python -m ruff check .`
5. Submit a PR with a clear description

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
