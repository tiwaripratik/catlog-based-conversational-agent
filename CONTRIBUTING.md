# Contributing

Thank you for your interest in contributing! Please follow these steps to make contributions smooth and consistent.

- Fork the repository and create a topic branch: `git checkout -b feat/short-description`.
- Keep changes focused and small; open a PR against `main` when ready.
- Run tests locally before opening a PR: `pytest -q`.
- Run formatting and linters via `pre-commit` (recommended):

```bash
pip install -r requirements.txt
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

- Use clear commit messages and reference issues when applicable.

If you're unsure where to start, check open issues or open a discussion.
