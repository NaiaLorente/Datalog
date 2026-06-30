# Contributing to datalog

Thank you for your interest in contributing! datalog is a small, focused tool and contributions of all sizes are welcome — from typo fixes to new features.

## Getting started

1. **Fork and clone** the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/datalog.git
   cd datalog
   ```

2. **Install in development mode** (editable install, no virtual environment required but recommended):
   ```bash
   pip install -e .
   ```

3. **Run the smoke test** to confirm everything works:
   ```bash
   python -c "
   import datalog
   with datalog.run('contrib-test', tags=['test']):
       datalog.log(accuracy=0.99, loss=0.01)
       datalog.note('Smoke test from contributor setup')
   runs = datalog.get_runs()
   assert len(runs) >= 1
   assert runs[0]['name'] == 'contrib-test'
   assert 'test' in runs[0]['tags']
   print('Setup verified. Ready to contribute!')
   "
   ```

## Project layout

```
datalog/
  core.py              # SQLite backend — run(), log(), note(), tag(), get_runs(), ...
  cli.py               # Click CLI commands
  dashboard/
    server.py          # Flask API routes
    templates/
      index.html       # Single-page dashboard (vanilla JS + Chart.js)
pyproject.toml
README.md
CONTRIBUTING.md
```

## Making changes

- **New core features** go in `datalog/core.py`.
- **New CLI commands** go in `datalog/cli.py` (use Click, follow existing patterns).
- **Dashboard changes** go in `datalog/dashboard/templates/index.html` — it's a self-contained SPA, no build step needed.
- **New API endpoints** go in `datalog/dashboard/server.py`.

Please keep external dependencies minimal. The existing stack (Flask, Click, Rich) covers almost everything needed.

## Submitting a pull request

1. Create a branch: `git checkout -b my-feature`
2. Make your changes, run the smoke test
3. Write a clear PR description explaining what you changed and why
4. Open the PR against the `main` branch

All PRs are reviewed promptly. If you're unsure whether a change fits the project's scope, open an issue first to discuss.

## Code style

- Python: follow the existing style (no strict formatter enforced yet)
- Keep functions small and focused
- Add a docstring to any new public function

## Questions?

Open an [issue](https://github.com/NaiaLorente/datalog/issues) — no question is too small.
