# mltrackr

> Track ML experiments in 2 lines of code. No server. No account. No config.

[![CI](https://github.com/NaiaLorente/Datalog/actions/workflows/ci.yml/badge.svg)](https://github.com/NaiaLorente/Datalog/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/badge/PyPI-mltrackr-blue)](https://pypi.org/project/mltrackr/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://pypi.org/project/mltrackr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/NaiaLorente/Datalog?style=social)](https://github.com/NaiaLorente/Datalog)

You're running a training loop. You want to know which hyperparameters worked best. You don't want to:

- Set up a tracking server
- Create an account on any service
- Write to a cloud API
- Configure environment variables
- Install 47 dependencies

**mltrackr is the answer.** Install it, wrap your loop, open a beautiful local dashboard. Done.

![mltrackr dashboard](assets/demo.gif)

---

## Quickstart (5 steps)

**1. Install**
```bash
pip install mltrackr
```
> This installs the `mltrackr` command and `import mltrackr` Python package.

**2. Generate a ready-to-run example**
```bash
python -m mltrackr init --framework plain -o demo.py
```
> On most systems `mltrackr init` works directly. If not, use `python -m mltrackr` instead.

**3. Run the demo (creates 6 fake training runs)**
```bash
python demo.py
```

**4. Inspect results in the terminal**
```bash
python -m mltrackr list
python -m mltrackr best accuracy
python -m mltrackr suggest accuracy
```

**5. Open the visual dashboard**
```bash
python -m mltrackr ui
```
Then open **http://localhost:7000** in your browser. Press `Ctrl+C` to stop.

---

## Your first real experiment

```python
import mltrackr

with mltrackr.run("resnet-baseline", tags=["cv", "baseline"]):
    mltrackr.log(lr=1e-3, batch_size=64, optimizer="adam")

    for epoch in range(50):
        loss, acc = train_one_epoch(model, dataloader)
        mltrackr.log(loss=loss, accuracy=acc, epoch=epoch)

    mltrackr.note("Solid baseline - try lr=5e-4 next")
```

```bash
# If 'mltrackr' works directly on your system:
mltrackr ui
mltrackr list
mltrackr best accuracy
mltrackr suggest accuracy
mltrackr report

# If not (e.g. Windows), use:
python -m mltrackr ui
python -m mltrackr list
python -m mltrackr best accuracy
```

Everything is saved locally in `~/.mltrackr/experiments.db`. A single SQLite file. Copy it, back it up, open it in any SQLite browser.

> **Got good results?** Run `mltrackr share accuracy` to generate a ready-to-post Twitter/X or Hacker News summary. If mltrackr saved you time, a ⭐ on GitHub goes a long way!

---

## Why mltrackr?

**The real problem:** you're hacking on a model, you want to log some metrics, but setting up MLflow takes 15 minutes and W&B wants you to create an account and send your data to the cloud. So you end up writing metrics to a text file or just... not tracking anything. Then you forget which hyperparameters worked. Then you run the same failed experiment again.

**mltrackr is the experiment tracker that's actually available when you need it.**

| | **mltrackr** | **MLflow** | **Weights & Biases** |
|---|---|---|---|
| Setup time | **5 seconds** | ~15 minutes | ~5 minutes |
| Requires account | ❌ No | ❌ No | ✅ Yes |
| Requires running server | ❌ No | ✅ Yes | ❌ No (cloud) |
| Works offline | ✅ Always | ⚠️ Partial | ❌ No |
| Data stays local | ✅ Always | ✅ Yes | ❌ No |
| Live anomaly detection | ✅ Built-in | ❌ No | ⚠️ Paid |
| Hyperparameter suggestions | ✅ Built-in | ❌ No | ⚠️ Paid |
| Auto-generated reports | ✅ Built-in | ❌ No | ❌ No |
| Free forever | ✅ MIT | ✅ Apache | ⚠️ Usage limits |

---

## Features you'll actually use

### ✅ Zero-friction tracking
Wrap any loop. Log any value. Works with every framework.
```python
import mltrackr

with mltrackr.run("gpt-finetune", tags=["nlp", "v3"]):
    mltrackr.log(lr=2e-5, epochs=3, model="gpt2")
    for step, batch in enumerate(dataloader):
        loss = model.train_step(batch)
        mltrackr.log(loss=loss.item(), step=step)
```

### ✅ Beautiful live dashboard
```bash
mltrackr ui
```
Opens at `http://localhost:7000` — a fast, dark-mode single-page app with:
- Searchable run list with **inline sparkline charts** in the sidebar
- **Trend indicators** (↑ ↓) showing whether each metric is improving
- **Side-by-side comparison** of any runs you select (best value highlighted)
- **Auto-generated time-series charts** with gradient fills
- **Metric progress bars** showing where the latest value sits in its historical range
- Global statistics view — success rate, most-logged metrics, run timeline
- Auto-refresh every 5 seconds — open while training, watch it update

### ✅ Live anomaly detection — catch bad runs early
```python
mltrackr.configure_watch(nan_check=True, divergence_window=5, plateau_window=15)

with mltrackr.run("training"):
    for epoch in range(100):
        mltrackr.log(loss=compute_loss())
        # Automatically warns if: loss → NaN, loss diverges for 5 epochs,
        # loss plateaus for 15 epochs (and suggests adjusting LR)
```
Stop wasting GPU hours on runs that are already failing.

### ✅ Hyperparameter suggestions
```bash
mltrackr suggest accuracy
```
Analyzes your run history and tells you which hyperparameter values are statistically correlated with better results. No black box — plain English insights like:
```
Best config: lr=0.001 → avg accuracy 0.943 (vs 0.871 for other values, +8.2%)
Next experiment: try batch_size=128 — larger batches correlated with +5.1% accuracy
```

### ✅ Auto-generated experiment reports
```bash
mltrackr report --output results.md
```
Generates a thesis-ready markdown report with:
- Summary statistics (total runs, completion rate, best configurations)
- Chronological experiment timeline
- Key findings (computed automatically)
- Notes from all your runs
- Optional AI narrative: `mltrackr report --ai` (uses local Ollama, no API keys)

### ✅ Generate a ready-to-run example
```bash
mltrackr init                           # plain Python example
mltrackr init --framework pytorch       # PyTorch training loop
mltrackr init --framework sklearn       # scikit-learn grid search
mltrackr init --framework keras         # Keras callback
```
Generates a complete working script you can run immediately.

### ✅ Works with every framework

| Framework | How |
|-----------|-----|
| **PyTorch** | `mltrackr.log(loss=loss.item(), acc=acc)` inside the training loop |
| **scikit-learn** | `mltrackr.log(**params, cv_score=score)` in your hyperparam loop |
| **Keras / TF** | One-file `TrainlogCallback` for `model.fit()` |
| **HuggingFace** | Custom `TrainerCallback` — see `examples/huggingface_example.py` |
| **XGBoost / LightGBM** | Log in the eval callback |
| **JAX / Flax** | Log at end of each training step |
| **Plain Python** | Anything that produces a number |

---

## Full API reference

### Python API

```python
import mltrackr

# ── Tracking ──────────────────────────────────────────────────────────────────
with mltrackr.run("name", tags=["tag1", "tag2"]) as run_id:
    mltrackr.log(accuracy=0.95, loss=0.05)          # log any key-value pairs
    mltrackr.note("Cosine LR schedule helped a lot") # attach plain-text notes

mltrackr.tag(run_id, "production")       # add tags after the fact
mltrackr.tag("experiment-name", "best")  # also works by name

# ── Querying ──────────────────────────────────────────────────────────────────
runs = mltrackr.get_runs()                           # all runs, newest first
best = mltrackr.get_best_run("accuracy")             # highest final value
best_low = mltrackr.get_best_run("loss", mode="min") # lowest final value
cmp = mltrackr.compare_runs(1, 2, 3)                 # list of run dicts

# ── Anomaly detection ─────────────────────────────────────────────────────────
mltrackr.configure_watch(
    nan_check=True,           # warn on NaN/Inf values
    divergence_window=5,      # warn if metric diverges for N steps
    plateau_window=15,        # warn if metric plateaus for N steps
    enabled=True,
)

# Or temporarily with a context manager:
with mltrackr.watch(divergence_window=3):
    # stricter watch for this block
    mltrackr.log(loss=0.5)

# ── Export & analysis ─────────────────────────────────────────────────────────
mltrackr.export_csv("results.csv")
mltrackr.export_json("results.json")
mltrackr.generate_report("report.md", use_ollama=False)
suggestions = mltrackr.suggest("accuracy", mode="max", top_n=3)
mltrackr.clear_all()  # deletes everything (irreversible)

# ── Config ────────────────────────────────────────────────────────────────────
mltrackr.configure(verbose=False)  # suppress auto-summary panels after each run
```

### CLI reference

```bash
# Dashboard
mltrackr ui                             # open at localhost:7000
mltrackr ui --port 8080 --no-browser    # custom port, no auto-open

# Inspect runs
mltrackr list                           # rich table, newest first
mltrackr list --limit 50
mltrackr compare 1 2 3                  # side-by-side metric comparison
mltrackr best accuracy                  # best run for a metric
mltrackr best loss --mode min

# Annotate
mltrackr tag 42 production tuned        # add tags to run #42
mltrackr note 42 "Try cosine annealing" # add note to run #42

# Analyse
mltrackr stats                          # aggregate statistics
mltrackr suggest accuracy               # hyperparameter recommendations
mltrackr suggest loss --mode min --top 5

# Generate
mltrackr report                         # write report.md
mltrackr report -o results.md --ai      # with Ollama AI narrative
mltrackr init --framework pytorch       # generate example script

# Export / clean
mltrackr export --format csv -o data.csv
mltrackr export --format json -o data.json
mltrackr clear                          # delete all (asks confirmation)

# Share
mltrackr share accuracy                 # generate Twitter/X + HN ready post
mltrackr share loss --mode min          # for metrics where lower is better
```

---

## How it works

- **SQLite** — `~/.mltrackr/experiments.db`. One file. No server. Inspect it with any SQLite browser. Back it up with `cp`.
- **Flask** — the dashboard is a local Flask server. Vanilla JS, Chart.js, zero npm, zero build step.
- **Thread-local state** — each training job in its own thread gets an isolated run context. Concurrent experiments just work.
- **Git-aware** — captures the current commit hash via `git rev-parse HEAD`. Silently skipped outside a git repo.
- **Watch hooks** — anomaly detection runs inside every `log()` call. Zero external services, works offline.

---

## Quickstart with examples

```bash
mltrackr init --framework pytorch -o train.py
python train.py
mltrackr ui
```

That's the whole flow. Five commands. Zero config.

---

## Roadmap

**Done ✅**
- Live anomaly detection (`configure_watch`)
- Auto-generated experiment reports (`mltrackr report`, Ollama support)
- Hyperparameter suggestions (`mltrackr suggest`)
- Quick-start example generator (`mltrackr init`)
- Sparkline charts in sidebar with trend indicators
- Metric progress bars and trend arrows in detail view
- Framework examples: PyTorch, scikit-learn, Keras, HuggingFace

**Coming up**
- [ ] `mltrackr.log_artifact("model.pt")` — save file paths alongside metrics
- [ ] Native PyTorch `TrainlogCallback` (pip-installable plugin)
- [ ] VS Code extension — inline run summary on hover
- [ ] `mltrackr serve` — shareable read-only dashboard URL (ngrok/localtunnel)
- [ ] Team sync via shared git-tracked SQLite
- [ ] Slack / Discord webhook on run completion

Have an idea? [Open a feature request](https://github.com/NaiaLorente/Datalog/issues/new) — or submit a PR.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). TL;DR: `pip install -e .`, make your change, open a PR.

All contributions welcome — typos, docs, features, bug fixes.

---

## License

MIT — use it however you want, forever.
