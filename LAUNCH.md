# Launch Kit

Ready-to-post copy for launching datalog. Pick the platform, copy, paste.

---

## Reddit — r/MachineLearning

**Title:**
> I built a zero-config ML experiment tracker that lives entirely on your laptop — no account, no server, no setup

**Body:**
```
I got tired of the MLflow "you need to spin up a tracking server" and W&B
"please enter your API key" dance every time I started a new project.

So I built datalog — a dead-simple experiment tracker that:

- Works the moment you pip install it (no server, no account, no .env)
- Stores everything in SQLite on your machine (~/.datalog/experiments.db)
- Opens a local dashboard at localhost:7000 (Chart.js charts, dark theme)
- Detects training anomalies automatically (NaN, divergence, plateau)
- Generates a publication-ready markdown report you can paste into your thesis
- Suggests next hyperparameters based on your run history

```python
import datalog

datalog.configure_watch()  # auto-detects NaN, diverging loss, plateaus

with datalog.run("resnet-baseline", tags=["v1"]):
    datalog.log(lr=0.001, batch_size=32)
    for epoch in range(50):
        loss, acc = train_epoch(model)
        datalog.log(loss=loss, accuracy=acc)
    datalog.note("First run — default config")
```

Then in terminal:
```
datalog list              # rich table of all runs
datalog compare 1 2 3    # side-by-side comparison, best value in green
datalog best accuracy     # find your top run
datalog suggest accuracy  # "lr < 0.0001 → avg acc 0.923 vs 0.871, try 0.0005"
datalog report            # generates a full markdown report for your thesis
```

It's MIT licensed, works offline, and has zero required cloud dependencies.

GitHub: https://github.com/NaiaLorente/datalog

Would love feedback — especially if there's a framework integration you'd find useful.
```

---

## Reddit — r/Python

**Title:**
> datalog: a pip install + one context manager = full ML experiment tracking, local dashboard, and auto-generated reports

**Body:**
```
Built this after getting annoyed at experiment tracking tools that need a server
or an account before you can log a single number.

pip install + one import:

```python
import datalog

with datalog.run("my-experiment", tags=["baseline"]):
    datalog.log(lr=0.001, accuracy=0.94, loss=0.12)
    datalog.note("First attempt")
```

That's it. Runs are stored in SQLite at ~/.datalog/experiments.db.

Then:
```
datalog ui       # local dashboard at localhost:7000
datalog report   # generates markdown report (paste into your thesis)
datalog suggest accuracy  # statistical hyperparameter recommendations
```

The watch feature is what I'm most proud of — it runs inside every log() call
and prints colored warnings if your loss is diverging or has gone NaN, before
you waste hours of compute finding out at the end.

MIT, zero cloud dependencies.
https://github.com/NaiaLorente/datalog
```

---

## Hacker News — Show HN

**Title:**
> Show HN: datalog – local ML experiment tracker, no server, no account, no config

**Body:**
```
I built datalog after hitting the same wall repeatedly: I want to track ML
experiments, but MLflow needs a running server and W&B needs an account and
API key. For laptop-based research, that's too much friction.

datalog is a single Python import that:
- Stores runs in SQLite at ~/.datalog/experiments.db
- Serves a local Flask/Chart.js dashboard at localhost:7000
- Logs your git commit hash automatically
- Detects NaN, diverging loss, and plateaued metrics during training
- Generates a markdown report for papers/theses (works with --ai for Ollama)
- Suggests next hyperparameter values based on correlation analysis

It's MIT licensed, has no cloud dependencies, and works fully offline.

https://github.com/NaiaLorente/datalog
```

---

## Twitter / X thread

**Tweet 1:**
```
I built datalog — a zero-config ML experiment tracker that lives entirely on your laptop.

No W&B account. No MLflow server. No .env file.

Just:
pip install .
import datalog

🧵 Here's what it does:
```

**Tweet 2:**
```
One context manager wraps your training loop:

with datalog.run("resnet-v2", tags=["baseline"]):
    datalog.log(lr=0.001)
    for epoch in range(50):
        datalog.log(accuracy=acc, loss=loss)

Everything saved to SQLite. No config needed.
```

**Tweet 3:**
```
datalog watch detects problems while your model trains:

⚠️ 'loss' has increased for 5 consecutive logs — possible divergence
⚠️ NaN/Inf detected in 'grad_norm' — consider stopping
ℹ️ 'accuracy' has plateaued for 15 logs — adjust your lr

Runs automatically inside every log() call.
```

**Tweet 4:**
```
datalog suggest accuracy tells you what's actually working:

lr < 0.0001 → avg accuracy 0.923 vs 0.871 (7.4% difference)
→ Try lr = 0.00005

No ML library. Pure statistics on your run history.
```

**Tweet 5:**
```
datalog report generates a full markdown experiment summary:

🥇 best-so-far — accuracy 0.9453 (↑ 12.8% vs baseline)
🥈 small-batch  — accuracy 0.9149
🥉 lower-lr     — accuracy 0.9132

Paste it into your thesis. Share it with your advisor.
```

**Tweet 6:**
```
MIT licensed. Works offline. 
No account. No server. No limits.

github.com/NaiaLorente/datalog

"The experiment tracker for people who hate setting up experiment trackers."
```

---

## LinkedIn post

```
I got tired of the setup overhead every time I started a new ML project.

MLflow needs a running server. Weights & Biases needs an account and API key.
For students and researchers working on a laptop, that's too much friction.

So I built datalog — a local-first ML experiment tracker that works the moment
you install it.

✅ Zero config (no server, no cloud, no environment variables)
✅ Local dashboard at localhost:7000 (Chart.js charts, dark theme)
✅ Live anomaly detection — warns you if your loss is diverging or NaN before
   you waste hours of compute time
✅ Generates a markdown report you can paste into your thesis or paper
✅ Suggests next hyperparameter values based on your run history
✅ MIT licensed, completely free, works offline

It's the experiment tracker I wish existed when I was a student.

→ github.com/NaiaLorente/datalog

#MachineLearning #DataScience #MLOps #Python #OpenSource
```

---

## Dev.to / Hashnode article title suggestions

- "Stop paying for experiment tracking — I built a free, zero-config alternative"
- "The experiment tracker that works before you finish your coffee"
- "How I track ML experiments without W&B, MLflow, or any cloud service"
- "datalog: dead-simple ML experiment tracking for students and researchers"

---

## GitHub topics to add manually

Go to your repo → About (gear icon) → Topics, and add:

```
machine-learning  experiment-tracking  mlops  data-science
python  sqlite  local-first  hyperparameter-tuning
pytorch  scikit-learn  huggingface  keras  jupyter
model-monitoring  research-tools  thesis  kaggle
```

These are the exact terms people search on GitHub. Adding them takes 60 seconds
and can 10x your organic reach.
