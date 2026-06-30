import sys
import click
from pathlib import Path


@click.group()
def cli():
    """mltrackr — the experiment tracker for people who hate setting up experiment trackers."""
    pass


@cli.command()
@click.option("--port", default=7000, show_default=True, help="Port to run the dashboard on.")
@click.option("--no-browser", is_flag=True, default=False, help="Don't open a browser window.")
def ui(port, no_browser):
    """Start the web dashboard."""
    import webbrowser
    import threading
    from mltrackr.dashboard.server import create_app

    app = create_app()

    url = f"http://127.0.0.1:{port}"
    click.echo(f"Starting mltrackr dashboard at {url}")

    if not no_browser:
        def _open():
            import time
            time.sleep(0.8)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    app.run(host="127.0.0.1", port=port, debug=False)


@cli.command(name="list")
@click.option("--limit", default=20, show_default=True, help="Number of runs to show.")
def list_runs(limit):
    """List recent experiment runs."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from mltrackr.core import get_runs, _compute_duration

    console = Console()
    runs = get_runs()[:limit]

    if not runs:
        console.print("[dim]No runs found. Start tracking with `with mltrackr.run('my-exp'):`[/dim]")
        return

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        title="[bold]Experiment Runs[/bold]",
        title_style="bold white",
    )
    table.add_column("ID", style="dim", width=5, justify="right")
    table.add_column("Name", style="bold white", min_width=20)
    table.add_column("Tags", style="cyan", min_width=14)
    table.add_column("Git Hash", style="yellow", width=9)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Status", justify="center", width=11)
    table.add_column("Metrics", justify="right", width=8)

    status_style = {
        "completed": "[bold green]completed[/bold green]",
        "failed": "[bold red]failed[/bold red]",
        "running": "[bold yellow]running[/bold yellow]",
    }

    for r in runs:
        duration = _compute_duration(r["start_time"], r["end_time"])
        if duration is not None:
            if duration < 60:
                dur_str = f"{duration:.1f}s"
            elif duration < 3600:
                dur_str = f"{duration / 60:.1f}m"
            else:
                dur_str = f"{duration / 3600:.1f}h"
        else:
            dur_str = "[dim]—[/dim]"

        git_hash = (r["git_commit"] or "")[:7] or "[dim]none[/dim]"
        status_rich = status_style.get(r["status"], r["status"])
        metric_count = sum(len(v) for v in r["metrics"].values())
        tags_str = " ".join(f"[cyan]{t}[/cyan]" for t in r.get("tags", [])) or "[dim]—[/dim]"

        table.add_row(
            str(r["id"]),
            r["name"],
            tags_str,
            git_hash,
            dur_str,
            status_rich,
            str(metric_count),
        )

    console.print()
    console.print(table)
    console.print()


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["csv", "json"]), default="csv", show_default=True)
@click.option("--output", "-o", default=None, help="Output file path.")
def export(fmt, output):
    """Export experiment data to CSV or JSON."""
    from mltrackr.core import export_csv, export_json

    if output is None:
        output = f"mltrackr_export.{fmt}"

    output_path = Path(output)

    if fmt == "csv":
        export_csv(str(output_path))
    else:
        export_json(str(output_path))

    click.echo(f"Exported to {output_path.resolve()}")


@cli.command()
def clear():
    """Clear all experiment data (irreversible)."""
    from mltrackr.core import clear_all

    click.confirm(
        "This will permanently delete ALL experiment runs and metrics. Continue?",
        abort=True,
    )
    clear_all()
    click.echo("All experiment data cleared.")


@cli.command()
@click.argument("run_ids", nargs=-1, type=int, required=True)
def compare(run_ids):
    """Compare multiple runs side-by-side in a rich table."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from mltrackr.core import compare_runs

    console = Console()

    if len(run_ids) < 2:
        console.print("[red]Please provide at least 2 run IDs to compare.[/red]")
        raise SystemExit(1)

    runs = compare_runs(*run_ids)

    if not runs:
        console.print("[red]No runs found for the given IDs.[/red]")
        raise SystemExit(1)

    # Collect all metric keys across all runs
    all_keys = []
    seen = set()
    for r in runs:
        for k in r["metrics"]:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        title="[bold]Run Comparison[/bold]",
        title_style="bold white",
    )
    table.add_column("Metric", style="bold", min_width=18)
    for r in runs:
        tags = " ".join(r.get("tags", []))
        header = f"#{r['id']} {r['name']}"
        if tags:
            header += f"\n[dim]{tags}[/dim]"
        table.add_column(header, justify="right", min_width=14)

    for key in all_keys:
        # Gather final values
        vals = []
        for r in runs:
            entries = r["metrics"].get(key, [])
            if entries:
                v = entries[-1]["value"]
                vals.append(v if isinstance(v, (int, float)) else None)
            else:
                vals.append(None)

        # Find best numeric value for highlighting (min for loss-type, max otherwise)
        from mltrackr.core import _metric_direction
        numeric_vals = [v for v in vals if v is not None]
        if numeric_vals:
            best_val = min(numeric_vals) if _metric_direction(key) == "lower" else max(numeric_vals)
        else:
            best_val = None

        cells = []
        for v in vals:
            if v is None:
                cells.append("[dim]—[/dim]")
            else:
                formatted = f"{v:.6g}" if isinstance(v, float) else str(v)
                if isinstance(v, (int, float)) and v == best_val:
                    cells.append(f"[bold green]{formatted}[/bold green]")
                else:
                    cells.append(formatted)

        table.add_row(key, *cells)

    console.print()
    console.print(table)
    console.print()


@cli.command()
@click.argument("metric")
@click.option("--mode", type=click.Choice(["max", "min"]), default="max", show_default=True)
def best(metric, mode):
    """Show the best run for a given metric."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from mltrackr.core import get_best_run, _compute_duration

    console = Console()
    run = get_best_run(metric, mode=mode)

    if run is None:
        console.print(f"[dim]No runs found with metric '{metric}'.[/dim]")
        return

    entries = run["metrics"].get(metric, [])
    final_val = entries[-1]["value"] if entries else "—"

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", border_style="dim",
                  title=f"[bold]Best run for '{metric}' ({mode})[/bold]", title_style="bold white")
    table.add_column("Field", style="dim", min_width=14)
    table.add_column("Value", style="bold white", min_width=24)

    duration = _compute_duration(run["start_time"], run["end_time"])
    dur_str = f"{duration:.1f}s" if duration is not None else "—"

    table.add_row("ID", str(run["id"]))
    table.add_row("Name", run["name"])
    table.add_row("Tags", " ".join(run.get("tags", [])) or "—")
    table.add_row("Status", run["status"])
    table.add_row("Duration", dur_str)
    table.add_row(f"{metric} (final)", f"[bold green]{final_val}[/bold green]")
    table.add_row("Git commit", (run["git_commit"] or "")[:12] or "—")

    console.print()
    console.print(table)
    console.print()


@cli.command(name="tag")
@click.argument("run_id", type=int)
@click.argument("tags", nargs=-1, required=True)
def tag_run(run_id, tags):
    """Add tags to a run."""
    from mltrackr.core import tag as core_tag

    core_tag(run_id, *tags)
    click.echo(f"Tagged run #{run_id} with: {', '.join(tags)}")


@cli.command(name="note")
@click.argument("run_id", type=int)
@click.argument("text")
def note_run(run_id, text):
    """Add a note to a run (outside a run context)."""
    from mltrackr.core import _connect
    import json

    with _connect() as conn:
        row = conn.execute("SELECT id, notes FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            click.echo(f"Run #{run_id} not found.", err=True)
            raise SystemExit(1)
        existing = row["notes"] if row["notes"] else ""
        separator = "\n\n" if existing else ""
        new_notes = existing + separator + text
        conn.execute("UPDATE runs SET notes = ? WHERE id = ?", (new_notes, run_id))
        conn.commit()

    click.echo(f"Note added to run #{run_id}.")


@cli.command()
def stats():
    """Show summary statistics across all runs."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from collections import Counter
    from mltrackr.core import get_runs

    console = Console()
    runs = get_runs()

    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    total = len(runs)
    completed = sum(1 for r in runs if r["status"] == "completed")
    failed = sum(1 for r in runs if r["status"] == "failed")
    running = sum(1 for r in runs if r["status"] == "running")
    success_rate = (completed / total * 100) if total > 0 else 0.0

    metric_counter = Counter()
    for r in runs:
        for key, entries in r["metrics"].items():
            metric_counter[key] += len(entries)

    top_metrics = metric_counter.most_common(5)

    # Summary table
    summary = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", border_style="dim",
                    title="[bold]Experiment Statistics[/bold]", title_style="bold white")
    summary.add_column("Metric", style="dim", min_width=18)
    summary.add_column("Value", style="bold white", min_width=14, justify="right")

    summary.add_row("Total runs", str(total))
    summary.add_row("Completed", f"[bold green]{completed}[/bold green]")
    summary.add_row("Failed", f"[bold red]{failed}[/bold red]")
    summary.add_row("Running", f"[bold yellow]{running}[/bold yellow]")
    summary.add_row("Success rate", f"[bold cyan]{success_rate:.1f}%[/bold cyan]")

    console.print()
    console.print(summary)

    if top_metrics:
        metrics_table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan",
                              border_style="dim", title="[bold]Top 5 Metrics[/bold]", title_style="bold white")
        metrics_table.add_column("Metric key", style="cyan", min_width=20)
        metrics_table.add_column("Data points", justify="right", min_width=12)

        for key, count in top_metrics:
            metrics_table.add_row(key, str(count))

        console.print(metrics_table)

    console.print()


@cli.command()
@click.argument("metric")
@click.option("--mode", type=click.Choice(["max", "min"]), default="max", show_default=True)
@click.option("--top", "top_n", default=3, show_default=True, help="Max top suggestions to show")
def suggest(metric, mode, top_n):
    """Analyse your runs and suggest next hyperparameter values to try."""
    from rich.console import Console
    from rich.panel import Panel
    from mltrackr.core import suggest as core_suggest

    console = Console()
    result = core_suggest(metric, mode=mode, top_n=top_n)

    lines = []

    br = result.get("best_run")
    if br:
        best_val = result.get("best_value")
        val_str = f"{best_val:.4g}" if isinstance(best_val, float) else str(best_val)
        lines.append(f"[bold green]Best run:[/bold green] #{br['id']} [cyan]{br['name']}[/cyan]  ->  {metric} = [bold]{val_str}[/bold]")
        lines.append("")

    for insight in result["insights"]:
        if insight.startswith("Best run"):
            lines.append(f"[bold green]Best config:[/bold green] {insight}")
        elif insight.startswith("Next experiment"):
            lines.append("")
            lines.append(f"[bold magenta]Recommendation:[/bold magenta] {insight}")
        else:
            lines.append(f"[yellow]Insight:[/yellow] {insight}")

    if not result["insights"]:
        lines.append("[dim]No insights available yet.[/dim]")

    console.print()
    console.print(Panel(
        "\n".join(lines),
        title=f"[bold]Hyperparameter Suggestions for '{metric}' ({mode})[/bold]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


@cli.command()
@click.option("--output", "-o", default="report.md", show_default=True, help="Output file path")
@click.option("--ai", "use_ollama", is_flag=True, help="Add AI narrative via Ollama (needs Ollama running)")
@click.option("--model", default="llama3", show_default=True, help="Ollama model to use with --ai")
def report(output, use_ollama, model):
    """Generate a markdown experiment report from all your runs."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from mltrackr.core import generate_report

    console = Console()

    if use_ollama:
        console.print("[dim]Connecting to Ollama for AI narrative... (falls back to template if unavailable)[/dim]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console, transient=True) as progress:
        progress.add_task("Generating report...", total=None)
        md = generate_report(output_path=output, use_ollama=use_ollama, ollama_model=model)

    lines = md.count("\n")
    console.print(f"\n[bold green]✓[/bold green] Report saved to [cyan]{output}[/cyan] ({lines} lines)\n")
    console.print("[dim]Tip: paste it into your thesis, share with your advisor, or upload to GitHub.[/dim]\n")


@cli.command(name="init")
@click.option("--framework", type=click.Choice(["pytorch", "sklearn", "keras", "plain"]), default="plain", show_default=True, help="Framework to generate example for.")
@click.option("--output", "-o", default="train_example.py", show_default=True, help="Output file path.")
def init_example(framework, output):
    """Generate a ready-to-run example training script."""
    from rich.console import Console
    console = Console()

    pytorch_code = '''\
# -*- coding: utf-8 -*-
import mltrackr
import math, random

# Warn you before GPU hours are wasted
mltrackr.configure_watch(nan_check=True, divergence_window=5)

with mltrackr.run("my-first-experiment", tags=["baseline", "pytorch"]):
    mltrackr.log(lr=1e-3, batch_size=64, hidden_dim=256)

    for epoch in range(20):
        # Replace with real loss/accuracy from your model
        loss = 2.0 * math.exp(-0.15 * epoch) + random.uniform(-0.05, 0.05)
        acc  = 1 - loss / 2.5 + random.uniform(-0.02, 0.02)
        mltrackr.log(loss=round(loss, 4), accuracy=round(acc, 4), epoch=epoch)

    mltrackr.note("Baseline run. Try lr=5e-4 or larger batch next.")

print("\\nRun: python -m mltrackr ui")
best = mltrackr.get_best_run("accuracy")
if best:
    print(f"Best accuracy so far: {best['metrics']['accuracy'][-1]['value']:.4f}")
'''

    sklearn_code = '''\
# -*- coding: utf-8 -*-
import mltrackr
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)

configs = [
    {"n_estimators": 50,  "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": None},
]

for cfg in configs:
    name = f"rf-n{cfg['n_estimators']}-d{cfg['max_depth']}"
    with mltrackr.run(name, tags=["sklearn", "random-forest"]):
        mltrackr.log(**cfg)
        clf = RandomForestClassifier(**cfg, random_state=42)
        scores = cross_val_score(clf, X, y, cv=5)
        mltrackr.log(cv_mean=round(scores.mean(), 4), cv_std=round(scores.std(), 4))
        mltrackr.note(f"Cross-val: {scores.mean():.3f} +/- {scores.std():.3f}")

best = mltrackr.get_best_run("cv_mean")
print(f"Best config: {best['name']}  cv_mean={best['metrics']['cv_mean'][-1]['value']:.4f}")
print("Run: python -m mltrackr suggest cv_mean")
'''

    plain_code = '''\
# -*- coding: utf-8 -*-
import mltrackr
import math, random

for lr in [1e-2, 1e-3, 1e-4]:
    for bs in [32, 64]:
        name = f"exp-lr{lr}-bs{bs}"
        with mltrackr.run(name, tags=["sweep", "grid-search"]):
            mltrackr.log(lr=lr, batch_size=bs)
            for step in range(30):
                noise = random.uniform(-0.03, 0.03)
                loss = (1.5 / (1 + lr * 200 * step)) + noise
                acc  = min(0.99, max(0.0, 0.5 + lr * 150 * step / (1 + lr * 50 * step)) + noise)
                mltrackr.log(loss=round(max(0.0, loss), 4), accuracy=round(acc, 4), step=step)
            mltrackr.note(f"lr={lr} bs={bs} - check dashboard for curves")

print("\\nTop 3 results:")
for r in mltrackr.get_runs()[:3]:
    acc = r["metrics"].get("accuracy", [{}])[-1].get("value", "?")
    print(f"  {r['name']:30s}  accuracy={acc}")

print("\\nNext steps:")
print("  python -m mltrackr ui              - open dashboard")
print("  python -m mltrackr best accuracy   - find winner")
print("  python -m mltrackr suggest accuracy - get next config")
'''

    keras_code = '''\
# -*- coding: utf-8 -*-
import mltrackr
import math, random

class MltrackrCallback:
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mltrackr.log(epoch=epoch, **{k: round(v, 4) for k, v in logs.items()})

try:
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Flatten

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    with mltrackr.run("mnist-baseline", tags=["keras", "mnist"]):
        mltrackr.log(lr=1e-3, epochs=5, hidden=128)
        model = Sequential([Flatten(input_shape=(28,28)), Dense(128, activation="relu"), Dense(10, activation="softmax")])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cb = MltrackrCallback()
        for epoch in range(5):
            h = model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), verbose=0)
            cb.on_epoch_end(epoch, h.history)
        mltrackr.note("Adam + 128 hidden units baseline on MNIST")

except ImportError:
    print("keras not found - running simulation instead")
    with mltrackr.run("keras-sim", tags=["keras"]):
        mltrackr.log(lr=1e-3, epochs=10)
        for e in range(10):
            mltrackr.log(epoch=e,
                loss=round(1.2*math.exp(-0.3*e)+random.uniform(-0.05,0.05), 4),
                accuracy=round(0.6+0.04*e+random.uniform(-0.02,0.02), 4))

print("Run: python -m mltrackr ui")
'''

    code_map = {"pytorch": pytorch_code, "sklearn": sklearn_code, "keras": keras_code, "plain": plain_code}
    code = code_map[framework]

    Path(output).write_text(code, encoding="utf-8")
    console.print(f"\n[bold green]✓[/bold green] Created [cyan]{output}[/cyan]\n")
    console.print(f"  [dim]Run it:[/dim]  [bold white]python {output}[/bold white]")
    console.print(f"  [dim]Then:[/dim]    [bold white]mltrackr ui[/bold white]  [dim]# open the dashboard[/dim]")
    console.print(f"           [bold white]mltrackr list[/bold white]  [dim]# see your runs[/dim]")
    console.print(f"           [bold white]mltrackr suggest accuracy[/bold white]  [dim]# get next config to try[/dim]\n")


@cli.command()
def demo():
    """Populate the database with realistic fake experiments so you can explore the dashboard immediately."""
    import math
    import random
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    import mltrackr

    console = Console()
    console.print("\n[bold cyan]mltrackr demo[/bold cyan] — generating sample experiments...\n")
    mltrackr.configure(verbose=False)

    configs = [
        {"name": "transformer-base",    "lr": 3e-4, "batch_size": 32, "hidden_dim": 256, "dropout": 0.1, "tags": ["transformer", "baseline"]},
        {"name": "transformer-large",   "lr": 1e-4, "batch_size": 16, "hidden_dim": 512, "dropout": 0.1, "tags": ["transformer", "large"]},
        {"name": "cnn-baseline",        "lr": 1e-3, "batch_size": 64, "hidden_dim": 128, "dropout": 0.2, "tags": ["cnn", "baseline"]},
        {"name": "cnn-dropout-heavy",   "lr": 5e-4, "batch_size": 64, "hidden_dim": 128, "dropout": 0.5, "tags": ["cnn", "regularized"]},
        {"name": "lstm-seq2seq",        "lr": 2e-3, "batch_size": 32, "hidden_dim": 256, "dropout": 0.3, "tags": ["lstm", "seq2seq"]},
        {"name": "mlp-small",           "lr": 5e-3, "batch_size": 128,"hidden_dim": 64,  "dropout": 0.0, "tags": ["mlp", "fast"]},
        {"name": "resnet-pretrained",   "lr": 1e-4, "batch_size": 16, "hidden_dim": 512, "dropout": 0.2, "tags": ["resnet", "transfer"]},
        {"name": "vit-small",           "lr": 5e-5, "batch_size": 8,  "hidden_dim": 384, "dropout": 0.1, "tags": ["vit", "attention"]},
    ]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
        task = progress.add_task("Creating runs...", total=len(configs))

        for cfg in configs:
            name = cfg["name"]
            lr = cfg["lr"]
            tags = cfg["tags"]
            epochs = random.randint(15, 30)

            with mltrackr.run(name, tags=tags):
                mltrackr.log(lr=lr, batch_size=cfg["batch_size"], hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"])
                peak_acc = random.uniform(0.78, 0.96)
                for epoch in range(epochs):
                    t = epoch / epochs
                    loss = (1.8 + random.uniform(-0.1, 0.1)) * math.exp(-lr * 2000 * t) + random.uniform(-0.02, 0.02)
                    acc = peak_acc * (1 - math.exp(-4 * t)) + random.uniform(-0.01, 0.01)
                    val_loss = loss * random.uniform(1.05, 1.2)
                    val_acc = acc * random.uniform(0.97, 1.0)
                    mltrackr.log(
                        epoch=epoch,
                        loss=round(max(0.01, loss), 4),
                        accuracy=round(min(0.999, max(0, acc)), 4),
                        val_loss=round(max(0.01, val_loss), 4),
                        val_accuracy=round(min(0.999, max(0, val_acc)), 4),
                    )
                mltrackr.note(f"Demo run. peak_acc={peak_acc:.3f}, lr={lr}, epochs={epochs}")

            progress.advance(task)

    console.print(f"[bold green]✓[/bold green] Created [bold]{len(configs)}[/bold] demo experiment runs.\n")
    console.print("  [dim]Now run:[/dim]  [bold white]python -m mltrackr ui[/bold white]  [dim]to explore the dashboard[/dim]")
    console.print("  [dim]Or try:[/dim]  [bold white]python -m mltrackr list[/bold white]")
    console.print("           [bold white]python -m mltrackr best val_accuracy[/bold white]")
    console.print("           [bold white]python -m mltrackr suggest val_accuracy[/bold white]\n")


@cli.command()
@click.argument("metric", default="accuracy")
@click.option("--mode", type=click.Choice(["max", "min"]), default="max", show_default=True)
def share(metric, mode):
    """Print a shareable text summary of your best result (great for Twitter/X, HN, Slack)."""
    from rich.console import Console
    from rich.panel import Panel
    from mltrackr.core import get_best_run, get_stats, _compute_duration
    import datetime as _dt

    console = Console()
    best = get_best_run(metric, mode=mode)
    stats = get_stats()

    if best is None:
        console.print(f"[dim]No runs found with metric '{metric}'.[/dim]")
        return

    entries = best["metrics"].get(metric, [])
    best_val = entries[-1]["value"] if entries else "?"
    val_str = f"{best_val:.4f}" if isinstance(best_val, float) else str(best_val)
    dur = _compute_duration(best["start_time"], best["end_time"])
    dur_str = f"{dur:.0f}s" if dur and dur < 60 else (f"{dur/60:.1f}m" if dur and dur < 3600 else (f"{dur/3600:.1f}h" if dur else ""))
    tags_str = " ".join(f"#{t}" for t in best.get("tags", []))

    outcome_kw = ("loss","error","mse","mae","acc","accuracy","score","f1","auc","r2")
    other_metrics = {
        k: v[-1]["value"] for k, v in best["metrics"].items()
        if k != metric and any(kw in k.lower() for kw in outcome_kw) and isinstance(v[-1]["value"], (int, float))
    }
    others_str = "  ".join(f"{k}: {v:.4g}" for k, v in list(other_metrics.items())[:3])

    tweet = (
        f"Just trained '{best['name']}' with mltrackr 🧪\n\n"
        f"  {metric}: {val_str}"
        + (f"  ({others_str})" if others_str else "")
        + (f"\n  ⏱ {dur_str}" if dur_str else "")
        + (f"\n  {tags_str}" if tags_str else "")
        + f"\n\n{stats['total_runs']} runs tracked locally — no server, no account, no config.\n"
        f"pip install mltrackr  →  github.com/NaiaLorente/Datalog"
    )

    hn_post = (
        f"mltrackr — track ML experiments in 2 lines of code\n\n"
        f"Best run: '{best['name']}'  {metric}={val_str}"
        + (f"  ({others_str})" if others_str else "")
        + f"\n\nZero setup: pip install mltrackr, wrap your loop, open the dashboard. "
        f"SQLite file, works offline, live anomaly detection built in.\n\n"
        f"https://github.com/NaiaLorente/Datalog"
    )

    console.print()
    console.print(Panel(
        f"[bold white]Twitter/X:[/bold white]\n[dim]{tweet}[/dim]",
        title="[bold cyan]📣 Share your results[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print(Panel(
        f"[bold white]Hacker News / Reddit:[/bold white]\n[dim]{hn_post}[/dim]",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()
    console.print("[dim]Tip: run [bold]mltrackr report[/bold] for a full markdown summary to paste anywhere.[/dim]\n")


if __name__ == "__main__":
    cli()
