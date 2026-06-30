import sqlite3
import json
import csv
import math
import threading
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_DIR = Path.home() / ".mltrackr"
DB_PATH = DB_DIR / "experiments.db"

_local = threading.local()

def _get_current_run_id():
    stack = getattr(_local, "run_stack", [])
    return stack[-1] if stack else None

def _push_run_id(run_id):
    if not hasattr(_local, "run_stack"):
        _local.run_stack = []
    _local.run_stack.append(run_id)

def _pop_run_id():
    if hasattr(_local, "run_stack") and _local.run_stack:
        _local.run_stack.pop()

# ── Watch configuration ──────────────────────────────────────────────────────

_watch_config = {
    "enabled": False,
    "nan_check": True,
    "divergence_window": 5,
    "plateau_window": 15,
}

# Per-run metric history for watch (keyed by run_id -> metric -> list of values)
_watch_history: dict = {}

# Anomaly counter per run_id
_anomaly_counts: dict = {}


def configure_watch(
    nan_check: bool = True,
    divergence_window: int = 5,
    plateau_window: int = 15,
    enabled: bool = True,
):
    """Configure automatic anomaly detection. Call once before training."""
    _watch_config["enabled"] = enabled
    _watch_config["nan_check"] = nan_check
    _watch_config["divergence_window"] = divergence_window
    _watch_config["plateau_window"] = plateau_window


@contextmanager
def watch(**kwargs):
    """Temporarily enable watch with given config, restore after."""
    old_config = dict(_watch_config)
    configure_watch(**kwargs)
    try:
        yield
    finally:
        _watch_config.update(old_config)


def _metric_direction(name: str):
    """Return 'lower' if lower is better, 'higher' if higher is better, 'unknown' otherwise."""
    lower_keywords = ("loss", "error", "mse", "mae", "rmse")
    higher_keywords = ("acc", "score", "f1", "auc", "precision", "recall", "r2")
    # Step/epoch/iteration counters are always increasing by design — skip anomaly checks
    counter_keywords = ("epoch", "step", "iter", "batch", "sample")
    name_lower = name.lower()
    if any(k in name_lower for k in counter_keywords):
        return "counter"
    if any(k in name_lower for k in lower_keywords):
        return "lower"
    if any(k in name_lower for k in higher_keywords):
        return "higher"
    return "unknown"


def _run_watch_checks(run_id: int, key: str, value):
    """Run anomaly detection for a newly logged value. Called from log()."""
    try:
        from rich.console import Console
        console = Console(stderr=True)
    except ImportError:
        return

    if run_id not in _watch_history:
        _watch_history[run_id] = {}
    if key not in _watch_history[run_id]:
        _watch_history[run_id][key] = []

    history = _watch_history[run_id][key]
    history.append(value)

    if run_id not in _anomaly_counts:
        _anomaly_counts[run_id] = 0

    # NaN / Inf check
    if _watch_config["nan_check"]:
        try:
            fv = float(value)
            if math.isnan(fv) or math.isinf(fv):
                console.print(
                    f"[bold yellow][mltrackr WARNING] NaN/Inf detected in '{key}' "
                    f"-- consider stopping this run[/bold yellow]"
                )
                _anomaly_counts[run_id] += 1
                return
        except (TypeError, ValueError):
            return  # non-numeric, skip further checks

    try:
        fv = float(value)
    except (TypeError, ValueError):
        return

    # Skip divergence/plateau checks for counter metrics (epoch, step, iter...)
    if _metric_direction(key) == "counter":
        return

    n_div = _watch_config["divergence_window"]
    n_plat = _watch_config["plateau_window"]

    # Divergence check
    if len(history) >= n_div:
        recent = [float(v) for v in history[-n_div:]]
        direction = _metric_direction(key)

        def strictly_increasing(vals):
            return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

        def strictly_decreasing(vals):
            return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))

        flagged = False
        if direction == "lower" and strictly_increasing(recent):
            console.print(
                f"[bold yellow][mltrackr WARNING] '{key}' has increased for {n_div} consecutive logs "
                f"-- possible divergence (current: {fv:.4g})[/bold yellow]"
            )
            flagged = True
        elif direction == "higher" and strictly_decreasing(recent):
            console.print(
                f"[bold yellow][mltrackr WARNING] '{key}' has decreased for {n_div} consecutive logs "
                f"-- possible divergence (current: {fv:.4g})[/bold yellow]"
            )
            flagged = True
        elif direction == "unknown":
            if strictly_increasing(recent) or strictly_decreasing(recent):
                console.print(
                    f"[bold yellow][mltrackr WARNING] '{key}' has been monotonically "
                    f"changing for {n_div} consecutive logs (current: {fv:.4g})[/bold yellow]"
                )
                flagged = True

        if flagged:
            _anomaly_counts[run_id] += 1

    # Plateau check
    if len(history) >= n_plat:
        recent = [float(v) for v in history[-n_plat:]]
        mean_val = sum(recent) / len(recent)
        if mean_val != 0:
            spread = (max(recent) - min(recent)) / abs(mean_val)
            if spread < 0.001:  # < 0.1% variation
                console.print(
                    f"[bold blue][mltrackr INFO] '{key}' has plateaued for {n_plat} logs "
                    f"(value: {mean_val:.4g}) -- consider adjusting lr[/bold blue]"
                )


# ── DB helpers ───────────────────────────────────────────────────────────────

def _get_db_path():
    return DB_PATH


def _connect():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                git_commit TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                tags TEXT,
                notes TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)
        conn.commit()

    # Migration: add tags and notes columns if they don't exist yet
    with _connect() as conn:
        for col in ("tags", "notes"):
            try:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} TEXT")
                conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists


_init_db()


def _get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


_verbose = True  # global; set False to suppress run-completion summaries


def configure(verbose: bool = True):
    """Configure global mltrackr behaviour. Call once at the top of your script."""
    global _verbose
    _verbose = verbose


def _print_run_summary(run_id: int, name: str, status: str, start_time: str, end_time: str):
    """Print a beautiful Rich summary panel after a run completes."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
    except ImportError:
        return

    console = Console()

    with _connect() as conn:
        metrics_rows = conn.execute(
            "SELECT key, value FROM metrics WHERE run_id = ? ORDER BY timestamp",
            (run_id,),
        ).fetchall()

    metrics: dict = {}
    for m in metrics_rows:
        key = m["key"]
        try:
            value = json.loads(m["value"])
        except Exception:
            value = m["value"]
        if isinstance(value, (int, float)):
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)

    # Compute duration
    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)
        dur_s = (end - start).total_seconds()
        dur_str = f"{dur_s:.1f}s" if dur_s < 60 else f"{dur_s/60:.1f}m" if dur_s < 3600 else f"{dur_s/3600:.1f}h"
    except Exception:
        dur_str = "—"

    status_color = {"completed": "green", "failed": "red"}.get(status, "yellow")
    status_icon = {"completed": "✓", "failed": "✗"}.get(status, "~")

    lines = [f"[bold {status_color}]{status_icon} {status}[/bold {status_color}]  [dim]#{run_id}[/dim]  [{status_color}]{name}[/{status_color}]  [dim]({dur_str})[/dim]"]

    if metrics:
        outcome_kw = ("loss", "error", "mse", "mae", "acc", "accuracy", "score", "f1", "auc", "r2")
        outcome_keys = [k for k in metrics if any(kw in k.lower() for kw in outcome_kw)]
        show_keys = (outcome_keys or list(metrics.keys()))[:6]
        parts = []
        for k in show_keys:
            vals = metrics[k]
            v = vals[-1]
            fmt = f"{v:.4g}" if isinstance(v, float) else str(v)
            parts.append(f"[cyan]{k}[/cyan]=[bold white]{fmt}[/bold white]")
        lines.append("  " + "  ".join(parts))

    lines.append(f"\n[dim]  → mltrackr ui   mltrackr best accuracy   mltrackr suggest accuracy[/dim]")

    console.print(Panel(
        "\n".join(lines),
        border_style=status_color,
        padding=(0, 1),
        expand=False,
    ))


@contextmanager
def run(name: str, tags: list = None):
    """Context manager that creates a new experiment run."""
    git_commit = _get_git_commit()
    start_time = datetime.now(timezone.utc).isoformat()
    tags_json = json.dumps(tags) if tags else None

    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO runs (name, git_commit, start_time, status, tags) VALUES (?, ?, ?, ?, ?)",
            (name, git_commit, start_time, "running", tags_json),
        )
        run_id = cursor.lastrowid
        conn.commit()

    _push_run_id(run_id)
    status = "completed"
    try:
        yield run_id
    except Exception:
        status = "failed"
        raise
    finally:
        end_time = datetime.now(timezone.utc).isoformat()
        with _connect() as conn:
            conn.execute(
                "UPDATE runs SET end_time = ?, status = ? WHERE id = ?",
                (end_time, status, run_id),
            )
            conn.commit()
        _pop_run_id()
        _watch_history.pop(run_id, None)
        if _verbose and not _get_current_run_id():  # only print for outermost run
            _print_run_summary(run_id, name, status, start_time, end_time)


def log(**kwargs):
    """Log key-value metrics to the current run."""
    run_id = _get_current_run_id()
    if run_id is None:
        raise RuntimeError("No active run. Use `with mltrackr.run('name'):` first.")

    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        for key, value in kwargs.items():
            conn.execute(
                "INSERT INTO metrics (run_id, key, value, timestamp) VALUES (?, ?, ?, ?)",
                (run_id, key, json.dumps(value), timestamp),
            )
        conn.commit()

    # Run anomaly detection if watch is configured
    if _watch_config["enabled"]:
        for key, value in kwargs.items():
            _run_watch_checks(run_id, key, value)


def note(text: str):
    """Append a text note to the current run (must be inside run context)."""
    run_id = _get_current_run_id()
    if run_id is None:
        raise RuntimeError("No active run. Use `with mltrackr.run('name'):` first.")

    with _connect() as conn:
        row = conn.execute("SELECT notes FROM runs WHERE id = ?", (run_id,)).fetchone()
        existing = row["notes"] if row and row["notes"] else ""
        separator = "\n\n" if existing else ""
        new_notes = existing + separator + text
        conn.execute("UPDATE runs SET notes = ? WHERE id = ?", (new_notes, run_id))
        conn.commit()


def tag(run_id_or_name, *tags):
    """Add tags to a run. Can be called inside or outside a run context."""
    with _connect() as conn:
        if isinstance(run_id_or_name, int):
            row = conn.execute("SELECT id, tags FROM runs WHERE id = ?", (run_id_or_name,)).fetchone()
        else:
            row = conn.execute(
                "SELECT id, tags FROM runs WHERE name = ? ORDER BY start_time DESC LIMIT 1",
                (run_id_or_name,),
            ).fetchone()

        if row is None:
            raise ValueError(f"No run found: {run_id_or_name!r}")

        existing = json.loads(row["tags"]) if row["tags"] else []
        merged = list(dict.fromkeys(existing + list(tags)))  # deduplicate, preserve order
        conn.execute("UPDATE runs SET tags = ? WHERE id = ?", (json.dumps(merged), row["id"]))
        conn.commit()


def get_runs():
    """Return list of all runs with their metrics."""
    with _connect() as conn:
        runs = conn.execute(
            "SELECT * FROM runs ORDER BY start_time DESC"
        ).fetchall()

        result = []
        for r in runs:
            metrics_rows = conn.execute(
                "SELECT key, value, timestamp FROM metrics WHERE run_id = ? ORDER BY timestamp",
                (r["id"],),
            ).fetchall()

            metrics = {}
            for m in metrics_rows:
                key = m["key"]
                try:
                    value = json.loads(m["value"])
                except Exception:
                    value = m["value"]
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append({"value": value, "timestamp": m["timestamp"]})

            tags_raw = r["tags"]
            tags_list = json.loads(tags_raw) if tags_raw else []

            result.append({
                "id": r["id"],
                "name": r["name"],
                "git_commit": r["git_commit"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "status": r["status"],
                "tags": tags_list,
                "notes": r["notes"] or "",
                "metrics": metrics,
            })
        return result


def get_best_run(metric: str, mode: str = "max") -> dict:
    """Return the run with the highest (mode='max') or lowest (mode='min') final value for a metric."""
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'")

    runs = get_runs()
    best = None
    best_val = None

    for r in runs:
        if metric not in r["metrics"]:
            continue
        entries = r["metrics"][metric]
        if not entries:
            continue
        final_val = entries[-1]["value"]
        if not isinstance(final_val, (int, float)):
            continue
        if best_val is None:
            best_val = final_val
            best = r
        elif mode == "max" and final_val > best_val:
            best_val = final_val
            best = r
        elif mode == "min" and final_val < best_val:
            best_val = final_val
            best = r

    return best


def compare_runs(*run_ids) -> list:
    """Return a list of run dicts for the given IDs, for side-by-side comparison."""
    runs = get_runs()
    id_set = set(run_ids)
    ordered = {r["id"]: r for r in runs if r["id"] in id_set}
    return [ordered[rid] for rid in run_ids if rid in ordered]


def _compute_duration(start_time, end_time):
    if not end_time:
        return None
    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)
        return (end - start).total_seconds()
    except Exception:
        return None


def export_csv(path: str):
    """Export all runs and metrics to a CSV file."""
    runs = get_runs()
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "run_name", "git_commit", "start_time", "end_time",
            "status", "duration_seconds", "tags", "notes",
            "metric_key", "metric_value", "metric_timestamp"
        ])
        for r in runs:
            duration = _compute_duration(r["start_time"], r["end_time"])
            tags_str = ",".join(r.get("tags", []))
            if r["metrics"]:
                for key, entries in r["metrics"].items():
                    for entry in entries:
                        writer.writerow([
                            r["id"], r["name"], r["git_commit"],
                            r["start_time"], r["end_time"], r["status"],
                            duration, tags_str, r.get("notes", ""),
                            key, entry["value"], entry["timestamp"]
                        ])
            else:
                writer.writerow([
                    r["id"], r["name"], r["git_commit"],
                    r["start_time"], r["end_time"], r["status"],
                    duration, tags_str, r.get("notes", ""),
                    None, None, None
                ])


def export_json(path: str):
    """Export all runs and metrics to a JSON file."""
    runs = get_runs()
    path = Path(path)
    for r in runs:
        r["duration_seconds"] = _compute_duration(r["start_time"], r["end_time"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, default=str)


def clear_all():
    """Delete all experiment data."""
    with _connect() as conn:
        conn.execute("DELETE FROM metrics")
        conn.execute("DELETE FROM runs")
        conn.commit()


def get_stats() -> dict:
    """Return aggregate statistics across all runs."""
    runs = get_runs()
    total = len(runs)
    completed = sum(1 for r in runs if r["status"] == "completed")
    failed = sum(1 for r in runs if r["status"] == "failed")
    running = sum(1 for r in runs if r["status"] == "running")
    metric_counts: dict = {}
    for r in runs:
        for key, entries in r["metrics"].items():
            metric_counts[key] = metric_counts.get(key, 0) + len(entries)
    top_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        "total_runs": total,
        "completed": completed,
        "failed": failed,
        "running": running,
        "success_rate": round(completed / total * 100, 1) if total else 0.0,
        "top_metrics": [{"key": k, "count": v} for k, v in top_metrics],
    }


# ── Report generation ────────────────────────────────────────────────────────

def _fmt_date_range(runs):
    """Return a human-readable date range string for the given runs."""
    times = []
    for r in runs:
        try:
            times.append(datetime.fromisoformat(r["start_time"]))
        except Exception:
            pass
    if not times:
        return "unknown"
    earliest = min(times)
    latest = max(times)
    if earliest.date() == latest.date():
        return earliest.strftime("%b %d, %Y").replace(' 0', '  ')
    return f"{earliest.strftime('%b %d').replace(' 0', '  ')} - {latest.strftime('%b %d, %Y').replace(' 0', '  ')}"


def _fmt_duration(seconds):
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _compute_key_findings(runs, all_metric_keys):
    """Compute key findings from runs data. Returns list of finding strings."""
    findings = []
    completed = [r for r in runs if r["status"] == "completed"]

    # 1. Most logged metric
    metric_dp_counts = {}
    for r in completed:
        for k, entries in r["metrics"].items():
            metric_dp_counts[k] = metric_dp_counts.get(k, 0) + len(entries)

    if metric_dp_counts:
        most_logged = max(metric_dp_counts, key=metric_dp_counts.get)
        findings.append(
            f"**Most logged metric**: `{most_logged}` "
            f"({metric_dp_counts[most_logged]} data points across {len(completed)} completed runs)"
        )

    # 2. Biggest improvement from chronologically first run to best run
    if len(completed) >= 2:
        chron = sorted(completed, key=lambda r: r["start_time"])
        first_run = chron[0]
        for metric_name in all_metric_keys:
            first_entries = first_run["metrics"].get(metric_name, [])
            if not first_entries:
                continue
            first_val = first_entries[-1]["value"]
            if not isinstance(first_val, (int, float)):
                continue
            direction = _metric_direction(metric_name)
            best_run = None
            best_val_here = None
            for r in completed:
                entries = r["metrics"].get(metric_name, [])
                if not entries:
                    continue
                v = entries[-1]["value"]
                if not isinstance(v, (int, float)) or math.isnan(v):
                    continue
                if best_val_here is None:
                    best_val_here = v
                    best_run = r
                elif direction == "lower" and v < best_val_here:
                    best_val_here = v
                    best_run = r
                elif direction != "lower" and v > best_val_here:
                    best_val_here = v
                    best_run = r

            if best_run and best_run["id"] != first_run["id"] and first_val != 0:
                pct = (best_val_here - first_val) / abs(first_val) * 100
                sign = "+" if pct > 0 else ""
                findings.append(
                    f"**Biggest improvement**: Run #{first_run['id']} `{first_run['name']}` "
                    f"-> #{best_run['id']} `{best_run['name']}` -- "
                    f"`{metric_name}` {sign}{pct:.1f}%"
                )
            break  # only analyze first eligible metric

    # 3. Most stable metric (lowest coefficient of variation across runs)
    stability = {}
    for metric_name in all_metric_keys:
        vals = []
        for r in completed:
            entries = r["metrics"].get(metric_name, [])
            if entries:
                v = entries[-1]["value"]
                if isinstance(v, (int, float)) and not math.isnan(v):
                    vals.append(v)
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(variance)
            cv = std / abs(mean) if mean != 0 else float("inf")
            stability[metric_name] = (std, cv)

    if stability:
        most_stable = min(stability, key=lambda k: stability[k][1])
        std_val, _ = stability[most_stable]
        findings.append(
            f"**Most stable metric**: `{most_stable}` "
            f"(std dev {std_val:.4g} across {len(completed)} completed runs)"
        )

    # 4. Best configuration from best run's first-timestamp logged params
    if completed and all_metric_keys:
        main_metric = all_metric_keys[0]
        direction = _metric_direction(main_metric)
        best_run = None
        best_val_main = None
        for r in completed:
            entries = r["metrics"].get(main_metric, [])
            if not entries:
                continue
            v = entries[-1]["value"]
            if not isinstance(v, (int, float)):
                continue
            if best_val_main is None:
                best_val_main = v
                best_run = r
            elif direction == "lower" and v < best_val_main:
                best_val_main = v
                best_run = r
            elif direction != "lower" and v > best_val_main:
                best_val_main = v
                best_run = r

        if best_run:
            timestamps = [ents[0]["timestamp"] for ents in best_run["metrics"].values() if ents]
            if timestamps:
                first_ts = min(timestamps)
                first_params = {}
                for k, ents in best_run["metrics"].items():
                    for e in ents:
                        if e["timestamp"] == first_ts and isinstance(e["value"], (int, float)):
                            first_params[k] = e["value"]
                if first_params:
                    params_str = ", ".join(
                        f"{k}={v:.4g}" for k, v in list(first_params.items())[:6]
                    )
                    findings.append(
                        f"**Best configuration**: {params_str} "
                        f"(run `{best_run['name']}` -- `{main_metric}` = {best_val_main:.4g})"
                    )

    return findings


def generate_report(output_path: str = None, use_ollama: bool = False, ollama_model: str = "llama3") -> str:
    """Generate a markdown experiment report. Returns the markdown string."""
    runs = get_runs()
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y").replace(' 0', '  ')

    completed = [r for r in runs if r["status"] == "completed"]
    failed = [r for r in runs if r["status"] == "failed"]
    running_runs = [r for r in runs if r["status"] == "running"]
    total = len(runs)
    success_rate = (len(completed) / total * 100) if total > 0 else 0.0
    date_range = _fmt_date_range(runs)

    # Collect all metric keys in order of frequency
    metric_freq = {}
    for r in runs:
        for k in r["metrics"]:
            metric_freq[k] = metric_freq.get(k, 0) + 1
    all_metric_keys = sorted(metric_freq, key=metric_freq.get, reverse=True)

    # Separate outcome metrics (accuracy, loss, f1…) from hyperparam-like keys
    _OUTCOME_KW = ("loss","error","mse","mae","rmse","mape","acc","accuracy",
                   "score","f1","auc","precision","recall","r2","map","ndcg")
    def _is_outcome(k):
        kl = k.lower()
        return any(kw in kl for kw in _OUTCOME_KW)
    outcome_keys = [k for k in all_metric_keys if _is_outcome(k)]
    search_keys = outcome_keys if outcome_keys else all_metric_keys

    # Find best run for the most representative outcome metric
    overall_best_run = None
    overall_best_metric = None
    overall_best_val = None
    for metric_name in search_keys:
        direction = _metric_direction(metric_name)
        for r in completed:
            entries = r["metrics"].get(metric_name, [])
            if not entries:
                continue
            v = entries[-1]["value"]
            if not isinstance(v, (int, float)) or math.isnan(v):
                continue
            if overall_best_run is None:
                overall_best_run = r
                overall_best_metric = metric_name
                overall_best_val = v
            elif metric_name == overall_best_metric:
                if (direction == "lower" and v < overall_best_val) or \
                   (direction != "lower" and v > overall_best_val):
                    overall_best_run = r
                    overall_best_val = v
        if overall_best_run:
            break

    # Best result line with % improvement vs baseline
    best_line = ""
    if overall_best_run:
        chron = sorted(
            [r for r in completed if overall_best_metric in r["metrics"]],
            key=lambda r: r["start_time"],
        )
        pct_str = ""
        if chron and chron[0]["id"] != overall_best_run["id"]:
            baseline_entries = chron[0]["metrics"].get(overall_best_metric, [])
            if baseline_entries:
                baseline_val = baseline_entries[-1]["value"]
                if isinstance(baseline_val, (int, float)) and baseline_val != 0:
                    pct = (overall_best_val - baseline_val) / abs(baseline_val) * 100
                    direction_word = "up" if pct > 0 else "down"
                    pct_str = f" ({direction_word} {abs(pct):.1f}% vs baseline)"
        best_line = (
            f"- **Best result**: `{overall_best_run['name']}` -- "
            f"{overall_best_metric} **{overall_best_val:.4g}**{pct_str}"
        )

    lines = [
        "# Experiment Report",
        f"*Generated by mltrackr on {date_str}*",
        "",
        "## Summary",
        f"- **Total runs**: {total}",
        f"- **Completed**: {len(completed)} ({success_rate:.1f}% success rate)",
        f"- **Failed**: {len(failed)}",
        f"- **Running**: {len(running_runs)}",
        f"- **Date range**: {date_range}",
    ]
    if best_line:
        lines.append(best_line)
    lines.append("")

    # Best Runs by Metric
    medals = ["🥇", "🥈", "🥉"]
    lines.append("## Best Runs by Metric")
    lines.append("")

    report_keys = (outcome_keys if outcome_keys else all_metric_keys)[:6]
    for metric_name in report_keys:
        direction = _metric_direction(metric_name)
        metric_runs = []
        for r in completed:
            entries = r["metrics"].get(metric_name, [])
            if not entries:
                continue
            v = entries[-1]["value"]
            if isinstance(v, (int, float)):
                dur = _compute_duration(r["start_time"], r["end_time"])
                tags_str = ", ".join(r.get("tags", [])) or "none"
                metric_runs.append((v, r["name"], dur, tags_str))

        if not metric_runs:
            continue

        reverse = (direction != "lower")
        metric_runs.sort(key=lambda x: x[0], reverse=reverse)
        top3 = metric_runs[:3]

        dir_label = "lowest" if direction == "lower" else "highest"
        lines.append(f"### {metric_name} ({dir_label})")
        lines.append("")
        lines.append("| Rank | Run | Value | Duration | Tags |")
        lines.append("|------|-----|-------|----------|------|")
        for i, (val, rname, dur, tags_str) in enumerate(top3):
            rank = medals[i] if i < len(medals) else f"#{i+1}"
            lines.append(f"| {rank} | {rname} | {val:.4g} | {_fmt_duration(dur)} | {tags_str} |")
        lines.append("")

    # Experiment Timeline
    lines.append("## Experiment Timeline")
    lines.append("")
    lines.append("| # | Run | Status | Started | Duration | Tags |")
    lines.append("|---|-----|--------|---------|----------|------|")
    status_sym = {"completed": "[ok]", "failed": "[FAIL]", "running": "[running]"}
    chron_runs = sorted(runs, key=lambda r: r["start_time"])
    for r in chron_runs:
        sym = status_sym.get(r["status"], "?")
        dur = _compute_duration(r["start_time"], r["end_time"])
        start_dt = datetime.fromisoformat(r["start_time"]).strftime("%b %d %H:%M").replace(' 0', '  ')
        tags_str = ", ".join(r.get("tags", [])) or "none"
        lines.append(
            f"| {r['id']} | `{r['name']}` | {sym} {r['status']} | {start_dt} "
            f"| {_fmt_duration(dur)} | {tags_str} |"
        )
    lines.append("")

    # Key Findings
    findings = _compute_key_findings(runs, all_metric_keys)
    if findings:
        lines.append("## Key Findings")
        lines.append("")
        for finding in findings:
            lines.append(f"- {finding}")
        lines.append("")

    # Failed Runs
    if failed:
        lines.append("## Failed Runs")
        lines.append("")
        lines.append("| # | Run | Started | Duration | Notes |")
        lines.append("|---|-----|---------|----------|-------|")
        for r in failed:
            dur = _compute_duration(r["start_time"], r["end_time"])
            start_dt = datetime.fromisoformat(r["start_time"]).strftime("%b %d %H:%M").replace(' 0', '  ')
            notes_text = (r.get("notes") or "").replace("\n", " ")
            notes_short = notes_text[:60] + ("..." if len(notes_text) > 60 else "")
            lines.append(
                f"| {r['id']} | `{r['name']}` | {start_dt} | {_fmt_duration(dur)} | {notes_short or 'none'} |"
            )
        lines.append("")

    # Notes & Observations
    notes_entries = [(r["id"], r["name"], r.get("notes", "")) for r in runs if r.get("notes")]
    if notes_entries:
        lines.append("## Notes & Observations")
        lines.append("")
        for run_id, run_name, run_notes in notes_entries:
            lines.append(f"**Run #{run_id} -- {run_name}**")
            lines.append("")
            for note_line in run_notes.strip().splitlines():
                lines.append(f"> {note_line}")
            lines.append("")

    report_md = "\n".join(lines)

    # Ollama narrative (prepended if successful)
    if use_ollama:
        ollama_narrative = _try_ollama_narrative(runs, report_md, ollama_model)
        if ollama_narrative:
            report_md = ollama_narrative + "\n\n---\n\n" + report_md

    if output_path:
        Path(output_path).write_text(report_md)

    return report_md


def _try_ollama_narrative(runs, report_md: str, model: str) -> str:
    """Try to get a narrative summary from Ollama. Returns empty string on failure."""
    try:
        import requests
    except ImportError:
        return ""

    completed = [r for r in runs if r["status"] == "completed"]
    total = len(runs)

    prompt = (
        "You are an ML experiment analyst. Given the following experiment report data, "
        "write a concise 2-paragraph narrative summary highlighting what worked, what didn't, "
        "and what the key takeaways are. Be specific and data-driven.\n\n"
        f"Total runs: {total}, Completed: {len(completed)}, Failed: {total - len(completed)}\n\n"
        f"Report excerpt:\n{report_md[:2000]}\n\n"
        "Write only the 2-paragraph narrative, no headers."
    )

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        if resp.ok:
            data = resp.json()
            narrative = data.get("response", "").strip()
            if narrative:
                return f"## AI Narrative Summary\n*Generated by {model} via Ollama*\n\n{narrative}"
    except Exception:
        pass
    return ""


# ── Suggest ──────────────────────────────────────────────────────────────────

def suggest(target_metric: str, mode: str = "max", top_n: int = 3) -> dict:
    """
    Analyze completed runs and suggest next hyperparameter values to try.

    Returns dict with:
    - 'target': metric name
    - 'best_run': the best run dict
    - 'best_value': best value achieved
    - 'insights': list of insight strings
    - 'suggestions': list of suggestion dicts {param, recommended_value, reasoning}
    """
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'")

    runs = get_runs()
    completed = [r for r in runs if r["status"] == "completed" and target_metric in r["metrics"]]

    if len(completed) < 3:
        return {
            "target": target_metric,
            "best_run": None,
            "best_value": None,
            "insights": [
                f"Only {len(completed)} completed run(s) with '{target_metric}' found. "
                "Run at least 3 experiments to get meaningful suggestions."
            ],
            "suggestions": [],
        }

    # For each run: collect first-timestamp params + final target_metric value
    run_data = []
    for r in completed:
        entries = r["metrics"].get(target_metric, [])
        if not entries:
            continue
        result_val = entries[-1]["value"]
        if not isinstance(result_val, (int, float)):
            continue

        timestamps = [ents[0]["timestamp"] for ents in r["metrics"].values() if ents]
        if not timestamps:
            continue
        first_ts = min(timestamps)

        params = {}
        for k, ents in r["metrics"].items():
            for e in ents:
                if e["timestamp"] == first_ts and isinstance(e["value"], (int, float)):
                    params[k] = e["value"]
        params.pop(target_metric, None)

        run_data.append({"run": r, "params": params, "result": result_val})

    if len(run_data) < 3:
        return {
            "target": target_metric,
            "best_run": None,
            "best_value": None,
            "insights": [
                "Not enough numeric data. Log hyperparameters in the first log() call of each run."
            ],
            "suggestions": [],
        }

    best_entry = (
        max(run_data, key=lambda x: x["result"])
        if mode == "max"
        else min(run_data, key=lambda x: x["result"])
    )
    best_run = best_entry["run"]
    best_result = best_entry["result"]

    # Find params present in >= 3 runs
    param_freq = {}
    for rd in run_data:
        for k in rd["params"]:
            param_freq[k] = param_freq.get(k, 0) + 1
    eligible_params = [k for k, cnt in param_freq.items() if cnt >= 3]

    insights = []
    suggestions = []

    for param in eligible_params:
        pairs = [(rd["params"][param], rd["result"]) for rd in run_data if param in rd["params"]]
        if len(pairs) < 3:
            continue

        param_vals_sorted = sorted(p[0] for p in pairs)
        median_val = param_vals_sorted[len(param_vals_sorted) // 2]

        low_group = [(pv, rv) for pv, rv in pairs if pv < median_val]
        high_group = [(pv, rv) for pv, rv in pairs if pv >= median_val]

        if not low_group or not high_group:
            continue

        low_avg = sum(rv for _, rv in low_group) / len(low_group)
        high_avg = sum(rv for _, rv in high_group) / len(high_group)

        base = abs(low_avg) if low_avg != 0 else (abs(high_avg) if high_avg != 0 else None)
        if base is None:
            continue
        diff_pct = abs(high_avg - low_avg) / base * 100

        if diff_pct < 5:
            continue

        if mode == "max":
            if low_avg > high_avg:
                better_side = "lower"
                better_avg, worse_avg = low_avg, high_avg
                recommend = median_val * 0.5
            else:
                better_side = "higher"
                better_avg, worse_avg = high_avg, low_avg
                recommend = median_val * 1.5
        else:
            if low_avg < high_avg:
                better_side = "lower"
                better_avg, worse_avg = low_avg, high_avg
                recommend = median_val * 0.5
            else:
                better_side = "higher"
                better_avg, worse_avg = high_avg, low_avg
                recommend = median_val * 1.5

        if better_side == "lower":
            insight = (
                f"{param} < {median_val:.4g} -> avg {target_metric} {better_avg:.4g} | "
                f"{param} >= {median_val:.4g} -> avg {target_metric} {worse_avg:.4g} "
                f"(try lower {param})"
            )
        else:
            insight = (
                f"{param} >= {median_val:.4g} -> avg {target_metric} {better_avg:.4g} | "
                f"{param} < {median_val:.4g} -> avg {target_metric} {worse_avg:.4g} "
                f"(try higher {param})"
            )

        insights.append(insight)
        suggestions.append({
            "param": param,
            "recommended_value": recommend,
            "reasoning": insight,
        })

        if len(suggestions) >= top_n:
            break

    # Summarize best run's params
    best_params = best_entry["params"]
    if best_params:
        params_str = ", ".join(f"{k}={v:.4g}" for k, v in list(best_params.items())[:6])
        insights.append(
            f"Best run `{best_run['name']}` params: {params_str} "
            f"-> {target_metric} = {best_result:.4g}"
        )

    # Suggest next experiment values (best params nudged by insight direction)
    next_exp_parts = []
    for k, v in list(best_params.items())[:4]:
        nudge = 0.8
        for s in suggestions:
            if s["param"] == k:
                nudge = 1.2 if "higher" in s["reasoning"] else 0.8
                break
        if v == 0:
            nudged = 0.1 if nudge > 1 else 0.0
        else:
            nudged = v * nudge
        if isinstance(v, int) or (isinstance(v, float) and v == int(v) and abs(v) >= 1):
            next_exp_parts.append(f"{k}={int(round(nudged))}")
        else:
            next_exp_parts.append(f"{k}={nudged:.4g}")
    if next_exp_parts:
        insights.append(f"Next experiment suggestion: {', '.join(next_exp_parts)}")

    return {
        "target": target_metric,
        "best_run": best_run,
        "best_value": best_result,
        "insights": insights,
        "suggestions": suggestions,
    }
