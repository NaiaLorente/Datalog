import json
from pathlib import Path
from flask import Flask, jsonify, render_template, abort, request
from mltrackr.core import (
    get_runs, export_csv, export_json, _compute_duration, compare_runs,
    generate_report, suggest,
)
import os

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app():
    app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/runs")
    def api_runs():
        runs = get_runs()
        for r in runs:
            r["duration_seconds"] = _compute_duration(r["start_time"], r["end_time"])
            r["git_commit_short"] = (r["git_commit"] or "")[:7] or None
        return jsonify(runs)

    @app.route("/api/runs/<int:run_id>")
    def api_run_detail(run_id):
        runs = get_runs()
        for r in runs:
            if r["id"] == run_id:
                r["duration_seconds"] = _compute_duration(r["start_time"], r["end_time"])
                r["git_commit_short"] = (r["git_commit"] or "")[:7] or None
                return jsonify(r)
        abort(404)

    @app.route("/api/runs/compare")
    def api_compare():
        ids_param = request.args.get("ids", "")
        try:
            run_ids = [int(x) for x in ids_param.split(",") if x.strip()]
        except ValueError:
            return jsonify({"error": "Invalid IDs"}), 400

        if len(run_ids) < 2:
            return jsonify({"error": "Provide at least 2 run IDs"}), 400

        runs = compare_runs(*run_ids)
        for r in runs:
            r["duration_seconds"] = _compute_duration(r["start_time"], r["end_time"])
            r["git_commit_short"] = (r["git_commit"] or "")[:7] or None
        return jsonify(runs)

    @app.route("/api/stats")
    def api_stats():
        from collections import Counter
        runs = get_runs()
        total = len(runs)
        completed = sum(1 for r in runs if r["status"] == "completed")
        failed = sum(1 for r in runs if r["status"] == "failed")
        running_count = sum(1 for r in runs if r["status"] == "running")

        metric_counter = Counter()
        for r in runs:
            for key, entries in r["metrics"].items():
                metric_counter[key] += len(entries)

        top_metrics = [{"key": k, "count": c} for k, c in metric_counter.most_common(10)]

        return jsonify({
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running_count,
            "success_rate": round(completed / total * 100, 1) if total > 0 else 0.0,
            "top_metrics": top_metrics,
        })

    @app.route("/api/export/csv")
    def api_export_csv():
        from flask import Response
        import io
        import csv as csv_mod

        runs = get_runs()
        output = io.StringIO()
        writer = csv_mod.writer(output)
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

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=mltrackr_export.csv"},
        )

    @app.route("/api/export/json")
    def api_export_json():
        from flask import Response
        runs = get_runs()
        for r in runs:
            r["duration_seconds"] = _compute_duration(r["start_time"], r["end_time"])
        return Response(
            json.dumps(runs, indent=2, default=str),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=mltrackr_export.json"},
        )

    @app.route("/api/report")
    def api_report():
        md = generate_report()
        return jsonify({"markdown": md})

    @app.route("/api/suggest")
    def api_suggest():
        metric = request.args.get("metric", "")
        mode = request.args.get("mode", "max")
        if not metric:
            return jsonify({"error": "metric parameter required"}), 400
        if mode not in ("max", "min"):
            mode = "max"
        result = suggest(metric, mode=mode)
        # best_run contains metrics which may be large; simplify for JSON
        if result.get("best_run"):
            br = result["best_run"]
            result["best_run"] = {
                "id": br["id"],
                "name": br["name"],
                "status": br["status"],
                "tags": br.get("tags", []),
            }
        return jsonify(result)

    return app
