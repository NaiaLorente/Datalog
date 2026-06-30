"""
mltrackr + scikit-learn example
-------------------------------
Track a hyperparameter search over a RandomForest classifier on the Iris dataset.
Run: python examples/sklearn_example.py
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import mltrackr

# Enable anomaly detection
mltrackr.configure_watch(divergence_window=3, plateau_window=5)

dataset = load_iris()
X, y = dataset.data, dataset.target

configs = [
    {"n_estimators": 10,  "max_depth": 3,    "min_samples_split": 2},
    {"n_estimators": 50,  "max_depth": 5,    "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 5,    "min_samples_split": 5},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
]

for cfg in configs:
    name = f"rf-n{cfg['n_estimators']}-d{cfg['max_depth']}"

    with mltrackr.run(name, tags=["sklearn", "random-forest", "iris"]):
        # Log hyperparameters first
        mltrackr.log(**cfg)
        mltrackr.note(f"RandomForest with {cfg}")

        clf = RandomForestClassifier(**cfg, random_state=42)
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

        # Log results
        mltrackr.log(
            accuracy_mean=round(scores.mean(), 4),
            accuracy_std=round(scores.std(), 4),
            accuracy_min=round(scores.min(), 4),
            accuracy_max=round(scores.max(), 4),
        )

        print(f"{name}: accuracy={scores.mean():.4f} ± {scores.std():.4f}")

# Show results
print("\n--- Results ---")
best = mltrackr.get_best_run("accuracy_mean")
print(f"Best run: {best['name']} — accuracy={best['metrics']['accuracy_mean'][-1]['value']:.4f}")

print("\n--- Suggestions ---")
result = mltrackr.suggest("accuracy_mean")
for ins in result.get("insights", []):
    print(f"  {ins}")

print("\nGenerating report...")
mltrackr.generate_report("sklearn_report.md")
print("Report saved to sklearn_report.md")
