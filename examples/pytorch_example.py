"""
mltrackr + PyTorch example
--------------------------
Tracks a simple MLP training loop on synthetic data.
Run: python examples/pytorch_example.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mltrackr

# Watch for NaN gradients, diverging loss, plateaus
mltrackr.configure_watch(divergence_window=5, plateau_window=10)

# Synthetic binary classification dataset
torch.manual_seed(42)
X = torch.randn(1000, 20)
y = (X[:, 0] + X[:, 1] > 0).float()
dataset = TensorDataset(X, y)

configs = [
    {"lr": 1e-2, "hidden": 64,  "dropout": 0.0, "epochs": 30},
    {"lr": 1e-3, "hidden": 64,  "dropout": 0.2, "epochs": 30},
    {"lr": 1e-3, "hidden": 128, "dropout": 0.2, "epochs": 30},
    {"lr": 5e-4, "hidden": 128, "dropout": 0.3, "epochs": 30},
]


class MLP(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze()


for cfg in configs:
    name = f"mlp-lr{cfg['lr']}-h{cfg['hidden']}-d{cfg['dropout']}"
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    with mltrackr.run(name, tags=["pytorch", "mlp", "binary-classification"]):
        # Log hyperparameters
        mltrackr.log(lr=cfg["lr"], hidden=cfg["hidden"],
                    dropout=cfg["dropout"], epochs=cfg["epochs"])

        model = MLP(cfg["hidden"], cfg["dropout"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        criterion = nn.BCELoss()

        for epoch in range(cfg["epochs"]):
            model.train()
            total_loss, correct, n = 0.0, 0, 0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct += ((pred > 0.5) == yb).sum().item()
                n += len(yb)

            # mltrackr.watch automatically checks for NaN, divergence, plateau
            mltrackr.log(
                epoch=epoch,
                loss=round(total_loss / n, 4),
                accuracy=round(correct / n, 4),
            )

        print(f"{name}: final loss={total_loss/n:.4f}, acc={correct/n:.4f}")

print("\n--- Best run ---")
best = mltrackr.get_best_run("accuracy")
print(f"  {best['name']} — accuracy={best['metrics']['accuracy'][-1]['value']:.4f}")

print("\nGenerating report...")
mltrackr.generate_report("pytorch_report.md")
print("Saved to pytorch_report.md")
