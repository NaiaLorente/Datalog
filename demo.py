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
                mltrackr.log(loss=round(loss, 4), accuracy=round(acc, 4), step=step)
            mltrackr.note(f"lr={lr} bs={bs} - check dashboard for curves")

print("\nTop 3 results:")
for r in mltrackr.get_runs()[:3]:
    acc = r["metrics"].get("accuracy", [{}])[-1].get("value", "?")
    print(f"  {r['name']:30s}  accuracy={acc}")

print("\nNext steps:")
print("  python -m mltrackr ui              - open dashboard")
print("  python -m mltrackr best accuracy   - find winner")
print("  python -m mltrackr suggest accuracy - get next config")
