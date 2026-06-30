"""
mltrackr + HuggingFace Transformers example
--------------------------------------------
Wraps a HuggingFace Trainer to log metrics to mltrackr after each epoch.
Run: pip install transformers datasets && python examples/huggingface_example.py
"""

import mltrackr


class DatalogCallback:
    """
    HuggingFace TrainerCallback that logs eval metrics to mltrackr.

    Usage:
        trainer = Trainer(..., callbacks=[DatalogCallback(run_id)])
    """

    def __init__(self, run_id: int):
        self.run_id = run_id

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            import mltrackr
            from mltrackr.core import _push_run_id, _pop_run_id
            _push_run_id(self.run_id)
            try:
                mltrackr.log(**{k.replace("eval_", ""): v
                                for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            finally:
                _pop_run_id()


def train_with_mltrackr(model_name: str, dataset_name: str, **training_kwargs):
    """
    Train a HuggingFace model with mltrackr experiment tracking.

    Example:
        train_with_mltrackr(
            "distilbert-base-uncased",
            "imdb",
            learning_rate=2e-5,
            num_train_epochs=3,
            per_device_train_batch_size=16,
        )
    """
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer)
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=128)

    tokenized = dataset.map(tokenize, batched=True)

    with mltrackr.run(
        f"{model_name.split('/')[-1]}-{dataset_name}",
        tags=["huggingface", "transformers", model_name.split("/")[-1]],
    ) as run_id:
        mltrackr.log(model=model_name, dataset=dataset_name, **training_kwargs)
        mltrackr.note(f"Fine-tuning {model_name} on {dataset_name}")

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            **training_kwargs,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"].select(range(1000)),
            eval_dataset=tokenized["test"].select(range(200)),
            callbacks=[DatalogCallback(run_id)],
        )
        trainer.train()

    print(f"Run complete. View results: mltrackr ui")


# ── Demo (runs without real HuggingFace models) ──────────────────────────────

if __name__ == "__main__":
    import random

    print("Running simulated HuggingFace fine-tuning demo...")
    print("(Replace with real train_with_mltrackr() call in your project)\n")

    model_configs = [
        ("distilbert-base-uncased", 2e-5, 16),
        ("distilbert-base-uncased", 3e-5, 16),
        ("bert-base-uncased",       2e-5, 8),
    ]

    for model_name, lr, batch in model_configs:
        with mltrackr.run(
            f"{model_name}-lr{lr}",
            tags=["huggingface", "transformers", "simulated"],
        ) as run_id:
            mltrackr.log(model=model_name, learning_rate=lr,
                        batch_size=batch, epochs=3)

            random.seed(hash(model_name + str(lr)))
            base_acc = 0.82 + random.uniform(0, 0.08)
            for epoch in range(1, 4):
                acc = base_acc + epoch * 0.02 + random.uniform(-0.01, 0.01)
                loss = 0.6 - epoch * 0.12 + random.uniform(-0.02, 0.02)
                mltrackr.log(epoch=epoch, accuracy=round(acc, 4),
                            eval_loss=round(loss, 4))

    best = mltrackr.get_best_run("accuracy")
    print(f"Best model: {best['name']} — accuracy={best['metrics']['accuracy'][-1]['value']:.4f}")
    mltrackr.generate_report("hf_report.md")
    print("Report saved to hf_report.md")
    print("\nOpen dashboard: mltrackr ui")
