"""
mltrackr + Keras / TensorFlow example
--------------------------------------
Uses a custom Keras callback to log metrics to mltrackr after each epoch.
Run: pip install tensorflow && python examples/keras_example.py
"""

import mltrackr


class DatalogCallback:
    """
    Keras callback that streams epoch metrics to mltrackr.

    Usage:
        model.fit(..., callbacks=[DatalogCallback()])
    """

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mltrackr.log(epoch=epoch, **{k: round(float(v), 4)
                                        for k, v in logs.items()})


def build_and_train(units: int, dropout: float, lr: float, epochs: int = 20):
    """Build and train a simple Keras model, logging to mltrackr."""
    import numpy as np

    # Lazy import so the file is importable without TF installed
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        raise ImportError("pip install tensorflow")

    np.random.seed(42)
    X_train = np.random.randn(800, 20).astype("float32")
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype("float32")
    X_val   = np.random.randn(200, 20).astype("float32")
    y_val   = (X_val[:, 0]   + X_val[:, 1]   > 0).astype("float32")

    model = keras.Sequential([
        keras.layers.Dense(units, activation="relu", input_shape=(20,)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(units, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    name = f"keras-u{units}-d{dropout}-lr{lr}"
    with mltrackr.run(name, tags=["keras", "tensorflow", "mlp"]):
        mltrackr.log(units=units, dropout=dropout, lr=lr, epochs=epochs)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            verbose=0,
            callbacks=[DatalogCallback()],
        )
        val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
        mltrackr.log(final_val_accuracy=round(val_acc, 4))
        print(f"{name}: val_accuracy={val_acc:.4f}")


if __name__ == "__main__":
    import random, math

    print("Running simulated Keras training demo...")
    print("(Replace with real build_and_train() calls for TF)\n")

    mltrackr.configure_watch(divergence_window=5, plateau_window=8)

    configs = [
        (64,  0.0, 1e-3),
        (64,  0.2, 1e-3),
        (128, 0.2, 5e-4),
        (128, 0.3, 1e-4),
    ]

    for units, dropout, lr in configs:
        name = f"keras-u{units}-d{dropout}-lr{lr}"
        with mltrackr.run(name, tags=["keras", "simulated"]):
            mltrackr.log(units=units, dropout=dropout, lr=lr)
            random.seed(hash(name))
            base = 0.75 + units / 1000 + dropout * 0.1 - math.log10(lr) * 0.02
            for epoch in range(20):
                acc = min(0.99, base + epoch * 0.008 + random.uniform(-0.01, 0.01))
                loss = max(0.01, 0.8 - epoch * 0.03 + random.uniform(-0.02, 0.02))
                mltrackr.log(epoch=epoch, accuracy=round(acc, 4), loss=round(loss, 4))

    best = mltrackr.get_best_run("accuracy")
    print(f"\nBest: {best['name']} — accuracy={best['metrics']['accuracy'][-1]['value']:.4f}")
    mltrackr.generate_report("keras_report.md")
    print("Report → keras_report.md  |  Dashboard → mltrackr ui")
