# Examples

Working examples showing datalog with popular ML frameworks.

| File | Framework | What it shows |
|------|-----------|---------------|
| [`sklearn_example.py`](sklearn_example.py) | scikit-learn | Hyperparameter search over RandomForest |
| [`pytorch_example.py`](pytorch_example.py) | PyTorch | MLP training loop with per-epoch logging |
| [`huggingface_example.py`](huggingface_example.py) | HuggingFace Transformers | Custom `TrainerCallback` for fine-tuning |
| [`keras_example.py`](keras_example.py) | Keras / TensorFlow | Custom Keras callback |

All examples use `datalog.configure_watch()` for live anomaly detection and end with `datalog.generate_report()`.

## Run any example

```bash
pip install .                        # install datalog
python examples/sklearn_example.py  # no extra deps needed
python examples/pytorch_example.py  # needs: pip install torch
python examples/keras_example.py    # needs: pip install tensorflow
python examples/huggingface_example.py  # needs: pip install transformers datasets
```

Then open the dashboard to see all runs:

```bash
datalog ui
```
