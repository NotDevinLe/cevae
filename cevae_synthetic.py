"""Run experiments on the synthetic dataset.

This is a thin wrapper around run_experiment.py kept for backwards
compatibility.  The ``run_experiment`` function is still importable so
that ``run_sample_size_experiments.py`` keeps working.

Prefer using ``run_experiment.py`` directly for new work::

    python run_experiment.py --dataset synthetic --models all
"""

from run_experiment import run, build_models, ALL_MODELS, OUTCOME_TYPE  # noqa: F401

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import SyntheticDataset


def run_experiment(dataset, epochs=100, learning_rate=1e-3, wd=1e-4,
                   verbose=True):
    """Backwards-compatible entry point used by run_sample_size_experiments."""
    scores = run(
        dataset,
        model_names=list(ALL_MODELS),
        outcome="binary",
        epochs=epochs,
        lr=learning_rate,
        wd=wd,
        verbose=verbose,
    )
    display_names = {"cevae": "CEVAE", "tarnet": "TARNet", "lr1": "LR-1", "lr2": "LR-2"}
    return {display_names[k]: v["test"] for k, v in scores.items()}


if __name__ == "__main__":
    dataset = SyntheticDataset(replications=1)
    scores = run_experiment(dataset)

    print("\n====== Final Test Scores ======")
    for name, arr in scores.items():
        m = arr.mean(axis=0)
        print(f"{name:8s}  ITE: {m[0]:.3f}  ATE: {m[1]:.3f}  "
              f"PEHE: {m[2]:.3f}  ATT: {m[3]:.3f}")

    names = list(scores.keys())
    ates = [scores[n][:, 1].mean() for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, ates, color=["C0", "C1", "C2", "C3"])
    ax.set_ylabel("Absolute ATE Error")
    ax.set_title("Synthetic Data — Test ATE Error by Model")
    for bar, v in zip(bars, ates):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("ate_error_synthetic.png", dpi=150)
    print("\nPlot saved to ate_error_synthetic.png")
    plt.show()
