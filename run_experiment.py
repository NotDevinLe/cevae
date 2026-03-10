"""Unified experiment runner for causal inference models.

Examples
--------
# Run CEVAE and TARNet on IHDP for 10 replications:
    python run_experiment.py --dataset ihdp --models cevae tarnet --replications 10

# Run all models on synthetic data:
    python run_experiment.py --dataset synthetic --models all

# Run only LR baselines on synthetic with 5 replications and 5000 samples:
    python run_experiment.py --dataset synthetic --models lr1 lr2 -n 5000 --replications 5

# Sample-size sweep (runs all selected models at each size):
    python run_experiment.py --dataset synthetic --models cevae tarnet --sweep-sizes 500 1000 5000 10000

# Proxy noise sweep:
    python run_experiment.py --dataset synthetic --models all --sweep-noise 0.0 0.1 0.2 0.3 0.4 0.45

# Single experiment with binary proxy model at a specific noise level:
    python run_experiment.py --dataset synthetic --models all --flip-prob 0.2 -n 5000

# Run all models on Jobs dataset:
    python run_experiment.py --dataset jobs --models all --replications 10

# Save a bar plot of ATE errors:
    python run_experiment.py --dataset ihdp --models all --save-plot results.png
"""

import argparse
import sys
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem

from cevae import CEVAE
from tarnet import TARNet
from lr import LR1, LR2
from datasets import IHDP, SyntheticDataset, Jobs
from evaluation import Evaluator, JobsEvaluator

ALL_MODELS = ["cevae", "tarnet", "lr1", "lr2"]

OUTCOME_TYPE = {
    "synthetic": "binary",
    "ihdp": "continuous",
    "jobs": "binary",
}

DISPLAY_NAMES = {
    "cevae": "CEVAE",
    "tarnet": "TARNet",
    "lr1": "LR-1",
    "lr2": "LR-2",
}


def build_models(names, n_bin, n_cont, outcome):
    """Instantiate only the requested models."""
    factories = {
        "cevae": lambda: CEVAE(n_bin, n_cont),
        "tarnet": lambda: TARNet(input_dim=n_bin + n_cont),
        "lr1": lambda: LR1(outcome=outcome),
        "lr2": lambda: LR2(outcome=outcome),
    }
    return OrderedDict((n, factories[n]()) for n in names)


def run(dataset, model_names, outcome, *, epochs=100, lr=1e-3, wd=1e-4,
        batch_size=100, verbose=True):
    """Train and evaluate *model_names* on *dataset*.

    Returns ``{name: {'train': np.array(R,4), 'test': np.array(R,4)}}``
    where the four columns are (ITE, ATE, PEHE, ATT).
    """
    replications = dataset.replications
    scores = {
        m: {
            "train": np.zeros((replications, 4)),
            "test":  np.zeros((replications, 4)),
        }
        for m in model_names
    }

    for i, (train, val, test, contfeats, binfeats) in enumerate(
        dataset.get_train_valid_test()
    ):
        if verbose:
            print(f"\n--- Replication {i + 1}/{replications} ---")

        (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
        (xva, tva, yva), (y_cfva, mu0va, mu1va) = val
        (xte, tte, yte), (y_cfte, mu0te, mu1te) = test

        perm = binfeats + contfeats
        xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]
        n_bin, n_cont = len(binfeats), len(contfeats)

        xall = np.concatenate([xtr, xva])
        tall = np.concatenate([ttr, tva])
        yall = np.concatenate([ytr, yva])

        y_cf_all = np.concatenate([y_cftr, y_cfva])
        mu0_all = np.concatenate([mu0tr, mu0va])
        mu1_all = np.concatenate([mu1tr, mu1va])

        eval_tr = Evaluator(yall, tall, y_cf=y_cf_all, mu0=mu0_all, mu1=mu1_all)
        eval_te = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

        models = build_models(model_names, n_bin, n_cont, outcome)

        for name, model in models.items():
            if name in ("lr1", "lr2"):
                model.fit(xall, tall, yall)
            else:
                model.fit(xtr, ttr, ytr, xva, tva, yva,
                          epochs=epochs, lr=lr, wd=wd,
                          batch_size=batch_size, verbose=verbose)

        for name, model in models.items():
            y0_te, y1_te = model.predict(xte, y=yte)
            scores[name]["test"][i, :] = eval_te.calc_stats(y1_te, y0_te)

            y0_tr, y1_tr = model.predict(xall, y=yall)
            scores[name]["train"][i, :] = eval_tr.calc_stats(y1_tr, y0_tr)

            if verbose:
                s = scores[name]["test"][i]
                print(f"[{name:8s}] ITE: {s[0]:.3f}  ATE: {s[1]:.3f}  "
                      f"PEHE: {s[2]:.3f}  ATT: {s[3]:.3f}")

    return scores


def run_jobs(dataset, model_names, *, epochs=100, lr=1e-3, wd=1e-4,
             batch_size=100, verbose=True):
    """Train and evaluate *model_names* on the Jobs dataset.

    Returns ``{name: {'train': np.array(R,3), 'test': np.array(R,3)}}``
    where the three columns are (R_pol, ATE_err, ATT_err).
    """
    replications = dataset.replications
    true_ate = dataset.true_ate
    scores = {
        m: {
            "train": np.zeros((replications, 3)),
            "test":  np.zeros((replications, 3)),
        }
        for m in model_names
    }

    for i, (train, valid, test, contfeats, binfeats) in enumerate(
        dataset.get_train_valid_test()
    ):
        if verbose:
            print(f"\n--- Replication {i + 1}/{replications} ---")

        xtr, ttr, ytr, etr = train
        xva, tva, yva, eva = valid
        xte, tte, yte, ete = test

        perm = binfeats + contfeats
        xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]
        n_bin, n_cont = len(binfeats), len(contfeats)

        xall = np.concatenate([xtr, xva])
        tall = np.concatenate([ttr, tva])
        yall = np.concatenate([ytr, yva])
        eall = np.concatenate([etr, eva])

        eval_tr = JobsEvaluator(yall, tall, eall, true_ate)
        eval_te = JobsEvaluator(yte, tte, ete, true_ate)

        models = build_models(model_names, n_bin, n_cont, "binary")

        for name, model in models.items():
            if name in ("lr1", "lr2"):
                model.fit(xall, tall, yall)
            else:
                model.fit(xtr, ttr, ytr, xva, tva, yva,
                          epochs=epochs, lr=lr, wd=wd,
                          batch_size=batch_size, verbose=verbose)

        for name, model in models.items():
            y0_te, y1_te = model.predict(xte, y=yte)
            scores[name]["test"][i, :] = eval_te.calc_stats(y1_te, y0_te)

            y0_tr, y1_tr = model.predict(xall, y=yall)
            scores[name]["train"][i, :] = eval_tr.calc_stats(y1_tr, y0_tr)

            if verbose:
                s = scores[name]["test"][i]
                print(f"[{name:8s}] R_pol: {s[0]:.3f}  "
                      f"ATE: {s[1]:.3f}  ATT: {s[2]:.3f}")

    return scores


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------

def _fmt(arr, col):
    """Format mean +/- SEM for a specific metric column."""
    m = arr[:, col].mean()
    s = sem(arr[:, col]) if arr.shape[0] > 1 else 0.0
    return f"{m:.3f}\u00b1{s:.3f}"


def print_table(scores):
    """Paper-style table: in-sample and out-of-sample epsilon_ATE and sqrt(PEHE)."""
    col_w = 14
    name_w = 10
    header = (f"{'Method':<{name_w}s} | "
              f"{'In-sample':^{2 * col_w + 1}s} | "
              f"{'Out-of-sample':^{2 * col_w + 1}s}")
    sub = (f"{'':>{name_w}s} | "
           f"{'e_ATE':>{col_w}s} {'sqrtPEHE':>{col_w}s} | "
           f"{'e_ATE':>{col_w}s} {'sqrtPEHE':>{col_w}s}")
    width = len(header)
    sep = "-" * width

    print(f"\n{sep}")
    print(header)
    print(sub)
    print(sep)
    for name, d in scores.items():
        tr, te = d["train"], d["test"]
        label = DISPLAY_NAMES.get(name, name)
        print(f"{label:<{name_w}s} | "
              f"{_fmt(tr, 1):>{col_w}s} {_fmt(tr, 2):>{col_w}s} | "
              f"{_fmt(te, 1):>{col_w}s} {_fmt(te, 2):>{col_w}s}")
    print(sep)


def print_summary(scores):
    """Compact summary (test set only)."""
    print("\n====== Final Test Scores ======")
    cols = ("ITE", "ATE", "PEHE", "ATT")
    for name, d in scores.items():
        arr = d["test"]
        means = arr.mean(axis=0)
        stds = sem(arr, axis=0) if arr.shape[0] > 1 else np.zeros(4)
        parts = [f"{c}: {m:.3f}\u00b1{s:.3f}" for c, m, s in zip(cols, means, stds)]
        label = DISPLAY_NAMES.get(name, name)
        print(f"  {label:8s}  {',  '.join(parts)}")


def print_jobs_table(scores):
    """Paper-style table for Jobs: R_pol, e_ATE, e_ATT."""
    col_w = 14
    name_w = 10
    header = (f"{'Method':<{name_w}s} | "
              f"{'In-sample':^{3 * col_w + 2}s} | "
              f"{'Out-of-sample':^{3 * col_w + 2}s}")
    sub = (f"{'':>{name_w}s} | "
           f"{'R_pol':>{col_w}s} {'e_ATE':>{col_w}s} {'e_ATT':>{col_w}s} | "
           f"{'R_pol':>{col_w}s} {'e_ATE':>{col_w}s} {'e_ATT':>{col_w}s}")
    width = len(header)
    sep = "-" * width

    print(f"\n{sep}")
    print(header)
    print(sub)
    print(sep)
    for name, d in scores.items():
        tr, te = d["train"], d["test"]
        label = DISPLAY_NAMES.get(name, name)
        print(f"{label:<{name_w}s} | "
              f"{_fmt(tr, 0):>{col_w}s} {_fmt(tr, 1):>{col_w}s} "
              f"{_fmt(tr, 2):>{col_w}s} | "
              f"{_fmt(te, 0):>{col_w}s} {_fmt(te, 1):>{col_w}s} "
              f"{_fmt(te, 2):>{col_w}s}")
    print(sep)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def save_ate_bar(scores, path, title="ATE Error by Model"):
    names = list(scores.keys())
    ates = [scores[n]["test"][:, 1].mean() for n in names]
    labels = [DISPLAY_NAMES.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, ates, color=[f"C{i}" for i in range(len(names))])
    ax.set_ylabel("Absolute ATE Error")
    ax.set_title(title)
    for bar, v in zip(bars, ates):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {path}")


def save_sweep_plot(sweep_results, x_values, path, *,
                    xlabel="X", ylabel="Absolute ATE Error",
                    title="Test ATE Error", xlog=False):
    """Generic line-plot for sweeps (sample-size or noise)."""
    markers = {"cevae": "o", "tarnet": "D", "lr1": "s", "lr2": "^"}
    styles  = {"cevae": "-", "tarnet": "-", "lr1": "--", "lr2": "--"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, values in sweep_results.items():
        label = DISPLAY_NAMES.get(name, name)
        ax.plot(x_values, values,
                marker=markers.get(name, "x"),
                linestyle=styles.get(name, "-"),
                label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlog:
        ax.set_xscale("log")
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(s) for s in x_values])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run causal inference experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--dataset", choices=["synthetic", "ihdp", "jobs"],
                   default="synthetic",
                   help="Which dataset to use (default: synthetic)")
    p.add_argument("--models", nargs="+", default=["all"],
                   choices=ALL_MODELS + ["all"],
                   help="Which models to train (default: all)")

    g = p.add_argument_group("hyperparameters")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--wd", type=float, default=1e-4)
    g.add_argument("--batch-size", type=int, default=100)
    g.add_argument("--replications", type=int, default=None,
                   help="Number of replications (default: 1 for synthetic, 10 for IHDP)")

    g2 = p.add_argument_group("synthetic dataset options")
    g2.add_argument("-n", type=int, default=None,
                    help="Sample size for synthetic data (default: use datasets/synthetic_data.json)")
    g2.add_argument("--seed", type=int, default=42)
    g2.add_argument("--flip-prob", type=float, default=None,
                    help="Use binary proxy model with this noise/flip probability")
    g2.add_argument("--n-proxies", type=int, default=5,
                    help="Number of binary proxy variables (default: 5)")

    g3 = p.add_argument_group("IHDP dataset options")
    g3.add_argument("--ihdp-path", default="datasets/IHDP/csv",
                    help="Path to IHDP CSV folder")

    g3b = p.add_argument_group("Jobs dataset options")
    g3b.add_argument("--jobs-path", default="datasets/jobs",
                     help="Path to Jobs NPZ folder")

    g4 = p.add_argument_group("sample-size sweep (synthetic only)")
    g4.add_argument("--sweep-sizes", nargs="+", type=int, default=None,
                    metavar="N",
                    help="Run a sample-size sweep instead of a single experiment")

    g5 = p.add_argument_group("proxy noise sweep (synthetic only)")
    g5.add_argument("--sweep-noise", nargs="+", type=float, default=None,
                    metavar="P",
                    help="Sweep proxy flip probabilities (e.g. 0.0 0.1 0.2 0.3 0.4 0.45)")

    p.add_argument("--save-plot", default=None, metavar="PATH",
                   help="Save an ATE bar/line plot to PATH")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-epoch output")

    args = p.parse_args(argv)

    if "all" in args.models:
        args.models = list(ALL_MODELS)
    args.models = [m.lower().replace("-", "") for m in args.models]

    if args.replications is None:
        args.replications = 10 if args.dataset in ("ihdp", "jobs") else 1

    return args


def make_dataset(args, n_override=None):
    n = n_override or args.n
    if args.dataset == "synthetic":
        flip_prob = getattr(args, "flip_prob", None)
        n_proxies = getattr(args, "n_proxies", 5)
        return SyntheticDataset(n=n, seed=args.seed,
                                replications=args.replications,
                                flip_prob=flip_prob, n_proxies=n_proxies)
    elif args.dataset == "jobs":
        return Jobs(path_data=args.jobs_path,
                    replications=args.replications)
    else:
        return IHDP(path_data=args.ihdp_path, replications=args.replications)


def main(argv=None):
    args = parse_args(argv)
    outcome = OUTCOME_TYPE[args.dataset]
    verbose = not args.quiet

    if args.sweep_sizes and args.sweep_noise:
        sys.exit("Error: --sweep-sizes and --sweep-noise cannot be used together")

    # --- Sample-size sweep ---
    if args.sweep_sizes:
        if args.dataset != "synthetic":
            sys.exit("Error: --sweep-sizes only works with --dataset synthetic")

        sweep_results = {m: [] for m in args.models}
        for n in args.sweep_sizes:
            print(f"\n{'=' * 60}")
            print(f"  Sample size n = {n}")
            print(f"{'=' * 60}")
            ds = SyntheticDataset(n=n, seed=args.seed,
                                  replications=args.replications)
            scores = run(ds, args.models, outcome,
                         epochs=args.epochs, lr=args.lr, wd=args.wd,
                         batch_size=args.batch_size, verbose=verbose)
            for name in args.models:
                ate = scores[name]["test"][:, 1].mean()
                sweep_results[name].append(ate)
                print(f"  {name:8s} test ATE error: {ate:.4f}")

        path = args.save_plot or "ate_vs_sample_size.png"
        save_sweep_plot(sweep_results, args.sweep_sizes, path,
                        xlabel="Sample Size (n)",
                        title="Synthetic Data \u2014 Test ATE Error vs Sample Size",
                        xlog=True)
        return

    # --- Noise sweep ---
    if args.sweep_noise:
        if args.dataset != "synthetic":
            sys.exit("Error: --sweep-noise only works with --dataset synthetic")

        n = args.n or 10000
        n_proxies = args.n_proxies
        sweep_results = {m: [] for m in args.models}

        for noise in args.sweep_noise:
            print(f"\n{'=' * 60}")
            print(f"  Proxy noise (flip_prob) = {noise}")
            print(f"{'=' * 60}")
            ds = SyntheticDataset(n=n, seed=args.seed,
                                  replications=args.replications,
                                  flip_prob=noise, n_proxies=n_proxies)
            scores = run(ds, args.models, outcome,
                         epochs=args.epochs, lr=args.lr, wd=args.wd,
                         batch_size=args.batch_size, verbose=verbose)
            for name in args.models:
                ate = scores[name]["test"][:, 1].mean()
                sweep_results[name].append(ate)
                print(f"  {name:8s} test ATE error: {ate:.4f}")

        path = args.save_plot or "ate_vs_noise.png"
        save_sweep_plot(sweep_results, args.sweep_noise, path,
                        xlabel="Proxy Noise (flip probability)",
                        title="Synthetic Data \u2014 ATE Error vs Proxy Noise")
        return

    # --- Standard single experiment ---
    ds = make_dataset(args)

    if args.dataset == "jobs":
        scores = run_jobs(ds, args.models,
                          epochs=args.epochs, lr=args.lr, wd=args.wd,
                          batch_size=args.batch_size, verbose=verbose)
        print_jobs_table(scores)
    else:
        scores = run(ds, args.models, outcome,
                     epochs=args.epochs, lr=args.lr, wd=args.wd,
                     batch_size=args.batch_size, verbose=verbose)
        print_table(scores)

    if args.save_plot:
        title = f"{args.dataset.upper()} \u2014 ATE Error by Model"
        save_ate_bar(scores, args.save_plot, title=title)


if __name__ == "__main__":
    main()
