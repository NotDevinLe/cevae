# CEVAE — Causal Effect Variational Autoencoder

A PyTorch implementation of the **Causal Effect Variational Autoencoder** (Louizos et al., 2017) with **TARNet** and **logistic-regression baselines** (LR-1 / LR-2), evaluated on synthetic and semi-synthetic causal inference benchmarks.

## Project Structure

```
├── cevae.py                  # CEVAE model (encoder + decoder + fit/predict)
├── tarnet.py                 # TARNet model (shared repr + two heads + fit/predict)
├── lr.py                     # LR-1 (S-learner) and LR-2 (T-learner) baselines
├── evaluation.py             # Evaluator — ITE, ATE, PEHE, ATT metrics
├── datasets.py               # Dataset loaders: IHDP, SyntheticDataset, TwinsDataset
│
├── run_experiment.py         # ★ Unified CLI runner — pick models, dataset, hyperparams
├── cevae_synthetic.py        # Thin wrapper (backward compat)
├── cevae_ihdp.py             # Thin wrapper (backward compat)
├── run_sample_size_experiments.py  # Sweep sample sizes on synthetic data
│
└── datasets/
    ├── create_synthetic.py   # Generate synthetic_data.json
    ├── synthetic_data.json   # Pre-generated synthetic dataset (n=10 000)
    ├── IHDP/                 # IHDP semi-synthetic benchmark (10 CSV splits)
    └── TWINS/                # Twins birth-weight dataset
```

## Models

### CEVAE (`cevae.py`)

Variational autoencoder that models a latent confounder **z**. The generative model factorises as p(z) p(x|z) p(t|z) p(y|z,t), and an inference network q(z|x,t,y) is trained by maximising a modified ELBO. Because the encoder recovers z, CEVAE can estimate treatment effects even when the confounder is hidden from the observed covariates.

### TARNet (`tarnet.py`)

Treatment-Agnostic Representation Network. A shared representation network maps covariates x into a latent space, then two separate heads predict potential outcomes y(0) and y(1). Trained with the factual MSE loss: only the head matching the observed treatment receives gradient.

### LR-1 / LR-2 (`lr.py`)

Simple logistic-regression baselines from the paper (called OLS-1 / OLS-2 when the outcome is continuous):

- **LR-1 (S-learner):** One model trained on [X, T] → Y. Predicts potential outcomes by setting T=0 or T=1 at inference.
- **LR-2 (T-learner):** Two separate models, one per treatment arm. Each predicts Y from X for its group.

## Datasets

### Synthetic (`datasets.py → SyntheticDataset`)

A binary-confounder DGP where the hidden variable z drives both treatment and outcome:

- z ~ Bernoulli(0.5)
- x ~ Normal(z, 5z + 3(1−z)), cast to int
- t ~ Bernoulli(0.75z + 0.25(1−z))
- y ~ Bernoulli(σ(3(z + 2(2t−1))))

Can be loaded from JSON or generated in-memory at any sample size.
Supports a **binary proxy noise model** for ablation studies: each proxy
is a noisy copy of z, flipped with probability `flip_prob`.

```python
SyntheticDataset(path_data="datasets/synthetic_data.json")  # from file (continuous proxy)
SyntheticDataset(n=5000, seed=42)                            # generate on the fly
SyntheticDataset(n=5000, flip_prob=0.2, n_proxies=5)         # 5 binary proxies, 20% noise
```

### IHDP (`datasets.py → IHDP`)

The Infant Health and Development Program semi-synthetic benchmark (Hill, 2011). 747 subjects, 25 covariates (6 continuous, 19 binary), continuous outcomes, 10 dataset replications.

### Twins (`datasets.py → TwinsDataset`)

Twin births dataset (Louizos et al., 2017 §4.3). 11 984 twin pairs (both < 2 kg), binary mortality outcome. GESTAT10 is held out as a hidden confounder; noisy one-hot proxies are appended as observed features. Treatment is assigned via a logistic model on the covariates and confounder.

## Evaluation Metrics (`evaluation.py`)

All metrics compare predicted potential outcomes (ŷ₀, ŷ₁) against ground truth (μ₀, μ₁):

| Metric | Description |
|--------|-------------|
| **ITE** | RMSE of individual treatment effects |
| **ATE** | Absolute error of the average treatment effect |
| **PEHE** | Precision in estimation of heterogeneous effects |
| **ATT** | Absolute error of the average treatment effect on the treated |

## Uniform Model Interface

Every model exposes the same two methods, keeping experiment code minimal:

```python
model.fit(x_tr, t_tr, y_tr, x_va, t_va, y_va, epochs=100, lr=1e-3, wd=1e-4)
y0, y1 = model.predict(x_test, y=y_test)   # y= needed for CEVAE, ignored by others
```

LR models use a simpler fit signature (no validation split, no epochs):

```python
model.fit(x, t, y)
```

## Running Experiments

All experiments go through **`run_experiment.py`**, which lets you pick the
dataset, models, and hyperparameters from the command line.

### Experiment 1 — Synthetic noise sweep

Sweep over proxy noise levels to show CEVAE's robustness to degraded
proxy variables.  Produces a line plot of ATE error vs flip probability.

```bash
python run_experiment.py --dataset synthetic --models all \
    --sweep-noise 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 \
    -n 10000 --n-proxies 5 --quiet
```

### Experiment 2 — IHDP benchmark

Run all models on IHDP across 10 replications.  Prints a paper-style
table with in-sample and out-of-sample ε_ATE and √PEHE.

```bash
python run_experiment.py --dataset ihdp --models all --replications 10
```

### More examples

```bash
# See all options
python run_experiment.py --help

# Run only CEVAE and TARNet on IHDP
python run_experiment.py --dataset ihdp --models cevae tarnet --replications 10

# Run all four models on synthetic data
python run_experiment.py --dataset synthetic --models all

# Single experiment with binary proxies at a specific noise level
python run_experiment.py --dataset synthetic --models all --flip-prob 0.2 -n 5000

# Sample-size sweep
python run_experiment.py --dataset synthetic --models cevae tarnet --sweep-sizes 500 1000 5000 10000

# Save a bar chart
python run_experiment.py --dataset ihdp --models all --save-plot results.png

# Suppress per-epoch output
python run_experiment.py --dataset ihdp --models cevae --quiet
```

### Key flags

| Flag | Description |
|------|-------------|
| `--dataset {synthetic,ihdp}` | Which benchmark to use |
| `--models {cevae,tarnet,lr1,lr2,all}` | One or more models (space-separated) |
| `--epochs`, `--lr`, `--wd`, `--batch-size` | Training hyperparameters |
| `--replications` | Number of data splits (default: 1 synthetic, 10 IHDP) |
| `-n` | Synthetic sample size (omit to use `datasets/synthetic_data.json`) |
| `--flip-prob P` | Use binary proxy model with this noise/flip probability |
| `--n-proxies N` | Number of binary proxy variables (default: 5) |
| `--sweep-sizes N [N ...]` | Sweep over sample sizes |
| `--sweep-noise P [P ...]` | Sweep over proxy flip probabilities |
| `--save-plot PATH` | Save an ATE bar chart (or sweep line plot) |
| `--quiet` | Suppress per-epoch training logs |

### Legacy scripts (still work)

The old entry points delegate to `run_experiment.py` and are kept for
backward compatibility:

```bash
python cevae_synthetic.py          # same as --dataset synthetic --models all
python cevae_ihdp.py               # same as --dataset ihdp --models cevae --replications 10
python run_sample_size_experiments.py  # sample-size sweep, all models
```