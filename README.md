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
├── cevae_synthetic.py        # Run all four models on synthetic data
├── cevae_ihdp.py             # Run CEVAE on IHDP (10 replications)
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

Can be loaded from JSON or generated in-memory at any sample size:

```python
SyntheticDataset(path_data="datasets/synthetic_data.json")  # from file
SyntheticDataset(n=5000, seed=42)                            # generate on the fly
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

**Synthetic — all models, single run:**

```bash
python cevae_synthetic.py
```

Trains CEVAE, TARNet, LR-1, and LR-2, prints test ITE/ATE/PEHE/ATT, and saves a bar chart to `ate_error_synthetic.png`.

**Synthetic — sample-size sweep:**

```bash
python run_sample_size_experiments.py
```

Runs all four models at n = 500, 1000, 2000, 5000, 10000 and plots ATE error vs. sample size to `ate_vs_sample_size.png`.

**IHDP — CEVAE only:**

```bash
python cevae_ihdp.py
```

Runs CEVAE across 10 IHDP replications and reports aggregate ITE/ATE/PEHE.