"""Run CEVAE, TARNet, LR-1, and LR-2 on synthetic data at varying sample sizes.

Produces a plot of final test ATE error vs. sample size for all four models.
"""

import numpy as np
import matplotlib.pyplot as plt

from datasets import SyntheticDataset
from cevae_synthetic import run_experiment

SAMPLE_SIZES = [500, 1000, 2000, 5000, 10000]
SEED = 42
EPOCHS = 100

model_names = ['CEVAE', 'TARNet', 'LR-1', 'LR-2']
ate_results = {m: [] for m in model_names}

for n in SAMPLE_SIZES:
    print(f"\n{'='*60}")
    print(f"  Sample size n = {n}")
    print(f"{'='*60}")

    dataset = SyntheticDataset(n=n, seed=SEED, replications=1)
    scores = run_experiment(dataset, epochs=EPOCHS, verbose=False)

    for name in model_names:
        ate = scores[name][:, 1].mean()
        ate_results[name].append(ate)
        print(f"  {name:8s} test ATE error: {ate:.4f}")

# --- Plot ---
markers = {'CEVAE': 'o', 'TARNet': 'D', 'LR-1': 's', 'LR-2': '^'}
styles = {'CEVAE': '-', 'TARNet': '-', 'LR-1': '--', 'LR-2': '--'}

fig, ax = plt.subplots(figsize=(9, 5))
for name in model_names:
    ax.plot(SAMPLE_SIZES, ate_results[name],
            marker=markers[name], linestyle=styles[name], label=name)
ax.set_xlabel('Sample Size (n)')
ax.set_ylabel('Absolute ATE Error')
ax.set_title('Synthetic Data — Test ATE Error vs Sample Size')
ax.set_xscale('log')
ax.set_xticks(SAMPLE_SIZES)
ax.set_xticklabels([str(s) for s in SAMPLE_SIZES])
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('ate_vs_sample_size.png', dpi=150)
print(f'\nPlot saved to ate_vs_sample_size.png')
plt.show()
