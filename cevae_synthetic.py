import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from cevae import CEVAE
from tarnet import TARNet
from datasets import SyntheticDataset
from evaluation import Evaluator
from lr import LR1, LR2


def run_experiment(dataset, epochs=100, learning_rate=1e-3, wd=1e-4,
                   verbose=True):
    """Run CEVAE, TARNet, LR-1, and LR-2 on *dataset*.

    Returns ``{model_name: np.array(replications, 4)}`` where the four
    columns are (ITE, ATE, PEHE, ATT) on the test set.
    """
    replications = dataset.replications
    model_names = ['CEVAE', 'TARNet', 'LR-1', 'LR-2']
    scores = {m: np.zeros((replications, 4)) for m in model_names}

    for i, (train, val, test, contfeats, binfeats) in enumerate(
        dataset.get_train_valid_test()
    ):
        if verbose:
            print(f"\n--- Replication {i + 1}/{replications} ---")

        (xtr, ttr, ytr), (_, mu0tr, mu1tr) = train
        (xva, tva, yva), (_, mu0va, mu1va) = val
        (xte, tte, yte), (y_cfte, mu0te, mu1te) = test

        perm = binfeats + contfeats
        xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]
        n_bin, n_cont = len(binfeats), len(contfeats)

        xall = np.concatenate([xtr, xva])
        tall = np.concatenate([ttr, tva])
        yall = np.concatenate([ytr, yva])

        eval_te = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

        # -- Build models --
        models = OrderedDict([
            ('CEVAE',  CEVAE(n_bin, n_cont)),
            ('TARNet', TARNet(input_dim=n_bin + n_cont)),
            ('LR-1',   LR1(outcome="binary")),
            ('LR-2',   LR2(outcome="binary")),
        ])

        # -- Fit --
        models['LR-1'].fit(xall, tall, yall)
        models['LR-2'].fit(xall, tall, yall)
        models['TARNet'].fit(xtr, ttr, ytr, xva, tva, yva,
                             epochs=epochs, lr=learning_rate, wd=wd,
                             verbose=verbose)
        models['CEVAE'].fit(xtr, ttr, ytr, xva, tva, yva,
                            epochs=epochs, lr=learning_rate, wd=wd,
                            verbose=verbose)

        # -- Evaluate --
        for name, model in models.items():
            y0, y1 = model.predict(xte, y=yte)
            scores[name][i, :] = eval_te.calc_stats(y1, y0)
            if verbose:
                s = scores[name][i]
                print(f"[{name:8s}] ITE: {s[0]:.3f}  ATE: {s[1]:.3f}  "
                      f"PEHE: {s[2]:.3f}  ATT: {s[3]:.3f}")

    return scores


# ------------------------------------------------------------------
# Standalone run
# ------------------------------------------------------------------
if __name__ == '__main__':
    dataset = SyntheticDataset(replications=1)
    scores = run_experiment(dataset)

    print('\n====== Final Test Scores ======')
    for name, arr in scores.items():
        m = arr.mean(axis=0)
        print(f'{name:8s}  ITE: {m[0]:.3f}  ATE: {m[1]:.3f}  '
              f'PEHE: {m[2]:.3f}  ATT: {m[3]:.3f}')

    names = list(scores.keys())
    ates = [scores[n][:, 1].mean() for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, ates, color=['C0', 'C1', 'C2', 'C3'])
    ax.set_ylabel('Absolute ATE Error')
    ax.set_title('Synthetic Data — Test ATE Error by Model')
    for bar, v in zip(bars, ates):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('ate_error_synthetic.png', dpi=150)
    print('\nPlot saved to ate_error_synthetic.png')
    plt.show()
