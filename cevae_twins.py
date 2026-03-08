import torch
import numpy as np
from scipy.stats import sem

from cevae import CEVAE
from datasets import TwinsDataset
from evaluation import Evaluator

replications = 10
epochs = 100
lr = 1e-3
wd = 1e-4
early = 10
print_every = 10

dataset = TwinsDataset(replications=replications)

scores_test = np.zeros((replications, 4))
scores_train = np.zeros((replications, 4))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i, (train, val, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
    print(f"\nReplication {i + 1}/{replications}")

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

    ym, ys = ytr.mean(), ytr.std()
    ytr, yva = (ytr - ym) / ys, (yva - ym) / ys

    best_logp_valid = -np.inf
    best_state = None

    model = CEVAE(n_bin, n_cont).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    xtr_t = torch.tensor(xtr, dtype=torch.float32, device=device)
    ttr_t = torch.tensor(ttr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    xva_t = torch.tensor(xva, dtype=torch.float32, device=device)
    tva_t = torch.tensor(tva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)

    xall_t = torch.tensor(xall, dtype=torch.float32, device=device)
    xte_t = torch.tensor(xte, dtype=torch.float32, device=device)
    yall_norm_t = torch.tensor((yall - ym) / ys, dtype=torch.float32, device=device)
    yte_norm_t = torch.tensor((yte - ym) / ys, dtype=torch.float32, device=device)

    n_iter_per_epoch = 10 * int(xtr.shape[0] / 100)
    idx = np.arange(xtr.shape[0])

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(idx)
        avg_loss = 0.0

        for j in range(n_iter_per_epoch):
            batch = np.random.choice(idx, 100)
            xb = xtr_t[batch]
            tb = ttr_t[batch]
            yb = ytr_t[batch]

            loss = model(xb, tb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        avg_loss /= n_iter_per_epoch

        if epoch % early == 0 or epoch == epochs - 1:
            model.eval()
            logp_valid = model.log_p_valid(xva_t, tva_t, yva_t)
            if logp_valid >= best_logp_valid:
                print(f'Improved validation bound, old: {best_logp_valid:.3f}, new: {logp_valid:.3f}')
                best_logp_valid = logp_valid
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % print_every == 0:
            model.eval()
            y0, y1 = model.predict_y(xall_t, yall_norm_t)
            y0, y1 = y0 * ys + ym, y1 * ys + ym
            score_tr = eval_tr.calc_stats(y1, y0)

            y0t, y1t = model.predict_y(xte_t, yte_norm_t)
            y0t, y1t = y0t * ys + ym, y1t * ys + ym
            score_te = eval_te.calc_stats(y1t, y0t)

            print(f"Epoch: {epoch + 1}/{epochs}, log p(x) >= {avg_loss:.3f}, "
                  f"ite_tr: {score_tr[0]:.3f}, ate_tr: {score_tr[1]:.3f}, pehe_tr: {score_tr[2]:.3f}, att_tr: {score_tr[3]:.3f}, "
                  f"ite_te: {score_te[0]:.3f}, ate_te: {score_te[1]:.3f}, pehe_te: {score_te[2]:.3f}, att_te: {score_te[3]:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    y0, y1 = model.predict_y(xall_t, yall_norm_t, n_samples=100)
    y0, y1 = y0 * ys + ym, y1 * ys + ym
    score = eval_tr.calc_stats(y1, y0)
    scores_train[i, :] = score

    y0t, y1t = model.predict_y(xte_t, yte_norm_t, n_samples=100)
    y0t, y1t = y0t * ys + ym, y1t * ys + ym
    score_test = eval_te.calc_stats(y1t, y0t)
    scores_test[i, :] = score_test

    print(f'Replication: {i + 1}/{replications}, tr_ite: {score[0]:.3f}, tr_ate: {score[1]:.3f}, '
          f'tr_pehe: {score[2]:.3f}, te_ite: {score_test[0]:.3f}, te_ate: {score_test[1]:.3f}, '
          f'te_pehe: {score_test[2]:.3f}')

print('\nCEVAE model total scores')
means, stds = np.mean(scores_train, axis=0), sem(scores_train, axis=0)
print(f'train ITE: {means[0]:.3f}+-{stds[0]:.3f}, train ATE: {means[1]:.3f}+-{stds[1]:.3f}, '
      f'train PEHE: {means[2]:.3f}+-{stds[2]:.3f}')

means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
print(f'test ITE: {means[0]:.3f}+-{stds[0]:.3f}, test ATE: {means[1]:.3f}+-{stds[1]:.3f}, '
      f'test PEHE: {means[2]:.3f}+-{stds[2]:.3f}')
