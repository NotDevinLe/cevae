#!/usr/bin/env python
"""Simplified CEVAE on IHDP — PyTorch implementation."""

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, kl_divergence

from datasets import IHDP
from evaluation import Evaluator

import numpy as np
from scipy.stats import sem
from argparse import ArgumentParser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(in_dim, hidden_dims, out_dim, final_act=None):
    layers = []
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ELU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    if final_act:
        layers.append(final_act)
    return nn.Sequential(*layers)


def to_t(arr):
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


class CEVAE(nn.Module):
    def __init__(self, n_bin, n_cont, d=20, h=200, nh=3):
        super().__init__()
        x_dim = n_bin + n_cont
        self.n_bin = n_bin

        # p(x, t, y | z)
        self.p_x_bin = mlp(d, [h] * nh, n_bin) # p(x|z) for binary
        self.p_x_mu = mlp(d, [h] * nh, n_cont) #p(x|z) for continuous and gaussian
        self.p_x_sig = mlp(d, [h] * nh, n_cont, nn.Softplus()) # connects to above
        self.p_t = mlp(d, [h], 1) # p(t|z)
        self.p_y0 = mlp(d, [h] * nh, 1) # mu_t=0
        self.p_y1 = mlp(d, [h] * nh, 1) # mu_t=1

        # q(z, t, y | x)
        self.q_t = mlp(x_dim, [d], 1) # q(t|x)
        self.q_y0 = mlp(x_dim, [h] * nh, 1) # mu_i when t=0
        self.q_y1 = mlp(x_dim, [h] * nh, 1) # mu_i when t=1
        self.q_z_mu0 = mlp(x_dim + 1, [h] * nh, d) #g1
        self.q_z_mu1 = mlp(x_dim + 1, [h] * nh, d) # g2
        self.q_z_sig0 = mlp(x_dim + 1, [h] * nh, d, nn.Softplus()) #g3
        self.q_z_sig1 = mlp(x_dim + 1, [h] * nh, d, nn.Softplus()) #g1

    def encode(self, x, t, y):
        qt = Bernoulli(logits=self.q_t(x))
        mu_y = t * self.q_y1(x) + (1 - t) * self.q_y0(x)
        qy = Normal(mu_y, 1.0)

        xy = torch.cat([x, y], 1)
        mu_z = t * self.q_z_mu1(xy) + (1 - t) * self.q_z_mu0(xy)
        sig_z = t * self.q_z_sig1(xy) + (1 - t) * self.q_z_sig0(xy)
        qz = Normal(mu_z, sig_z)
        return qt, qy, qz

    def decode(self, z, t):
        x_bin_logits = self.p_x_bin(z)
        x_cont_mu = self.p_x_mu(z)
        x_cont_sig = self.p_x_sig(z)
        t_logits = self.p_t(z)
        mu_y = t * self.p_y1(z) + (1 - t) * self.p_y0(z)
        return x_bin_logits, x_cont_mu, x_cont_sig, t_logits, mu_y

    def forward(self, x, t, y):
        qt, qy, qz = self.encode(x, t, y)
        z = qz.rsample()
        xb_lo, xc_mu, xc_sig, t_lo, y_mu = self.decode(z, t)

        x_bin, x_cont = x[:, : self.n_bin], x[:, self.n_bin :]
        log_p = (
            Bernoulli(logits=xb_lo).log_prob(x_bin).sum(1)
            + Normal(xc_mu, xc_sig).log_prob(x_cont).sum(1)
            + Bernoulli(logits=t_lo).log_prob(t).sum(1)
            + Normal(y_mu, 1.0).log_prob(y).sum(1)
        )
        kl = kl_divergence(qz, Normal(0.0, 1.0)).sum(1)
        aux = qt.log_prob(t).sum(1) + qy.log_prob(y).sum(1)
        return -(log_p - kl + aux).mean()

    @torch.no_grad()
    def log_p_valid(self, x, t, y):
        _, _, qz = self.encode(x, t, y)
        z = qz.mean
        xb_lo, xc_mu, xc_sig, t_lo, y_mu = self.decode(z, t)

        x_bin, x_cont = x[:, : self.n_bin], x[:, self.n_bin :]
        log_p = (
            Bernoulli(logits=xb_lo).log_prob(x_bin).sum(1)
            + Normal(xc_mu, xc_sig).log_prob(x_cont).sum(1)
            + Bernoulli(logits=t_lo).log_prob(t).sum(1)
            + Normal(y_mu, 1.0).log_prob(y).sum(1)
        )
        pz = Normal(0.0, 1.0)
        return (log_p + pz.log_prob(z).sum(1) - qz.log_prob(z).sum(1)).mean().item()

    @torch.no_grad()
    def predict_y(self, x, y, n_samples=1):
        n = x.shape[0]
        t0 = torch.zeros(n, 1, device=x.device)
        t1 = torch.ones(n, 1, device=x.device)
        y0s, y1s = [], []
        for _ in range(n_samples):
            _, _, qz0 = self.encode(x, t0, y)
            _, _, qz1 = self.encode(x, t1, y)
            y0s.append(self.decode(qz0.rsample(), t0)[-1])
            y1s.append(self.decode(qz1.rsample(), t1)[-1])
        return (
            torch.stack(y0s).mean(0).cpu().numpy(),
            torch.stack(y1s).mean(0).cpu().numpy(),
        )


def main():
    parser = ArgumentParser()
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--early", type=int, default=10)
    parser.add_argument("--print_every", type=int, default=10)
    args = parser.parse_args()

    dataset = IHDP(replications=args.reps)
    scores_tr = np.zeros((args.reps, 3))
    scores_te = np.zeros((args.reps, 3))

    for i, (train, valid, test, contfeats, binfeats) in enumerate(
        dataset.get_train_valid_test()
    ):
        print(f"\n=== Replication {i + 1}/{args.reps} ===")
        (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
        (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
        (xte, tte, yte), (y_cfte, mu0te, mu1te) = test

        perm = binfeats + contfeats
        xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]
        n_bin, n_cont = len(binfeats), len(contfeats)

        xall = np.concatenate([xtr, xva])
        tall = np.concatenate([ttr, tva])
        yall = np.concatenate([ytr, yva])

        eval_tr = Evaluator(
            yall, tall,
            y_cf=np.concatenate([y_cftr, y_cfva]),
            mu0=np.concatenate([mu0tr, mu0va]),
            mu1=np.concatenate([mu1tr, mu1va]),
        )
        eval_te = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

        ym, ys = ytr.mean(), ytr.std()
        ytr_n, yva_n = (ytr - ym) / ys, (yva - ym) / ys

        torch.manual_seed(1)
        np.random.seed(1)

        model = CEVAE(n_bin, n_cont).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        xva_t, tva_t, yva_t = to_t(xva), to_t(tva), to_t(yva_n)
        xall_t = to_t(xall)
        yall_t = to_t((yall - ym) / ys)
        xte_t = to_t(xte)
        yte_t = to_t((yte - ym) / ys)

        best_logp, best_state = -np.inf, None
        n_iter = 10 * (xtr.shape[0] // 100)
        idx = np.arange(xtr.shape[0])

        for epoch in range(args.epochs):
            model.train()
            np.random.shuffle(idx)
            losses = []

            for _ in range(n_iter):
                b = np.random.choice(idx, 100)
                loss = model(to_t(xtr[b]), to_t(ttr[b]), to_t(ytr_n[b]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())

            if epoch % args.early == 0 or epoch == args.epochs - 1:
                model.eval()
                logp = model.log_p_valid(xva_t, tva_t, yva_t)
                if logp > best_logp:
                    print(f"  valid bound: {best_logp:.3f} -> {logp:.3f}")
                    best_logp = logp
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % args.print_every == 0:
                model.eval()
                y0, y1 = model.predict_y(xte_t, yte_t)
                y0, y1 = y0 * ys + ym, y1 * ys + ym
                s = eval_te.calc_stats(y1, y0)
                print(
                    f"  epoch {epoch + 1}: loss={np.mean(losses):.3f}  "
                    f"ite={s[0]:.3f} ate={s[1]:.3f} pehe={s[2]:.3f}"
                )

        if best_state:
            model.load_state_dict(best_state)
        model.to(DEVICE).eval()

        y0, y1 = model.predict_y(xall_t, yall_t, n_samples=100)
        y0, y1 = y0 * ys + ym, y1 * ys + ym
        scores_tr[i] = eval_tr.calc_stats(y1, y0)

        y0, y1 = model.predict_y(xte_t, yte_t, n_samples=100)
        y0, y1 = y0 * ys + ym, y1 * ys + ym
        scores_te[i] = eval_te.calc_stats(y1, y0)

    for name, sc in [("train", scores_tr), ("test", scores_te)]:
        m, s = sc.mean(0), sem(sc, 0)
        print(f"  {name}: ITE={m[0]:.3f}+-{s[0]:.3f}  ATE={m[1]:.3f}+-{s[1]:.3f}  PEHE={m[2]:.3f}+-{s[2]:.3f}")


if __name__ == "__main__":
    main()