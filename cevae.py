import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, kl_divergence


def mlp(in_dim, hidden_dims, out_dim, final_act=None):
    layers = []
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ELU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    if final_act:
        layers.append(final_act)
    return nn.Sequential(*layers)


def _trunk(in_dim, hidden_dims):
    """Hidden layers with ELU — no output projection."""
    layers = []
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ELU()]
        in_dim = h
    return nn.Sequential(*layers)


class _DualHead(nn.Module):
    """Shared hidden layer that branches into two output projections."""
    def __init__(self, in_dim, hidden_dim, out_dim, act_b=None):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ELU())
        self.head_a = nn.Linear(hidden_dim, out_dim)
        self.head_b = (nn.Sequential(nn.Linear(hidden_dim, out_dim), act_b)
                       if act_b else nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        h = self.shared(x)
        return self.head_a(h), self.head_b(h)


class CEVAE(nn.Module):
    def __init__(self, n_bin, n_cont, d=20, h=200, nh=3):
        super().__init__()
        self.n_bin = n_bin
        self.n_cont = n_cont
        x_dim = n_bin + n_cont

        # ---- Decoder: p(x, t, y | z) ----

        # p(x|z): shared (nh-1)-layer trunk → binary head + continuous (mu,sig) head
        self.p_x_trunk = _trunk(d, [h] * (nh - 1))
        self.p_x_bin_head = mlp(h, [h], n_bin) if n_bin > 0 else None
        self.p_x_cont_head = (_DualHead(h, h, n_cont, nn.Softplus())
                              if n_cont > 0 else None)

        self.p_t = mlp(d, [h], 1)                   # p(t|z)
        self.p_y0 = mlp(d, [h] * nh, 1)             # p(y|z, t=0)
        self.p_y1 = mlp(d, [h] * nh, 1)             # p(y|z, t=1)

        # ---- Encoder: q(z, t, y | x) ----

        self.q_t = mlp(x_dim, [d], 1)               # q(t|x)

        # q(y|x,t): shared trunk → per-arm heads
        self.q_y_trunk = _trunk(x_dim, [h] * (nh - 1))
        self.q_y_head0 = mlp(h, [h], 1)
        self.q_y_head1 = mlp(h, [h], 1)

        # q(z|x,y,t): shared trunk → per-arm (mu,sig) dual-heads
        self.q_z_trunk = _trunk(x_dim + 1, [h] * (nh - 1))
        self.q_z_head0 = _DualHead(h, h, d, nn.Softplus())
        self.q_z_head1 = _DualHead(h, h, d, nn.Softplus())

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x, t, y):
        qt = Bernoulli(logits=self.q_t(x))

        hqy = self.q_y_trunk(x)
        mu_y = t * self.q_y_head1(hqy) + (1 - t) * self.q_y_head0(hqy)
        qy = Normal(mu_y, 1.0)

        hqz = self.q_z_trunk(torch.cat([x, y], 1))
        mu_z0, sig_z0 = self.q_z_head0(hqz)
        mu_z1, sig_z1 = self.q_z_head1(hqz)
        mu_z = t * mu_z1 + (1 - t) * mu_z0
        sig_z = t * sig_z1 + (1 - t) * sig_z0
        qz = Normal(mu_z, sig_z)

        return qt, qy, qz

    def decode(self, z, t):
        hx = self.p_x_trunk(z)
        x_bin_logits = self.p_x_bin_head(hx) if self.p_x_bin_head else None
        if self.p_x_cont_head is not None:
            x_cont_mu, x_cont_sig = self.p_x_cont_head(hx)
        else:
            x_cont_mu, x_cont_sig = None, None

        t_logits = self.p_t(z)
        mu_y = t * self.p_y1(z) + (1 - t) * self.p_y0(z)

        return x_bin_logits, x_cont_mu, x_cont_sig, t_logits, mu_y
    
    def forward(self, x, t, y):
        qt, qy, qz = self.encode(x, t, y)
        z = qz.rsample()
        xb_lo, xc_mu, xc_sig, t_lo, y_mu = self.decode(z, t)

        # p(x,t,y|z)
        log_p = torch.zeros(x.shape[0], device=x.device)
        if self.n_cont > 0:
            x_cont = x[:, self.n_bin:]
            log_p = log_p + Normal(xc_mu, xc_sig).log_prob(x_cont).sum(1)
        if xb_lo is not None:
            x_bin = x[:, :self.n_bin]
            log_p = log_p + Bernoulli(logits=xb_lo).log_prob(x_bin).sum(1)
        log_p = (
            log_p
            + Bernoulli(logits=t_lo).log_prob(t).sum(1)
            + Normal(y_mu, 1.0).log_prob(y).sum(1)
        )

        # get the kl divergence
        kl = kl_divergence(qz, Normal(0.0, 1.0)).sum(1)
        # their additional term
        aux = qt.log_prob(t).sum(1) + qy.log_prob(y).sum(1)
        return -(log_p - kl + aux).mean() #finally :))

    @torch.no_grad()
    def log_p_valid(self, x, t, y):
        _, _, qz = self.encode(x, t, y)
        z = qz.mean
        xb_lo, xc_mu, xc_sig, t_lo, y_mu = self.decode(z, t)

        log_p = torch.zeros(x.shape[0], device=x.device)
        if self.n_cont > 0:
            x_cont = x[:, self.n_bin:]
            log_p = log_p + Normal(xc_mu, xc_sig).log_prob(x_cont).sum(1)
        if xb_lo is not None:
            x_bin = x[:, :self.n_bin]
            log_p = log_p + Bernoulli(logits=xb_lo).log_prob(x_bin).sum(1)
        log_p = (
            log_p
            + Bernoulli(logits=t_lo).log_prob(t).sum(1)
            + Normal(y_mu, 1.0).log_prob(y).sum(1)
        )
        pz = Normal(0.0, 1.0)
        return (log_p + pz.log_prob(z).sum(1) - qz.log_prob(z).sum(1)).mean().item()

    @torch.no_grad()
    def predict_y(self, x, n_samples=1):
        n = x.shape[0]
        t0 = torch.zeros(n, 1, device=x.device)
        t1 = torch.ones(n, 1, device=x.device)
        y0s, y1s = [], []
        for _ in range(n_samples):
            # Sample qt → qy → qz from the variational model (no observed y),
            # then decode with forced t=0 / t=1 using the SAME z.
            qt = Bernoulli(logits=self.q_t(x)).sample()

            hqy = self.q_y_trunk(x)
            mu_y = qt * self.q_y_head1(hqy) + (1 - qt) * self.q_y_head0(hqy)
            qy = Normal(mu_y, 1.0).sample()

            hqz = self.q_z_trunk(torch.cat([x, qy], 1))
            mu_z0, sig_z0 = self.q_z_head0(hqz)
            mu_z1, sig_z1 = self.q_z_head1(hqz)
            mu_z = qt * mu_z1 + (1 - qt) * mu_z0
            sig_z = qt * sig_z1 + (1 - qt) * sig_z0
            z = Normal(mu_z, sig_z).sample()

            y0s.append(self.decode(z, t0)[-1])
            y1s.append(self.decode(z, t1)[-1])
        return (
            torch.stack(y0s).mean(0).cpu().numpy(),
            torch.stack(y1s).mean(0).cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # High-level train / predict API
    # ------------------------------------------------------------------

    def fit(self, x_tr, t_tr, y_tr, x_va, t_va, y_va, *,
            epochs=100, lr=1e-3, wd=1e-4, check_every=10,
            batch_size=100, seed=1, verbose=True, eval_callback=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(seed)
        np.random.seed(seed)
        self._reset_parameters()
        self.to(device)

        self._ym, self._ys = float(y_tr.mean()), float(y_tr.std())
        ytr_n = (y_tr - self._ym) / self._ys
        yva_n = (y_va - self._ym) / self._ys

        xtr_t = torch.tensor(x_tr, dtype=torch.float32, device=device)
        ttr_t = torch.tensor(t_tr, dtype=torch.float32, device=device)
        ytr_t = torch.tensor(ytr_n, dtype=torch.float32, device=device)
        xva_t = torch.tensor(x_va, dtype=torch.float32, device=device)
        tva_t = torch.tensor(t_va, dtype=torch.float32, device=device)
        yva_t = torch.tensor(yva_n, dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        n = x_tr.shape[0]
        bs = min(batch_size, n)
        n_iter = max(1, 10 * (n // bs))
        idx = np.arange(n)
        best_logp = -np.inf
        best_state = None

        for epoch in range(epochs):
            self.train()
            np.random.shuffle(idx)
            for _ in range(n_iter):
                b = np.random.choice(idx, bs)
                loss = self.forward(xtr_t[b], ttr_t[b], ytr_t[b])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % check_every == 0 or epoch == epochs - 1:
                self.eval()
                logp = self.log_p_valid(xva_t, tva_t, yva_t)
                improved = logp >= best_logp
                if verbose:
                    tag = '*' if improved else ' '
                    print(f'[CEVAE] Epoch {epoch+1:3d}: '
                          f'val bound {logp:.3f} (best {best_logp:.3f}){tag}')
                if improved:
                    best_logp = logp
                    best_state = {k: v.clone()
                                  for k, v in self.state_dict().items()}

            if eval_callback and epoch % check_every == 0:
                self.eval()
                eval_callback(epoch + 1, self)

        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()

    def predict(self, x, y=None, n_samples=100):
        """Numpy in / numpy out.  Returns *(y0, y1)* on the original scale."""
        device = next(self.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        y0, y1 = self.predict_y(x_t, n_samples=n_samples)
        return y0 * self._ys + self._ym, y1 * self._ys + self._ym