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

class CEVAE(nn.Module):
    def __init__(self, n_bin, n_cont, d=20, h=200, nh=3):
        super().__init__()
        self.n_bin = n_bin # num binary dimensions
        self.n_cont = n_cont # num cont dimensions
        x_dim = n_bin + n_cont

        # p(x, t, y | z)
        self.p_x_bin = mlp(d, [h] * nh, n_bin) if n_bin > 0 else None
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
        self.q_z_sig1 = mlp(x_dim + 1, [h] * nh, d, nn.Softplus()) #g4

    def encode(self, x, t, y):
        # first we have to get the treaatment based on x
        qt = Bernoulli(logits=self.q_t(x))
        # now we get the outcome or rather q(y|x,t)
        mu_y = t * self.q_y1(x) + (1 - t) * self.q_y0(x)
        qy = Normal(mu_y, 1.0) # assume 1 std

        xy = torch.cat([x,y], 1)
        mu_z = t * self.q_z_mu1(xy) + (1 - t) * self.q_z_mu0(xy)
        sig_z = t * self.q_z_sig1(xy) + (1 - t) * self.q_z_sig0(xy)

        qz = Normal(mu_z, sig_z)
        return qt, qy, qz

    def decode(self, z, t):
        x_bin_logits = self.p_x_bin(z) if self.p_x_bin is not None else None
        x_cont_mu = self.p_x_mu(z)
        x_cont_sig = self.p_x_sig(z)

        t_logits = self.p_t(z)
        mu_y = t * self.p_y1(z) + (1 - t) * self.p_y0(z)

        return x_bin_logits, x_cont_mu, x_cont_sig, t_logits, mu_y
    
    def forward(self, x, t, y):
        qt, qy, qz = self.encode(x, t, y)
        z = qz.sample()  # sample from the normal distribution that we found in our encoder
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

    # ------------------------------------------------------------------
    # High-level train / predict API
    # ------------------------------------------------------------------

    def fit(self, x_tr, t_tr, y_tr, x_va, t_va, y_va, *,
            epochs=100, lr=1e-3, wd=1e-4, early=10,
            batch_size=100, verbose=True, eval_callback=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

            if epoch % early == 0 or epoch == epochs - 1:
                self.eval()
                logp = self.log_p_valid(xva_t, tva_t, yva_t)
                if logp >= best_logp:
                    if verbose:
                        print(f'[CEVAE] Epoch {epoch+1}: val bound '
                              f'{best_logp:.3f} -> {logp:.3f}')
                    best_logp = logp
                    best_state = {k: v.clone()
                                  for k, v in self.state_dict().items()}

            if eval_callback and epoch % early == 0:
                self.eval()
                eval_callback(epoch + 1, self)

        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()

    def predict(self, x, y=None, n_samples=100):
        """Numpy in / numpy out.  Returns *(y0, y1)* on the original scale."""
        device = next(self.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        y_t = torch.tensor((y - self._ym) / self._ys,
                           dtype=torch.float32, device=device)
        y0, y1 = self.predict_y(x_t, y_t, n_samples=n_samples)
        return y0 * self._ys + self._ym, y1 * self._ys + self._ym