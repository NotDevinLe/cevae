import numpy as np
import torch
import torch.nn as nn


class TARNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=50):
        super(TARNet, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.head0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        rep = self.shared(x)
        return self.head0(rep), self.head1(rep)

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
        mse = nn.MSELoss()

        n = x_tr.shape[0]
        bs = min(batch_size, n)
        n_iter = max(1, 10 * (n // bs))
        idx = np.arange(n)
        best_val = np.inf
        best_state = None

        for epoch in range(epochs):
            self.train()
            np.random.shuffle(idx)
            for _ in range(n_iter):
                b = np.random.choice(idx, bs)
                xb, tb, yb = xtr_t[b], ttr_t[b], ytr_t[b]
                y0p, y1p = self.forward(xb)
                loss = mse(tb * y1p + (1 - tb) * y0p, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % early == 0 or epoch == epochs - 1:
                self.eval()
                with torch.no_grad():
                    y0v, y1v = self.forward(xva_t)
                    val_loss = mse(tva_t * y1v + (1 - tva_t) * y0v,
                                   yva_t).item()
                if val_loss < best_val:
                    if verbose:
                        print(f'[TARNet] Epoch {epoch+1}: val loss '
                              f'{best_val:.4f} -> {val_loss:.4f}')
                    best_val = val_loss
                    best_state = {k: v.clone()
                                  for k, v in self.state_dict().items()}

            if eval_callback and epoch % early == 0:
                self.eval()
                eval_callback(epoch + 1, self)

        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()

    @torch.no_grad()
    def predict(self, x, y=None, n_samples=None):
        """Numpy in / numpy out.  Returns *(y0, y1)* on the original scale."""
        device = next(self.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        y0, y1 = self.forward(x_t)
        return (y0.cpu().numpy() * self._ys + self._ym,
                y1.cpu().numpy() * self._ys + self._ym)
