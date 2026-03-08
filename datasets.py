import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class IHDP(object):
    def __init__(self, path_data="cevae/datasets/IHDP/csv", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in range(25) if i not in self.binfeats]

    def __iter__(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            # this binary feature is in {1, 2}
            x[:, 13] -= 1
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats


class SyntheticDataset(object):
    def __init__(self, path_data="datasets/synthetic_data.json", replications=1):
        self.path_data = path_data
        self.replications = replications
        self.binfeats = []
        self.contfeats = [0]

        with open(path_data) as f:
            raw = json.load(f)
        self.data = np.array(raw, dtype=np.float64)

    @staticmethod
    def _sigmoid(a):
        return 1.0 / (1.0 + np.exp(-a))

    def _potential_outcomes(self, z):
        mu_0 = self._sigmoid(3 * (z + 2 * (2 * 0 - 1)))[:, np.newaxis]
        mu_1 = self._sigmoid(3 * (z + 2 * (2 * 1 - 1)))[:, np.newaxis]
        return mu_0, mu_1

    def __iter__(self):
        for i in range(self.replications):
            z = self.data[:, 0]
            x = self.data[:, 1:2]
            t = self.data[:, 2]
            y = self.data[:, 3:4]

            mu_0, mu_1 = self._potential_outcomes(z)
            y_cf = t[:, np.newaxis] * mu_0 + (1 - t[:, np.newaxis]) * mu_1

            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in range(self.replications):
            z = self.data[:, 0]
            x = self.data[:, 1:2]
            t = self.data[:, 2:3]
            y = self.data[:, 3:4]

            mu_0, mu_1 = self._potential_outcomes(z)
            y_cf = t * mu_0 + (1 - t) * mu_1

            idxtrain, ite = train_test_split(
                np.arange(x.shape[0]), test_size=0.1, random_state=i
            )
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=i)

            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats


class TwinsDataset(object):
    """Twins dataset following Louizos et al. (2017) Section 4.3.

    GESTAT10 is held out as the hidden confounder z. Treatment is assigned
    via t_i ~ Bern(sigmoid(w_o^T x + w_h(z/10 - 0.1))). Noisy one-hot
    proxies of z (replicated 3x, bits flipped with probability noise_level)
    are appended to the observed covariates.
    """

    _BIN_NAMES = frozenset([
        'alcohol', 'anemia', 'cardiac', 'chyper', 'csex', 'diabetes',
        'dmar', 'eclamp', 'hemo', 'herpes', 'hydra', 'incervix', 'lung',
        'othermr', 'phyper', 'pre4000', 'preterm', 'renal', 'rh',
        'tobacco', 'uterine', 'bord_0', 'bord_1',
    ])

    _DROP_COLS = ['Unnamed: 0.1', 'Unnamed: 0', 'infant_id_0', 'infant_id_1']

    def __init__(self, path_data="datasets/TWINS", replications=10,
                 noise_level=0.1):
        self.path_data = path_data
        self.replications = replications
        self.noise_level = noise_level

        X = pd.read_csv(f"{path_data}/twin_pairs_X_3years_samesex.csv")
        T_bw = pd.read_csv(f"{path_data}/twin_pairs_T_3years_samesex.csv")
        Y = pd.read_csv(f"{path_data}/twin_pairs_Y_3years_samesex.csv")

        mask = (T_bw['dbirwt_0'] < 2000) & (T_bw['dbirwt_1'] < 2000)
        X, Y = X[mask].reset_index(drop=True), Y[mask].reset_index(drop=True)

        self.y0_all = Y['mort_0'].values.astype(np.float64)[:, np.newaxis]
        self.y1_all = Y['mort_1'].values.astype(np.float64)[:, np.newaxis]

        X = X.drop(columns=[c for c in self._DROP_COLS if c in X.columns])
        self.z = X['gestat10'].values.astype(np.float64)
        X = X.drop(columns=['gestat10'])

        X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean(numeric_only=True))
        self.x_covariates = X.values.astype(np.float64)
        self.covar_names = list(X.columns)
        self.n = self.x_covariates.shape[0]

        self._covar_bin_idx = [
            i for i, c in enumerate(self.covar_names) if c in self._BIN_NAMES
        ]
        self._covar_cont_idx = [
            i for i in range(len(self.covar_names))
            if i not in set(self._covar_bin_idx)
        ]

    @staticmethod
    def _sigmoid(a):
        return 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))

    def _create_proxies(self, rng):
        """One-hot encode gestat10, replicate 3x, flip bits with noise."""
        z_idx = (self.z - 1).astype(int).clip(0, 9)
        onehot = np.zeros((self.n, 10))
        onehot[np.arange(self.n), z_idx] = 1.0
        proxies = np.tile(onehot, (1, 3))
        flip = rng.binomial(1, self.noise_level, proxies.shape).astype(np.float64)
        return np.abs(proxies - flip)

    def _assign_treatment(self, rng):
        """t_i ~ Bern(sigmoid(w_o^T x + w_h*(z/10 - 0.1)))."""
        x_norm = (self.x_covariates - self.x_covariates.mean(0)) / (
            self.x_covariates.std(0) + 1e-8
        )
        w_o = rng.normal(0, 0.1, x_norm.shape[1])
        w_h = rng.normal(5, 0.1)
        logit = x_norm @ w_o + w_h * (self.z / 10.0 - 0.1)
        return rng.binomial(1, self._sigmoid(logit)).astype(np.float64)

    def get_train_valid_test(self):
        for rep in range(self.replications):
            rng = np.random.RandomState(rep)

            proxies = self._create_proxies(rng)
            x = np.concatenate([self.x_covariates, proxies], axis=1)

            n_orig = self.x_covariates.shape[1]
            proxy_bin_idx = list(range(n_orig, n_orig + 30))
            binfeats = self._covar_bin_idx + proxy_bin_idx
            contfeats = self._covar_cont_idx

            t = self._assign_treatment(rng)[:, np.newaxis]
            y = t * self.y1_all + (1 - t) * self.y0_all
            y_cf = t * self.y0_all + (1 - t) * self.y1_all
            mu_0, mu_1 = self.y0_all.copy(), self.y1_all.copy()

            idx_tr, idx_te = train_test_split(
                np.arange(x.shape[0]), test_size=0.1, random_state=rep
            )
            itr, iva = train_test_split(idx_tr, test_size=0.3, random_state=rep)

            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[idx_te], t[idx_te], y[idx_te]), (y_cf[idx_te], mu_0[idx_te], mu_1[idx_te])
            yield train, valid, test, contfeats, binfeats