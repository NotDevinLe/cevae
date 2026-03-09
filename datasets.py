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
    def __init__(self, path_data=None, n=None, replications=1, seed=42):
        self.replications = replications
        self.binfeats = []
        self.contfeats = [0]

        if n is not None:
            self.data = self._generate(n, seed)
        else:
            if path_data is None:
                path_data = "datasets/synthetic_data.json"
            with open(path_data) as f:
                raw = json.load(f)
            self.data = np.array(raw, dtype=np.float64)

    @staticmethod
    def _generate(n, seed):
        rng = np.random.RandomState(seed)
        sigma_z1, sigma_z0 = 5, 3
        z = rng.binomial(1, 0.5, n)
        x = rng.normal(z, sigma_z1 * z + sigma_z0 * (1 - z)).astype(int)
        t = rng.binomial(1, 0.75 * z + 0.25 * (1 - z))
        prob = 1.0 / (1.0 + np.exp(-3 * (z + 2 * (2 * t - 1))))
        y = rng.binomial(1, prob)
        return np.column_stack([z, x, t, y]).astype(np.float64)

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