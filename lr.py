import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression


class LR1:
    """S-learner: one model trained on [X, T] -> Y.

    Predicts potential outcomes by setting T=0 or T=1 at inference time.
    Uses LogisticRegression for binary outcomes, LinearRegression for
    continuous outcomes (OLS-1).
    """

    def __init__(self, outcome="binary"):
        self.outcome = outcome
        self.model = (
            LogisticRegression(max_iter=10_000)
            if outcome == "binary"
            else LinearRegression()
        )

    def fit(self, x, t, y):
        t_flat = t.ravel()
        y_flat = y.ravel()
        xt = np.concatenate([x, t_flat[:, np.newaxis]], axis=1)
        if self.outcome == "binary":
            y_flat = y_flat.astype(int)
        self.model.fit(xt, y_flat)

    def predict(self, x, y=None, n_samples=None):
        n = x.shape[0]
        x0 = np.concatenate([x, np.zeros((n, 1))], axis=1)
        x1 = np.concatenate([x, np.ones((n, 1))], axis=1)

        if self.outcome == "binary":
            y0 = self.model.predict_proba(x0)[:, 1][:, np.newaxis]
            y1 = self.model.predict_proba(x1)[:, 1][:, np.newaxis]
        else:
            y0 = self.model.predict(x0)[:, np.newaxis]
            y1 = self.model.predict(x1)[:, np.newaxis]
        return y0, y1

    predict_y = predict


class LR2:
    """T-learner: two separate models, one per treatment arm.

    Model_0 is trained on control samples (T=0), Model_1 on treated (T=1).
    Uses LogisticRegression for binary outcomes, LinearRegression for
    continuous outcomes (OLS-2).
    """

    def __init__(self, outcome="binary"):
        self.outcome = outcome
        if outcome == "binary":
            self.model_0 = LogisticRegression(max_iter=10_000)
            self.model_1 = LogisticRegression(max_iter=10_000)
        else:
            self.model_0 = LinearRegression()
            self.model_1 = LinearRegression()

    def fit(self, x, t, y):
        t_flat = t.ravel()
        y_flat = y.ravel()
        if self.outcome == "binary":
            y_flat = y_flat.astype(int)
        idx0, idx1 = t_flat == 0, t_flat == 1
        self._const_0 = self._fit_arm(self.model_0, x[idx0], y_flat[idx0])
        self._const_1 = self._fit_arm(self.model_1, x[idx1], y_flat[idx1])

    @staticmethod
    def _fit_arm(model, x, y):
        """Fit one arm; return the constant class value if only one class present."""
        classes = np.unique(y)
        if len(classes) < 2 and isinstance(model, LogisticRegression):
            return float(classes[0])
        model.fit(x, y)
        return None

    def _predict_arm(self, model, const, x):
        if const is not None:
            return np.full((x.shape[0], 1), const)
        if self.outcome == "binary":
            return model.predict_proba(x)[:, 1][:, np.newaxis]
        return model.predict(x)[:, np.newaxis]

    def predict(self, x, y=None, n_samples=None):
        y0 = self._predict_arm(self.model_0, self._const_0, x)
        y1 = self._predict_arm(self.model_1, self._const_1, x)
        return y0, y1

    predict_y = predict
