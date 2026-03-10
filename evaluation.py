import numpy as np


class JobsEvaluator:
    """Evaluator for the Jobs dataset (binary outcome, no counterfactuals).

    Metrics computed on the experimental (randomized) subset where treatment
    assignment is independent of covariates:
      - Policy risk  (R_pol)
      - ATE error    |predicted ATE - true ATE|
      - ATT error    |predicted ATT - true ATT|
    """

    def __init__(self, y, t, e, true_ate):
        self.y = y.flatten()
        self.t = t.flatten()
        self.e = e.flatten()
        self.true_ate = true_ate

        exp_t1 = (self.e == 1) & (self.t == 1)
        exp_t0 = (self.e == 1) & (self.t == 0)
        self.true_att = self.y[exp_t1].mean() - self.y[exp_t0].mean()

    def policy_risk(self, ypred1, ypred0):
        """R_pol estimated on the experimental subset (Shalit et al., 2017)."""
        policy = ((ypred1 - ypred0).flatten() > 0).astype(float)
        exp = self.e == 1
        n_exp = exp.sum()
        if n_exp == 0:
            return float('nan')

        p1 = exp & (policy == 1)
        p0 = exp & (policy == 0)
        p1_t1 = p1 & (self.t == 1)
        p0_t0 = p0 & (self.t == 0)

        risk = 1.0
        if p1_t1.sum() > 0 and p1.sum() > 0:
            risk -= self.y[p1_t1].mean() * p1.sum() / n_exp
        if p0_t0.sum() > 0 and p0.sum() > 0:
            risk -= self.y[p0_t0].mean() * p0.sum() / n_exp
        return float(risk)

    def abs_ate(self, ypred1, ypred0):
        return float(abs(np.mean(ypred1 - ypred0) - self.true_ate))

    def abs_att(self, ypred1, ypred0):
        treated = self.t == 1
        if treated.sum() == 0:
            return float('nan')
        pred_att = float(np.mean((ypred1 - ypred0).flatten()[treated]))
        return abs(pred_att - self.true_att)

    def calc_stats(self, ypred1, ypred0):
        """Returns (policy_risk, ate_error, att_error)."""
        return (
            self.policy_risk(ypred1, ypred0),
            self.abs_ate(ypred1, ypred0),
            self.abs_att(ypred1, ypred0),
        )


class Evaluator(object):
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0

    def rmse_ite(self, ypred1, ypred0):
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.sqrt(np.mean(np.square(self.true_ite - pred_ite)))

    def abs_ate(self, ypred1, ypred0):
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))

    def pehe(self, ypred1, ypred0):
        return np.sqrt(np.mean(np.square((self.mu1 - self.mu0) - (ypred1 - ypred0))))

    def y_errors(self, y0, y1):
        ypred = (1 - self.t) * y0 + self.t * y1
        ypred_cf = self.t * y0 + (1 - self.t) * y1
        return self.y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(self, ypred, ypred_cf):
        rmse_factual = np.sqrt(np.mean(np.square(ypred - self.y)))
        rmse_cfactual = np.sqrt(np.mean(np.square(ypred_cf - self.y_cf)))
        return rmse_factual, rmse_cfactual

    def calc_stats(self, ypred1, ypred0):
        ite = self.rmse_ite(ypred1, ypred0)
        ate = self.abs_ate(ypred1, ypred0)
        pehe = self.pehe(ypred1, ypred0)

        treated_idx = np.where(self.t == 1)[0]
        true_att = np.mean(self.true_ite[treated_idx])
        pred_ite = ypred1 - ypred0
        pred_att = np.mean(pred_ite[treated_idx])
        return ite, ate, pehe, np.abs(pred_att - true_att)

