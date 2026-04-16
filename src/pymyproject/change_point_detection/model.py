import numpy as np
import pymc as pm
import pytensor.tensor as pt


class BayesianChangePointDetection:
    def __init__(
        self, 
        data, 
        n_cps, 
        seed,
    ):
        self.data = data
        self.n_cps = n_cps
        self.seed = seed

        self.model = pm.Model()
        self.prepare()
        self.build()

    def fit(
        self, 
        draws, 
        tune, 
        target_accept,
        chains,
        lag,
    ):
        with self.model:
            kwargs = dict(
                draws=draws, 
                tune=tune,
                target_accept=target_accept,
                chains=chains,
                idata_kwargs={"log_likelihood": True},
                random_seed=self.seed,
            )
            trace = pm.sample(**kwargs)
            trace_thinned = trace.sel(draw=slice(None, None, lag))

        return trace_thinned

    def predict(
        self, 
        trace,
    ):
        with self.model:
            kwargs = dict(
                trace=trace,
                var_names=["obs"],
                random_seed=self.seed,
            )
            ppc = pm.sample_posterior_predictive(**kwargs)

        return ppc

    def prepare(self):
        self.y = np.asarray(self.data)
        self.n = len(self.y)
        self.idx = np.arange(self.n)

    def build(self):
        with self.model:
            # ===== prior of change points =====
            # prior of tau_k: DiscreteUniform(1, n-2), k=1,2,...,n_cps
            # raw discrete change points
            tau = self.CHANGEPOINT()
            # ===== prior of segment dist. params =====
            # prior of mu: N(0,5^2)
            mus, sigma = self.SEGMENT()
            # ===== segment-wise mean assignment =====
            mu = self.ASSIGNMENT(mus, tau)
            # ===== likelihood =====
            # y_t: N(mu_t, sigma^2)
            # mu_t = mu_k if tau_k-1 <= t < tau_k
            obs = self.LIKELIHOOD(mu, sigma)

    def CHANGEPOINT(self):
        # ===== prior of change points =====
        # prior of tau_k: DiscreteUniform(1, n-2), k=1,2,...,n_cps
        # raw discrete change points
        tau_raw = pm.DiscreteUniform(
            name="tau_raw",
            lower=1,
            upper=self.n-2,
            shape=self.n_cps,
        )
        # order: tau0 < tau1 < tau2 < ...
        tau = pm.Deterministic(
            name="tau",
            var=pt.sort(tau_raw),
        )
        return tau

    def SEGMENT(self):
        # ===== prior of segment dist. params =====
        # prior of mu: N(0,5^2)
        mus = pm.Normal(
            name="mus",
            mu=0,
            sigma=5,
            shape=self.n_cps+1,
        )
        # prior of sigma: halfnormal(2)
        sigma = pm.HalfNormal(
            name="sigma", 
            sigma=2,
        )
        return mus, sigma

    def ASSIGNMENT(self, mus, tau):
        # ===== segment-wise mean assignment =====
        mu = mus[0]
        for k in range(self.n_cps):
            mu = pm.math.switch(
                self.idx >= tau[k],
                mus[k+1],
                mu,
            )
        return mu

    def LIKELIHOOD(self, mu, sigma):
        # ===== likelihood =====
        # y_t: N(mu_t, sigma^2)
        # mu_t = mu_k if tau_k-1 <= t < tau_k
        obs = pm.Normal(
            name="obs", 
            mu=mu, 
            sigma=sigma, 
            observed=self.y,
        )
        return obs