from collections import defaultdict
import pandas as pd
from .model import BayesianChangePointDetection


def main(scores, n_cps, draws, tune, target_accept, chains, lag, seed):
    # date column
    if isinstance(scores.index, pd.PeriodIndex):
        pass
    else:
        scores.index = pd.to_datetime(scores.index).to_period("M")
    scores = scores.sort_index(ascending=True)

    result = defaultdict(dict)

    for factor in scores.columns:
        for k in n_cps:
            kwargs = dict(
                data=scores[factor].values, 
                n_cps=k, 
                seed=seed,
            )
            model = BayesianChangePointDetection(**kwargs)

            kwargs = dict(
                draws=draws, 
                tune=tune,
                target_accept=target_accept,
                chains=chains,
                lag=lag,
            )
            trace = model.fit(**kwargs)
            ppc = model.predict(trace)
            trace.extend(ppc)
            
            result[factor][k] = trace

    return result