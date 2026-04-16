import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis


def clr_transform(X, eps=1e-10):
    X = X + eps
    gmean = np.exp(np.mean(np.log(X), axis=1, keepdims=True))
    X_clr = np.log(X / gmean)
    return X_clr


def engine(X, num_factors, date, topic_labels, seed):
    kwargs = dict(
        n_components=num_factors,
        random_state=seed,
    )
    fa = FactorAnalysis(**kwargs)

    # factor score
    scores = pd.DataFrame(
        data=fa.fit_transform(X),
        index=date,
        columns=[f"factor_{i+1}" for i in range(num_factors)],
    )

    # factor loading
    loadings = pd.DataFrame(
        data=fa.components_.T,
        index=topic_labels,
        columns=[f"factor_{i+1}" for i in range(num_factors)],
    )
    loadings.index.name = "topic"

    return {
        "model": fa,
        "log_likelihood": fa.score(X),
        "scores": scores,
        "loadings": loadings,
    }


def main(df, num_factors, seed):
    # preprocessing
    X = df.values
    X_clr = clr_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clr)

    # date column
    if isinstance(df.index, pd.PeriodIndex):
        date = df.index
    else:
        date = pd.to_datetime(df.index).to_period("M")

    result = {}

    for k in num_factors:
        kwargs = dict(
            X=X_scaled, 
            num_factors=k, 
            date=date, 
            topic_labels=list(df.columns), 
            seed=seed,
        )
        result[k] = engine(**kwargs)

        print(
            f"NUM FACTOR: {k}",
            f"LIKELIHOOD: {result[k]['log_likelihood']:.4f}",
            sep="\t",
        )

    print("FACTOR ANALYSIS FINISHED")

    return result