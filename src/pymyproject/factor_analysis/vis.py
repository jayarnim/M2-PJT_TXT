import pandas as pd
import matplotlib.pyplot as plt


def plot_log_likelihood(result, figsize):
    Y = [obj["log_likelihood"] for k, obj in result.items()]
    X = range(len(result))
    K_LIST = list(result.keys())

    fig = plt.figure(figsize=figsize)

    plt.plot(X, Y, marker='o')

    for x, y in zip(X, Y):
        kwargs = dict(
            x=x, 
            y=y+0.001, 
            s=f"{y:.4f}", 
            ha="center", 
            rotation=50,
        )
        plt.text(**kwargs)

    plt.xticks(ticks=X, labels=K_LIST)
    
    plt.xlabel("number of factors")
    plt.ylabel("log likelihood")
    plt.title("log likelihood by number of factors")
    
    plt.grid(True)

    plt.tight_layout()

    return fig


def plot_factor_loadings(result, num_factors, figsize):
    COLOR_MAP = "coolwarm"

    loadings = result[num_factors]["loadings"]
    
    fig = plt.figure(figsize=figsize)

    im = plt.imshow(X=loadings.values, aspect="auto", cmap=COLOR_MAP)

    plt.colorbar(im, label="loadings")

    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            kwargs = dict(
                x=j,
                y=i,
                s=f"{loadings.iloc[i, j]:.2f}",
                ha="center", 
                va="center", 
                fontsize=10,
            )
            plt.text(**kwargs)

    plt.xticks(ticks=range(loadings.shape[1]), labels=loadings.columns)
    plt.yticks(ticks=range(loadings.shape[0]), labels=loadings.index)

    plt.xlabel("factor")
    plt.ylabel("topic")
    plt.title(f"factor loadings heatmap (k={num_factors})")

    plt.tight_layout()

    return fig


def plot_factor_scores(result, num_factors):
    scores = result[num_factors]["scores"]

    if isinstance(scores.index, pd.PeriodIndex):
        pass
    else:
        scores.index = pd.to_datetime(scores.index).to_period("M")
    scores = scores.sort_index(ascending=True)
    date = scores.index.to_timestamp()

    fig, axes = plt.subplots(
        nrows=num_factors, 
        ncols=1,
        figsize=(16, 4 * num_factors),
        squeeze=False,
    )

    for ax, factor_idx in zip(axes.flatten(), range(num_factors)):
        y = scores.iloc[:,factor_idx].values

        ax.plot(date, y)
        ax.set_title(f"factor {factor_idx+1}")
        ax.grid(alpha=0.3)
    
    TITLE = f'factor scores (k={num_factors})'
    plt.suptitle(t=TITLE, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return (fig, axes)