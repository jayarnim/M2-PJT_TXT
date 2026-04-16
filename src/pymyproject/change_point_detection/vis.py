from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az


def search_cps(trace, date, n_cps):
    kwargs = dict(
        a=trace.posterior["tau"].values.reshape(-1, n_cps),
        axis=0,
        keepdims=False,
    )
    cps_idx = stats.mode(**kwargs).mode.tolist()
    cps = date[cps_idx]

    return cps


# =========================
# PLOT SEGMENT WEIGHTED SHIFTS
# =========================
def calc_segment_weighted_shifts(doc_topic_mat, loadings, cps):
    if isinstance(doc_topic_mat.index, pd.PeriodIndex):
        pass
    else:
        doc_topic_mat.index = pd.to_datetime(doc_topic_mat.index).to_period("M")
    doc_topic_mat = doc_topic_mat.sort_index(ascending=True)
    date = doc_topic_mat.index
    
    topic_labels = loadings.index.tolist()
    doc_topic_mat = doc_topic_mat[topic_labels]

    boundaries = [date.min()] + list(cps) + [date.max()]

    data = dict()
    data["loadings"] = loadings

    for i in range(len(boundaries)-2):
        left = pd.Period(boundaries[i], freq="M")
        cp = pd.Period(boundaries[i + 1], freq="M")
        right = pd.Period(boundaries[i + 2], freq="M")

        COND = (date >= left) & (date < cp)
        pre = doc_topic_mat.loc[COND,:].mean()

        COND = (date >= cp) & (date < right)
        post = doc_topic_mat.loc[COND,:].mean()

        data[f"cp_{i}"] = loadings * (post - pre)

    return pd.DataFrame(data=data, index=topic_labels)


def plot_segment_weighted_shifts(result, doc_topic_mat, scores, loadings, threshold, factor, n_cps):
    kwargs = dict(
        trace=result[factor][n_cps],
        date=scores[factor].index.to_timestamp().to_period("M"),
        n_cps=n_cps,
    )
    cps = search_cps(**kwargs)

    kwargs = dict(
        doc_topic_mat=doc_topic_mat,
        loadings=loadings[factor],
        cps=cps,
    )
    shifts = calc_segment_weighted_shifts(**kwargs)

    X_MIN = shifts.stack().min()
    X_MAX = shifts.stack().max()
    PAD = 0.05
    XLIM = (X_MIN - PAD, X_MAX + PAD)

    fig, axes = plt.subplots(
        nrows=n_cps, 
        ncols=1,
        figsize=(16, 6 * n_cps),
        squeeze=False,
    )

    LOADINGS_COL = "loadings"
    CP_COLS = shifts.columns.drop(LOADINGS_COL).tolist()

    for cp_idx, (ax, cp_col) in enumerate(zip(axes.flatten(), CP_COLS)):
        segment = shifts[[LOADINGS_COL, cp_col]].sort_values(cp_col)
        colors = [
            "red" if abs(v) >= threshold else "gray"
            for v in segment[LOADINGS_COL]
        ]

        bars = ax.barh(segment.index, segment[cp_col], color=colors)
        ax.set_xlim(XLIM)
        ax.axvline(0, color="black")

        for bar in bars:
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2

            kwargs = dict(
                x=width,
                y=y,
                s=f"{width:.4f}",
                va="center",
                ha="left" if width >= 0 else "right"
            )
            ax.text(**kwargs)
        
        ax.set_xlabel("contribution shift")
        ax.set_ylabel("topic")
        ax.set_title(f"cp {cp_idx+1} ({cps[cp_idx]})")

    TITLE = f'{factor} score shift decomposition across change points (n_cps={n_cps})'
    plt.suptitle(t=TITLE, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return (fig, axes)


# =========================
# PLOT PREDICTION
# =========================
def build_prediction_data(trace, scores, n_cps):

    date = scores.index.to_timestamp()

    y_obs = scores.values
    y_fit = (
        trace
        .posterior_predictive["obs"]
        .mean(dim=["chain", "draw"])
        .values
    )

    kwargs = dict(
        a=trace.posterior["tau"].values.reshape(-1, n_cps),
        axis=0,
        keepdims=False,
    )
    cps_idx = stats.mode(**kwargs).mode.tolist()
    cps = date[cps_idx]

    return dict(
        x=date, 
        y_obs=y_obs, 
        y_fit=y_fit, 
        cps=cps,
    )


def draw_prediction_plot(ax, x, y_obs, y_fit, cps):
    ax.plot(
        x, y_obs, 
        label="observed value",
    )
    ax.plot(
        x, y_fit, 
        color="red", 
        label="fitted value",
    )
    for cp in cps:
        ax.axvline(
            x=cp, 
            linestyle="--", 
            color="black", 
            alpha=0.7,
        )
    ax.legend()
    ax.set_title(f"cps={len(cps)}")

    return ax


def plot_prediction(result, scores, factor):
    result = result[factor]
    scores = scores[factor]
    N_CPS = len(result)

    fig, axes = plt.subplots(
        nrows=N_CPS, 
        ncols=1,
        figsize=(16, 4 * N_CPS),
    )

    for ax, (n_cps, trace) in zip(axes.flatten(), result.items()):
        kwargs = build_prediction_data(trace, scores, n_cps)
        ax = draw_prediction_plot(ax, **kwargs)
    
    TITLE = f'change points of factor score ({factor})'
    plt.suptitle(t=TITLE, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return (fig, axes)


# =========================
# PLOT CHANGE POINTS
# =========================
def build_change_points_data(result, scores, factor, n_cps):
    trace = result[factor][n_cps]
    scores = scores[factor]

    date = scores.index.to_timestamp()
    y_obs = scores.values

    kwargs = dict(
        trace=trace,
        date=date,
        n_cps=n_cps,
    )
    cps = search_cps(**kwargs)

    mus = (
        trace
        .posterior["mus"]
        .mean(dim=["chain", "draw"])
        .values
    )

    return dict(
        x=date, 
        y=y_obs, 
        mus=mus, 
        cps=cps,
    )


def plot_change_points(result, scores, factor, n_cps):
    kwargs = dict(
        result=result,
        scores=scores, 
        factor=factor, 
        n_cps=n_cps,
    )
    data = build_change_points_data(**kwargs)

    print(data["cps"])
    print([type(cp) for cp in data["cps"]])

    fig = plt.figure(figsize=(16, 6))

    # y observation
    plt.plot(
        data["x"], data["y"], 
        label="observed",
    )

    # change points
    for cp in data["cps"]:
        kwargs = dict(
            x=cp,
            color="black",
            linestyle="--",
            lw=1,
        )
        plt.axvline(**kwargs)

    # segment mean
    boundaries = [data["x"][0]] + list(data["cps"]) + [data["x"][-1]]

    for i in range(len(data["mus"])):
        kwargs = dict(
            y=data["mus"][i],
            xmin=boundaries[i],
            xmax=boundaries[i+1],
            colors="red",
            linewidth=2,
        )
        plt.hlines(**kwargs)

    plt.xlabel("yearmonth")
    plt.ylabel("factor scores")
    plt.title(f"{factor} change points (n_cps={n_cps})")
    
    plt.tight_layout()

    return fig


# =========================
# PLOT TAU
# =========================
def draw_posterior(samples, date, idx, ax):
    sns.histplot(
        data=date[samples[:, idx]],
        bins=50,
        stat="probability",
        ax=ax,
    )

    mode_idx = stats.mode(
        a=samples[:, idx],
        axis=0,
        keepdims=False,
    ).mode.item()

    ax.axvline(
        x=date[mode_idx],
        # linestyle="--",
        color="red",
        # alpha=0.7,
    )

    mode_date = date[mode_idx].strftime("%Y-%m")

    ax.set_xlabel("yearmonth")
    ax.set_ylabel("density")
    ax.set_title(f"cp {idx+1} posterior (mode: {mode_date})")

    return ax


def draw_autocorr(trace, idx, ax):
    az.plot_autocorr(
        data=trace,
        var_names=["tau"],
        coords={"tau_dim_0": [idx]},
        ax=ax,
    )
    ax.set_title(f"cp {idx+1} autocorr")
    return ax


def plot_tau_diagnostics(result, date, factor, n_cps):
    trace = result[factor][n_cps]
    tau_samples = trace.posterior["tau"].values.reshape(-1, n_cps)

    fig, axes = plt.subplots(
        nrows=n_cps,
        ncols=2,
        figsize=(18, 4 * n_cps),
        squeeze=False,
    )

    for i in range(n_cps):
        ax_post = axes[i, 0]
        ax_auto = axes[i, 1]

        # LEFT: posterior
        kwargs = dict(
            samples=tau_samples,
            date=date,
            idx=i,
            ax=ax_post,
        )
        ax_post = draw_posterior(**kwargs)

        # RIGHT: autocorrelation
        kwargs = dict(
            trace=trace,
            idx=i,
            ax=ax_auto,
        )
        ax_auto = draw_autocorr(**kwargs)

    TITLE = f"{factor} tau diagnostics (n_cp={n_cps})"
    plt.suptitle(t=TITLE, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    return fig, axes