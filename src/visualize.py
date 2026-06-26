"""
visualize.py
------------
Chart helpers using matplotlib + seaborn. They all take a PANDAS dataframe
(small, already aggregated) and save a PNG into outputs/figures so we can drop
the screenshots straight into the report.

In the notebook we do:  spark_df.toPandas()  first, then call these.

There are a lot of chart types now (bar, line, grouped bar, stacked bar,
horizontal bar, histogram, boxplot, scatter, heatmap, treemap, pie, dual-axis,
lollipop, radar) so the report can show many different visualisations.

Geographic MAPS live in a separate file: maps.py (they use plotly).
"""

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    _HAS_SNS = True
except Exception:  # seaborn is optional, charts still work without it
    _HAS_SNS = False

from src import config

# a small, consistent colour palette so all charts look like a set
PALETTE = ["#2a6f97", "#e07a5f", "#81b29a", "#f2cc8f", "#3d405b",
           "#9d4edd", "#588157", "#bc4749", "#4cc9f0", "#ff9f1c"]


def _save(fig, filename: str):
    """Save a figure into outputs/figures (and leave it open to show inline)."""
    path = config.FIG_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved -> {path}")


# ---------------------------------------------------------------------------
# BASIC: BAR & LINE  (kept from before, slightly nicer)
# ---------------------------------------------------------------------------
def bar_chart(pdf, x, y, title, xlabel, ylabel, filename, color="#2a6f97"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(pdf[x].astype(str), pdf[y], color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    _save(fig, filename)
    return fig


def line_chart(pdf, x, y, title, xlabel, ylabel, filename, color="#e07a5f"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pdf[x], pdf[y], marker="o", color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _save(fig, filename)
    return fig


def hbar_chart(pdf, x, y, title, xlabel, ylabel, filename, color="#3d405b", top=None):
    """Horizontal bar - good for long category names (airports, routes)."""
    data = pdf.head(top) if top else pdf
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(data))))
    ax.barh(data[x].astype(str)[::-1], data[y][::-1], color=color)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel(xlabel)
    _save(fig, filename)
    return fig


def lollipop_chart(pdf, x, y, title, xlabel, ylabel, filename, color="#bc4749"):
    """Like a bar chart but cleaner for rankings (stem + dot)."""
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(pdf))))
    labels = pdf[x].astype(str)[::-1]
    vals = pdf[y][::-1]
    ax.hlines(y=labels, xmin=0, xmax=vals, color=color, alpha=0.6)
    ax.plot(vals, labels, "o", color=color)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# GROUPED & STACKED BARS
# ---------------------------------------------------------------------------
def grouped_bar(pdf, x, ys, title, xlabel, ylabel, filename, labels=None):
    """Several bars per category (e.g. avg dep vs avg arr delay)."""
    labels = labels or ys
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(ys)
    width = 0.8 / n
    idx = np.arange(len(pdf))
    for i, (col, lab) in enumerate(zip(ys, labels)):
        ax.bar(idx + i * width, pdf[col], width, label=lab, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(idx + width * (n - 1) / 2)
    ax.set_xticklabels(pdf[x].astype(str), rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    _save(fig, filename)
    return fig


def stacked_bar(pdf, x, ys, title, xlabel, ylabel, filename, labels=None):
    """Stacked bars - e.g. delay-cause minutes per month."""
    labels = labels or ys
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(pdf))
    for i, (col, lab) in enumerate(zip(ys, labels)):
        ax.bar(pdf[x].astype(str), pdf[col], bottom=bottom,
               label=lab, color=PALETTE[i % len(PALETTE)])
        bottom += pdf[col].values
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    _save(fig, filename)
    return fig


def dual_axis(pdf, x, y_left, y_right, title, l_label, r_label, filename):
    """Two metrics with different scales (flights vs avg delay)."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(pdf[x].astype(str), pdf[y_left], color="#2a6f97", alpha=0.7)
    ax1.set_ylabel(l_label, color="#2a6f97")
    ax1.set_xlabel(x)
    plt.xticks(rotation=45, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(pdf[x].astype(str), pdf[y_right], color="#e07a5f", marker="o")
    ax2.set_ylabel(r_label, color="#e07a5f")
    ax1.set_title(title)
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# DISTRIBUTIONS
# ---------------------------------------------------------------------------
def histogram(pdf, bin_col, count_col, title, xlabel, ylabel, filename, color="#588157"):
    """Bar-style histogram from already-binned data."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(pdf[bin_col], pdf[count_col], width=pdf[bin_col].diff().median() or 10,
           color=color, align="edge", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, filename)
    return fig


def boxplot_by_group(pdf, group_col, value_col, title, filename, max_groups=12):
    """
    Boxplot of a value split by a category. pdf is RAW (not aggregated) sampled
    rows, e.g. df.select('season','arr_delay').sample(...).toPandas().
    """
    groups = pdf[group_col].value_counts().head(max_groups).index.tolist()
    data = [pdf.loc[pdf[group_col] == g, value_col].dropna().values for g in groups]
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        ax.boxplot(data, tick_labels=groups, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=groups, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(value_col)
    plt.xticks(rotation=45, ha="right")
    _save(fig, filename)
    return fig


def scatter(pdf, x, y, title, xlabel, ylabel, filename, color="#9d4edd", alpha=0.3):
    """Scatter plot (e.g. dep_delay vs arr_delay to show propagation)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pdf[x], pdf[y], s=8, alpha=alpha, color=color)
    # add a y=x reference line
    lim = [min(pdf[x].min(), pdf[y].min()), max(pdf[x].max(), pdf[y].max())]
    ax.plot(lim, lim, "k--", alpha=0.5, label="arr = dep")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# HEATMAPS & MATRICES
# ---------------------------------------------------------------------------
def heatmap(pivot, title, filename, xlabel="", ylabel="", cmap="RdYlGn_r", fmt=".0f"):
    """Heatmap from a pivoted pandas table (index x columns)."""
    fig, ax = plt.subplots(figsize=(11, 7))
    if _HAS_SNS:
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                    linewidths=0.4, cbar_kws={"label": "minutes"})
    else:
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        fig.colorbar(im, ax=ax, label="minutes")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, filename)
    return fig


def correlation_heatmap(corr_pdf, title, filename):
    """Correlation matrix heatmap (-1..1)."""
    fig, ax = plt.subplots(figsize=(9, 8))
    if _HAS_SNS:
        sns.heatmap(corr_pdf, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, square=True, linewidths=0.4)
    else:
        im = ax.imshow(corr_pdf.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_pdf.columns)))
        ax.set_xticklabels(corr_pdf.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr_pdf.index)))
        ax.set_yticklabels(corr_pdf.index)
        fig.colorbar(im, ax=ax)
    ax.set_title(title)
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# PARTS-OF-WHOLE: PIE & TREEMAP
# ---------------------------------------------------------------------------
def pie_chart(pdf, label_col, value_col, title, filename):
    """Pie / donut for shares (e.g. delay-cause split, delay bands)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(pdf[value_col], labels=pdf[label_col], autopct="%1.1f%%",
           colors=PALETTE, wedgeprops={"width": 0.45})  # donut
    ax.set_title(title)
    _save(fig, filename)
    return fig


def treemap(pdf, label_col, value_col, title, filename):
    """
    Treemap of categories sized by value (e.g. flights per airline).
    Uses the squarify library; if it is missing we fall back to a bar chart.
    """
    try:
        import squarify
    except Exception:
        print("squarify not installed -> drawing a bar chart instead")
        return bar_chart(pdf, label_col, value_col, title, label_col,
                         value_col, filename)
    fig, ax = plt.subplots(figsize=(11, 7))
    labels = [f"{l}\n{int(v):,}" for l, v in zip(pdf[label_col], pdf[value_col])]
    squarify.plot(sizes=pdf[value_col], label=labels, ax=ax,
                  color=PALETTE * (len(pdf) // len(PALETTE) + 1), alpha=0.85,
                  text_kwargs={"fontsize": 8})
    ax.axis("off")
    ax.set_title(title)
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# RADAR (spider) - nice for comparing a few airlines on several metrics
# ---------------------------------------------------------------------------
def radar_chart(pdf, label_col, metric_cols, title, filename, top=5):
    """
    Compare the top rows on several normalised metrics on a radar/spider chart.
    Each metric is scaled 0-1 so they fit on the same axes.
    """
    data = pdf.head(top).copy()
    norm = data[metric_cols].copy()
    for c in metric_cols:
        rng = norm[c].max() - norm[c].min()
        norm[c] = (norm[c] - norm[c].min()) / rng if rng else 0.5

    angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for i, (_, row) in enumerate(norm.iterrows()):
        vals = row[metric_cols].tolist()
        vals += vals[:1]
        label = str(data.iloc[i][label_col])
        ax.plot(angles, vals, color=PALETTE[i % len(PALETTE)], label=label)
        ax.fill(angles, vals, color=PALETTE[i % len(PALETTE)], alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_cols)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# CLUSTERING: scatter coloured by cluster
# ---------------------------------------------------------------------------
def cluster_scatter(pdf, x, y, cluster_col, title, xlabel, ylabel, filename):
    """Scatter of airports coloured by their K-Means cluster."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in sorted(pdf[cluster_col].unique()):
        sub = pdf[pdf[cluster_col] == c]
        ax.scatter(sub[x], sub[y], s=40, alpha=0.8,
                   color=PALETTE[c % len(PALETTE)], label=f"Cluster {c}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    _save(fig, filename)
    return fig
