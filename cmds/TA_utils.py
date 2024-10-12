#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------------
# FINM 36700 - UTILITY FUNCTIONS
# ------------------------------
# Disclaimer & Usage Notes:
# ------------------------------------------------------------------------------
# While a decent amount of effort has gone into ensuring correctness,
# there is no guarantee that these functions are 100% correct. You are
# free to use these functions, without attribution, in your own work.
#
# The general structure in terms of naming conventions is that any function
# beginning with "calc_", is a function that calculates something, and will
# return a DataFrame or a dict. Any function beginning with "plot_", is a
# function that plots something, and will return a matplotlib axes object.
#
# Any function that begins with "print_", is a function that prints something
# and returns nothing.
# ------------------------------------------------------------------------------


import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# --------------------
# REGRESSION FUNCTIONS
# --------------------


def calc_univariate_regression(y, X, intercept=True, adj=12):
    """
    Calculate a univariate regression of y on X. Note that both X and y
    need to be one-dimensional.

    Args:
        y : target variable
        X : independent variable
        intercept (bool, optional): Fit the regression with an intercept or not. Defaults to True.
        adj (int, optional): What to adjust the returns by. Defaults to 12.

    Returns:
        DataFrame: Summary of regression results
    """
    X_down = X[y < 0]
    y_down = y[y < 0]
    if intercept:
        X = sm.add_constant(X)
        X_down = sm.add_constant(X_down)

    model = sm.OLS(y, X, missing="drop")
    results = model.fit()

    inter = results.params.iloc[0] if intercept else 0
    beta = results.params.iloc[1] if intercept else results.params.iloc[0]

    summary = dict()

    summary["Alpha"] = inter * adj
    summary["Beta"] = beta

    down_mod = sm.OLS(y_down, X_down, missing="drop").fit()
    summary["Downside Beta"] = down_mod.params.iloc[1] if intercept else down_mod.params.iloc[0]

    summary["R-Squared"] = results.rsquared
    summary["Treynor Ratio"] = (y.mean() / beta) * adj
    summary["Information Ratio"] = (inter / results.resid.std()) * np.sqrt(adj)
    summary["Tracking Error"] = (
        inter / summary["Information Ratio"]
        if intercept
        else results.resid.std() * np.sqrt(adj)
    )
    
    if isinstance(y, pd.Series):
        return pd.DataFrame(summary, index=[y.name])
    else:
        return pd.DataFrame(summary, index=y.columns)

def calc_multivariate_regression(y, X, intercept=True, adj=12):
    """
    Calculate a multivariate regression of y on X. Adds useful metrics such
    as the Information Ratio and Tracking Error. Note that we can't calculate
    Treynor Ratio or Downside Beta here.

    Args:
        y : target variable
        X : independent variables
        intercept (bool, optional): Defaults to True.
        adj (int, optional): Annualization factor. Defaults to 12.

    Returns:
        DataFrame: Summary of regression results
    """
    if intercept:
        X = sm.add_constant(X)

    model = sm.OLS(y, X, missing="drop")
    results = model.fit()
    summary = dict()

    inter = results.params.iloc[0] if intercept else 0
    betas = results.params.iloc[1:] if intercept else results.params

    summary["Alpha"] = inter * adj
    summary["R-Squared"] = results.rsquared

    X_cols = X.columns[1:] if intercept else X.columns

    for i, col in enumerate(X_cols):
        summary[f"{col} Beta"] = betas[i]

    summary["Information Ratio"] = (inter / results.resid.std()) * np.sqrt(adj)
    summary["Tracking Error"] = results.resid.std() * np.sqrt(adj)
    
    if isinstance(y, pd.Series):
        return pd.DataFrame(summary, index=[y.name])
    else:
        return pd.DataFrame(summary, index=y.columns)

def calc_iterative_regression(y, X, intercept=True, one_to_many=False, adj=12):
    """
    Iterative regression for checking one X column against many different y columns,
    or vice versa. "one_to_many=True" means that we are checking one X column against many
    y columns, and "one_to_many=False" means that we are checking many X columns against a
    single y column.

    To enforce dynamic behavior in terms of regressors and regressands, we
    check that BOTH X and y are DataFrames

    Args:
        y : Target variable(s)
        X : Independent variable(s)
        intercept (bool, optional): Defaults to True.
        one_to_many (bool, optional): Which way to run the regression. Defaults to False.
        adj (int, optional): Annualization. Defaults to 12.

    Returns:
        DataFrame : Summary of regression results.
    """

    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame):
        raise TypeError("X and y must both be DataFrames.")

    if one_to_many:
        if isinstance(X, pd.Series) or X.shape[1] > 1:
            summary = pd.concat(
                [
                    calc_multivariate_regression(y[col], X, intercept, adj)
                    for col in y.columns
                ],
                axis=0,
            )
        else:
            summary = pd.concat(
                [
                    calc_univariate_regression(y[col], X, intercept, adj)
                    for col in y.columns
                ],
                axis=0,
            )
        summary.index = y.columns
        return summary
    else:
        summary = pd.concat(
            [
                calc_univariate_regression(y, X[col], intercept, adj)
                for col in X.columns
            ],
            axis=0,
        )
        summary.index = X.columns
        return summary


# -----------------------
# RISK & RETURN FUNCTIONS
# -----------------------


def calc_return_metrics(data, as_df=False, adj=12):
    """
    Calculate return metrics for a DataFrame of assets.

    Args:
        data (pd.DataFrame): DataFrame of asset returns.
        as_df (bool, optional): Return a DF or a dict. Defaults to False (return a dict).
        adj (int, optional): Annualization. Defaults to 12.

    Returns:
        Union[dict, DataFrame]: Dict or DataFrame of return metrics.
    """
    summary = dict()
    if adj == 12:
        periodicity = "Annualized"
    elif adj == 4:
        periodicity = "Quarterly"
    else:
        periodicity = "Monthly"
    
    summary[f"{periodicity} Return"] = data.mean() * adj
    summary[f"{periodicity} Volatility"] = data.std() * np.sqrt(adj)
    summary[f"{periodicity} Sharpe Ratio"] = (
        summary[f"{periodicity} Return"] / summary["Annualized Volatility"]
    )
    summary[f"{periodicity} Sortino Ratio"] = summary["Annualized Return"] / (
        data[data < 0].std() * np.sqrt(adj)
    )
    
    return pd.DataFrame(summary, index=data.columns) if as_df else summary


def calc_risk_metrics(data, as_df=False, var=0.05):
    """
    Calculate risk metrics for a DataFrame of assets.

    Args:
        data (pd.DataFrame): DataFrame of asset returns.
        as_df (bool, optional): Return a DF or a dict. Defaults to False.
        adj (int, optional): Annualizatin. Defaults to 12.
        var (float, optional): VaR level. Defaults to 0.05.

    Returns:
        Union[dict, DataFrame]: Dict or DataFrame of risk metrics.
    """
    summary = dict()
    summary["Skewness"] = data.skew()
    summary["Excess Kurtosis"] = data.kurtosis()
    summary[f"VaR ({var})"] = data.quantile(var, axis=0)
    summary[f"CVaR ({var})"] = data[data <= data.quantile(var, axis=0)].mean()
    summary["Min"] = data.min()
    summary["Max"] = data.max()

    wealth_index = 1000 * (1 + data).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    summary["Max Drawdown"] = drawdowns.min()

    summary["Bottom"] = drawdowns.idxmin()
    summary["Peak"] = previous_peaks.idxmax()

    recovery_date = []
    for col in wealth_index.columns:
        prev_max = previous_peaks[col][: drawdowns[col].idxmin()].max()
        recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin() :]]).T
        recovery_date.append(
            recovery_wealth[recovery_wealth[col] >= prev_max].index.min()
        )
    summary["Recovery"] = ["-" if pd.isnull(i) else i for i in recovery_date]

    summary["Duration (days)"] = [
        (i - j).days if i != "-" else "-"
        for i, j in zip(summary["Recovery"], summary["Bottom"])
    ]

    return pd.DataFrame(summary, index=data.columns) if as_df else summary


def calc_performance_metrics(data, adj=12, var=0.05):
    """
    Aggregating function for calculating performance metrics. Returns both
    risk and performance metrics.

    Args:
        data (pd.DataFrame): DataFrame of asset returns.
        adj (int, optional): Annualization. Defaults to 12.
        var (float, optional): VaR level. Defaults to 0.05.

    Returns:
        DataFrame: DataFrame of performance metrics.
    """
    summary = {
        **calc_return_metrics(data=data, adj=adj),
        **calc_risk_metrics(data=data, var=var),
    }
    summary["Calmar Ratio"] = summary["Annualized Return"] / abs(
        summary["Max Drawdown"]
    )
    return pd.DataFrame(summary, index=data.columns)


# -----------------------
# CORRELATION HELPERS
# -----------------------


def plot_correlation_matrix(corrs, ax=None):
    if ax:
        sns.heatmap(
            corrs,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.7,
            annot_kws={"size": 10},
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.75},
            ax=ax,
        )
    # Correlation helper function.
    else:
        ax = sns.heatmap(
            corrs,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.7,
            annot_kws={"size": 10},
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.75},
        )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    return ax


def print_max_min_correlation(corrs):
    # Correlation helper function. Prints the min/max/absolute value
    # for the correlation matrix.
    corr_series = corrs.unstack()
    corr_series = corr_series[corr_series != 1]

    max_corr = corr_series.abs().agg(["idxmax", "max"]).T
    min_corr = corr_series.abs().agg(["idxmin", "min"]).T
    min_corr_raw = corr_series.agg(["idxmin", "min"]).T
    max_corr, max_corr_val = max_corr["idxmax"], max_corr["max"]
    min_corr, min_corr_val = min_corr["idxmin"], min_corr["min"]
    min_corr_raw, min_corr_raw_val = min_corr_raw["idxmin"], min_corr_raw["min"]

    print(
        f"Max Corr (by absolute value): {max_corr[0]} and {max_corr[1]} with a correlation of {max_corr_val:.2f}"
    )
    print(
        f"Min Corr (by absolute value): {min_corr[0]} and {min_corr[1]} with a correlation of {min_corr_val:.2f}"
    )
    print(
        f"Min Corr (raw): {min_corr_raw[0]} and {min_corr_raw[1]} with a correlation of {min_corr_raw_val:.2f}"
    )


# -------------------
# PORTFOLIO FUNCTIONS
# -------------------


def calc_tangency_portfolio(mean_rets, cov_matrix):
    """
    NOTE: This *does not* assume access to the risk-free rate. Use
        Mark's portfolio.py for tangency/GMV/etc. portfolios.

    Function to calculate tangency portfolio weights. Comes from the
    formula seen in class.

    Args:
        mean_rets: Vector of mean returns.
        cov_matrix: Covariance matrix of returns.

    Returns:
        Vector of tangency portfolio weights.
    """
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(mean_rets.shape)
    return (inv_cov @ mean_rets) / (ones.T @ (inv_cov @ mean_rets))


def calc_gmv_portfolio(cov_matrix):
    """
    NOTE: This *does not* assume access to the risk-free rate. Use
        Mark's portfolio.py for tangency/GMV/etc. portfolios.

    Function to calculate the weights of the global minimum variance portfolio.

    Args:
        cov_matrix : Covariance matrix of returns.

    Returns:
        Vector of GMV portfolio weights.
    """
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except TypeError:
        cov_inv = np.linalg.inv(np.array(cov_matrix))

    one_vector = np.ones(len(cov_matrix.index))
    return cov_inv @ one_vector / (one_vector @ cov_inv @ (one_vector))


def calc_mv_portfolio(mean_rets, cov_matrix, target=None):
    """
    NOTE: This *does not* assume access to the risk-free rate. Use
        Mark's portfolio.py for tangency/GMV/etc. portfolios.

    Function to calculate the weights of the mean-variance portfolio. If
    target is not specified, then the function will return the tangency portfolio.
    If target is specified, then we return the MV-efficient portfolio with the target
    return.

    Args:
        mean_rets : Vector of mean returns.
        cov_matrix : Covariance matrix of returns.
        target (optional):  Target mean return. Defaults to None. Note: must be adjusted for
                            annualization the same time-frequency as the mean returns. If the
                            mean returns are monthly, the target must be monthly as well.

    Returns:
        Vector of MV portfolio weights.
    """
    w_tan = calc_tangency_portfolio(mean_rets, cov_matrix)

    if target is None:
        return w_tan

    w_gmv = calc_gmv_portfolio(cov_matrix)
    delta = (target - mean_rets @ w_gmv) / (mean_rets @ w_tan - mean_rets @ w_gmv)
    return delta * w_tan + (1 - delta) * w_gmv


# -----------------------------------------------
# MISC. FUNCTIONS
#
# These are functions that you, as a student, will
# not use, but are used to illustrate some of the
# concepts in the class, via TA code.
# -----------------------------------------------


def plot_capm_regression(assets, market, ret_cross=False, adj=12, fig=None, ax=None):
    """
    Plot CAPM regressions on a scatter plot.

    Args:
        assets : DataFrame of asset returns.
        market : DataFrame of market returns.
        ret_cross (bool, optional): Return the cross-sectional regression results. Defaults to False.
        adj (int, optional): Annualization. Defaults to 12.
        fig (optional): Figure object. Defaults to None.
        ax (optional): Axes object. Defaults to None.

    Returns:
        fig, ax, beta_mean: Figure and axes objects for the plot. And, cross-sectional regression results.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    betas = []
    means = []

    market_const = sm.add_constant(market)
    for asset in assets.columns:
        regr = sm.OLS(assets[asset], market_const).fit()
        ax.scatter(regr.params[1], assets[asset].mean() * adj, label=asset, zorder=2)
        ax.annotate(asset, (regr.params[1], assets[asset].mean() * adj), zorder=3)
        betas.append(regr.params[1])
        means.append(assets[asset].mean() * adj)

    ax.plot([0, 1.4], [0, market.mean()[0] * adj * 1.4], c="black", zorder=1, alpha=0.5)
    beta_mean_regr = sm.OLS(means, sm.add_constant(betas)).fit()

    ax.plot(
        np.arange(0, 1.6, 0.1),
        beta_mean_regr.params[0] + beta_mean_regr.params[1] * np.arange(0, 1.6, 0.1),
        zorder=1,
        alpha=0.7,
    )

    ax.set_yticks(np.arange(0, 0.15, 0.02))
    ax.set_xticks(np.arange(0, 1.6, 0.2))
    ax.set_title("Mean Return vs. Beta")
    ax.set_xlabel("Beta")
    ax.set_ylabel("Mean Return")

    if ret_cross:
        return fig, ax, beta_mean_regr
    else:
        return fig, ax


def plot_mv_frontier(rets, delta=2, plot_tan=True, adj=12, fig=None, ax=None):
    """
    Plot MV frontier, and the tangency and GMV portfolios.

    Args:
        rets : Returns DataFrame
        delta (int, optional): Delta range (from -delta to +delta). Defaults to 2. Use to make
                                the plot look nicer, and keep the MV frontier within a reasonable range.
        plot_tan (bool, optional): Set to False if the tangency gives an extreme value. Defaults to True.
        adj (int, optional): Annualization. Defaults to 12.
        fig (optional): Figure object. Defaults to None.
        ax (optional): Axes object. Defaults to None.

    Returns:
        fig, ax: Figure and axes objects for the plot.
    """
    omega_tan = pd.Series(
        calc_tangency_portfolio(rets.mean(), rets.cov()), index=rets.columns
    )

    omega_gmv = pd.Series(calc_gmv_portfolio(rets.cov()), index=rets.columns)
    omega = pd.concat([omega_tan, omega_gmv], axis=1)
    omega.columns = ["tangency", "gmv"]

    delta_grid = np.linspace(-delta, delta, 150)
    mv_frame = pd.DataFrame(columns=["mean", "vol"], index=delta_grid)
    for i, delta in enumerate(delta_grid):
        omega_mv = delta * omega_tan + (1 - delta) * omega_gmv
        rets_p = rets @ omega_mv
        mv_frame["mean"].iloc[i] = rets_p.mean() * adj
        mv_frame["vol"].iloc[i] = rets_p.std() * np.sqrt(adj)

    rets_special = pd.DataFrame(index=rets.index)
    rets_special["tan"] = rets @ omega_tan.values
    rets_special["gmv"] = rets @ omega_gmv.values

    mv_assets = pd.concat([rets.mean() * adj, rets.std() * np.sqrt(adj)], axis=1)
    mv_special = pd.concat(
        [rets_special.mean() * adj, rets_special.std() * np.sqrt(adj)], axis=1
    )
    mv_assets.columns = ["mean", "vol"]
    mv_special.columns = ["mean", "vol"]

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    colors = [
        "#348ABD",
        "#7A68A6",
        "#A60628",
        "#467821",
        "#CF4457",
        "#188487",
        "#E24A33",
        "#ECD078",
        "#D95B43",
        "#C02942",
        "#ECD078",
        "#D95B43",
        "#C02942",
        "#2A044A",
        "#0B2E59",
        "#0D6759",
        "#7AB317",
        "#A0C55F",
        "#2A044A",
        "#0B2E59",
        "#0D6759",
    ]
    ax.set_prop_cycle("color", colors)

    ax.plot(
        mv_frame["vol"],
        mv_frame["mean"],
        c="k",
        linewidth=3,
        label="MV Frontier",
        zorder=1,
    )
    if plot_tan:
        ax.scatter(
            mv_special["vol"][0],
            mv_special["mean"][0],
            c="g",
            linewidth=3,
            # label="Tangency Portfolio",
        )
        text = ax.text(
            x=mv_special["vol"][0] + 0.0005,
            y=mv_special["mean"][0] + 0.0005,
            s="Tangency",
            fontsize=11,
            c="w",
        )
        text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])

    ax.scatter(
        mv_special["vol"][1],
        mv_special["mean"][1],
        c="r",
        linewidth=3,
        # label="GMV Portfolio",
    )
    text = ax.text(
        x=mv_special["vol"][1] + 0.0005,
        y=mv_special["mean"][1] + 0.0005,
        s="GMV",
        fontsize=11,
        c="w",
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])
    for _, val in mv_assets.iterrows():
        ax.scatter(
            val["vol"],
            val["mean"],
            linewidth=3,
            label=val.name,
        )
    ax.set_xlabel("Volatility (Annualized)")
    ax.set_ylabel("Mean (Annualized)")
    ax.legend()
    return fig, ax


def plot_pairplot(rets):
    """
    Plot a pairplot of returns. Add a vertical line at 0 -- this is useful
    for visualizing the skewness of the returns.

    Args:
        rets : DataFrame of returns.

    Returns:
        axes : Axes object for the plot.
    """

    axes = sns.pairplot(rets, diag_kind="kde", plot_kws={"alpha": 0.5})
    for _, ax in enumerate(axes.diag_axes):
        ax.axvline(0, c="k", lw=1, alpha=0.7)

    return axes