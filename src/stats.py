"""
stats.py - regression with calendar-day Driscoll-Kraay covariance, an independent brute-force check,
Benjamini-Hochberg, and the resampling tools used by the event comparison.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def ols_dk(X: np.ndarray, y: np.ndarray, day: np.ndarray, lags=(7, 14, 28)):
    """OLS; Driscoll-Kraay covariance: scores X_i*e_i summed within each CALENDAR day (days with no
    observation contribute zero rows, so lag k always means k calendar days), Bartlett kernel.
    No small-sample scaling (n_days ~ 700 >> number of parameters)."""
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise np.linalg.LinAlgError("rank-deficient design")
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    nd = int(day.max()) + 1
    S0 = np.zeros((nd, X.shape[1]))
    np.add.at(S0, day, X * e[:, None])
    bread = np.linalg.inv(X.T @ X)
    se = {}
    for L in lags:
        S = S0.T @ S0
        for k in range(1, min(L, nd - 1) + 1):
            G = S0[k:].T @ S0[:-k]
            S += (1 - k / (L + 1)) * (G + G.T)
        se[L] = np.sqrt(np.clip(np.diag(bread @ S @ bread), 0, None))
    return b, e, se


def dk_bruteforce(X, e, day, L):
    """Independent O(n_days^2) implementation of the same estimator (for verification only)."""
    days = np.unique(day)
    g = {d: (X[day == d] * e[day == d, None]).sum(0) for d in days}
    S = np.zeros((X.shape[1], X.shape[1]))
    for t in days:
        for s in days:
            k = abs(int(t) - int(s))
            if k <= L:
                S += (1 - k / (L + 1)) * np.outer(g[t], g[s])
    bread = np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(bread @ S @ bread))


def p_normal(z):
    return math.erfc(abs(z) / math.sqrt(2)) if np.isfinite(z) else np.nan


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def fit_expected(pool: pd.DataFrame):
    A = np.c_[np.ones(len(pool)), pool["level0"], pool["rise"]]
    coef, *_ = np.linalg.lstsq(A, pool["change"].to_numpy(), rcond=None)
    return coef


def predict(coef, df):
    return np.c_[np.ones(len(df)), df["level0"], df["rise"]] @ coef


def block_resample(pool: pd.DataFrame, B: int, rng) -> pd.DataFrame:
    """Moving-block bootstrap on calendar time: a block = all pool rows whose date lies in
    [d, d+B-1] for a randomly drawn pool date d. Blocks are drawn until the original size is reached."""
    # integer day numbers: avoids Timestamp vs numpy.datetime64 key mismatches across pandas versions
    row_day = ((pd.to_datetime(pool["date"]) - pd.Timestamp("2000-01-01")).dt.days).to_numpy()
    days = np.unique(row_day)
    rows_by_day = {int(k): np.flatnonzero(row_day == k) for k in days}
    out, n = [], 0
    while n < len(pool):
        i = rng.integers(0, len(days))
        sel = days[(days >= days[i]) & (days < days[i] + B)]
        idx = np.concatenate([rows_by_day[int(k)] for k in sel])
        out.append(idx)
        n += len(idx)
    return pool.iloc[np.concatenate(out)]


def signflip_p(x):
    """Exact two-sided sign-flip test of mean(x)=0 (assumes independent, symmetric excesses under H0)."""
    x = np.asarray(x, float)
    k = len(x)
    obs = abs(x.mean())
    signs = ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1) * 2 - 1
    return float((np.abs((signs * x).mean(1)) >= obs - 1e-12).mean())
