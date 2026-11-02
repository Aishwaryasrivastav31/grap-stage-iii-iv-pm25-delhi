"""
analysis.py - the revised analyses (v2). Every function returns tables; run_all.py writes them.

  Test 1  Adjusted stage-day associations (pooled OLS, calendar-day Driscoll-Kraay SEs)
  Test 2  Regression-to-the-mean-adjusted event comparison (entry episodes, clean pool,
          episode x moving-block bootstrap, sign-flip and seasonal placebo-in-time tests)
  Test 3  Missing-data comparison (weekly block bootstrap)
  Test 4  Cross-pollutant associations (secondary, BH within the family of 6 contrasts)
  Severe  Severe-day descriptive analysis (PM2.5 sub-index bands)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stats import ols_dk, p_normal, bh, fit_expected, predict, block_resample, signflip_p
from timeline import daily_exposure, episodes, classify_starts, intervals

START, END = "2024-01-01", "2025-12-31"
WEATHER = ["at", "rh", "ws"]
FOURIER_K = 3
SEED = 2026
WINTER = (10, 11, 12, 1, 2)


# ---------------------------------------------------------------------------- design
def design(d: pd.DataFrame, weather: bool, stages: bool, season: str = "fourier") -> pd.DataFrame:
    t = (d["date"] - pd.Timestamp(START)).dt.days.to_numpy(float)
    X = pd.DataFrame({"intercept": 1.0, "trend_years": t / 365.25}, index=d.index)
    if season == "fourier":
        doy = d["date"].dt.dayofyear.to_numpy(float)
        for k in range(1, FOURIER_K + 1):
            X[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
            X[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    else:                                                   # month fixed effects
        for m in sorted(d["date"].dt.month.unique())[1:]:
            X[f"month_{m}"] = (d["date"].dt.month == m).astype(float)
    for k in range(1, 7):
        X[f"weekday_{k}"] = (d["date"].dt.dayofweek == k).astype(float)
    for s in sorted(d["station"].unique())[1:]:
        X[f"station_{s}"] = (d["station"] == s).astype(float)
    if weather:
        for c in WEATHER:
            X[c] = d[c]
    if stages:
        X["stage_III"] = (d["level"] == 3).astype(float)
        X["stage_IV"] = (d["level"] == 4).astype(float)
    return X


def day_index(d):
    return (d["date"] - pd.Timestamp(START)).dt.days.to_numpy()


# ---------------------------------------------------------------------------- Test 1
def test1(daily: pd.DataFrame, name: str, outcome="pm25", weather=True, season="fourier", subset=None,
          balanced=False):
    cols = [outcome] + (WEATHER if weather else [])
    x = daily if subset is None else daily[subset]
    x = x.dropna(subset=cols).copy()
    if balanced:                                            # keep days on which every station in x is present
        n_st = x["station"].nunique()
        ok = x.groupby("date")["station"].transform("nunique") == n_st
        x = x[ok]
    if x.empty or (x["level"] >= 3).sum() < 5:
        return [], None
    X = design(x, weather, True, season)
    b, e, se = ols_dk(X.to_numpy(float), np.log1p(x[outcome].to_numpy(float)), day_index(x))
    info = {"model": name, "outcome": outcome, "station_days": len(x), "calendar_days": x["date"].nunique(),
            "stations": ", ".join(sorted(x["station"].unique())),
            "stage_III_station_days": int((x.level == 3).sum()), "stage_IV_station_days": int((x.level == 4).sum())}
    rows = []
    for term in ("stage_III", "stage_IV"):
        j = list(X.columns).index(term)
        s = se[14][j]
        lo, hi = b[j] - 1.96 * s, b[j] + 1.96 * s
        rows.append({"model": name, "outcome": outcome, "term": term, "beta_log1p": b[j], "se_dk14": s,
                     "se_dk7": se[7][j], "se_dk28": se[28][j],
                     "ratio_pct": 100 * (math.exp(b[j]) - 1), "ci95_low_pct": 100 * (math.exp(lo) - 1),
                     "ci95_high_pct": 100 * (math.exp(hi) - 1), "p_value": p_normal(b[j] / s),
                     "station_days": len(x)})
    return rows, info


def verdict(lo, hi, outcome):
    lab = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "co": "CO"}.get(outcome, outcome)
    return (f"lower {lab} (CI below 0)" if hi < 0 else f"higher {lab} (CI above 0)" if lo > 0
            else "CI includes 0")


# ---------------------------------------------------------------------------- Test 2 building blocks
def residual_panel(daily: pd.DataFrame, weather: bool, fit_on: str = "all", season="fourier"):
    """Station-day residuals from the adjustment model WITHOUT stage terms.
    fit_on='all': coefficients estimated on all days (original); 'nonstage': on days below Stage III only,
    then applied to all days."""
    cols = ["pm25"] + (WEATHER if weather else [])
    x = daily.dropna(subset=cols).copy()
    X = design(x, weather, False, season)
    y = np.log1p(x["pm25"].to_numpy(float))
    m = np.ones(len(x), bool) if fit_on == "all" else (x["level"] < 3).to_numpy()
    b, *_ = np.linalg.lstsq(X.to_numpy(float)[m], y[m], rcond=None)
    x["resid"] = y - X.to_numpy(float) @ b
    return x[["date", "station", "resid", "pm25", "level"]]


def network(resid: pd.DataFrame):
    full = pd.date_range(START, END, freq="D")
    g = resid.groupby("date")
    return pd.DataFrame({"r": g["resid"].mean().reindex(full), "n_st": g["station"].nunique().reindex(full).fillna(0),
                         "stations": g["station"].apply(lambda s: ",".join(sorted(s))).reindex(full).fillna("")})


def min_obs(W):
    return max(2, math.ceil(3 * W / 7))


def window_stats(r: pd.Series, d0, W):
    pre = r.loc[d0 - pd.Timedelta(days=W): d0 - pd.Timedelta(days=1)]
    post = r.loc[d0 + pd.Timedelta(days=1): d0 + pd.Timedelta(days=W)]
    lev = r.get(d0, np.nan)
    if pre.notna().sum() < min_obs(W) or post.notna().sum() < min_obs(W) or not np.isfinite(lev):
        return np.nan, np.nan, np.nan, int(pre.notna().sum()), int(post.notna().sum())
    return post.mean() - pre.mean(), lev, lev - pre.mean(), int(pre.notna().sum()), int(post.notna().sum())


def pool_days(level: pd.Series, W: int, months=WINTER, exclude_dates=()):
    s0, s1 = pd.Timestamp(START) + pd.Timedelta(days=W), pd.Timestamp(END) - pd.Timedelta(days=W)
    hi = (level >= 3).astype(int)
    roll = hi.rolling(2 * W + 1, center=True).max()          # any Stage III/IV day in d-W..d+W
    days = [d for d in level.index if s0 <= d <= s1 and d.month in months and roll.get(d, 1) == 0]
    if exclude_dates:
        ex = pd.DatetimeIndex(exclude_dates)
        days = [d for d in days if not any(abs((d - e).days) <= W + 3 for e in ex)]
    return days


def build_rows(series_by_unit: dict, dates, W):
    rows = []
    for unit, r in series_by_unit.items():
        for d in dates:
            ch, lev, rise, npre, npost = window_stats(r, d, W)
            rows.append({"date": d, "unit": unit, "change": ch, "level0": lev, "rise": rise,
                         "n_pre": npre, "n_post": npost})
    return pd.DataFrame(rows)


def event_estimate(ev: pd.DataFrame, pool: pd.DataFrame):
    coef = fit_expected(pool)
    ev = ev.copy()
    ev["expected"] = predict(coef, ev)
    ev["excess"] = ev["change"] - ev["expected"]
    per_event = ev.groupby("date").agg(change=("change", "mean"), expected=("expected", "mean"),
                                       excess=("excess", "mean"), level0=("level0", "mean"), rise=("rise", "mean"),
                                       units=("unit", "nunique"))
    return per_event, coef


def bootstrap(ev: pd.DataFrame, pool: pd.DataFrame, B: int, reps: int, rng):
    """Episode x block bootstrap: entry events (one per episode) resampled with replacement; comparison
    pool resampled in calendar blocks of B days; the expected-change model is refitted every replicate."""
    dates = ev["date"].unique()
    out = []
    for _ in range(reps):
        pick = rng.choice(dates, len(dates), replace=True)
        e_b = pd.concat([ev[ev["date"] == d] for d in pick])
        p_b = block_resample(pool, B, rng)
        coef = fit_expected(p_b)
        ex = e_b["change"].to_numpy() - predict(coef, e_b)
        # average within event first, then across events (as in the point estimate)
        e_b = e_b.assign(ex=ex, k=np.repeat(np.arange(len(pick)), [int((ev["date"] == d).sum()) for d in pick]))
        out.append(e_b.groupby("k")["ex"].mean().mean())
    return np.array(out)


def placebo_in_time(ev: pd.DataFrame, pool: pd.DataFrame, W: int, reps: int, rng):
    """Seasonal placebo: for each real event draw a pool date from the same calendar month, compute the
    'excess' of these pseudo-events with the model refitted on pool rows whose windows do not overlap them.
    Returns the null distribution of the mean excess."""
    months = pd.to_datetime(ev["date"].unique()).month
    pdates = pd.to_datetime(pool["date"].unique())
    out = []
    placebo_in_time.same_month = []                           # share of pseudo-events in the event's own month
    for _ in range(reps):
        pick, same = [], 0
        for m in months:
            near_m = {m, (m % 12) + 1, ((m - 2) % 12) + 1}        # same month, else neighbouring, else any Oct-Feb
            for allowed in ({m}, near_m, set(WINTER)):
                cand = [d for d in pdates[np.isin(pdates.month, list(allowed))]
                        if all(abs((d - q).days) > W for q in pick)]
                if cand:
                    break
            if not cand:
                break
            choice = cand[rng.integers(0, len(cand))]
            same += choice.month == m
            pick.append(choice)
        if len(pick) < len(months):
            continue
        placebo_in_time.same_month.append(same / len(months))
        near = pool["date"].apply(lambda d: any(abs((d - q).days) <= 2 * W for q in pick))
        coef = fit_expected(pool[~near])
        pe = pool[pool["date"].isin(pick)]
        exc = pe["change"].to_numpy() - predict(coef, pe)
        out.append(pd.Series(exc, index=pe["date"].to_numpy()).groupby(level=0).mean().mean())
    return np.array(out)


def support_flags(per_event: pd.DataFrame, pool: pd.DataFrame):
    lo0, hi0 = pool["level0"].quantile([0.01, 0.99])
    lor, hir = pool["rise"].quantile([0.01, 0.99])
    f = per_event.copy()
    f["level0_pctile_in_pool"] = [100 * (pool["level0"] < v).mean() for v in f["level0"]]
    f["rise_pctile_in_pool"] = [100 * (pool["rise"] < v).mean() for v in f["rise"]]
    f["supported"] = (f["level0"].between(lo0, hi0)) & (f["rise"].between(lor, hir))
    return f


def to_pct(v):
    return 100 * (math.exp(v) - 1)
