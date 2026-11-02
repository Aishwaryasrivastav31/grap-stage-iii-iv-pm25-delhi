"""
timeline.py - GRAP timeline, daily exposure coding and episode construction.

Conventions (fixed before any revised outcome was examined)
  * Intervals are half-open [start, end) in IST, at minute resolution.
  * A day's exposure to level >= L = total hours of that calendar day (IST, 00:00-24:00) covered by the
    union of all orders with stage >= L. Overlapping III and IV orders are therefore handled by union,
    and when Stage IV is revoked while III continues the hours fall back to level III automatically.
  * Daily level = the highest L in {III, IV} with exposure >= threshold hours (primary 12 h);
    otherwise 0 = "no Stage III/IV" (this includes no stage, Stage I/II and any unknown lower stage:
    Stage I/II are not coded because their history is incomplete).
  * Timing scenarios: 'late' (primary) = start/end at the documented time (the time by which the order
    is known to be in effect / revoked); 'early' = earliest plausible effective time; 'max' = early
    start + late end; 'min' = late start + early end.
  * Episode = maximal run of Stage III/IV days, runs separated by <= GAP days below III merged
    (primary GAP = 7; sensitivity 3 and 14).
  * Entry event = first Stage III/IV day of an episode that starts inside the study and whose W days
    before are all below III (and inside the study). Other starts are classified as escalation
    (III -> IV while III in force) or re-entry (new start inside an existing episode).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}
SCENARIOS = {"late": ("late", "late"), "early": ("early", "early"),
             "max": ("early", "late"), "min": ("late", "early")}


def load_timeline(path: Path, study_end: str) -> pd.DataFrame:
    t = pd.read_csv(path, dtype=str).fillna("")
    t["level"] = t["stage"].map(ROMAN)
    if t["level"].isna().any():
        raise ValueError("unknown stage in timeline")
    for c in ("effective_start_early_ist", "effective_start_late_ist", "effective_end_early_ist", "effective_end_late_ist"):
        t[c] = pd.to_datetime(t[c].replace("", np.nan))
    used = t[t["use_in_analysis"].str.lower() == "yes"].copy()
    far = pd.Timestamp(study_end) + pd.Timedelta(days=30)
    for c in ("effective_end_early_ist", "effective_end_late_ist"):
        used[c] = used[c].fillna(far)
    bad = (used.effective_start_early_ist > used.effective_start_late_ist) | \
          (used.effective_end_early_ist > used.effective_end_late_ist)
    if bad.any():
        raise ValueError(f"early bound after late bound in rows {list(used.loc[bad, 'order_id'])}")
    return used


def intervals(tl: pd.DataFrame, scenario: str = "late") -> pd.DataFrame:
    s, e = SCENARIOS[scenario]
    out = pd.DataFrame({"order_id": tl["order_id"], "level": tl["level"],
                        "start": tl[f"effective_start_{s}_ist"], "end": tl[f"effective_end_{e}_ist"]})
    return out[out["end"] > out["start"]].reset_index(drop=True)


def _union(iv: list[tuple[pd.Timestamp, pd.Timestamp]]):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def daily_exposure(iv: pd.DataFrame, start: str, end: str, threshold_h: float = 12.0) -> pd.DataFrame:
    """One row per calendar day: hours at level>=III, hours at level>=IV, daily level (0/3/4)."""
    days = pd.date_range(start, end, freq="D")
    res = pd.DataFrame(index=days)
    for L in (3, 4):
        u = _union([(r.start, r.end) for r in iv.itertuples() if r.level >= L])
        hrs = np.zeros(len(days))
        for a, b in u:
            for i, d in enumerate(days):
                lo, hi = max(a, d), min(b, d + pd.Timedelta(days=1))
                if hi > lo:
                    hrs[i] += (hi - lo).total_seconds() / 3600
        res[f"hours_ge{L}"] = hrs
    lvl = np.zeros(len(days), int)
    lvl[res["hours_ge3"].to_numpy() >= threshold_h] = 3
    lvl[res["hours_ge4"].to_numpy() >= threshold_h] = 4
    res["level"] = lvl
    res.index.name = "date"
    return res


def episodes(level: pd.Series, gap: int = 7) -> pd.DataFrame:
    """Maximal runs of level>=3 days, merging runs separated by <= gap days below III."""
    on = level >= 3
    runs, cur = [], None
    for d, v in on.items():
        if v:
            if cur is not None and (d - cur[1]).days - 1 <= gap:
                cur[1] = d
            else:
                cur = [d, d]
                runs.append(cur)
    return pd.DataFrame([{"episode": f"P{i + 1}", "first_day": a, "last_day": b,
                          "stage_days": int(on[a:b].sum()), "stage_iv_days": int((level[a:b] == 4).sum())}
                         for i, (a, b) in enumerate(runs)])


def classify_starts(level: pd.Series, eps: pd.DataFrame, study_start: str, window: int = 7) -> pd.DataFrame:
    """Every day on which the daily level rises (0->3, 0->4, 3->4) is a start; classify it."""
    prev = level.shift(1, fill_value=0)
    rows = []
    s0 = pd.Timestamp(study_start)
    for d in level.index[(level > prev)]:
        ep = eps[(eps.first_day <= d) & (eps.last_day >= d)].iloc[0]
        frm, to = int(prev[d]), int(level[d])
        if d == ep.first_day:
            pre = level[d - pd.Timedelta(days=window): d - pd.Timedelta(days=1)]
            left_censored = (d - pd.Timedelta(days=window)) < s0 or d == level.index[0]
            kind = "entry" if not left_censored else "entry (left-censored)"
            clean = (not left_censored) and bool((pre < 3).all())
        elif frm >= 3:
            kind, clean = "escalation", False
        else:
            kind, clean = "re-entry", False
        rows.append({"day0": d, "episode": ep.episode, "type": kind, "from_level": frm, "to_level": to,
                     "pre_window_clean": clean})
    return pd.DataFrame(rows)
