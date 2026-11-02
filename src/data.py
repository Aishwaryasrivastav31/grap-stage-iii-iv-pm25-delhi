"""
data.py - read CPCB station exports, run checks, flag questionable readings, aggregate to daily.

Rules: offsets in the file ("+0000") are stripped and the clock digits read as IST, after a check that the
solar radiation peak (10-14 h) or ozone peak (12-17 h) falls where it should in IST; checked per year.
Masked (set missing): negative pollutants, RH outside 0-100, wind speed < 0 or > 30 m/s, wind direction
outside 0-360. Flagged (kept in the primary analysis, dropped in a sensitivity run): PM2.5 >= 985 ug/m3
(near the instrument ceiling) and PM2.5 > PM10 at the same timestamp.
Hourly mean needs >= MIN_QH of 4 readings; daily mean needs >= MIN_HOURS of 24 valid hours, per variable.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

REGISTRY = {"site_301": "Anand Vihar", "site_1420": "Ashok Vihar", "site_108": "Aya Nagar", "site_1560": "Bawana"}
COLS = {"PM2.5": "pm25", "PM10": "pm10", "NO2": "no2", "CO": "co", "Ozone": "o3",
        "AT": "at", "RH": "rh", "WS": "ws", "SR": "sr"}
VARS = ["pm25", "pm10", "no2", "co", "at", "rh", "ws"]
WS_MAX, PM_CEILING = 30.0, 985.0


class DataError(Exception):
    pass


def read_station(path: str, start: str, end: str, assume_ist: bool = False):
    fname = os.path.basename(path)
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    keep = {}
    for c in raw.columns:
        base = re.split(r"\s*\(", c)[0].strip()
        if c in ("Station ID", "Station Name", "Timestamp"):
            keep[c] = {"Station ID": "station_id", "Station Name": "station_name", "Timestamp": "ts"}[c]
        elif base in COLS:
            keep[c] = COLS[base]
    df = raw[list(keep)].rename(columns=keep)
    df["ts"] = pd.to_datetime(df["ts"].str.replace(r"(Z|[+-]\d{2}:?\d{2})$", "", regex=True),
                              format="ISO8601", errors="coerce")
    n_bad_ts = int(df["ts"].isna().sum())
    df = df[df["ts"].notna()].copy()
    for c in [c for c in COLS.values() if c in df]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ids = df["station_id"].dropna().unique()
    if len(ids) != 1 or ids[0] not in REGISTRY:
        raise DataError(f"{fname}: station id(s) {list(ids)} not in registry {list(REGISTRY)}")
    station = REGISTRY[ids[0]]
    if not df["station_name"].dropna().str.startswith(station).all():
        raise DataError(f"{fname}: station name does not match registry ({station})")

    # clock check per year
    clock = []
    for yr, g in df.groupby(df["ts"].dt.year):
        h = g["ts"].dt.hour
        if "sr" in g and g["sr"].notna().sum() > 1000:
            pk, how, ok = int(g["sr"].groupby(h).mean().idxmax()), "solar radiation", None
            ok = 10 <= pk <= 14
        elif "o3" in g and g["o3"].notna().sum() > 1000:
            pk, how = int(g["o3"].groupby(h).mean().idxmax()), "ozone"
            ok = 12 <= pk <= 17
        else:
            pk, how, ok = None, "none", assume_ist
        if not ok:
            raise DataError(f"{fname} {yr}: clock check failed ({how} peak {pk}); use --assume-ist only if the source is IST")
        clock.append(f"{yr}: {how} peak {pk}:00")

    # duplicates
    dup = df.duplicated("ts", keep=False)
    n_dup = int(dup.sum())
    if dup.any():
        vals = [c for c in COLS.values() if c in df]
        if (df[dup].groupby("ts")[vals].nunique(dropna=False) > 1).any(axis=None):
            raise DataError(f"{fname}: conflicting duplicate timestamps")
        df = df.drop_duplicates("ts")

    df = df[(df.ts >= start) & (df.ts < pd.Timestamp(end) + pd.Timedelta(days=1))].copy()
    audit = {"file": fname, "station": station, "station_id": ids[0], "rows_in_period": len(df),
             "unparseable_timestamps": n_bad_ts, "duplicate_rows_identical": n_dup, "clock_check": "; ".join(clock)}
    for c in ("pm25", "pm10", "no2", "co", "o3"):
        if c in df:
            audit[f"{c}_negative_masked"] = int((df[c] < 0).sum())
            df.loc[df[c] < 0, c] = np.nan
    if "rh" in df:
        audit["rh_out_of_range_masked"] = int(((df.rh < 0) | (df.rh > 100)).sum())
        df.loc[(df.rh < 0) | (df.rh > 100), "rh"] = np.nan
    if "ws" in df:
        audit["ws_out_of_range_masked"] = int(((df.ws < 0) | (df.ws > WS_MAX)).sum())
        df.loc[(df.ws < 0) | (df.ws > WS_MAX), "ws"] = np.nan
    df["flag_ceiling"] = df["pm25"] >= PM_CEILING
    df["flag_pm25_gt_pm10"] = (df["pm25"] > df["pm10"]) if "pm10" in df else False
    audit["pm25_flag_ceiling"] = int(df["flag_ceiling"].sum())
    audit["pm25_flag_gt_pm10"] = int(df["flag_pm25_gt_pm10"].sum())
    audit["pm25_valid_readings"] = int(df["pm25"].notna().sum())
    df["station"] = station
    return df, audit


def to_daily(df: pd.DataFrame, start: str, end: str, min_qh: int = 3, min_hours: int = 18,
             drop_flagged: bool = False) -> pd.DataFrame:
    df = df.copy()
    if drop_flagged:
        df.loc[df["flag_ceiling"] | df["flag_pm25_gt_pm10"], "pm25"] = np.nan
    vars_ = [c for c in VARS if c in df]
    full = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(days=1), freq="15min", inclusive="left")
    s = df.set_index("ts").reindex(full)
    h = s[vars_].resample("h")
    hourly = h.mean().where(h.count() >= min_qh)
    d = hourly.resample("D")
    daily = d.mean().where(d.count() >= min_hours)
    for c in VARS:
        if c not in daily:
            daily[c] = np.nan
    daily["pm25_valid_hours"] = hourly["pm25"].resample("D").count()
    daily["station"] = df["station"].iat[0]
    return daily.rename_axis("date").reset_index()
