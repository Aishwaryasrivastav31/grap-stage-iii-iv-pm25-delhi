"""figures.py - all report figures. Grey for context, one blue accent for what the title is about."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, GREY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.6, "legend.frameon": False})
STAGE3, STAGE4 = (0.92, 0.41, 0.20, 0.14), (0.92, 0.41, 0.20, 0.34)


def pct(v):
    return 100 * (np.exp(v) - 1)


def _shade(ax, level, ymax=None):
    on = level[level >= 3]
    for d, l in on.items():
        ax.axvspan(d, d + pd.Timedelta(days=1), color=STAGE4 if l == 4 else STAGE3, lw=0, zorder=0)


def _save(fig, path, note=None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def timeline(daily, level, path):
    net = daily.groupby("date")["pm25"].mean()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    _shade(ax, level)
    ax.plot(net.index, net.values, color=GREY, lw=0.8, label="daily PM2.5, mean of stations with data")
    ax.plot(net.index, net.rolling(7, min_periods=4, center=True).mean(), color=BLUE, lw=1.6, label="7-day mean")
    ax.axhline(250, color=INK2, lw=0.8, ls="--")
    ax.annotate("250: 'Severe'\n(sub-index 400)", xy=(1.005, 250), xycoords=("axes fraction", "data"),
                fontsize=7, color=INK2, va="center", ha="left")
    ax.axhline(60, color=INK2, lw=0.6, ls=":")
    ax.annotate("60: 24-h\nNAAQS", xy=(1.005, 60), xycoords=("axes fraction", "data"), fontsize=7, color=INK2,
                va="center", ha="left")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Stage III/IV is invoked at the winter peaks; many severe days fall outside it")
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [Patch(color=STAGE3), Patch(color=STAGE4)], l + ["Stage III day", "Stage IV day"],
              loc="upper center", ncol=4, fontsize=7.5)
    ax.set_ylim(0, max(700, net.max() * 1.05))
    _save(fig, path, "Source: CPCB 15-min data, 4 stations; GRAP days from grap_timeline_v2.csv (late-bound timing, >=12 h rule).")


def overlap(el, W, path):
    el = el.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(el) + 1.4))
    col = {"included (clean entry)": BLUE}
    for i, r in el.iterrows():
        d0 = pd.Timestamp(r["day0"])
        y = len(el) - 1 - i
        c = col.get(r["inclusion"], GREY)
        ax.barh(y, W, left=d0 - pd.Timedelta(days=W), height=0.5, color=c, alpha=0.35, lw=0)
        ax.barh(y, W, left=d0 + pd.Timedelta(days=1), height=0.5, color=c, alpha=0.8, lw=0)
        ax.plot([d0 + pd.Timedelta(hours=12)], [y], marker="|", ms=12, color=INK)
        ax.text(d0 + pd.Timedelta(days=W + 2), y, f"{r['event_id']} {r['day0']}  {r['type']} ({r['episode']})",
                va="center", fontsize=7.5, color=INK if c == BLUE else INK2)
    ax.set_yticks([])
    ax.set_title(f"Only first entries with a clean {W}-day pre-window are used; escalations and re-entries overlap")
    ax.grid(axis="y", visible=False)
    starts = pd.to_datetime(el["day0"])
    ax.set_xlim(starts.min() - pd.Timedelta(days=12), starts.max() + pd.Timedelta(days=75))
    _save(fig, path, f"Pale bar = {W} days before day 0, solid bar = {W} days after. Blue = included; grey = excluded (reason in table 20).")


def support(per_event, pool, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax = axs[0]
    ax.scatter(pool["level0"], pool["rise"], s=8, color=GREY, alpha=0.5, lw=0, label=f"comparison rows (n={len(pool)})")
    sup = per_event["supported"]
    ax.scatter(per_event.loc[sup, "level0"], per_event.loc[sup, "rise"], s=46, color=BLUE, edgecolor=SURF, lw=1.5,
               zorder=3, label="entry event inside pool range")
    ax.scatter(per_event.loc[~sup, "level0"], per_event.loc[~sup, "rise"], s=46, facecolor=SURF, edgecolor=BLUE,
               lw=1.8, zorder=3, label="entry event outside pool 1st-99th pct (extrapolated)")
    for d, r in per_event.iterrows():
        ax.annotate(d.strftime("%d %b %y"), (r.level0, r.rise), xytext=(5, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("day-0 residual level (log scale)")
    ax.set_ylabel("rise over the previous week")
    ax.set_title("Most entry events sit at the edge of the comparison days")
    ax.legend(fontsize=7.5, loc="lower right")
    ax2 = axs[1]
    m = pd.to_datetime(pool["date"]).dt.month.value_counts().reindex([10, 11, 12, 1, 2]).fillna(0)
    me = pd.Series(per_event.index.month).value_counts().reindex([10, 11, 12, 1, 2]).fillna(0)
    xs = np.arange(5)
    ax2.bar(xs - 0.2, m / m.sum(), 0.38, color=GREY, label="comparison")
    ax2.bar(xs + 0.2, me / me.sum(), 0.38, color=BLUE, label="events")
    ax2.set_xticks(xs, ["Oct", "Nov", "Dec", "Jan", "Feb"])
    ax2.set_title("Calendar month (share)")
    ax2.legend(fontsize=7.5)
    _save(fig, path, "Residuals from the weather/season/trend model without stage terms, 3-station network mean.")


def event_forest(pe, P, sens_t, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    ax = axs[0]
    pe = pe.sort_index()
    ys = np.arange(len(pe))[::-1]
    ax.axvline(0, color=INK2, lw=0.8)
    ax.scatter(pe["excess_pct"], ys, color=BLUE, s=36, zorder=3)
    ax.set_yticks(ys, [f"{d} ({e})" for d, e in zip(pe.index, pe["episode"])])
    est, lo, hi = P["estimate_pct"], P["ci_low_pct"], P["ci_high_pct"]
    ax.errorbar([est], [-1.2], xerr=[[est - lo], [hi - est]], fmt="D", color=INK, ms=6, capsize=3)
    ax.text(est, -1.9, f"mean {est:+.1f}% [{lo:+.1f}, {hi:+.1f}]", ha="center", fontsize=7.5)
    ax.set_ylim(-2.5, len(pe))
    ax.set_xlabel("excess change, % on the 1+PM2.5 scale")
    ax.set_title("Event-specific excess changes")
    ax2 = axs[1]
    loo = P["loo"].sort_values("left_out_event")
    ys2 = np.arange(len(loo))[::-1]
    ax2.axvline(0, color=INK2, lw=0.8)
    ax2.axvline(est, color=BLUE, lw=0.8, ls="--")
    ax2.scatter(loo["estimate_pct"], ys2, color=GREY, s=30)
    ax2.set_yticks(ys2, [f"without {d}" for d in loo["left_out_event"]])
    ax2.set_title("Leave one episode out")
    _save(fig, path, "Interval: episode x 14-day block bootstrap, expected-change model refitted in each replicate. Few episodes: treat as approximate.")


def sensitivity(s, path):
    s = s.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(s) + 1.2))
    ax.axvline(0, color=INK2, lw=0.8)
    for i, r in s.iterrows():
        c = BLUE if r["analysis"].startswith("Primary (") else GREY
        ax.plot([r.ci95_low_pct, r.ci95_high_pct], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.scatter([r.estimate_pct], [i], color=c, s=26, zorder=3)
        ax.text(max(s.ci95_high_pct) + 2, i, f"n={r.n_events}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(s)), s["analysis"], fontsize=7.5)
    ax.set_xlabel("excess change after Stage III/IV entry, % (1+PM2.5 scale)")
    ax.set_title("No specification shows a clear reduction; intervals are wide throughout")
    _save(fig, path, "Sensitivity runs use 1,000 bootstrap replicates; primary uses 4,000.")


def coverage(cov, cov_w, event_days, path):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    st = list(cov.columns)
    for i, s in enumerate(st):
        ok = cov[s]
        ax.scatter(ok.index[ok], np.full(ok.sum(), i), marker="|", s=40, color=BLUE, lw=0.8)
        okw = cov_w[s] if s in cov_w else ok & False
        ax.scatter(okw.index[okw], np.full(okw.sum(), i - 0.28), marker="|", s=12, color=GREY, lw=0.6)
    for d in event_days:
        ax.axvline(d, color=ORANGE, lw=0.8, ls="--")
    ax.set_yticks(range(len(st)), st)
    ax.set_ylim(-0.7, len(st) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_title("PM2.5 coverage is near-complete except Anand Vihar gaps; Aya Nagar never has weather")
    _save(fig, path, "Blue = eligible PM2.5 day (>=18 valid hours); grey = eligible wind-speed day; dashed = entry events.")


def test1_plot(t1, path):
    t = t1[t1.term == "stage_III"].iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.34 * len(t) + 1.3))
    ax.axvline(0, color=INK2, lw=0.8)
    for i, r in t.iterrows():
        c = BLUE if "primary" in r["model"] else GREY
        ax.plot([r.ci95_low_pct, r.ci95_high_pct], [i, i], color=c, lw=2)
        ax.scatter([r.ratio_pct], [i], color=c, s=26, zorder=3)
    ax.set_yticks(range(len(t)), [m.replace("T1 ", "") for m in t["model"]], fontsize=7.5)
    ax.set_xlabel("Stage III days vs days without Stage III/IV, % on the 1+PM2.5 scale")
    ax.set_title("Stage III days are dirtier in every specification, as expected when stages follow pollution")
    _save(fig, path, "Associations only: stages are invoked because pollution is high. 95% CI from Driscoll-Kraay (14-day) SEs.")


def severe(sev, level, path):
    m = sev[sev.pm25.notna()].copy()
    m["mon"] = m["date"].dt.to_period("M")
    m["sev_stage"] = m["severe"] & (m["level"] >= 3)
    m["sev_other"] = m["severe"] & (m["level"] < 3)
    g = m.groupby("mon")[["sev_stage", "sev_other"]].sum()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    xs = np.arange(len(g))
    ax.bar(xs, g["sev_stage"], color=BLUE, width=0.75, label="severe station-days under Stage III/IV")
    ax.bar(xs, g["sev_other"], bottom=g["sev_stage"], color=GREY, width=0.75, label="severe station-days without III/IV")
    ax.set_xticks(xs, [p.strftime("%b\n%y") if p.month in (1, 11) else p.strftime("%b") for p in g.index], fontsize=7)
    ax.set_ylabel("station-days, PM2.5 > 250")
    ax.set_title("A large share of severe station-days occur without Stage III/IV in force")
    ax.legend(fontsize=7.5)
    _save(fig, path, "Severe = daily PM2.5 > 250 µg/m³ (PM2.5 sub-index > 400). Sum over the 4 stations.")


def missingness(t3, path):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ys = np.arange(len(t3))[::-1]
    ax.axvline(0, color=INK2, lw=0.8)
    for y, (_, r) in zip(ys, t3.iterrows()):
        ax.plot([r.ci95_low_pp, r.ci95_high_pp], [y, y], color=BLUE, lw=2)
        ax.scatter([r.difference_pp], [y], color=BLUE, s=30, zorder=3)
    ax.set_yticks(ys, t3["station"])
    ax.set_xlabel("missing PM2.5 hours: Stage III/IV days minus other Oct-Feb days (percentage points)")
    ax.set_title("No station has more missing data on Stage III/IV days")
    _save(fig, path, "95% CI from a weekly block bootstrap.")


def event_paths(P, W, path):
    r = P["net"]["r"]
    pool = P["pool"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ks = range(-W, W + 1)
    prof = lambda days: pd.DataFrame({k: [r.get(d + pd.Timedelta(days=k), np.nan) for d in days] for k in ks})
    ev = prof(P["per_event"].index)
    # nearest comparison rows in (level0, rise): 5 per event, for illustration only
    near = []
    for _, e in P["per_event"].iterrows():
        dist = (pool.level0 - e.level0) ** 2 + (pool.rise - e.rise) ** 2
        near += list(pool.loc[dist.nsmallest(5).index, "date"])
    cp = prof(near)
    for i in range(len(ev)):
        ax.plot(list(ks), ev.iloc[i], color=BLUE, alpha=0.18, lw=0.8)
    ax.plot(list(ks), ev.mean(), color=BLUE, lw=2, marker="o", ms=3, label=f"entry events (n={len(ev)})")
    ax.plot(list(ks), cp.mean(), color=GREY, lw=2, ls="--", marker="s", ms=3, label="5 most similar comparison days per event")
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("days from day 0")
    ax.set_ylabel("weather-adjusted log(1+PM2.5) residual")
    ax.set_title("Entries follow a sharper rise; afterwards PM2.5 falls back, but not faster than after similar days")
    ax.legend(fontsize=7.5)
    _save(fig, path, "Illustration only; the estimate uses the expected-change model, not these nearest neighbours.")


def placebo(P, path):
    null = P.get("placebo_null")
    if null is None or len(null) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(pct(null), bins=40, color=GREY)
    ax.axvline(P["estimate_pct"], color=BLUE, lw=2)
    ax.text(P["estimate_pct"], ax.get_ylim()[1] * 0.92, f"  observed {P['estimate_pct']:+.1f}%", color=BLUE, fontsize=8)
    ax.set_xlabel("mean 'excess' of pseudo-events drawn from comparison days (same or nearest month), %")
    ax.set_title("The observed excess is within the range seen on dates with no GRAP entry")
    _save(fig, path, f"Seasonal placebo-in-time, {len(null)} draws; two-sided p = {P.get('placebo_p', np.nan):.2f}. "
          f"{100 * P.get('placebo_same_month_share', np.nan):.0f}% of pseudo-events fall in the event's own month (few Nov-Dec comparison days).")
