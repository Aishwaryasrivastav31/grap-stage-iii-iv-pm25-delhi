# GRAP Stage III/IV and PM2.5 in Delhi, 2024–2025: revised study (v2)

Four CPCB stations: Anand Vihar, Ashok Vihar, Aya Nagar, Bawana. 15-minute data, 1 Jan 2024 – 31 Dec 2025.

**Start with `reports/REPORT.md`.** It contains the findings, what each analysis estimates, and its limits.

## How to run

Requirements: Python 3.10+, numpy, pandas, matplotlib. No statsmodels or scipy needed.

```
pip install -r requirements.txt
python tests/test_exposure.py        # 8 tests: exposure coding, episodes, timeline bounds
python tests/test_stats.py           # 5 tests: covariance vs brute force, calibration, BH, sign-flip, blocks
python run_all.py                    # full analysis, about 10-20 min (4,000 bootstrap replicates)
python run_all.py --quick            # about 1 min, fewer replicates, for checking the setup
python original/grap_fair_test.py    # optional: the original v1 analysis, unchanged
```

Windows PowerShell: run the same commands from the project folder; paths are handled by Python.

## Folder layout

| Path | What it is |
|---|---|
| `data/raw/` | The four CPCB station exports (unchanged) |
| `data/external/grap_timeline_v2.csv` | GRAP orders: order date, early/late effective times, sources, verification status |
| `data/external/co_interventions.csv` | Other policies and events in force (Winter Action Plans, Diwali, revised GRAP schedule) |
| `src/` | `data.py` (reading, checks, aggregation) · `timeline.py` (exposure, episodes) · `stats.py` (regression, covariance, resampling) · `analysis.py` (tests) · `figures.py` |
| `tests/` | Unit tests |
| `run_all.py` | Runs everything; the primary specification is the `PLAN` dictionary at the top |
| `reports/REPORT.md` | Research report |
| `reports/AUDIT.md` | Issue register from the audit of v1 |
| `reports/ANALYSIS_PLAN.md` | Original plan, corrections, amendments, exploratory parts |
| `reports/CHANGELOG.md` | Each correction and its effect on the results |
| `reports/v2/tables/`, `reports/v2/figures/` | All outputs (table numbers are cited in the report) |
| `original/` | The v1 script and its exact outputs, kept for comparison |

## Before publishing

- Every GRAP time in `grap_timeline_v2.csv` comes from PIB, AIR or news reports. Check each one against the CAQM order on caqm.nic.in and update `verification_status`.
- The Stage I/II dates and the 21 Nov 2025 schedule revision came from the author and are unverified.
