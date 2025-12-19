# Data download + monthly alignment for:
# - S&P 500 daily (Yahoo Finance ^GSPC)
# - Risk-free rate (try FRED DGS3MO daily; fallback to Yahoo ^IRX)
# - Monthly Sunspot Number (SILSO SN_m_tot_V2.0.txt)
#
# Output: a monthly DataFrame with (log) excess returns and lagged sunspot features,
# plus z-score normalized SN for coefficient interpretability.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

# -----------------------------
# 0) Date range: past 20 years from 19/12/2025
# -----------------------------
END_DATE = pd.Timestamp("2025-12-19")
START_DATE = END_DATE - pd.DateOffset(years=20)

# -----------------------------
# 1) Download daily S&P 500 (^GSPC)
# -----------------------------
spx = yf.download(
    "^GSPC",
    start=START_DATE.strftime("%Y-%m-%d"),
    end=(END_DATE + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    auto_adjust=False,
    progress=False,
)
if spx.empty:
    raise RuntimeError("Failed to download ^GSPC from yfinance.")

# Ensure Adj Close exists
if "Adj Close" not in spx.columns:
    raise RuntimeError("Adj Close not found in ^GSPC download. Check yfinance output.")

spx = spx[["Adj Close"]].rename(columns={"Adj Close": "spx_adj_close"}).dropna()
spx.index = pd.to_datetime(spx.index)

# Daily log returns
spx["spx_log_ret_d"] = np.log(spx["spx_adj_close"]).diff()

# Monthly price (last trading day of month) and monthly log return
spx_m = spx["spx_adj_close"].resample("M").last().to_frame("spx_adj_close_m")
spx_m["spx_log_ret_m"] = np.log(spx_m["spx_adj_close_m"]).diff()

# -----------------------------
# 2) Download interest rate data to compute excess returns
#    Preferred: FRED daily DGS3MO (3-month Treasury constant maturity rate, % p.a.)
#    Fallback: Yahoo Finance ^IRX (13-week T-bill yield, % p.a.)
# -----------------------------
def get_rf_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    # Try FRED (no API key usually required)
    try:
        from pandas_datareader.data import DataReader
        rf = DataReader("DGS3MO", "fred", start, end)  # percent per annum
        rf = rf.rename(columns={"DGS3MO": "rf_yield_pct_pa"})
        rf.index = pd.to_datetime(rf.index)
        return rf["rf_yield_pct_pa"]
    except Exception as e:
        print(f"[Info] FRED download failed ({type(e).__name__}). Falling back to yfinance ^IRX.")
        irx = yf.download(
            "^IRX",
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )
        if irx.empty or "Adj Close" not in irx.columns:
            raise RuntimeError("Failed to download risk-free proxy from both FRED and yfinance.")
        irx = irx[["Adj Close"]].rename(columns={"Adj Close": "rf_yield_pct_pa"}).dropna()
        irx.index = pd.to_datetime(irx.index)
        return irx["rf_yield_pct_pa"]

rf_d = get_rf_daily(START_DATE, END_DATE)

# Convert daily yields (% p.a.) into monthly log risk-free returns
# Approximation: monthly simple rf ≈ (annual_yield/100) / 12, then log(1 + rf_monthly_simple)
rf_m = rf_d.resample("M").mean().to_frame("rf_yield_pct_pa_m")
rf_m["rf_log_ret_m"] = np.log1p((rf_m["rf_yield_pct_pa_m"] / 100.0) / 12.0)

# Monthly log excess return target
# excess = market log return - risk-free log return
monthly = spx_m.join(rf_m, how="inner")
monthly["excess_log_ret_m"] = monthly["spx_log_ret_m"] - monthly["rf_log_ret_m"]

# -----------------------------
# 3) Download monthly Sunspot Numbers (SILSO / SIDC)
# -----------------------------
SILSO_SN_MONTHLY_URL = "https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.txt"

resp = requests.get(SILSO_SN_MONTHLY_URL, timeout=30)
resp.raise_for_status()

# File format (whitespace-delimited) typically:
# YEAR  MONTH  decimal_date  SN  SD  N_obs
sn_raw = pd.read_csv(
    StringIO(resp.text),
    delim_whitespace=True,
    header=None,
    comment="#",
)

# Try to name columns defensively based on expected structure
# If the format changes, you'll see it quickly here.
if sn_raw.shape[1] < 4:
    raise RuntimeError(f"Unexpected SILSO monthly SN file format: {sn_raw.shape[1]} columns found.")

# Use first 6 columns if available, otherwise name what we have
colnames = ["year", "month", "decimal_date", "sn", "sn_sd", "n_obs"]
sn_raw = sn_raw.iloc[:, : min(sn_raw.shape[1], len(colnames))]
sn_raw.columns = colnames[: sn_raw.shape[1]]

# Create a month-end timestamp
sn_raw["date"] = pd.to_datetime(dict(year=sn_raw["year"], month=sn_raw["month"], day=1)) + pd.offsets.MonthEnd(0)
sn_raw = sn_raw.set_index("date").sort_index()

# Restrict to past 20 years (monthly span that overlaps)
sn = sn_raw.loc[START_DATE - pd.offsets.MonthEnd(1) : END_DATE].copy()

# Missing data handling:
# SILSO uses -1 to indicate missing SN; 0 is a valid value (solar minima months can be ~0).
# The prompt notes "occasionally marked as -1 or 0" — treat negative as missing; keep 0.
sn["sn_is_missing"] = sn["sn"] < 0
missing_months = sn["sn_is_missing"].sum()

# Replace missing with NaN
sn.loc[sn["sn_is_missing"], "sn"] = np.nan

# Build a PeriodIndex (YYYY-MM) if you prefer that representation
sn["period"] = sn.index.to_period("M")

# Check missing months in the monthly index itself
expected_months = pd.period_range(sn.index.min().to_period("M"), sn.index.max().to_period("M"), freq="M")
missing_periods = expected_months.difference(sn["period"].unique())

print(f"[SN] Missing SN values flagged (sn < 0): {int(missing_months)}")
print(f"[SN] Missing monthly periods in index range: {len(missing_periods)}")
if len(missing_periods) > 0:
    print("[SN] Example missing periods:", list(missing_periods[:10]))

# -----------------------------
# 4) Z-score normalization for SN (for interpretability)
# -----------------------------
# Full-sample z-score (fine for interpretation; for strict OOS prediction later, consider expanding z-score)
sn_mean = sn["sn"].mean(skipna=True)
sn_std = sn["sn"].std(skipna=True, ddof=0)  # population std for clean scaling
sn["sn_z"] = (sn["sn"] - sn_mean) / sn_std

# Optional: expanding z-score that avoids using future information (useful later in OOS tests)
# (This uses only information up to *that month*. You will still lag SN by 1 month for the return month.)
exp_mean = sn["sn"].expanding(min_periods=12).mean()
exp_std = sn["sn"].expanding(min_periods=12).std(ddof=0)
sn["sn_z_expanding"] = (sn["sn"] - exp_mean) / exp_std

# Basic additional SN features (optional but commonly useful)
sn["sn_delta"] = sn["sn"].diff()
sn["sn_delta_z"] = sn["sn_z"].diff()
sn["sn_ma12"] = sn["sn"].rolling(12).mean()
sn["sn_ma12_z"] = (sn["sn_ma12"] - sn["sn_ma12"].mean(skipna=True)) / sn["sn_ma12"].std(skipna=True, ddof=0)

# -----------------------------
# 5) Align monthly SN with monthly returns WITHOUT look-ahead:
#    SN observed for month M can only be used for returns from month M+1 onward.
#    Since monthly returns are indexed by month-end, we shift SN forward by 1 month to match return month.
# -----------------------------
sn_features = sn[["sn", "sn_z", "sn_z_expanding", "sn_delta", "sn_delta_z", "sn_ma12", "sn_ma12_z"]].copy()
sn_features = sn_features.add_prefix("sn_")

# Lag SN by 1 month for predictive use in month m returns
sn_features_lag1 = sn_features.shift(1).add_suffix("_lag1")

monthly = monthly.join(sn_features_lag1, how="left")

# Final sanity checks / summary
out = monthly.dropna(subset=["excess_log_ret_m"]).copy()

print("\n[Monthly dataset]")
print("Start:", out.index.min().date(), "End:", out.index.max().date())
print("Rows:", len(out))
print("\nColumns:", list(out.columns))

# 'out' is your clean monthly dataset with:
# - spx_log_ret_m
# - rf_log_ret_m
# - excess_log_ret_m (initial target)
# - lagged sunspot features (sn_*_lag1), including z-scores
#
# Example: use out['excess_log_ret_m'] as target, out['sn_sn_z_lag1'] as predictor.
