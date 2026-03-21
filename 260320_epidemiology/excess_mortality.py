"""
Excess mortality analysis – Brussels Region
Model: log(deaths) ~ constant + annual harmonic
Envelope: 97.72nd percentile of residuals (≈ +2 SD)
COVID mask: March 2020 – December 2020 (excluded from fit & residual quantile)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── 1. Load & prepare ─────────────────────────────────────────────────────────
df = pd.read_csv("df_mortality_brussels.csv", sep=";",
                 names=["week", "deaths"], header=0, encoding="latin-1")

df["date"]   = pd.to_datetime(df["week"], format="%d-%m-%y")
df           = df.sort_values("date").reset_index(drop=True)
df["t"]      = np.arange(1, len(df) + 1)          # integer week index

# ── 2. COVID mask ─────────────────────────────────────────────────────────────
covid_start = pd.Timestamp("2020-03-01")
covid_end   = pd.Timestamp("2020-12-31")
df["covid"] = (df["date"] >= covid_start) & (df["date"] <= covid_end)

# ── 3. Log-transform & annual harmonic terms ──────────────────────────────────
T_weeks          = 365.25 / 7          # ≈ 52.18 weeks per year
df["log_deaths"] = np.log(df["deaths"])
df["sin1"]       = np.sin(2 * np.pi * df["t"] / T_weeks)
df["cos1"]       = np.cos(2 * np.pi * df["t"] / T_weeks)

# ── 4. Fit OLS on non-COVID weeks ─────────────────────────────────────────────
train = df[~df["covid"]].copy()
X_train = np.column_stack([np.ones(len(train)), train["sin1"], train["cos1"]])
y_train = train["log_deaths"].values

result = np.linalg.lstsq(X_train, y_train, rcond=None)
coef   = result[0]                     # [intercept, sin1, cos1]

print("Model coefficients:")
print(f"  Intercept : {coef[0]:.4f}")
print(f"  sin1      : {coef[1]:.4f}")
print(f"  cos1      : {coef[2]:.4f}")

# R² on training data
y_hat_train = X_train @ coef
ss_res = np.sum((y_train - y_hat_train) ** 2)
ss_tot = np.sum((y_train - y_train.mean()) ** 2)
print(f"  R²        : {1 - ss_res / ss_tot:.4f}")

# ── 5. Predict for all weeks & compute excess envelope ────────────────────────
X_all = np.column_stack([np.ones(len(df)), df["sin1"], df["cos1"]])
df["fitted_log"] = X_all @ coef

# Residuals on training weeks only
train_resid = y_train - y_hat_train
thresh      = np.quantile(train_resid, 0.9772)
print(f"\n97.72th percentile of residuals (log scale): {thresh:.4f}")

df["envelope_log"] = df["fitted_log"] + thresh
df["fitted"]       = np.exp(df["fitted_log"])
df["envelope"]     = np.exp(df["envelope_log"])

# ── 6. Flag excess weeks ──────────────────────────────────────────────────────
df["excess"] = df["deaths"] > df["envelope"]

print(f"Excess weeks (outside COVID mask): {(df['excess'] & ~df['covid']).sum()}")
print(f"Excess weeks (inside  COVID mask): {(df['excess'] &  df['covid']).sum()}")

# ── 7. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

# COVID exclusion band
ax.axvspan(covid_start, covid_end + pd.Timedelta(days=6),
           color="#BBBBBB", alpha=0.30, zorder=1)

# Excess mortality — one rect per week
week_days = pd.Timedelta(days=6)
for _, row in df[df["excess"]].iterrows():
    ax.fill_betweenx(
        [row["envelope"], row["deaths"]],
        row["date"], row["date"] + week_days,
        color="#D62728", alpha=0.55, linewidth=0, zorder=2
    )

# Observed deaths
ax.plot(df["date"], df["deaths"],
        color="black", linewidth=0.6, zorder=3, label="Observed deaths")

# Seasonal model
ax.plot(df["date"], df["fitted"],
        color="#1F77B4", linewidth=1.2, zorder=4, label="Seasonal model")

# Excess envelope
ax.plot(df["date"], df["envelope"],
        color="#D62728", linewidth=0.9, linestyle="--", zorder=5,
        label="97.72 % excess envelope")

# Legend handles
legend_handles = [
    mpatches.Patch(color="black",   label="Observed deaths"),
    mpatches.Patch(color="#1F77B4", label="Seasonal model"),
    mpatches.Patch(color="#D62728", linestyle="--", label="97.72 % excess envelope"),
    mpatches.Patch(color="#D62728", alpha=0.55, label="Excess mortality weeks"),
    mpatches.Patch(color="#BBBBBB", alpha=0.60, label="COVID-19 mask (Mar–Dec 2020)"),
]

ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.8)
ax.set_xlabel("Date")
ax.set_ylabel("Deaths per week")
ax.set_title("Weekly all-cause mortality — Brussels Region")
ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
ax.grid(axis="y", linewidth=0.4, alpha=0.5)

plt.tight_layout()
plt.savefig("excess_mortality_brussels_py.png", dpi=150)
print("\nPlot saved to excess_mortality_brussels_py.png")
plt.show()
