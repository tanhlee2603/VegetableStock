import pandas as pd, numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm

DATA = "vietnam_coffee_price_dataset.csv"

# ---------- 1. Load merged dataset directly ----------
df = pd.read_csv(DATA, encoding="utf-8-sig")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['province', 'date']).reset_index(drop=True)
print("Loaded:", df.shape, "missing values:", df.isnull().sum().sum())

# ---------- 2. ADF stationarity test (Dak Lak, longest series) ----------
print("=" * 70)
print("AUGMENTED DICKEY-FULLER (ADF) TEST - Dak Lak")
print("=" * 70)
dl = df[df['province'] == 'Dak Lak'].sort_values('date')
level = dl['coffee_price_vnd_kg'].dropna()
diff = level.diff().dropna()
for name, series in [("Price level", level), ("First difference", diff)]:
    stat, pval, *_ = adfuller(series, autolag='AIC')
    concl = "STATIONARY" if pval < 0.05 else "NON-STATIONARY"
    print(f"{name}: ADF={stat:.3f}, p={pval:.4f} -> {concl}")

# ---------- 3. Correlation analysis (level & differenced), incl. oil/fuel breakdown ----------
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS (province-pooled, level & differenced)")
print("=" * 70)
corr_vars = {
    'London futures (converted)': 'london_price_vnd_kg',
    'USD/VND exchange rate': 'usd_vnd_rate',
    'Brent crude (converted)': 'brent_price_vnd_liter_converted',
    'WTI crude (converted)': 'wti_price_vnd_liter_converted',
    'RON95 retail': 'ron95_price_vnd_liter',
    'E5RON92 retail': 'e5ron92_price_vnd_liter',
    'Diesel retail': 'diesel_price_vnd_liter',
    'Kerosene retail': 'kerosene_price_vnd_liter',
    'Mean temperature': 'temperature_2m_mean',
}
dcorr = df.sort_values(['province', 'date']).copy()
gp = dcorr.groupby('province')
dcorr['price_diff'] = gp['coffee_price_vnd_kg'].diff()
for label, col in corr_vars.items():
    dcorr[col + '_diff'] = gp[col].diff()
    dcorr[col + '_diff_lag1'] = dcorr.groupby('province')[col + '_diff'].shift(1)

print(f"{'Variable':30s} {'Level r':>10s} {'Diff r (lag0)':>14s} {'Diff r (lag1)':>14s}")
corr_summary = []
for label, col in corr_vars.items():
    sub = dcorr.dropna(subset=['coffee_price_vnd_kg', col])
    lvl_r = sub['coffee_price_vnd_kg'].corr(sub[col])
    sub_d = dcorr.dropna(subset=['price_diff', col + '_diff'])
    diff_r0 = sub_d['price_diff'].corr(sub_d[col + '_diff'])
    sub_d1 = dcorr.dropna(subset=['price_diff', col + '_diff_lag1'])
    diff_r1 = sub_d1['price_diff'].corr(sub_d1[col + '_diff_lag1'])
    print(f"{label:30s} {lvl_r:10.3f} {diff_r0:14.3f} {diff_r1:14.3f}")
    corr_summary.append({"variable": label, "level_r": round(lvl_r,3), "diff_r_lag0": round(diff_r0,3), "diff_r_lag1": round(diff_r1,3)})
pd.DataFrame(corr_summary).to_csv("correlation_analysis.csv", index=False)

# ---------- 4. Feature engineering for benchmark/ablation ----------
df = df.sort_values(['province', 'date'])
g = df.groupby('province')
df['price_lag1'] = g['coffee_price_vnd_kg'].shift(1)
df['target_change'] = df['coffee_price_vnd_kg'] - df['price_lag1']
df['price_chg_lag1'] = g['coffee_price_vnd_kg'].diff().shift(1)
df['price_chg_lag7'] = g['coffee_price_vnd_kg'].diff(7).shift(1)

df['london_chg'] = g['london_price_vnd_kg'].diff()
df['london_chg_lag1'] = g['london_price_vnd_kg'].diff().shift(1)
df['usdvnd_chg'] = g['usd_vnd_rate'].diff()

df['brent_chg'] = g['brent_price_vnd_liter_converted'].diff()
df['wti_chg'] = g['wti_price_vnd_liter_converted'].diff()

df['ron95_chg'] = g['ron95_price_vnd_liter'].diff()
df['e5ron92_chg'] = g['e5ron92_price_vnd_liter'].diff()
df['diesel_chg'] = g['diesel_price_vnd_liter'].diff()
df['kerosene_chg'] = g['kerosene_price_vnd_liter'].diff()

df['dow'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df = pd.get_dummies(df, columns=['province'], prefix='mkt')
mkt_cols = [c for c in df.columns if c.startswith('mkt_')]

OIL_FEATS = ['brent_chg', 'wti_chg']
FUEL_FEATS = ['ron95_chg', 'e5ron92_chg', 'diesel_chg', 'kerosene_chg']
WEATHER_FEATS = ['temperature_2m_mean', 'precipitation_sum']
BASE_FEATS = ['price_chg_lag1', 'price_chg_lag7', 'dow', 'month'] + mkt_cols

FEATURE_GROUPS = {
    "own_lag_only":                BASE_FEATS,
    "own_lag+london":               BASE_FEATS + ['london_chg', 'london_chg_lag1'],
    "own_lag+fx":                    BASE_FEATS + ['usdvnd_chg'],
    "own_lag+weather":               BASE_FEATS + WEATHER_FEATS,
    "own_lag+world_oil":             BASE_FEATS + OIL_FEATS,
    "own_lag+vn_fuel":                BASE_FEATS + FUEL_FEATS,
    "own_lag+london+fx+weather":      BASE_FEATS + ['london_chg', 'london_chg_lag1', 'usdvnd_chg'] + WEATHER_FEATS,
    "full (+world_oil+vn_fuel)":      BASE_FEATS + ['london_chg', 'london_chg_lag1', 'usdvnd_chg'] + WEATHER_FEATS + OIL_FEATS + FUEL_FEATS,
}
all_feats = sorted(set(sum(FEATURE_GROUPS.values(), [])))
model_df = df.dropna(subset=all_feats + ['target_change', 'price_lag1', 'coffee_price_vnd_kg']).copy()
tr = model_df[model_df['split'] == 'train']
te = model_df[model_df['split'] == 'test']
print("\nTrain rows:", len(tr), "Test rows:", len(te))

def metrics_level(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    return mae, rmse, mape, smape

def diebold_mariano(e1, e2):
    d = e1**2 - e2**2
    dbar = d.mean()
    n = len(d)
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0 / n
    if var_d <= 0:
        return np.nan, np.nan
    dm_stat = dbar / np.sqrt(var_d)
    pval = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, pval

naive_pred_level_te = te['price_lag1'].values
naive_err_te = te['coffee_price_vnd_kg'].values - naive_pred_level_te
naive_mae, naive_rmse, naive_mape, naive_smape = metrics_level(te['coffee_price_vnd_kg'].values, naive_pred_level_te)

results = [{"feature_group": "(baseline)", "model": "Naive persistence",
            "MAE": round(naive_mae,1), "RMSE": round(naive_rmse,1), "MAPE%": round(naive_mape,3), "sMAPE%": round(naive_smape,3),
            "DM_stat_vs_naive": None, "DM_pvalue": None, "significant_better_than_naive(p<0.05)": None}]

for gname, feats in FEATURE_GROUPS.items():
    Xtr, ytr = tr[feats], tr['target_change']
    Xte = te[feats]
    for mname, model in [("LinearRegression", LinearRegression()),
                          ("RandomForest", RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1))]:
        model.fit(Xtr, ytr)
        pred_chg = model.predict(Xte)
        pred_level = te['price_lag1'].values + pred_chg
        mae, rmse, mape, smape = metrics_level(te['coffee_price_vnd_kg'].values, pred_level)
        err = te['coffee_price_vnd_kg'].values - pred_level
        dm_stat, dm_p = diebold_mariano(naive_err_te, err)
        results.append({
            "feature_group": gname, "model": mname,
            "MAE": round(mae,1), "RMSE": round(rmse,1), "MAPE%": round(mape,3), "sMAPE%": round(smape,3),
            "DM_stat_vs_naive": round(dm_stat,3) if not np.isnan(dm_stat) else None,
            "DM_pvalue": round(dm_p,4) if not np.isnan(dm_p) else None,
            "significant_better_than_naive(p<0.05)": bool(dm_stat > 0 and dm_p < 0.05) if not np.isnan(dm_p) else None
        })

res_df = pd.DataFrame(results)
res_df.to_csv("ablation_results.csv", index=False)
pd.set_option('display.width', 200)
print("\n" + "=" * 70)
print("ABLATION STUDY (target = Delta price, test set 2026) - incl. world oil + VN retail fuel")
print("=" * 70)
print(res_df.to_string(index=False))

# ---------- 5. Baseline comparison: price level vs. differenced target ----------
from sklearn.ensemble import GradientBoostingRegressor

FEATS_DIFF = FEATURE_GROUPS["own_lag+london+fx+weather"]
df['london_lvl'] = df['london_price_vnd_kg']
df['usdvnd_lvl'] = df['usd_vnd_rate']
FEATS_LEVEL = ['price_lag1', 'dow', 'month', 'london_lvl', 'usdvnd_lvl'] + WEATHER_FEATS + mkt_cols

model_df2 = df.dropna(subset=list(set(FEATS_DIFF + FEATS_LEVEL)) + ['target_change', 'price_lag1', 'coffee_price_vnd_kg']).copy()
tr2 = model_df2[model_df2['split'] == 'train']
te2 = model_df2[model_df2['split'] == 'test']
y_true2 = te2['coffee_price_vnd_kg'].values

print("\n" + "=" * 70)
print("BASELINE COMPARISON: Price level vs. Differenced target")
print("=" * 70)
naive_pred2 = te2['price_lag1'].values
mae, rmse, mape, smape = metrics_level(y_true2, naive_pred2)
print(f"{'Naive persistence':25s} level  MAE={mae:8.1f} MAPE={mape:.2f}%")

for mname, model in [("LinearRegression", LinearRegression()),
                      ("RandomForest", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)),
                      ("GradientBoosting", GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42))]:
    model.fit(tr2[FEATS_LEVEL], tr2['coffee_price_vnd_kg'])
    pred = model.predict(te2[FEATS_LEVEL])
    mae, rmse, mape, smape = metrics_level(y_true2, pred)
    print(f"{mname:25s} level  MAE={mae:8.1f} MAPE={mape:.2f}%")

for mname, model in [("LinearRegression", LinearRegression()),
                      ("RandomForest", RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1)),
                      ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03, random_state=42))]:
    model.fit(tr2[FEATS_DIFF], tr2['target_change'])
    pred_chg = model.predict(te2[FEATS_DIFF])
    pred_level = te2['price_lag1'].values + pred_chg
    mae, rmse, mape, smape = metrics_level(y_true2, pred_level)
    print(f"{mname:25s} diff   MAE={mae:8.1f} MAPE={mape:.2f}%")

# ---------- 6. Diagnostic decomposition: isolate the effect of world oil vs. VN retail fuel ----------
# Reproduces the "supplementary decomposition" referenced in Section 5.3 of the manuscript
# (own-lag+London+FX+weather = 719.3 VND/kg baseline combo; adding oil alone vs. fuel alone vs. both).
print("\n" + "=" * 70)
print("DIAGNOSTIC DECOMPOSITION: world oil vs. VN retail fuel, added to own_lag+london+fx+weather")
print("=" * 70)
LFX_W = FEATURE_GROUPS["own_lag+london+fx+weather"]
decomposition_variants = {
    "own_lag+london+fx+weather (baseline combo)": LFX_W,
    "+ world oil only": LFX_W + OIL_FEATS,
    "+ VN retail fuel only": LFX_W + FUEL_FEATS,
    "+ world oil + VN retail fuel (Full)": LFX_W + OIL_FEATS + FUEL_FEATS,
}
decomp_all_feats = sorted(set(sum(decomposition_variants.values(), [])))
decomp_df = df.dropna(subset=decomp_all_feats + ['target_change', 'price_lag1', 'coffee_price_vnd_kg']).copy()
tr3 = decomp_df[decomp_df['split'] == 'train']
te3 = decomp_df[decomp_df['split'] == 'test']
naive_pred3 = te3['price_lag1'].values
naive_err3 = te3['coffee_price_vnd_kg'].values - naive_pred3

decomp_results = []
print(f"{'Variant':42s} {'MAE':>10s} {'DM stat':>10s} {'p-value':>10s}")
for name, feats in decomposition_variants.items():
    model = LinearRegression()
    model.fit(tr3[feats], tr3['target_change'])
    pred_chg = model.predict(te3[feats])
    pred_level = te3['price_lag1'].values + pred_chg
    mae = float(np.mean(np.abs(te3['coffee_price_vnd_kg'].values - pred_level)))
    err = te3['coffee_price_vnd_kg'].values - pred_level
    dm_stat, dm_p = diebold_mariano(naive_err3, err)
    print(f"{name:42s} {mae:10.1f} {dm_stat:10.3f} {dm_p:10.4f}")
    decomp_results.append({"variant": name, "MAE": round(mae, 1),
                            "DM_stat_vs_naive": round(dm_stat, 3) if not np.isnan(dm_stat) else None,
                            "DM_pvalue": round(dm_p, 4) if not np.isnan(dm_p) else None})
pd.DataFrame(decomp_results).to_csv("diagnostic_decomposition_oil_vs_fuel.csv", index=False)
