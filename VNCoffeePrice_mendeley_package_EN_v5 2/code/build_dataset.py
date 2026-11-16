import pandas as pd, numpy as np

DATA = "raw_data/"

# ---------- 1. Domestic coffee price (VND/kg) + local weather ----------
def load_split(f, split):
    d = pd.read_csv(DATA + f, encoding="utf-8-sig")
    d['date'] = pd.to_datetime(d['date'], format="%d/%m/%Y")
    d['split'] = split
    return d

coffee = pd.concat([
    load_split("coffee_price_train.csv", "train"),
    load_split("coffee_price_val.csv", "val"),
    load_split("coffee_price_test.csv", "test"),
]).sort_values(['province', 'date']).reset_index(drop=True)

coffee = coffee.rename(columns={'price_vnd': 'coffee_price_vnd_kg'})

# ---------- 2. ICE London futures price (USD/ton) + USD/VND exchange rate ----------
def load_investing(fname, price_col, colname):
    d = pd.read_csv(DATA + fname, encoding="utf-8-sig")
    d['Date'] = pd.to_datetime(d['Date'], format="%m/%d/%Y")
    d[price_col] = d[price_col].astype(str).str.replace(",", "").astype(float)
    return d[['Date', price_col]].rename(columns={price_col: colname, 'Date': 'date'}).sort_values('date')

london = load_investing("london_robusta_futures_historical.csv", "Price (USD/ton)", "london_price_usd_ton")
fx = load_investing("usd_vnd_exchange_rate_historical.csv", "Price", "usd_vnd_rate")

# ---------- 3. World crude oil prices, Brent/WTI (USD/barrel) ----------
oil = pd.read_csv(DATA + "world_crude_oil_prices_2020_2026.csv", encoding="utf-8-sig")
oil['date'] = pd.to_datetime(oil['date'])
oil = oil.rename(columns={'brent_usd': 'brent_price_usd_barrel', 'wti_usd': 'wti_price_usd_barrel'})[
    ['date', 'brent_price_usd_barrel', 'wti_price_usd_barrel']]

# ---------- 4. Vietnam retail fuel prices (VND/liter), 4 fuel types ----------
vnfuel = pd.read_csv(DATA + "vietnam_fuel_prices_2020_2026.csv", encoding="utf-8-sig")
vnfuel['date'] = pd.to_datetime(vnfuel['date'])
vnfuel = vnfuel.rename(columns={
    'ron95_iii': 'ron95_price_vnd_liter',
    'e5_ron92': 'e5ron92_price_vnd_liter',
    'diesel_005s': 'diesel_price_vnd_liter',
    'kerosene': 'kerosene_price_vnd_liter',
})[['date', 'ron95_price_vnd_liter', 'e5ron92_price_vnd_liter', 'diesel_price_vnd_liter', 'kerosene_price_vnd_liter']]

# ---------- 5. Build full daily date axis + forward/backward-fill (weekends/holidays/adjustment cycles) ----------
date_min = coffee['date'].min()
date_max = coffee['date'].max()
full_dates = pd.DataFrame({'date': pd.date_range(date_min, date_max, freq='D')})

def ffill_bfill(df_src, cols):
    m = full_dates.merge(df_src, on='date', how='left').sort_values('date')
    for c in cols:
        m[c] = m[c].ffill().bfill()
    return m

london_ff = ffill_bfill(london, ['london_price_usd_ton'])
fx_ff = ffill_bfill(fx, ['usd_vnd_rate'])
oil_ff = ffill_bfill(oil, ['brent_price_usd_barrel', 'wti_price_usd_barrel'])
vnfuel_ff = ffill_bfill(vnfuel, ['ron95_price_vnd_liter', 'e5ron92_price_vnd_liter', 'diesel_price_vnd_liter', 'kerosene_price_vnd_liter'])

ext = london_ff.merge(fx_ff, on='date').merge(oil_ff, on='date').merge(vnfuel_ff, on='date')

# ---------- 6. Unit conversion for consistency ----------
# London: USD/ton -> VND/kg (1 ton = 1000 kg)
ext['london_price_vnd_kg'] = ext['london_price_usd_ton'] * ext['usd_vnd_rate'] / 1000.0
# World crude oil: USD/barrel -> converted VND/liter (1 barrel = 158.987 liters) -- note: this is the
# converted crude price only (excludes refining, taxation, distribution costs), for trend comparison only
ext['brent_price_vnd_liter_converted'] = ext['brent_price_usd_barrel'] * ext['usd_vnd_rate'] / 158.987
ext['wti_price_vnd_liter_converted'] = ext['wti_price_usd_barrel'] * ext['usd_vnd_rate'] / 158.987

# ---------- 7. Merge everything ----------
merged = coffee.merge(ext, on='date', how='left')
merged['domestic_london_price_gap_vnd_kg'] = merged['coffee_price_vnd_kg'] - merged['london_price_vnd_kg']

cols = ['date', 'split', 'province', 'commodity', 'coffee_price_vnd_kg',
        'london_price_usd_ton', 'usd_vnd_rate', 'london_price_vnd_kg', 'domestic_london_price_gap_vnd_kg',
        'brent_price_usd_barrel', 'wti_price_usd_barrel', 'brent_price_vnd_liter_converted', 'wti_price_vnd_liter_converted',
        'ron95_price_vnd_liter', 'e5ron92_price_vnd_liter', 'diesel_price_vnd_liter', 'kerosene_price_vnd_liter',
        'temperature_2m_mean', 'temperature_2m_min', 'temperature_2m_max',
        'precipitation_sum', 'rain_sum', 'lat', 'lon', 'coord_source']
merged = merged[cols].sort_values(['province', 'date']).reset_index(drop=True)

out_path = "vietnam_coffee_price_dataset.csv"
merged.to_csv(out_path, index=False, encoding="utf-8-sig")

print("Shape:", merged.shape)
print()
print("Missing values:")
print(merged.isnull().sum())
print()
print("Date range:", merged['date'].min(), "->", merged['date'].max())
print()
print("Sample:")
print(merged.head(3).to_string())
print()
print("Provinces:", merged['province'].unique())
