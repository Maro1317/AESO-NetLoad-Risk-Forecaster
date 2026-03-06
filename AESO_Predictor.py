import pandas as pd
import requests
import xgboost as xgb
import numpy as np
import warnings
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
FILE_PATH    = r"C:\Users\Clixw\Downloads\Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025 (1).csv"
TRAIN_CUTOFF = "2025-07-30"
TARGET_DATE  = "2025-07-31"
EXPORT_NAME  = r"C:\Users\Clixw\Desktop\AESO_Results.csv"

LOCATIONS = {
    "Pincher": {"lat": 49.48, "lon": -113.98},
    "MedHat":  {"lat": 50.03, "lon": -110.67},
    "Provost": {"lat": 52.35, "lon": -110.26},
    "Calgary": {"lat": 51.05, "lon": -114.07},
}

WIND_CODES  = ['AKE1','CR1','KHW1','NEP1','BUL1','BUL2','CRR1','CRR2',
               'RIV1','WHT1','WHT2','RTL1','FMG1','GDP1','GOC1']
SOLAR_CODES = ['TVS1','STR1','STR2','HYS1','JER1','BRK1','BRK2','COL1',
               'BUR1','BSC1','CLR1','CLR2','HUL1','INF1','VXH1']

FEATURES_WIND  = ['Wind_Pincher','Wind_MedHat','Wind_Provost','Wind_Calgary',
                 'Hour', 'Month']
FEATURES_SOLAR = ['Solar_MedHat','Solar_Pincher','Solar_Lag24','Hour', 'Month']
FEATURES_LOAD  = ['Temp_Calgary','Temp_MedHat','Load_Lag24','Hour','Is_Weekend', 'Month']


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
def process_data(file_path):
    print("Step 1: Processing AESO Historical Data...")
    df = pd.read_csv(file_path)

    df['Actual_Wind']  = df[[c for c in WIND_CODES  if c in df.columns]].sum(axis=1)
    df['Actual_Solar'] = df[[c for c in SOLAR_CODES if c in df.columns]].sum(axis=1)
    df['ACTUAL_AIL']   = pd.to_numeric(df['ACTUAL_AIL'],         errors='coerce')
    df['ACTUAL_POOL_PRICE'] = pd.to_numeric(df['ACTUAL_POOL_PRICE'], errors='coerce')
    df['Actual_Net_Load'] = df['ACTUAL_AIL'] - (df['Actual_Wind'] + df['Actual_Solar'])

    cols = ['Date_Begin_GMT','ACTUAL_AIL','Actual_Wind','Actual_Solar',
            'Actual_Net_Load','ACTUAL_POOL_PRICE']
    clean = df[cols].copy()
    clean['Date_Begin_GMT'] = pd.to_datetime(clean['Date_Begin_GMT'])
    return clean.sort_values('Date_Begin_GMT').reset_index(drop=True)


def get_weather(target_date_str, start_date="2025-01-01"):
    print(f"  > Fetching weather (start={start_date}, end={target_date_str})...")
    end_date = (pd.to_datetime(target_date_str) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    full_range = pd.date_range(
        start=start_date,
        end=pd.to_datetime(end_date) + pd.Timedelta(hours=23),
        freq='h'
    )
    master = pd.DataFrame({'Date_Begin_GMT': full_range})

    for name, coords in LOCATIONS.items():
        url    = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": coords["lat"], "longitude": coords["lon"],
            "start_date": start_date, "end_date": end_date,
            "hourly": ["temperature_2m","wind_speed_100m","shortwave_radiation"],
            "timezone": "GMT"
        }
        data = requests.get(url, params=params).json()
        tmp  = pd.DataFrame({
            'Date_Begin_GMT':     pd.to_datetime(data['hourly']['time']),
            f'Temp_{name}':       data['hourly']['temperature_2m'],
            f'Wind_{name}':       data['hourly']['wind_speed_100m'],
            f'Solar_{name}':      data['hourly']['shortwave_radiation'],
        })
        tmp[f'Wind_{name}_Cubed'] = tmp[f'Wind_{name}'] ** 3
        master = pd.merge(master, tmp, on='Date_Begin_GMT', how='left')

    return master.ffill()


def build_features(df_merged):
    """Add time features and lag columns."""
    df = df_merged.copy()
    df['Date_Begin_Local'] = df['Date_Begin_GMT'] - pd.Timedelta(hours=6)  # MDT = UTC-6
    df['Hour']       = df['Date_Begin_Local'].dt.hour
    df['Month']      = df['Date_Begin_Local'].dt.month
    df['Is_Weekend'] = (df['Date_Begin_Local'].dt.dayofweek >= 5).astype(int)
    df['Wind_Lag24'] = df['Actual_Wind'].shift(24)
    df['Solar_Lag24']= df['Actual_Solar'].shift(24)
    df['Load_Lag24'] = df['ACTUAL_AIL'].shift(24)
    return df


def train_models(train_df):
    train = train_df.dropna(subset=FEATURES_WIND + FEATURES_SOLAR + FEATURES_LOAD + ['Actual_Net_Load'])
    
    # PRODUCTION MODEL - Train on all available data for the most robust prediction
    m_wind  = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42).fit(train[FEATURES_WIND],  train['Actual_Wind'])
    m_solar = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42).fit(train[FEATURES_SOLAR], train['Actual_Solar'])
    m_load  = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42).fit(train[FEATURES_LOAD],  train['ACTUAL_AIL'])

    return m_wind, m_solar, m_load


def predict_day(final_df, target_dt, m_wind, m_solar, m_load):
    idx = final_df[final_df['Date_Begin_Local'].dt.date == target_dt].index
    for i in idx:
        row = final_df.loc[[i]]
        final_df.loc[i, 'Pred_Wind']   = max(0, m_wind.predict(row[FEATURES_WIND])[0])
        final_df.loc[i, 'Pred_Solar']  = max(0, m_solar.predict(row[FEATURES_SOLAR])[0])
        final_df.loc[i, 'Pred_Demand'] = m_load.predict(row[FEATURES_LOAD])[0]
    final_df['Pred_Net_Load'] = final_df['Pred_Demand'] - (final_df['Pred_Wind'] + final_df['Pred_Solar'])
    return final_df


# ──────────────────────────────────────────────
# MAIN VALIDATION (single target day)
# ──────────────────────────────────────────────
def run_validation():
    df_grid = process_data(FILE_PATH)
    print("Step 2: Fetching Weather Data...")
    weather = get_weather(TARGET_DATE)

    final_df = pd.merge(weather, df_grid, on='Date_Begin_GMT', how='left')
    final_df = build_features(final_df)

    print("Step 3: Training XGBoost Models...")
    cutoff_dt = pd.to_datetime(TRAIN_CUTOFF).replace(hour=23)
    train_df  = final_df[final_df['Date_Begin_Local'] <= cutoff_dt]
    m_wind, m_solar, m_load = train_models(train_df)

    target_dt = pd.to_datetime(TARGET_DATE).date()
    print(f"Step 4: Executing Forecast for {TARGET_DATE}...")
    final_df = predict_day(final_df, target_dt, m_wind, m_solar, m_load)

    results = final_df[final_df['Date_Begin_Local'].dt.date == target_dt].copy()

    # Calculate Error and Cost metrics for the single day
    results['AbsError_NetLoad'] = (results['Actual_Net_Load'] - results['Pred_Net_Load']).abs()
    results['Cost_of_Error']    = results['AbsError_NetLoad'] * results['ACTUAL_POOL_PRICE']

    export_cols = [
        'Hour','ACTUAL_AIL','Pred_Demand',
        'Actual_Wind','Pred_Wind','Wind_Pincher','Wind_MedHat','Wind_Provost',
        'Actual_Solar','Pred_Solar',
        'Actual_Net_Load','Pred_Net_Load','AbsError_NetLoad',
        'ACTUAL_POOL_PRICE','Cost_of_Error',
        'Temp_Calgary','Temp_MedHat','Temp_Pincher','Temp_Provost'
    ]
    validation_table = results[export_cols]

    print("\n" + "="*100)
    print(f" FINAL PROJECT VALIDATION: {TARGET_DATE}")
    print("="*100)
    print(validation_table.to_string(index=False, float_format="%.1f", na_rep="N/A"))

    valid = results.dropna(subset=['Actual_Net_Load'])
    if not valid.empty:
        mae = mean_absolute_error(valid['Actual_Net_Load'], valid['Pred_Net_Load'])
        # Calculate R-Squared for the specific 24-hour target day
        from sklearn.metrics import r2_score
        r2_demand = r2_score(valid['ACTUAL_AIL'], valid['Pred_Demand'])
        r2_wind = r2_score(valid['Actual_Wind'], valid['Pred_Wind'])
        r2_solar = r2_score(valid['Actual_Solar'], valid['Pred_Solar'])
        
        print(f"\nProject Performance MAE (for target day {TARGET_DATE}): {mae:.2f} MW")
        print(f"R-Squared (Demand): {r2_demand:.4f}")
        # Note: 24-hour Wind R2 can be extremely sensitive/negative due to low variance.
        print(f"R-Squared (Wind):   {r2_wind:.4f}")
        print(f"R-Squared (Solar):  {r2_solar:.4f}")
    else:
        print("\nProject Performance MAE: N/A")

    validation_table.to_csv(EXPORT_NAME, index=False)
    print(f"SUCCESS: Exported forecast to {EXPORT_NAME}")

    # Export all PowerBI supporting tables
    export_powerbi_tables(df_grid)


# ──────────────────────────────────────────────
# POWERBI EXPORT PIPELINE
# ──────────────────────────────────────────────
def export_powerbi_tables(df_grid=None):
    print("\n" + "="*100)
    print(" EXPORTING POWERBI TABLES")
    print("="*100)

    if df_grid is None:
        df_grid = process_data(FILE_PATH)

    # Rebuild full historical dataframe with time features
    df = df_grid.copy()
    df['Date_Begin_Local'] = df['Date_Begin_GMT'] - pd.Timedelta(hours=6)
    df['Hour']       = df['Date_Begin_Local'].dt.hour
    df['Date']       = df['Date_Begin_Local'].dt.date
    df['Month']      = df['Date_Begin_Local'].dt.month
    df['Year']       = df['Date_Begin_Local'].dt.year
    df['DayOfWeek']  = df['Date_Begin_Local'].dt.day_name()
    df['DayOfWeek_Num'] = df['Date_Begin_Local'].dt.dayofweek
    df['Month_Name'] = df['Date_Begin_Local'].dt.strftime('%b')
    df['Is_Weekend'] = (df['DayOfWeek_Num'] >= 5).astype(int)
    df['Hourly_Renewables'] = df['Actual_Wind'] + df['Actual_Solar']
    df['Renewable_Pct'] = (df['Hourly_Renewables'] / df['ACTUAL_AIL'].replace(0, pd.NA)) * 100
    df['Ramp_Rate']  = df['Actual_Net_Load'].diff()  # MW change hour-over-hour

    # ── TABLE 1: Daily History (5-year trend, Duck Curve, Chart 5) ────────────
    print("  > Building Table 1: Daily History (5-year trend)...")
    daily = df.groupby('Date').agg(
        Total_Demand_MWh      = ('ACTUAL_AIL',        'sum'),
        Total_Wind_MWh        = ('Actual_Wind',       'sum'),
        Total_Solar_MWh       = ('Actual_Solar',      'sum'),
        Total_Net_Load_MWh    = ('Actual_Net_Load',   'sum'),
        Avg_Pool_Price        = ('ACTUAL_POOL_PRICE',  'mean'),
        Max_Pool_Price        = ('ACTUAL_POOL_PRICE',  'max'),
        Avg_Renewable_Pct     = ('Renewable_Pct',     'mean'),
        Max_Ramp_Rate_MW_per_h= ('Ramp_Rate',         'max'),
    ).reset_index()
    daily['Date'] = daily['Date'].astype(str)
    daily['Renewable_Share_Pct'] = (
        (daily['Total_Wind_MWh'] + daily['Total_Solar_MWh']) / daily['Total_Demand_MWh']
    ) * 100
    out1 = r"C:\Users\Clixw\Desktop\AESO_1_Daily_History.csv"
    daily.to_csv(out1, index=False)
    print(f"     Saved → {out1}  ({len(daily)} rows)")

    # ── TABLE 2: Price vs Net Load hourly (Chart 4 correlation, Chart 2) ─────
    print("  > Building Table 2: Hourly Price vs Net Load (for correlation)...")
    price_load = df[['Date_Begin_Local','Date','Hour','Month','Month_Name','Year',
                     'DayOfWeek','DayOfWeek_Num','Is_Weekend',
                     'ACTUAL_AIL','Actual_Wind','Actual_Solar',
                     'Actual_Net_Load','Hourly_Renewables','Renewable_Pct',
                     'ACTUAL_POOL_PRICE','Ramp_Rate']].copy()
    price_load['Date_Begin_Local'] = price_load['Date_Begin_Local'].astype(str)
    price_load['Date']             = price_load['Date'].astype(str)
    price_load = price_load.dropna(subset=['ACTUAL_POOL_PRICE','Actual_Net_Load'])
    out2 = r"C:\Users\Clixw\Desktop\AESO_2_Price_vs_NetLoad.csv"
    price_load.to_csv(out2, index=False)
    print(f"     Saved → {out2}  ({len(price_load)} rows)")

    # ── TABLE 3: Average Hourly Renewable Profile by Month (Duck Curve avg) ───
    print("  > Building Table 3: Renewable Hourly Profiles by Month...")
    profile = df.groupby(['Month','Month_Name','Hour']).agg(
        Avg_Wind_MW       = ('Actual_Wind',     'mean'),
        Avg_Solar_MW      = ('Actual_Solar',    'mean'),
        Avg_Demand_MW     = ('ACTUAL_AIL',      'mean'),
        Avg_Net_Load_MW   = ('Actual_Net_Load', 'mean'),
        Avg_Renewable_Pct = ('Renewable_Pct',   'mean'),
        Avg_Ramp_Rate     = ('Ramp_Rate',       'mean'),
    ).reset_index()
    out3 = r"C:\Users\Clixw\Desktop\AESO_3_Hourly_Renewable_Profile.csv"
    profile.to_csv(out3, index=False)
    print(f"     Saved → {out3}  ({len(profile)} rows)")

    # ── TABLE 4: Risk / Price Spike Flags ─────────────────────────────────────
    print("  > Building Table 4: High-Risk Event Flags...")
    p90_price = df['ACTUAL_POOL_PRICE'].quantile(0.90)
    p90_load  = df['Actual_Net_Load'].quantile(0.90)
    risk = df[['Date_Begin_Local','Date','Hour','DayOfWeek','Month_Name','Year',
               'ACTUAL_AIL','Actual_Wind','Actual_Solar','Actual_Net_Load',
               'Hourly_Renewables','Renewable_Pct','Ramp_Rate',
               'ACTUAL_POOL_PRICE','Is_Weekend']].copy()
    risk['Price_Spike']      = (risk['ACTUAL_POOL_PRICE'] > p90_price).astype(int)
    risk['High_Gas_Stress']  = (risk['Actual_Net_Load']   > p90_load).astype(int)
    risk['Risk_Score']       = risk['Price_Spike'] + risk['High_Gas_Stress']
    risk['Risk_Label']       = risk['Risk_Score'].map({0:'Low',1:'Medium',2:'High'})
    risk['Date_Begin_Local'] = risk['Date_Begin_Local'].astype(str)
    risk['Date']             = risk['Date'].astype(str)
    risk = risk.dropna(subset=['ACTUAL_POOL_PRICE','Actual_Net_Load'])
    out4 = r"C:\Users\Clixw\Desktop\AESO_4_Risk_Flags.csv"
    risk.to_csv(out4, index=False)
    print(f"     Saved → {out4}  ({len(risk)} rows)")

    print(f"\n  Price Spike Threshold (P90): ${p90_price:.2f}/MWh")
    print(f"  High Gas Stress Threshold (P90): {p90_load:.1f} MW")
    
    print("\n  ALL POWERBI TABLES EXPORT COMPLETE.")





if __name__ == "__main__":
    run_validation()