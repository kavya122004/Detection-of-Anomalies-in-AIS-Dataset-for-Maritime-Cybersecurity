import pandas as pd
import numpy as np
from haversine import haversine, Unit
from sklearn.cluster import KMeans


# ---------------- DATA PREPARATION ----------------

def prepare_data(df):

    # 🔥 STEP 1: Clean column names
    df.columns = df.columns.str.strip()

    # 🔥 STEP 2: FORCE correct column names (VERY IMPORTANT FIX)
    if 'MMSI' not in df.columns and len(df.columns) >= 6:
        df.rename(columns={
            df.columns[0]: 'MMSI',
            df.columns[1]: 'Channel',
            df.columns[2]: 'Lat',
            df.columns[3]: 'Lon',
            df.columns[4]: 'Time',
            df.columns[5]: 'Speed'
        }, inplace=True)

    # 🔥 STEP 3: Continue processing
    df = df.sort_values(by=['MMSI', 'Time'])
    df['Time_diff'] = df.groupby('MMSI')['Time'].diff()

    df['Prev_Lat'] = df.groupby('MMSI')['Lat'].shift(1)
    df['Prev_Lon'] = df.groupby('MMSI')['Lon'].shift(1)

    df['Haversine_dist'] = df.apply(
        lambda row: haversine(
            (row['Prev_Lat'], row['Prev_Lon']),
            (row['Lat'], row['Lon']),
            unit=Unit.NAUTICAL_MILES
        ) if pd.notnull(row['Prev_Lat']) and pd.notnull(row['Prev_Lon']) else 0,
        axis=1
    )
    df.drop(columns=['Prev_Lat', 'Prev_Lon'], inplace=True)

    df['V_calc'] = df['Haversine_dist'] / df['Time_diff']
    df['V_calc'] = df['V_calc'].replace([np.inf, -np.inf], np.nan).fillna(0)

    df['V_rep'] = df['Speed']

    df['Var_V_calc'] = df.groupby('MMSI')['V_calc'].transform('var')
    df['Var_V_rep'] = df.groupby('MMSI')['V_rep'].transform('var')

    return df


# ---------------- FEATURE CALCULATION ----------------

def calculate_features(df):

    def calculate_svr(row):
        if row['Var_V_calc'] >= row['Var_V_rep']:
            return row['Var_V_calc'] / (row['Var_V_rep'] + 1e-6)
        else:
            return row['Var_V_rep'] / (row['Var_V_calc'] + 1e-6)

    df['SVR'] = df.apply(calculate_svr, axis=1)
    df['AASD'] = abs(df['V_calc'] - df['V_rep'])

    return df


# ---------------- TIMING ANALYSIS ----------------

def analyze_timing(df):

    bins = [0, 14, 23, np.inf]
    labels = ["<14", "14-23", ">23"]

    df['Speed_Category'] = pd.cut(df['Speed'], bins=bins, labels=labels, right=False)

    return df.groupby('Speed_Category')['Time_diff'].describe()


# ---------------- INPUT ----------------

def get_ship_input(mmsi, lat, lon, time, speed):

    return pd.DataFrame({
        'MMSI': [mmsi],
        'Channel': ['A'],
        'Lat': [lat],
        'Lon': [lon],
        'Time': [time],
        'Speed': [speed]
    })


def prepare_input_data(df, existing_df):

    # 🔥 Clean column names
    df.columns = df.columns.str.strip()
    existing_df.columns = existing_df.columns.str.strip()

    df = pd.concat([existing_df, df], ignore_index=True)
    df = df.sort_values(by=['MMSI', 'Time'])

    df['Time_diff'] = df.groupby('MMSI')['Time'].diff()

    df['Prev_Lat'] = df.groupby('MMSI')['Lat'].shift(1)
    df['Prev_Lon'] = df.groupby('MMSI')['Lon'].shift(1)

    df['Haversine_dist'] = df.apply(
        lambda row: haversine(
            (row['Prev_Lat'], row['Prev_Lon']),
            (row['Lat'], row['Lon']),
            unit=Unit.NAUTICAL_MILES
        ) if pd.notnull(row['Prev_Lat']) and pd.notnull(row['Prev_Lon']) else 0,
        axis=1
    )
    df.drop(columns=['Prev_Lat', 'Prev_Lon'], inplace=True)

    df['V_calc'] = df['Haversine_dist'] / df['Time_diff']
    df['V_calc'] = df['V_calc'].replace([np.inf, -np.inf], np.nan).fillna(0)

    df['V_rep'] = df['Speed']

    df['Var_V_calc'] = df.groupby('MMSI')['V_calc'].transform('var')
    df['Var_V_rep'] = df.groupby('MMSI')['V_rep'].transform('var')

    df = calculate_features(df)

    return df.iloc[[-1]]


# ---------------- SPEED CHECK ----------------

def assess_speed(input_df, timing_analysis):

    speed = input_df['Speed'].values[0]
    time_diff = input_df['Time_diff'].values[0]
    aasd = input_df['AASD'].values[0]
    svr = input_df['SVR'].values[0]

    var1 = var2 = var3 = None

    speed_category = pd.cut([speed], bins=[0, 14, 23, np.inf],
                            labels=["<14", "14-23", ">23"], right=False)[0]

    expected = timing_analysis.loc[speed_category]

    if not (expected['min'] <= time_diff <= expected['max']):
        var1 = "WARNING: Time interval abnormal"

    if aasd > 5:
        var2 = "WARNING: Speed mismatch"

    if svr > 2:
        var3 = "WARNING: High variation"

    msg = "Speed normal" if not (var1 or var2 or var3) else "Issues detected"

    return var1, var2, var3, msg


# ---------------- THREAT DETECTION ----------------

def assess_threat(input_df, existing_df):

    df = pd.concat([existing_df, input_df], ignore_index=True)

    features = df[['SVR', 'AASD']].fillna(0)
    features['SVR'] = np.log1p(features['SVR'])
    features['AASD'] = np.log1p(features['AASD'])

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    df['Anomaly_Label'] = kmeans.fit_predict(features)

    return "WARNING: Anomaly detected" if df.iloc[-1]['Anomaly_Label'] == 1 else "Normal"


# ---------------- MAIN ----------------

def start_test(mmsi, lat, lon, time, speed):

    from django.conf import settings
    import os

    path = os.path.join(settings.MEDIA_ROOT, 'AIS.csv')

    df = pd.read_csv(path)

    df = prepare_data(df)

    timing = analyze_timing(df)

    input_df = get_ship_input(mmsi, lat, lon, time, speed)

    input_df = prepare_input_data(input_df, df.copy())

    var1, var2, var3, msg = assess_speed(input_df, timing)

    threat = assess_threat(input_df, df.copy())

    return var1, var2, var3, msg, threat