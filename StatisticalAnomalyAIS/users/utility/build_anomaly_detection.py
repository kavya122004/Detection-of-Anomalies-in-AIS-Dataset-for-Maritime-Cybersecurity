import pandas as pd
import numpy as np
from haversine import haversine, Unit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#   Generate Synthetic Dataset
def generate_synthetic_data(num_records=5000, num_ships=50):
    """Generates a synthetic AIS dataset."""

    np.random.seed(42)  # for reproducibility

    data = []
    for mmsi in range(num_ships):
        #   Base latitude and longitude
        base_lat = np.random.uniform(59.4, 59.5)
        base_lon = np.random.uniform(24.6, 24.7)

        #   Base speed
        base_speed = np.random.uniform(5, 25)

        time = 1662700800  # Start time (UNIX timestamp)
        for _ in range(num_records // num_ships):
            #   Simulate some variations in position and speed
            lat = base_lat + np.random.normal(0, 0.001)
            lon = base_lon + np.random.normal(0, 0.001)
            speed = base_speed + np.random.normal(0, 2)

            #   Introduce some anomalies
            if np.random.rand() < 0.01:  # 1% chance of anomaly
                lat += np.random.uniform(0.005, 0.01)
                speed += np.random.uniform(5, 10)

            data.append({
                'MMSI': 276005280 + mmsi,  # Making MMSI realistic
                'Channel': 'A',
                'Lat': lat,
                'Lon': lon,
                'Time': time,
                'Speed': speed
            })
            time += np.random.uniform(2, 15)  # Varying time intervals
    return pd.DataFrame(data)


#   Data Preparation
def prepare_data(df):
    """Prepares the AIS data by calculating time differences,
    Haversine distance, and calculated speed.
    """

    df = df.sort_values(by=['MMSI', 'Time'])
    df['Time_diff'] = df.groupby('MMSI')['Time'].diff()

    #   Calculate Haversine distance (vectorized)
    def calculate_haversine_vectorized(group):
        lats = group['Lat'].values
        lons = group['Lon'].values

        distances = [0]  # Distance for the first point
        for i in range(1, len(group)):
            distances.append(haversine((lats[i - 1], lons[i - 1]), (lats[i], lons[i]), unit=Unit.NAUTICAL_MILES))

        group['Haversine_dist'] = distances
        return group

    df = df.groupby('MMSI').apply(calculate_haversine_vectorized)

    df['V_calc'] = df['Haversine_dist'] / df['Time_diff']
    df['V_calc'] = df['V_calc'].replace([np.inf, -np.inf], np.nan)  # Handle infinities
    df['V_calc'] = df['V_calc'].fillna(0)  # Fill NaN with 0

    df['V_rep'] = df['Speed']
    df['Var_V_calc'] = df.groupby('MMSI')['V_calc'].transform('var')
    df['Var_V_rep'] = df.groupby('MMSI')['V_rep'].transform('var')
    df.to_csv("AIS.csv", index=False)
    return df


#   Feature Calculation
def calculate_features(df):
    """Calculates SVR and AASD."""

    def calculate_svr(row):
        if row['Var_V_calc'] >= row['Var_V_rep']:
            return row['Var_V_calc'] / row['Var_V_rep']
        else:
            return row['Var_V_rep'] / row['Var_V_calc']

    df['SVR'] = df.apply(calculate_svr, axis=1)
    df['AASD'] = abs(df['V_calc'] - df['V_rep'])
    return df


#   Timing Analysis (Simplified)
def analyze_timing(df):
    """Performs a simplified timing analysis."""

    #   Define expected timing intervals (from Table I in the paper)
    bins = [0, 14, 23, np.inf]
    labels = ["<14", "14-23", ">23"]
    df['Speed_Category'] = pd.cut(df['Speed'], bins=bins, labels=labels, right=False)

    timing_analysis = df.groupby('Speed_Category')['Time_diff'].describe()

    print("Simplified Timing Analysis:")
    print(timing_analysis)
    return timing_analysis


#   Correlation Analysis
def analyze_correlation(df):
    """Calculates and prints the correlation between V_rep and V_calc."""

    correlation = df.groupby('MMSI')[['V_rep', 'V_calc']].corr().unstack().iloc[:, 1]
    print("\nCorrelation between V_rep and V_calc:")
    print(correlation)
    return correlation


#   Anomaly Detection (using SVR and AASD)
def detect_anomalies(df):
    """
    Performs anomaly detection using SVR and AASD with K-Means clustering.
    """

    #   Scaling features can be important for clustering
    features = df[['SVR', 'AASD']].fillna(0)  # Handle NaN values

    #   Simple scaling (can be improved)
    features['SVR'] = np.log1p(features['SVR'])
    features['AASD'] = np.log1p(features['AASD'])

    #   Basic K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)  # Explicitly set n_init
    df['Anomaly_Label'] = kmeans.fit_predict(features)

    print("\nAnomaly Detection Results:")
    print(df[['MMSI', 'SVR', 'AASD', 'Anomaly_Label']].head())
    return df


#   Visualization Functions
def plot_speed_comparison(df, mmsi_list=None):
    """Plots V_rep vs. V_calc for specified MMSIs."""

    if mmsi_list is None:
        mmsi_list = df['MMSI'].unique()[:3]  # Plot first 3 if none specified

    plt.figure(figsize=(12, 6 * len(mmsi_list)))
    for i, mmsi in enumerate(mmsi_list):
        plt.subplot(len(mmsi_list), 1, i + 1)
        ship_data = df[df['MMSI'] == mmsi]
        plt.scatter(ship_data['V_calc'], ship_data['V_rep'], s=5)
        plt.xlabel("Calculated Speed [kts]")
        plt.ylabel("Reported Speed [kts]")
        plt.title(f"MMSI: {mmsi}")
    plt.tight_layout()
    plt.show()


def plot_svr_aasd(df):
    """Plots SVR vs. AASD with anomaly labels."""

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df['AASD'], df['SVR'], c=df['Anomaly_Label'], cmap='viridis', s=5)
    plt.xlabel("Absolute Speed Difference [kts] (log scale)")
    plt.ylabel("Speed Variation Ratio (log scale)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("SVR vs. AASD with Anomaly Labels")
    plt.legend(*scatter.legend_elements(), title="Anomaly")
    plt.tight_layout()
    plt.show()


def plot_speed_time_diff(df, mmsi=276005280):  # plotting for a specific MMSI
    """Plots speed difference and time difference over time."""

    ship_data = df[df['MMSI'] == mmsi].sort_values('Time')  # Ensure time order

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Speed Difference [kts]', color=color)
    ax1.plot(ship_data.index, ship_data['V_calc'] - ship_data['V_rep'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Time Delay [s]', color=color)  # we already handled the x-label with ax1
    ax2.plot(ship_data.index, ship_data['Time_diff'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label would be slightly clipped
    plt.title(f"Speed Difference and Time Delay (MMSI: {mmsi})")
    plt.show()

#
# #   Main Execution
# if __name__ == "__main__":
#     df = generate_synthetic_data()
#     df = prepare_data(df)
#     timing_analysis = analyze_timing(df)
#     correlation = analyze_correlation(df)
#     df = calculate_features(df)
#     df_anomalies = detect_anomalies(df)
#
#     plot_speed_comparison(df, mmsi_list=[276005280, 276005285, 276005290])  # Example MMSIs
#     plot_svr_aasd(df_anomalies)
#     plot_speed_time_diff(df)
