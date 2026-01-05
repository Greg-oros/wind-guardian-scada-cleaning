import pandas as pd
import numpy as np
from reading_data import load_data
import matplotlib.pyplot as plt
dane = load_data()


def clean_data_1():

    # The Cut-in Speed (starting speed) for large turbines is usually around 3.0 - 3.5 m/s.
    # Below this speed the turbine can stand
    # Code that will remove data where: The wind is blowing strongly (e.g. > 3.5 m/s),
    # but the power is zero or negative (<= 0):
    error = (dane['Wind_Speed_ms'] > 3.5) & (dane['LV_Active_Power_kW'] <= 0)
    bad_data = dane[error]
    no_error_data = dane[~error]
    return no_error_data

#def review_clean_data_1():
    print(dane[:3])
    error = (dane['Wind_Speed_ms'] > 3.5) & (dane['LV_Active_Power_kW'] <= 0)
    bad_data = dane[error]
    no_error_data = dane[~error]
    print(len(no_error_data))
    print(f"Number of rows to delete: {len(bad_data)}")
    print(f"Number of rows clean: {len(no_error_data)}")
    # Alternative:
    # dane_czyste = dane[~((dane['Wind_Speed_ms'] > 3.5) & (dane['LV_Active_Power_kW'] <= 0))]
    # print(f"Number of rows clean: {len(dane_czyste)}")
    # Visualization "Before and After" on single chart
    plt.figure(figsize=(10, 6))
    # Good data - blue color (small dots, alpha=0.5 for transparency)
    plt.scatter(no_error_data["Wind_Speed_ms"], no_error_data['LV_Active_Power_kW'], c='blue', s=1, alpha=0.3, label='Praca normalna')
    # Bad data - red color
    plt.scatter(bad_data["Wind_Speed_ms"], bad_data['LV_Active_Power_kW'], c='red', s=1, label='Postój/Awaria (Do usunięcia)')
    plt.xlabel('Wind_Speed_ms')
    plt.ylabel('LV_Active_Power_kW')
    plt.title('Filtered data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# The function below is not used for cleaning, but provides conditions for cleaning, through
# expressing statistics, this can be seen in the analyze_data file in the group_agg_and_theory_vs_repower() function
# and the group_aggregation() function
def clean_data_2():
    dane_2 = clean_data_1()
    # Calculation of the mean curve and standard deviation from actual data
    # We group the data by wind speed intervals
    # np.arange creates a one-dimensional array from the minimum value to max with a step of 0.5
    # (+ 0.5 is added because np.arange doesn't include the last value, by numpy's definition)
    bins = np.arange(dane_2["Wind_Speed_ms"].min(), dane_2["Wind_Speed_ms"].max() + 0.5, 0.5)
    # creates a new column 'wind_bin' and places values from Wind_Speed_ms into bins:
    # for example: A point with wind of 3.2 m/s → will fall into the interval (3.0, 3.5]

    dane_2['wind_bin'] = pd.cut(dane_2["Wind_Speed_ms"], bins=bins)
    # when we already have these intervals, we create statistics for them, meaning a new data frame:
    # .agg only takes into account the LV_Active_Power_kW column, if there were no agg, pandas would do it for all columns
    statystyki = dane_2.groupby('wind_bin', observed=True)['LV_Active_Power_kW'].agg(['mean', 'std', 'count'])
    # Only groups with more than 5 observations
    statystyki = statystyki[statystyki['count'] > 5]
    return statystyki

# the polar plot wake_effect_polar() shows measurements that are close to the center, these are
# errors that can be cleaned, the function below uses clean_data_2()
# adds columns with statistics and creates a new data frame clean_data for use by
# the cleaning verification function that draws another polar plot
def marge_stats_2clean():
    dane_2 = clean_data_1().copy()
    # We create bins
    # We create intervals every 0.5 m/s
    bins = np.arange(0, dane_2['Wind_Speed_ms'].max() + 0.5, 0.5)
    dane_2['wind_bin'] = pd.cut(dane_2['Wind_Speed_ms'], bins=bins)

    # We calculate statistics for each bin
    # Important: reset_index() makes 'wind_bin' become a regular column, which will make merging easier
    stats = dane_2.groupby('wind_bin', observed=True)['LV_Active_Power_kW'].agg(['mean', 'std']).reset_index()

    # MERGING
    # We attach to each row in the original data information about the mean and std from its bin
    dane_merged = pd.merge(dane_2, stats, on='wind_bin', how='left')

    # We define the cleaning condition
    # Rule: We keep a row if its Power > (Mean_in_bin - 2 * Std_in_bin)
    limit_dolny = dane_merged['mean'] - (2 * dane_merged['std'])
    condition = dane_merged['LV_Active_Power_kW'] > limit_dolny

    # New DataFrames
    clean_data = dane_merged[condition]  # Good Dataset
    dirty_data = dane_merged[~condition]  # Garbage (Wake effect, failures, zeros)

    print(f"Original number of rows: {len(dane_2)}")
    print(f"Number of rows after cleaning: {len(clean_data)}")
    print(f"Deleted: {len(dirty_data)} rows ({(len(dirty_data) / len(dane_2)) * 100:.2f}%)")
    print(clean_data.head())
    return dirty_data, clean_data

