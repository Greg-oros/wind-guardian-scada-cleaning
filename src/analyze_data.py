import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reading_data import load_data
from clean_data import clean_data_1, clean_data_2, marge_stats_2clean
import matplotlib.pyplot as plt
dane = clean_data_1()
statystyki = clean_data_2()

def theory_vs_repower():
    # Exploratory Data Analysis" (EDA)
    # Comparison of actual data with the theoretical curve to identify anomalies.


    plt.figure(figsize=(12, 8)) # Trochę większy rozmiar dla czytelności

    # We draw the BACKGROUND (Actual data)
    # We change the color to gray ('gray' or 'black') and make small dots.
    # This allows focusing on the line, while treating the cloud as context.
    plt.scatter(dane["Wind_Speed_ms"], dane['LV_Active_Power_kW'],
                c='gray', s=3, alpha=0.2, label='Rzeczywiste Dane (Raw)')

    # We draw the LINE (Theory)
    # We must sort the data by the X axis (Wind), so the line is smooth and not zigzagged!
    dane_sorted = dane.sort_values(by="Wind_Speed_ms")

    plt.plot(dane_sorted["Wind_Speed_ms"], dane_sorted['Theory_Power_Curve_KWh'],
             color='red', linewidth=3, label='Krzywa Teoretyczna (Wzorzec)')

    # Axis labels (We change them to physically correct ones)
    plt.xlabel('Wind [m/s]', fontsize=12)
    plt.ylabel('Power [kW]', fontsize=12)
    plt.title('Real vs. Theory', fontsize=14)

    # Description
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

def group_aggregation():
    # Bin centers
    x_srednie = [(interval.left + interval.right) / 2 for interval in statystyki.index]
    y_srednie = statystyki['mean'].values
    y_std = statystyki['std'].values

    # Mean curve from actual data
    plt.plot(x_srednie, y_srednie, color='green', linewidth=4, label='Średnia rzeczywista')

    # std curve
    plt.plot(x_srednie, y_srednie + y_std, color='blue', linewidth=4,
             linestyle='--', label='Średnia ± 1 std')
    plt.plot(x_srednie, y_srednie - y_std, color='blue', linewidth=4,
             linestyle='--')

    plt.xlabel('Wind speed [m/s]')
    plt.ylabel('LV_Active_Power_kW')
    plt.title('Agregacja danych w binach')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def group_agg_and_theory_vs_repower():
    plt.figure(figsize=(12, 8))

    # Real data
    plt.scatter(
        dane["Wind_Speed_ms"],
        dane['LV_Active_Power_kW'],
        c='gray',
        s=3,
        alpha=0.2,
        label='Real data (Raw)'
    )

    # Theory curve
    dane_sorted = dane.sort_values(by="Wind_Speed_ms")

    plt.plot(
        dane_sorted["Wind_Speed_ms"],
        dane_sorted['Theory_Power_Curve_KWh'],
        color='red',
        linewidth=3,
        label='Theory curve'
    )

    # Stats aggregation

    # Bin centers
    x_srednie = [(interval.left + interval.right) / 2
                 for interval in statystyki.index]

    y_srednie = statystyki['mean'].values
    y_std = statystyki['std'].values

    # Mean curve real data
    plt.plot(
        x_srednie,
        y_srednie,
        color='green',
        linewidth=4,
        label='Średnia rzeczywista (bin)'
    )

    # std curve
    plt.plot(
        x_srednie,
        y_srednie + y_std,
        color='blue',
        linewidth=3,
        linestyle='--',
        label='Średnia ± 1 std'
    )

    plt.plot(
        x_srednie,
        y_srednie - y_std,
        color='blue',
        linewidth=3,
        linestyle='--'
    )

    # Desctiptions
    plt.xlabel('Wind[m/s]', fontsize=12)
    plt.ylabel('Powe [kW]', fontsize=12)
    plt.title('Real vs. Teoria + Bin aggregation', fontsize=14)

    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

def wake_effect():
    # looking for gaps on the ceiling
    plt.figure(figsize=(10, 6))
    # setting strong wind > 10 m/s
    strong_wind = (dane['Wind_Speed_ms'] > 10)
    new_data = dane[strong_wind]
    plt.scatter(new_data["Wind_direction"], new_data['LV_Active_Power_kW'], c='blue', s=1, alpha=0.3,
                label='Dziury w suficie')
    plt.xlabel('Wind_direction')
    plt.ylabel('LV_Active_Power_kW')
    plt.title('Filtered data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



  # polar chart
def wake_effect_polar():

    plt.figure(figsize=(8, 8))

    strong_wind = (dane['Wind_Speed_ms'] > 10)
    new_data = dane[strong_wind]

    theta = np.radians(new_data["Wind_direction"])
    r = new_data['LV_Active_Power_kW']

    # We create axes with polar projection
    ax = plt.subplot(111, projection='polar')

    # We draw on the 'ax' axes, not on the 'plt'
    ax.scatter(theta, r, c='blue', s=2, alpha=0.3, label='Pomiary > 10m/s')

    # Compass settings
    ax.set_theta_zero_location("N")  # N on top
    ax.set_theta_direction(-1)  # Direction (Clockwise)

    plt.title('Where does turbin lose power? (Wake Effect Analysis)', y=1.08)
    plt.legend(loc='lower right')
    plt.show()
    # points close to the center of the circle create wake_effect
def wake_effect_polar_after_cleaning():
    dirty_data, clean_data = marge_stats_2clean()

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # We draw the REJECTED data in red (as a background)
    # To see what we've discarded
    bad_wind = dirty_data[dirty_data['Wind_Speed_ms'] > 10]
    ax.scatter(np.radians(bad_wind["Wind_direction"]), bad_wind['LV_Active_Power_kW'],
               c='red', s=5, alpha=0.5, label='Deleted (Anomalie)')

    # Clean data - green

    good_wind = clean_data[clean_data['Wind_Speed_ms'] > 10]
    ax.scatter(np.radians(good_wind["Wind_direction"]), good_wind['LV_Active_Power_kW'],
               c='green', s=2, alpha=0.5, label='Clean data')

    plt.title('Cleaning effect Red = garbage, Green = dataset', y=1.08)
    plt.legend(loc='lower right')
    plt.show()




