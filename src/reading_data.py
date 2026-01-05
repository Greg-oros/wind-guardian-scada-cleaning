import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

def load_data():
    # Loading data from file:
    dane = pd.read_csv("go to https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset", header=0)
    # Printing first 3 rows
    # print(f"Printing first 3 rows: \n{dane[:3]}")
    # Changing date format na datetime and column names
    dane['Date/Time'] = pd.to_datetime(dane['Date/Time'], format="%d %m %Y %H:%M")
    dane.columns = ['Date_Time', 'LV_Active_Power_kW', 'Wind_Speed_ms','Theory_Power_Curve_KWh', 'Wind_direction']
    return dane
#  checking NaN, data types:
def review():
    dane = load_data()
    print(f"Type of data: \n{dane.dtypes}")
    print()
    print(f"Are there NaN's: \n{dane.isna().sum()}")
    # Changing float to int32 (Theory_Power_Curve_KWh) and decimal value (LV_Active_Power_kW)
    dane["LV_Active_Power_kW"] = dane["LV_Active_Power_kW"].round(1)
    dane["Wind_Speed_ms"] = dane["Wind_Speed_ms"].round(2)
    dane["Theory_Power_Curve_KWh"] = dane["Theory_Power_Curve_KWh"].astype("int32")
    print(f"Printing first 3 rows after changes: \n{dane[:3]}")
    # Drawing Scatter Plot:
    # For data < 100k, for > 100k using hexbin
    plt.scatter(dane["Wind_Speed_ms"], dane["LV_Active_Power_kW"], s=5, alpha = 0.3)
    plt.xlabel("Wind_Speed_ms")
    plt.ylabel("LV_Active_Power_kW")
    plt.show()

if __name__ == "__main__":
    review()

