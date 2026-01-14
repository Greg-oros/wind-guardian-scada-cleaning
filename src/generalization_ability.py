
from sklearn.linear_model import LinearRegression
from splitting_data import train_test_sets
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
from reading_data import load_data
import matplotlib.pyplot as plt
dane = load_data()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

model = joblib.load("model1_wind.pkl")

def model_check():
    # Przygotowujemy dane do sprawdzenia (CAŁA HISTORIA)
    #Pamiętaj o podwójnym nawiasie [[ ]], żeby zrobić tabelę 2D!
    X_whole = dane[["Wind_Speed_ms"]]
    # Używamy naszego złapanego modelu do oceny
    logging.info("Szukanie anomalii w całych danych...")
    # Tu model mówi: "Ile powinno być"
    predictions_whole = model.predict(X_whole)
    # Obliczamy odchylenie (Rzeczywistość - Oczekiwania)
    # Tworzymy nowe kolumny w oryginalnych danych, żeby móc to przeanalizować
    dane['Model_Prediction'] = predictions_whole
    dane['Deviation'] = dane['LV_Active_Power_kW'] - dane['Model_Prediction']
    # Wyświetlamy wyniki (Filtracja Awarie)
    # Szukamy momentów, gdzie turbina dała o 500 kW MNIEJ niż powinna
    awarie = dane[dane['Deviation'] < -500]
    print(f"Znaleziono {len(awarie)} potencjalnych awarii/przestojów.")
    # Wyświetlamy 5 przykładowych awarii
    print(awarie[['Date_Time', 'Wind_Speed_ms', 'LV_Active_Power_kW', 'Model_Prediction', 'Deviation']].head())
    return awarie


def plot_anomalies():

    plt.figure(figsize=(12, 8))

    # 1. Rysujemy WSZYSTKIE dane na szaro (tło)
    # To pokazuje normalną pracę + szum
    plt.scatter(dane['Wind_Speed_ms'], dane['LV_Active_Power_kW'],
                color='gray', s=1, alpha=0.5, label='Normalna praca')

    # 2. Rysujemy AWARIE na czerwono
    # To są punkty, które Twój model oznaczył jako Deviation < -500
    # Używamy zmiennej 'awarie', którą stworzyłeś wcześniej
    plt.scatter(awarie['Wind_Speed_ms'], awarie['LV_Active_Power_kW'],
                color='red', s=5, label='Wykryte Anomalie (Strata > 500kW)')

    # 3. Dodajemy krzywą modelu (opcjonalnie, żeby widzieć wzorzec)
    # Sortujemy dla ładnej linii
    dane_sorted = dane.sort_values('Wind_Speed_ms')
    plt.plot(dane_sorted['Wind_Speed_ms'], dane_sorted['Model_Prediction'],
             color='blue', linewidth=2, linestyle='--', label='Wzorzec Modelu')

    plt.xlabel('Prędkość wiatru [m/s]')
    plt.ylabel('Moc [kW]')
    plt.title(f'Raport Predictive Maintenance: Wykryto {len(awarie)} anomalii')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":

    awarie = model_check()
    plot_anomalies()
