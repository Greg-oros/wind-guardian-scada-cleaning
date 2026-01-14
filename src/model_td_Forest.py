
from sklearn.linear_model import LinearRegression
from splitting_data import train_test_sets
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# 1. Funkcja przyjmuje dane, a nie tworzy ich w środku!
def train_model(X_train, y_train):
    logging.info("Start treningu modelu")
    start_time = time.perf_counter()

    # model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=99) # model ma sople,
    # trzeba je wygładzić:
    model = RandomForestRegressor(
        n_estimators=200,  # Dajemy mu 200 drzew (było 100). Więcej = stabilniej.
        max_depth=15,  # Pozwalamy mu rosnąć dość głęboko...
        min_samples_leaf=14,  # ...ALE każdy liść musi mieć min. 20 obserwacji! (To jest nasze żelazko)
        random_state=99
    )

    model.fit(X_train, y_train)

    elapsed = time.perf_counter() - start_time
    logging.info(f"Zakończono trening | czas: {elapsed:.4f} s")
    return model


# 2. Funkcja przyjmuje model i dane
def make_predictions(model, X_test):
    logging.info("Start predykcji")
    start_time = time.perf_counter()

    y_pred = model.predict(X_test)

    elapsed = time.perf_counter() - start_time
    logging.info(f"Zakończono predykcję | czas: {elapsed:.4f} s")
    return y_pred


def plot_results(X_test, y_test, y_pred):
    # Sortowanie potrzebne tylko do ładnego narysowania linii
    # X_test to DataFrame, więc wyciągamy kolumnę do sortowania
    # Zakładam, że pierwsza kolumna to Wind Speed

    # Tworzymy pomocniczy DataFrame do sortowania, żeby nie pomylić indeksów
    data_to_plot = X_test.copy()
    data_to_plot['y_test'] = y_test
    data_to_plot['y_pred'] = y_pred

    # Sortujemy po wietrze (kolumna 0)
    data_to_plot = data_to_plot.sort_values(by=data_to_plot.columns[0])

    plt.figure(figsize=(10, 6))
    plt.scatter(data_to_plot.iloc[:, 0], data_to_plot['y_test'], color="red", s=1, alpha=0.3, label="Prawda (y_test)")

    # Rysujemy linię modelu (teraz wyjdzie prosta kreska, bo to Regresja Liniowa)
    plt.plot(data_to_plot.iloc[:, 0], data_to_plot['y_pred'], color="blue", linewidth=2, label="Opinia Modelu")

    plt.xlabel("Prędkość wiatru")
    plt.ylabel("Moc")
    plt.title("Regresja Liniowa vs Rzeczywistość (Underfitting)")
    plt.legend()
    plt.show()


# --- MAIN FLOW ---
if __name__ == "__main__":
    # 1. Dzielimy dane RAZ
    X_train, X_test, y_train, y_test = train_test_sets()
    # (Zakomentowane, bo nie mam Twojego pliku, ale tu jest miejsce na to wywołanie)

    # 2. Trenujemy
    model = train_model(X_train, y_train)
    joblib.dump(model, "model1_wind.pkl")
    # 3. Przewidujemy
    y_pred = make_predictions(model, X_test)

    # 4. Rysujemy
    plot_results(X_test, y_test, y_pred)