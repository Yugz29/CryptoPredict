"""
BTC Price Prediction - Version 2
Version de base simplifiée - Prédiction sur 24 heures avec données horaires
"""

import requests
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =============================================================================
# CONFIGURATION
# =============================================================================

SYMBOL = "BTCUSDT"
INTERVAL = "1h"                 # Données horaires
SEQ_LENGTH = 24                 # Fenêtre de 24 heures
PREDICTION_HORIZON = 24         # Prédire les 24 prochaines heures
DATA_LIMIT = 500
EPOCHS = 20
BATCH_SIZE = 32

# =============================================================================
# FONCTIONS
# =============================================================================

def get_historical_data(symbol, interval, limit=DATA_LIMIT):
    """
    Récupère les données historiques depuis l'API Binance.
    Version simple : uniquement le prix de clôture.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erreur API Binance: {e}")
        return pd.DataFrame()
    
    # Transformation en DataFrame
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    df = df[["close"]].astype(float)
    
    return df


def prepare_data(data, seq_length):
    """
    Prépare les données pour l'entraînement LSTM.
    
    - Normalise entre 0 et 1
    - Crée des séquences de seq_length heures
    - Chaque séquence prédit les 24 heures suivantes
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length - PREDICTION_HORIZON):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length:i + seq_length + PREDICTION_HORIZON])
    
    return np.array(X), np.array(y), scaler


def build_model(seq_length):
    """
    Construit un modèle LSTM simple à 2 couches.
    
    Architecture:
        - 2 couches LSTM de 50 neurones
        - Sortie dense : 24 valeurs (24 heures)
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50),
        Dense(PREDICTION_HORIZON)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_future(data, model, scaler, seq_length):
    """
    Prédit les 24 prochaines heures à partir des seq_length dernières heures.
    """
    last_sequence = data.values[-seq_length:]
    scaled_sequence = scaler.transform(last_sequence)
    scaled_sequence = scaled_sequence.reshape(1, seq_length, 1)
    
    # Prédiction
    predictions_scaled = model.predict(scaled_sequence, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    
    return predictions


# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BTC/USDT Price Prediction - Version 3")
    print("=" * 60)
    
    # Étape 1 : Récupération des données
    print("\n[1/4] Récupération des données historiques...")
    data = get_historical_data(SYMBOL, INTERVAL)
    
    if data.empty:
        print("❌ Échec de récupération des données.")
        exit(1)
    
    print(f"✓ {len(data)} points de données récupérés")
    print(f"  Période: {data.index[0]} → {data.index[-1]}")
    print(f"  Prix actuel: ${data['close'].iloc[-1]:.2f}")
    
    # Étape 2 : Préparation des données
    print("\n[2/4] Préparation des données pour LSTM...")
    X, y, scaler = prepare_data(data.values, SEQ_LENGTH)
    print(f"✓ {len(X)} séquences d'entraînement créées")
    print(f"  Shape X: {X.shape}")
    print(f"  Shape y: {y.shape}")
    
    # Étape 3 : Construction et entraînement du modèle
    print(f"\n[3/4] Entraînement du modèle LSTM ({EPOCHS} epochs)...")
    model = build_model(SEQ_LENGTH)
    history = model.fit(
        X, y, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.2,
        verbose=1
    )
    print("✓ Modèle entraîné")
    
    # Étape 4 : Prédiction des 24 prochaines heures
    print("\n[4/4] Prédiction des prochaines 24 heures...")
    predictions = predict_future(data, model, scaler, SEQ_LENGTH)
    
    # Affichage des résultats
    print("\n" + "=" * 60)
    print("RÉSULTATS DES PRÉDICTIONS")
    print("=" * 60)
    
    current_time = datetime.datetime.now()
    for i, price in enumerate(predictions):
        future_time = current_time + datetime.timedelta(hours=i + 1)
        print(f"{future_time.strftime('%d/%m/%Y %Hh00')} : ${price[0]:.2f}")
    
    print("\n" + "=" * 60)
    print("✓ Prédiction terminée")
    print("=" * 60)
