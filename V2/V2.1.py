"""
BTC Price Prediction - Version 2.1
Version temps réel avec visualisation Plotly - Prédiction sur 60 minutes
"""

import requests
import pandas as pd
import numpy as np
import datetime
import time
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =============================================================================
# CONFIGURATION
# =============================================================================

SYMBOL = "BTCUSDT"
INTERVAL = "1m"                 # Données minute par minute
SEQ_LENGTH = 24                 # Fenêtre de 24 minutes
PREDICTION_HORIZON = 60         # Prédire les 60 prochaines minutes
REFRESH_INTERVAL = 60           # Rafraîchir toutes les 60 secondes
DATA_LIMIT = 1000
EPOCHS = 50
BATCH_SIZE = 32

# =============================================================================
# FONCTIONS
# =============================================================================

def get_historical_data(symbol, interval, limit=DATA_LIMIT):
    """
    Récupère les données historiques depuis l'API Binance.
    Gère l'échantillonnage régulier et l'interpolation.
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
    
    # Conversion du timestamp et timezone
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df["timestamp"] = df["timestamp"].dt.tz_localize('UTC').dt.tz_convert('Europe/Paris')
    df.set_index("timestamp", inplace=True)
    df = df[["close"]].astype(float)
    
    # Assurer un échantillonnage régulier (1 minute)
    # Important : les LSTM nécessitent des intervalles constants
    df = df.resample('min').last()
    df["close"] = df["close"].interpolate(method='linear')
    
    return df


def prepare_data(data, seq_length, scaler_type='minmax'):
    """
    Prépare les données pour l'entraînement LSTM.
    
    - Normalise les données (MinMaxScaler ou StandardScaler)
    - Crée des séquences temporelles
    - X : seq_length dernières minutes
    - y : 60 prochaines minutes
    """
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length - PREDICTION_HORIZON):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length:i + seq_length + PREDICTION_HORIZON])
    
    return np.array(X), np.array(y), scaler


def build_model(seq_length):
    """
    Construit un modèle LSTM à 2 couches avec dropout.
    
    Architecture:
        - LSTM(100) + Dropout(0.2)
        - LSTM(100) + Dropout(0.2)
        - Dense(60) : sortie de 60 prédictions
    """
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(PREDICTION_HORIZON)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_future(data, model, scaler, seq_length):
    """
    Prédit les 60 prochaines minutes.
    Ajoute le dernier prix réel comme point de départ pour la continuité du graphique.
    """
    last_sequence = data.values[-seq_length:]
    scaled_sequence = scaler.transform(last_sequence)
    scaled_sequence = scaled_sequence.reshape(1, seq_length, 1)
    
    # Prédiction
    predictions_scaled = model.predict(scaled_sequence, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    
    # Ajouter le dernier prix réel pour assurer la continuité visuelle
    last_real_price = data.values[-1, 0]
    predictions = np.insert(predictions, 0, last_real_price)
    
    return predictions


def live_predictions(symbol=SYMBOL, interval=INTERVAL, seq_length=SEQ_LENGTH, refresh_interval=REFRESH_INTERVAL):
    """
    Boucle principale de prédiction en temps réel.
    
    FONCTIONNEMENT:
        1. Entraîne le modèle une fois au démarrage
        2. Boucle infinie :
           - Récupère les nouvelles données
           - Fait une prédiction (sans réentraîner)
           - Met à jour le graphique
           - Attend refresh_interval secondes
    
    NOTE: Le modèle n'est entraîné qu'une fois, pas de réentraînement continu.
    """
    print("=" * 60)
    print("BTC/USDT Live Predictions - Version 3.3")
    print("=" * 60)
    
    # Initialisation : récupération des données et entraînement initial
    print("\n[INIT] Récupération des données initiales...")
    data = get_historical_data(symbol, interval)
    
    if data.empty:
        print("❌ Échec de récupération des données.")
        return
    
    print(f"✓ {len(data)} points de données récupérés")
    
    # Préparation et entraînement du modèle
    print(f"\n[INIT] Entraînement du modèle ({EPOCHS} epochs)...")
    X, y, scaler = prepare_data(data.values, seq_length, scaler_type='minmax')
    model = build_model(seq_length)
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
    print("✓ Modèle entraîné\n")
    
    # Initialisation du graphique Plotly
    fig = go.Figure()
    fig.update_layout(
        title='Prédictions du Prix BTC/USDT (Prochaines 60 minutes)',
        xaxis_title='Temps',
        yaxis_title='Prix en $',
        showlegend=True,
        template='plotly_white'
    )
    
    print("🚀 Démarrage des prédictions en temps réel...")
    print(f"   Rafraîchissement toutes les {refresh_interval} secondes\n")
    
    # Boucle de prédiction en temps réel
    while True:
        try:
            time.sleep(refresh_interval)
            
            # Récupérer de nouvelles données
            new_data = get_historical_data(symbol, interval)
            
            if new_data.empty:
                print("⚠️  Échec de récupération des données, nouvelle tentative...")
                continue
            
            # Garder seulement les données récentes nécessaires
            new_data = new_data[-(seq_length + PREDICTION_HORIZON):]
            X_new, y_new, scaler = prepare_data(new_data.values, seq_length)
            
            # Prédire les 60 prochaines minutes
            predictions = predict_future(new_data, model, scaler, seq_length)
            
            # Calculer les timestamps pour le graphique
            current_time = datetime.datetime.now()
            past_times = new_data.index[-60:]
            real_prices = new_data["close"].values[-60:]
            future_times = [current_time + datetime.timedelta(minutes=i) for i in range(61)]
            
            # Mise à jour du graphique
            fig.data = []  # Effacer les anciennes traces
            
            # Courbe des prix passés (60 dernières minutes)
            fig.add_trace(go.Scatter(
                x=past_times,
                y=real_prices,
                mode='lines+markers',
                name='Prix Passés',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            # Courbe des prédictions futures (60 prochaines minutes)
            fig.add_trace(go.Scatter(
                x=future_times,
                y=predictions.flatten(),
                mode='lines+markers',
                name='Prédictions Futures',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=4)
            ))
            
            fig.update_layout(title=f"Prédictions BTC/USDT ({current_time.strftime('%H:%M:%S')})")
            fig.show(config={"displayModeBar": False})
            
            print(f"[{current_time.strftime('%H:%M:%S')}] ✓ Prédiction mise à jour | Prix actuel: ${real_prices[-1]:.2f}")
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Arrêt des prédictions.")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")
            continue


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    live_predictions()
