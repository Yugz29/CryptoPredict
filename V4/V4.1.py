"""
BTC Price Prediction - Version 4.1
Avec sauvegarde du modèle et bandes de confiance
"""

import requests
import pandas as pd
import numpy as np
import datetime
import time
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
SEQ_LENGTH = 24
PREDICTION_HORIZON = 60
REFRESH_INTERVAL = 30
DATA_LIMIT = 1000
EPOCHS = 5
BATCH_SIZE = 32
MODEL_PATH = "lstm_model.h5"    # Chemin de sauvegarde du modèle

# =============================================================================
# FONCTIONS
# =============================================================================

def get_historical_data(symbol, interval, limit=DATA_LIMIT):
    """
    Récupère les données historiques depuis l'API Binance.
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
    
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df["timestamp"] = df["timestamp"].dt.tz_localize('UTC').dt.tz_convert('Europe/Paris')
    df.set_index("timestamp", inplace=True)
    df = df[["close"]].astype(float)
    
    df = df.resample('min').last()
    df["close"] = df["close"].interpolate(method='linear')
    
    return df


def prepare_data(data, seq_length, scaler_type='minmax'):
    """
    Prépare les données pour l'entraînement LSTM.
    """
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length - PREDICTION_HORIZON):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length:i + seq_length + PREDICTION_HORIZON])
    
    X = np.array(X)
    y = np.array(y)
    
    # Ajustement de forme si nécessaire
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=-1)
    
    return X, y, scaler


def build_or_load_model(seq_length, model_path=MODEL_PATH):
    """
    Charge un modèle existant ou en crée un nouveau.
    
    AVANTAGE:
        Évite de réentraîner depuis zéro à chaque fois.
        Le modèle garde sa "mémoire" entre les itérations.
    """
    input_shape = (seq_length, 1)
    
    # Tentative de chargement du modèle existant
    if os.path.exists(model_path):
        try:
            print(f"📂 Chargement du modèle depuis {model_path}...")
            model = load_model(model_path)
            
            # Vérifier la compatibilité
            if model.input_shape[1:] != input_shape:
                raise ValueError("La forme du modèle ne correspond pas aux données.")
            
            print("   ✓ Modèle chargé avec succès")
            return model
            
        except Exception as e:
            print(f"   ⚠️  Erreur lors du chargement : {e}")
            print("   Création d'un nouveau modèle...")
    
    # Création d'un nouveau modèle
    print("🆕 Création d'un nouveau modèle...")
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(PREDICTION_HORIZON, activation='linear')
    ])
    
    # Utilisation de Huber Loss (moins sensible aux outliers que MSE)
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber())
    
    return model


def predict_future(data, model, scaler, seq_length):
    """
    Prédit les 60 prochaines minutes.
    """
    last_sequence = data.values[-seq_length:]
    scaled_sequence = scaler.transform(last_sequence)
    scaled_sequence = scaled_sequence.reshape(1, seq_length, 1)
    
    predictions_scaled = model.predict(scaled_sequence, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    
    last_real_price = data.values[-1, 0]
    predictions = np.insert(predictions, 0, last_real_price)
    
    return predictions


def live_predictions(symbol=SYMBOL, interval=INTERVAL, seq_length=SEQ_LENGTH, 
                     refresh_interval=REFRESH_INTERVAL, model_path=MODEL_PATH):
    """
    Boucle de prédiction en temps réel avec sauvegarde du modèle.
    
    AMÉLIORATION V4.2:
        - Le modèle est sauvegardé après chaque entraînement
        - Réutilisation du modèle existant (pas de reconstruction)
        - Ajout de bandes de confiance (écart-type des prédictions)
    """
    print("=" * 60)
    print("BTC/USDT Live Predictions - Version 4.2")
    print("Avec sauvegarde du modèle et bandes de confiance")
    print("=" * 60)
    
    # Initialisation du graphique
    fig = go.Figure()
    fig.update_layout(
        title='Prédictions du Prix BTC/USDT (Prochaines 60 minutes)',
        xaxis_title='Temps',
        yaxis_title='Prix en $',
        showlegend=True,
        template='plotly_white'
    )
    
    # Charger ou créer le modèle
    model = build_or_load_model(seq_length, model_path)
    
    print("\n🚀 Démarrage des prédictions...\n")
    
    # Boucle principale
    while True:
        try:
            # Récupérer les nouvelles données
            data = get_historical_data(symbol, interval)
            if data.empty:
                print("⚠️  Échec de récupération, nouvelle tentative...")
                time.sleep(refresh_interval)
                continue
            
            # Préparer les données
            X, y, scaler = prepare_data(data.values, seq_length, scaler_type='minmax')
            
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Données préparées : {X.shape}")
            
            # Réentraîner le modèle avec les nouvelles données
            if len(X) > 0:
                print(f"    Réentraînement ({EPOCHS} epochs)...")
                model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                         validation_split=0.2, verbose=0)
                model.save(model_path)
                print(f"    ✓ Modèle sauvegardé")
            
            # Prédire les 60 prochaines minutes
            predictions = predict_future(data, model, scaler, seq_length)
            
            # Calculer les bandes de confiance
            # (basées sur l'écart-type des prédictions)
            predictions_std = np.std(predictions)
            upper_band = predictions + predictions_std
            lower_band = predictions - predictions_std
            
            # Mise à jour du graphique
            current_time = datetime.datetime.now()
            past_times = data.index[-60:]
            real_prices = data["close"].values[-60:]
            future_times = [current_time + datetime.timedelta(minutes=i) for i in range(61)]
            
            fig.data = []
            
            # Prix passés
            fig.add_trace(go.Scatter(
                x=past_times,
                y=real_prices,
                mode='lines+markers',
                name='Prix Passés',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            # Prédictions futures
            fig.add_trace(go.Scatter(
                x=future_times,
                y=predictions.flatten(),
                mode='lines+markers',
                name='Prédictions',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ))
            
            # Bandes de confiance
            fig.add_trace(go.Scatter(
                x=future_times,
                y=upper_band.flatten(),
                mode='lines',
                name='Bande Supérieure',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=future_times,
                y=lower_band.flatten(),
                mode='lines',
                name='Bande Inférieure',
                line=dict(color='gray', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.2)',
                showlegend=True
            ))
            
            fig.update_layout(title=f"Prédictions BTC/USDT ({current_time.strftime('%H:%M:%S')})")
            fig.show(config={"displayModeBar": False})
            
            print(f"    ✓ Prix actuel: ${real_prices[-1]:.2f}\n")
            
            # Attendre avant le prochain cycle
            time.sleep(refresh_interval)
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Arrêt des prédictions.")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}\n")
            time.sleep(refresh_interval)
            continue


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    live_predictions()
