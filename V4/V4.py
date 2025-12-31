"""
BTC Price Prediction - Version 4
Prédiction temps réel avec réentraînement continu
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
INTERVAL = "1m"
SEQ_LENGTH = 24
PREDICTION_HORIZON = 60
REFRESH_INTERVAL = 30           # Rafraîchir toutes les 30 secondes
DATA_LIMIT = 1000
EPOCHS = 5                      # Peu d'epochs car réentraînement fréquent
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
    df["timestamp"] = df["timestamp"].dt.tz_localize('UTC').dt.tz_convert('Europe/Paris')
    df.set_index("timestamp", inplace=True)
    df = df[["close"]].astype(float)
    
    # Échantillonnage régulier et interpolation
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
    
    return np.array(X), np.array(y), scaler


def build_model(seq_length):
    """
    Construit un modèle LSTM à 2 couches.
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
    """
    last_sequence = data.values[-seq_length:]
    scaled_sequence = scaler.transform(last_sequence)
    scaled_sequence = scaled_sequence.reshape(1, seq_length, 1)
    
    predictions_scaled = model.predict(scaled_sequence, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    
    # Ajouter le dernier prix réel pour continuité
    last_real_price = data.values[-1, 0]
    predictions = np.insert(predictions, 0, last_real_price)
    
    return predictions


def live_predictions(symbol=SYMBOL, interval=INTERVAL, seq_length=SEQ_LENGTH, refresh_interval=REFRESH_INTERVAL):
    """
    Boucle de prédiction en temps réel.
    
    NOTE IMPORTANTE:
        Le modèle est RECONSTRUIT et RÉENTRAÎNÉ à chaque itération.
        C'est extrêmement coûteux en ressources mais permet au modèle
        de s'adapter continuellement aux nouvelles données.
    """
    print("=" * 60)
    print("BTC/USDT Live Predictions - Version 4")
    print("Réentraînement continu du modèle")
    print("=" * 60)
    print(f"\n⚠️  Le modèle sera réentraîné toutes les {refresh_interval}s")
    print("    Cela peut être très lent sur CPU\n")
    
    # Initialisation du graphique
    fig = go.Figure()
    fig.update_layout(
        title='Prédictions du Prix BTC/USDT (Prochaines 60 minutes)',
        xaxis_title='Temps',
        yaxis_title='Prix en $',
        showlegend=True,
        template='plotly_white'
    )
    
    print("🚀 Démarrage des prédictions...\n")
    
    # Boucle principale
    while True:
        try:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Récupération des données...")
            
            # Récupérer les nouvelles données
            data = get_historical_data(symbol, interval)
            if data.empty:
                print("⚠️  Échec de récupération, nouvelle tentative...")
                time.sleep(refresh_interval)
                continue
            
            # Préparer les données
            X, y, scaler = prepare_data(data.values, seq_length, scaler_type='minmax')
            
            # Construire et entraîner un NOUVEAU modèle
            print(f"    Entraînement du modèle ({EPOCHS} epochs)...")
            model = build_model(seq_length)
            model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=0)
            
            # Prédire les 60 prochaines minutes
            predictions = predict_future(data, model, scaler, seq_length)
            
            # Mise à jour du graphique
            current_time = datetime.datetime.now()
            past_times = data.index[-60:]
            real_prices = data["close"].values[-60:]
            future_times = [current_time + datetime.timedelta(minutes=i) for i in range(61)]
            
            fig.data = []
            fig.add_trace(go.Scatter(
                x=past_times,
                y=real_prices,
                mode='lines+markers',
                name='Prix Passés',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
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
            
            print(f"    ✓ Prédiction terminée | Prix actuel: ${real_prices[-1]:.2f}\n")
            
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
