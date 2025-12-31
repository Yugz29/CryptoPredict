"""
BTC Price Prediction - Version 3
Prédiction multi-features avec indicateurs techniques (librairie ta)
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
import ta

# =============================================================================
# CONFIGURATION
# =============================================================================

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
SEQ_LENGTH = 48                 # Fenêtre augmentée à 48 minutes
PREDICTION_HORIZON = 60
REFRESH_INTERVAL = 60
DATA_LIMIT = 1000
EPOCHS = 50
BATCH_SIZE = 32

# Paramètres des indicateurs techniques
RSI_WINDOW = 14
EMA_WINDOW = 20
ATR_WINDOW = 14

# =============================================================================
# FONCTIONS
# =============================================================================

def get_historical_data(symbol, interval, limit=DATA_LIMIT):
    """
    Récupère les données historiques et calcule les indicateurs techniques.
    
    Indicateurs ajoutés:
        - RSI (Relative Strength Index) : momentum
        - MACD (Moving Average Convergence Divergence) : trend
        - EMA 20 (Exponential Moving Average) : trend
        - ATR (Average True Range) : volatilité
    
    NOTE: La librairie 'ta' remplace 'talib' pour simplifier l'installation.
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
    
    # Garder les colonnes OHLCV (nécessaires pour les indicateurs)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    
    # Calcul des indicateurs techniques avec la librairie 'ta'
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=RSI_WINDOW).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=EMA_WINDOW).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=ATR_WINDOW
    ).average_true_range()
    
    # Échantillonnage régulier et interpolation
    df = df.resample('1min').last()
    df = df.interpolate(method='linear').fillna(method='bfill')
    
    return df


def prepare_data(data, seq_length, scaler_type='minmax'):
    """
    Prépare les données multi-features pour LSTM.
    
    IMPORTANT: 
        - On normalise toutes les features ensemble
        - On prédit uniquement le prix de clôture (colonne 3)
        - Les autres features servent de contexte au modèle
    """
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length - PREDICTION_HORIZON):
        X.append(scaled_data[i:i + seq_length])
        # y contient uniquement la colonne "close" (index 3 dans OHLCV)
        y.append(scaled_data[i + seq_length:i + seq_length + PREDICTION_HORIZON, 3])
    
    return np.array(X), np.array(y), scaler


def build_model(seq_length, feature_count):
    """
    Construit un modèle LSTM multi-features.
    
    Architecture:
        - LSTM(128) : couche plus large pour gérer plus de features
        - LSTM(64) : couche de réduction
        - Dropout(0.2) après chaque couche
        - Dense(60) : sortie de 60 prédictions
    
    NOTE: input_shape = (seq_length, feature_count)
          En V4, feature_count = 9 (open, high, low, close, volume, rsi, macd, ema, atr)
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, feature_count)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(PREDICTION_HORIZON)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_future(data, model, scaler, seq_length):
    """
    Prédit les 60 prochaines minutes en utilisant toutes les features.
    
    ASTUCE:
        Pour dénormaliser, on doit reconstruire une matrice avec toutes les features
        puis extraire uniquement la colonne "close".
    """
    last_sequence = data.values[-seq_length:]
    scaled_sequence = scaler.transform(last_sequence)
    scaled_sequence = scaled_sequence.reshape(1, seq_length, -1)
    
    # Prédiction
    predictions_scaled = model.predict(scaled_sequence, verbose=0)
    
    # Reconstruction de la matrice pour dénormalisation
    # On crée une matrice de zéros avec toutes les colonnes
    zeros = np.zeros((predictions_scaled.shape[1], data.shape[1]))
    zeros[:, 3] = predictions_scaled.reshape(-1)  # Insérer les prédictions dans "close"
    
    # Dénormaliser et extraire uniquement "close"
    predictions = scaler.inverse_transform(zeros)[:, 3]
    
    # Ajouter le dernier prix réel pour continuité
    predictions = np.insert(predictions, 0, data["close"].iloc[-1])
    
    return predictions


def live_predictions(symbol=SYMBOL, interval=INTERVAL, seq_length=SEQ_LENGTH, refresh_interval=REFRESH_INTERVAL):
    """
    Boucle de prédiction en temps réel avec multi-features.
    """
    print("=" * 60)
    print("BTC/USDT Live Predictions - Version 4")
    print("Multi-features: OHLCV + RSI + MACD + EMA + ATR")
    print("=" * 60)
    
    # Récupération initiale et entraînement
    print("\n[INIT] Récupération des données initiales...")
    data = get_historical_data(symbol, interval)
    
    if data.empty:
        print("❌ Échec de récupération des données.")
        return
    
    print(f"✓ {len(data)} points récupérés")
    print(f"  Features: {list(data.columns)}")
    
    print(f"\n[INIT] Entraînement du modèle ({EPOCHS} epochs)...")
    X, y, scaler = prepare_data(data.values, seq_length, scaler_type='minmax')
    feature_count = X.shape[2]
    print(f"  Shape X: {X.shape} (features: {feature_count})")
    
    model = build_model(seq_length, feature_count)
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
    print("✓ Modèle entraîné\n")
    
    # Initialisation du graphique
    fig = go.Figure()
    fig.update_layout(
        title='Prédictions BTC/USDT (Multi-features)',
        xaxis_title='Temps',
        yaxis_title='Prix en $',
        showlegend=True,
        template='plotly_white'
    )
    
    print("🚀 Démarrage des prédictions en temps réel...\n")
    
    # Boucle principale
    while True:
        try:
            time.sleep(refresh_interval)
            
            new_data = get_historical_data(symbol, interval)
            if new_data.empty:
                print("⚠️  Échec de récupération, nouvelle tentative...")
                continue
            
            new_data = new_data[-(seq_length + PREDICTION_HORIZON):]
            predictions = predict_future(new_data, model, scaler, seq_length)
            
            # Mise à jour du graphique
            current_time = datetime.datetime.now()
            past_times = new_data.index[-60:]
            real_prices = new_data["close"].values[-60:]
            future_times = [current_time + datetime.timedelta(minutes=i) for i in range(61)]
            
            fig.data = []
            fig.add_trace(go.Scatter(
                x=past_times,
                y=real_prices,
                mode='lines',
                name='Prix Passés',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=future_times,
                y=predictions,
                mode='lines',
                name='Prédictions Futures',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(title=f"Prédictions BTC/USDT ({current_time.strftime('%H:%M:%S')})")
            fig.show(config={"displayModeBar": False})
            
            print(f"[{current_time.strftime('%H:%M:%S')}] ✓ Mise à jour | Prix: ${real_prices[-1]:.2f}")
            
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
