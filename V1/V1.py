"""
BTC Price Prediction - Version 1
------------------------------------------------
- Pipeline LSTM avec indicateurs techniques (RSI, MACD, SMA, EMA)
- Prévision sur 60 prochaines minutes
- Expérimentation pédagogique, pas de modèle prêt pour trading réel
- Limites : réentraînement continu, simplifications des features
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
import talib

# =============================================================================
# CONFIGURATION
# =============================================================================

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
SEQ_LENGTH = 24                 # Fenêtre d'historique : 24 minutes
PREDICTION_HORIZON = 60         # Prédire les 60 prochaines minutes
REFRESH_INTERVAL = 30           # Rafraîchir toutes les 30 secondes
DATA_LIMIT = 1000

# Paramètres des indicateurs techniques
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_WINDOW = 30

# =============================================================================
# FONCTIONS
# =============================================================================

def get_historical_data(symbol, interval, limit=DATA_LIMIT):
    """
    Récupère les données historiques depuis l'API Binance.
    Ajoute les indicateurs techniques RSI, MACD, SMA, EMA.
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
    df = df[["close", "volume"]].astype(float)
    
    # Calcul des indicateurs techniques avec talib
    df['RSI'] = talib.RSI(df['close'], timeperiod=RSI_PERIOD)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(
        df['close'], 
        fastperiod=MACD_FAST, 
        slowperiod=MACD_SLOW, 
        signalperiod=MACD_SIGNAL
    )
    df['SMA'] = df['close'].rolling(window=MA_WINDOW).mean()
    df['EMA'] = df['close'].ewm(span=MA_WINDOW, adjust=False).mean()
    
    # Assurer un échantillonnage régulier (important pour les LSTM)
    df = df.resample('min').last()
    df = df.interpolate(method='linear')  # Combler les trous dans les données
    
    return df


def prepare_data(data, seq_length, scaler_type='minmax'):
    """
    Prépare les données pour l'entraînement LSTM.
    
    - Normalise les données (entre 0 et 1 avec MinMaxScaler)
    - Crée des séquences temporelles (X) et leurs cibles (y)
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


def build_optimized_model(seq_length):
    """
    Construit un modèle LSTM à 3 couches avec dropout.
    
    Architecture:
        - 3 couches LSTM de 100 neurones chacune
        - Dropout de 0.2 après chaque couche (éviter l'overfitting)
        - Sortie dense : 60 valeurs (60 minutes de prédictions)
    """
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(PREDICTION_HORIZON)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_future(data, model, scaler, seq_length):
    """
    Prédit les 60 prochaines minutes en utilisant les dernières seq_length minutes.
    """
    last_sequence = data.values[-seq_length:]
    scaled_sequence = scaler.transform(last_sequence)
    scaled_sequence = scaled_sequence.reshape(1, seq_length, 1)
    
    # Prédiction
    predictions_scaled = model.predict(scaled_sequence, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    
    # Ajouter le dernier prix réel comme point de départ de la courbe
    last_real_price = data.values[-1, 0]
    predictions = np.insert(predictions, 0, last_real_price)
    
    return predictions


def live_predictions(symbol=SYMBOL, interval=INTERVAL, seq_length=SEQ_LENGTH, refresh_interval=REFRESH_INTERVAL):
    """
    Boucle principale : récupère les données, entraîne le modèle, prédit et affiche.
    
    NOTE: Le modèle est réentraîné à chaque itération, ce qui est très coûteux.
    """
    # Initialisation du graphique Plotly
    fig = go.Figure()
    fig.update_layout(
        title='Prédictions du Prix BTC/USDT (Prochaines 60 minutes)',
        xaxis_title='Temps',
        yaxis_title='Prix en $',
        showlegend=True
    )

    while True:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Mise à jour des données...")
        
        # Récupération des données
        data = get_historical_data(symbol, interval)
        if data.empty:
            print("Aucune donnée récupérée, nouvelle tentative dans 30s...")
            time.sleep(refresh_interval)
            continue

        # Préparation des features (close + indicateurs techniques)
        features = data[['close', 'RSI', 'MACD', 'SMA', 'EMA', 'volume']].values
        X, y, scaler = prepare_data(features, seq_length, scaler_type='minmax')

        # Construction et entraînement du modèle
        print("Entraînement du modèle...")
        model = build_optimized_model(seq_length)
        model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)

        # Prédiction des 60 prochaines minutes
        predictions = predict_future(
            data[['close', 'RSI', 'MACD', 'SMA', 'EMA', 'volume']], 
            model, 
            scaler, 
            seq_length
        )

        # Mise à jour du graphique
        current_time = datetime.datetime.now()
        past_times = data.index[-60:]
        real_prices = data["close"].values[-60:]
        future_times = [current_time + datetime.timedelta(minutes=i) for i in range(61)]
        predictions_with_last = predictions.flatten()

        fig.data = []
        fig.add_trace(go.Scatter(
            x=past_times, 
            y=real_prices, 
            mode='lines+markers', 
            name='Prix Passés', 
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=future_times, 
            y=predictions_with_last, 
            mode='lines+markers', 
            name='Prédictions Futures', 
            line=dict(color='red')
        ))
        fig.update_layout(title=f"Prédictions du Prix BTC/USDT ({current_time.strftime('%H:%M:%S')})")
        fig.show(config={"displayModeBar": False})

        # Attendre avant le prochain rafraîchissement
        time.sleep(refresh_interval)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    print("Démarrage des prédictions en temps réel...")
    print(f"Symbole: {SYMBOL} | Intervalle: {INTERVAL} | Rafraîchissement: {REFRESH_INTERVAL}s")
    live_predictions()