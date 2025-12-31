"""
BTC Price Prediction - Version 5
Avec interface web interactive (Dash)
"""

import requests
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# =============================================================================
# CONFIGURATION
# =============================================================================

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
SEQ_LENGTH = 24
PREDICTION_HORIZON = 60
REFRESH_INTERVAL = 30           # Intervalle de rafraîchissement (secondes)
DATA_LIMIT = 1000
EPOCHS = 5
BATCH_SIZE = 32
MODEL_PATH = "lstm_model.h5"

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
    """
    input_shape = (seq_length, 1)
    
    if os.path.exists(model_path):
        try:
            print(f"Chargement du modèle depuis {model_path}...")
            model = load_model(model_path)
            # Recompiler au cas où
            model.compile(optimizer='adam', loss=MeanSquaredError())
            print("Modèle chargé avec succès")
            return model
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")
            print("Création d'un nouveau modèle...")
    
    # Créer un nouveau modèle
    print("Création d'un nouveau modèle...")
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(PREDICTION_HORIZON, activation='linear')
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    
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


def create_dash_app(symbol=SYMBOL, interval=INTERVAL, seq_length=SEQ_LENGTH, 
                    refresh_interval=REFRESH_INTERVAL, model_path=MODEL_PATH):
    """
    Crée et configure l'application Dash.
    
    NOUVEAUTÉ V5:
        Utilisation de Dash pour créer une interface web interactive.
        Le graphique se met à jour automatiquement sans recharger la page.
    """
    # Initialiser l'application Dash
    app = dash.Dash(__name__)
    
    # Layout de l'application
    app.layout = html.Div([
        html.H1("BTC/USDT Price Predictions", style={'textAlign': 'center'}),
        html.Div(id='notification', style={
            'textAlign': 'center', 
            'color': 'green', 
            'fontSize': 18,
            'marginBottom': 20
        }),
        dcc.Graph(id='price-graph'),
        dcc.Interval(
            id='interval-component',
            interval=refresh_interval * 1000,  # en millisecondes
            n_intervals=0
        )
    ], style={'padding': 20})
    
    # Charger ou construire le modèle une seule fois
    model = build_or_load_model(seq_length, model_path)
    
    # Callback pour mettre à jour le graphique
    @app.callback(
        [Output('price-graph', 'figure'), Output('notification', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_graph(n_intervals):
        """
        Fonction appelée automatiquement à chaque intervalle.
        Met à jour le graphique et affiche des notifications.
        """
        try:
            # Récupérer les nouvelles données
            data = get_historical_data(symbol, interval)
            
            if data.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Erreur de récupération des données")
                return empty_fig, "❌ Erreur de récupération des données"
            
            # Préparer les données
            X, y, scaler = prepare_data(data.values, seq_length, scaler_type='minmax')
            
            # Réentraîner le modèle avec les nouvelles données
            if len(X) > 0:
                model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                         validation_split=0.2, verbose=0)
                model.save(model_path)
            
            # Prédire les valeurs futures
            predictions = predict_future(data, model, scaler, seq_length)
            
            # Calculer les bandes de confiance
            predictions_std = np.std(predictions)
            upper_band = predictions + predictions_std
            lower_band = predictions - predictions_std
            
            # Préparer les données pour le graphique
            current_time = datetime.datetime.now()
            past_times = data.index[-60:]
            real_prices = data["close"].values[-60:]
            future_times = [current_time + datetime.timedelta(minutes=i) for i in range(61)]
            
            # Créer le graphique
            figure = {
                'data': [
                    go.Scatter(
                        x=past_times, 
                        y=real_prices, 
                        mode='lines+markers',
                        name='Prix Passés',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4)
                    ),
                    go.Scatter(
                        x=future_times,
                        y=predictions.flatten(),
                        mode='lines+markers',
                        name='Prédictions Futures',
                        line=dict(color='red', width=2),
                        marker=dict(size=4)
                    ),
                    go.Scatter(
                        x=future_times,
                        y=upper_band.flatten(),
                        mode='lines',
                        name='Bande Supérieure',
                        line=dict(color='gray', width=1, dash='dot')
                    ),
                    go.Scatter(
                        x=future_times,
                        y=lower_band.flatten(),
                        mode='lines',
                        name='Bande Inférieure',
                        line=dict(color='gray', width=1, dash='dot'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.2)'
                    ),
                ],
                'layout': go.Layout(
                    title=f"Prédictions BTC/USDT - {current_time.strftime('%H:%M:%S')}",
                    xaxis={'title': 'Temps'},
                    yaxis={'title': 'Prix en $'},
                    showlegend=True,
                    template='plotly_white',
                    hovermode='x unified'
                )
            }
            
            # Message de notification
            notification = f"✓ Mise à jour réussie | Prix actuel: ${real_prices[-1]:.2f} | Prédiction +1h: ${predictions[-1]:.2f}"
            
            return figure, notification
            
        except Exception as e:
            print(f"Erreur dans update_graph: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Erreur lors de la mise à jour")
            return empty_fig, f"❌ Erreur: {str(e)}"
    
    return app


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BTC/USDT Live Predictions - Version 5 (Dash)")
    print("=" * 60)
    print("\nInterface web interactive avec Dash")
    print(f"Rafraîchissement automatique toutes les {REFRESH_INTERVAL}s\n")
    
    # Créer et lancer l'application Dash
    app = create_dash_app()
    
    print("🚀 Démarrage du serveur Dash...")
    print("   Ouvrez votre navigateur à l'adresse: http://127.0.0.1:8050/\n")
    
    app.run_server(debug=True)
