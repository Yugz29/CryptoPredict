"""
Script d'entraînement du modèle LSTM
Entraîne un modèle sur des données historiques CSV
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_FILE = "btc_data.csv"
MODEL_PATH = "btc_lstm_model.h5"
SEQ_LENGTH = 60                 # Fenêtre d'observation (60 minutes)
EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# =============================================================================
# FONCTIONS
# =============================================================================

def load_data(filename=DATA_FILE):
    """
    Charge les données depuis un fichier CSV.
    
    FORMAT ATTENDU:
        - Index: timestamp (datetime)
        - Colonne: close (float)
    """
    if not os.path.exists(filename):
        print(f"❌ Fichier {filename} introuvable.")
        return None
    
    try:
        df = pd.read_csv(filename, index_col="timestamp", parse_dates=True)
        df = df[["close"]]
        print(f"✓ {len(df)} lignes chargées depuis {filename}")
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None


def prepare_data(df, seq_length=SEQ_LENGTH, scaler_type='minmax'):
    """
    Prépare les données pour l'entraînement LSTM.
    
    - Normalise les données entre 0 et 1
    - Crée des séquences de seq_length timesteps
    - Chaque séquence prédit le timestep suivant
    """
    # Normalisation
    scaler = MinMaxScaler(feature_range=(0, 1)) if scaler_type == 'minmax' else StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape pour LSTM : (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    print(f"✓ Données préparées : X shape = {X.shape}, y shape = {y.shape}")
    
    return X, y, scaler


def build_lstm_model(seq_length, input_shape):
    """
    Construit un modèle LSTM pour la prédiction de séries temporelles.
    
    Architecture:
        - 2 couches LSTM de 64 neurones avec dropout
        - 1 couche dense de sortie (1 neurone)
    """
    model = Sequential()
    
    # Première couche LSTM
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Deuxième couche LSTM
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    
    # Couche de sortie
    model.add(Dense(1))
    
    # Compilation
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError()
    )
    
    print("\n" + "=" * 60)
    print("ARCHITECTURE DU MODÈLE")
    print("=" * 60)
    model.summary()
    print("=" * 60 + "\n")
    
    return model


def train_model(df, seq_length=SEQ_LENGTH, epochs=EPOCHS, batch_size=BATCH_SIZE, model_path=MODEL_PATH):
    """
    Entraîne le modèle LSTM sur les données fournies.
    """
    print("\n[1/3] Préparation des données...")
    X, y, scaler = prepare_data(df, seq_length)
    
    print("\n[2/3] Construction du modèle...")
    model = build_lstm_model(seq_length, (X.shape[1], 1))
    
    print(f"[3/3] Entraînement du modèle ({epochs} epochs)...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )
    
    # Sauvegarder le modèle
    model.save(model_path)
    print(f"\n✓ Modèle entraîné et sauvegardé sous {model_path}")
    
    # Afficher les métriques finales
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"  Loss finale: {final_loss:.6f}")
    print(f"  Validation loss finale: {final_val_loss:.6f}")
    
    return model, scaler


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """
    Point d'entrée du script d'entraînement.
    """
    print("=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE LSTM - BTC PRICE PREDICTION")
    print("=" * 60)
    
    # Charger les données
    df = load_data(DATA_FILE)
    
    if df is None:
        print("\n❌ Impossible de charger les données. Arrêt du script.")
        return
    
    # Vérifier la quantité de données
    if len(df) < SEQ_LENGTH:
        print(f"\n❌ Données insuffisantes : {len(df)} lignes (minimum: {SEQ_LENGTH})")
        return
    
    # Entraîner le modèle
    try:
        model, scaler = train_model(
            df,
            seq_length=SEQ_LENGTH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            model_path=MODEL_PATH
        )
        
        print("\n" + "=" * 60)
        print("✓ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
        print("=" * 60)
        print(f"  Modèle sauvegardé : {MODEL_PATH}")
        print(f"  Vous pouvez maintenant utiliser ce modèle dans V5.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'entraînement : {e}")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    main()
