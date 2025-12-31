# V1 – Pipeline LSTM de base

## Architecture

Un pipeline complet qui va de la récupération des données à la visualisation des prédictions :

**Récupération des données** – API Binance pour récupérer 1000 bougies (1m), calcul des indicateurs RSI, MACD, SMA, EMA via `talib`.

**Préparation** – Normalisation MinMaxScaler, création de séquences de 24 minutes pour prédire les 60 suivantes.

**Modèle** – LSTM 3 couches (100 neurones chacune) avec dropout 0.2, sortie dense de 60 valeurs.

**Boucle de prédiction** – Réentraînement complet toutes les 30s, affichage Plotly en temps réel.

## Limites de cette version

- Réentraînement à chaque itération (très coûteux)
- Ratio historique/prédiction déséquilibré (24 minutes d'historique pour prédire 60 minutes)
- Indicateurs techniques calculés mais peu exploités par le modèle
- Pas de validation ni de métriques de performance