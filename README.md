# CryptoPredict

Projet de prédiction du prix du Bitcoin avec des LSTM, réalisé en 2023-2024 avant ma formation en développement.

> **⚠️ Avertissement** : Projet purement expérimental. Ne pas utiliser pour du trading réel.

## Contexte

En 2023, j'avais des bitcoins et je voulais créer un algo pour prédire les prix. Je ne connaissais rien au trading ni au développement à l'époque.

Ce projet a été réalisé avec l'aide de ChatGPT : je décrivais ce que je voulais, ChatGPT générait le code, je testais, ça plantait, je réitérais. Beaucoup d'allers-retours jusqu'à obtenir quelque chose qui fonctionne.

Cette expérience m'a donné envie de me reconvertir dans le dev. Ce repository documente mes premières expérimentations avec les réseaux de neurones LSTM appliqués aux séries temporelles financières.

## Les Versions

Le projet compte 5 versions, chacune explorant une approche ou résolvant un problème spécifique :

### [V1](./V1/) – Pipeline LSTM de base
Premier prototype : récupération des données Binance, indicateurs techniques (RSI, MACD, SMA, EMA), modèle LSTM simple. Réentraînement à chaque itération (très coûteux).

### [V2](./V2/) – Simplification et expérimentation
Deux fichiers distincts pour tester différentes approches : données horaires vs minute, ratio équilibré vs déséquilibré, entraînement unique vs continu.

### [V3](./V3/) – Modèle multi-features
Passage de 1 feature (prix) à 9 features (OHLCV + indicateurs techniques). Fenêtre élargie à 48 minutes, architecture LSTM plus large.

### [V4](./V4/) – Réentraînement continu
Tentatives pour adapter le modèle aux nouvelles données : réentraînement complet à chaque cycle (V4), puis sauvegarde/chargement du modèle avec bandes de confiance (V4.1).

### [V5](./V5/) – Interface web
Application web avec Dash pour une visualisation centralisée. Séparation de l'entraînement et de l'inférence en deux scripts distincts.

## Limites

Ce projet comporte de nombreuses simplifications :
- Pas de validation robuste (backtest, métriques)
- Ratios historique/prédiction parfois déséquilibrés
- Pas de gestion de la dérive du modèle
- Incompatibilités d'architecture entre certaines versions

L'objectif était de comprendre le fonctionnement des LSTM, pas de créer un système de trading fonctionnel.

## Évolution

Ce projet a inspiré **DevNote**, mon projet actuel : un outil pour centraliser notes, code et apprentissages pendant le développement.

## Technologies

Python, TensorFlow/Keras, Pandas, Plotly, Dash, scikit-learn, ta/talib