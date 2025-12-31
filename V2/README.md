# V2 – Simplification et expérimentation temps réel

La V2 se décline en deux fichiers distincts pour tester différentes approches.

## V2.py – Version de base (données horaires)

**Changements par rapport à V1** :
- Passage en **données horaires** (1h) au lieu de minutes
- Suppression des indicateurs techniques (RSI, MACD, SMA, EMA) → uniquement le prix de clôture
- Ratio équilibré : 24h d'historique pour prédire les 24h suivantes
- Modèle simplifié : 2 couches LSTM de 50 neurones (au lieu de 3x100)
- **Pas de boucle temps réel** : un seul entraînement, une seule prédiction, affichage console

**Architecture** :
- Récupération de 500 bougies horaires via API Binance
- Normalisation MinMaxScaler
- LSTM 2 couches (50 neurones chacune)
- 20 epochs, batch size 32

**Objectif** : Version épurée pour comprendre le comportement de base du LSTM sur une échelle de temps plus large, sans la complexité des indicateurs techniques.

## V2.1.py – Version temps réel (données minute)

**Évolution par rapport à V2** :
- Retour aux **données minute** (1m) pour du quasi-temps réel
- Prédiction sur 60 minutes (comme V1) mais avec un modèle simplifié
- Dropout ajouté (0.2) pour limiter le surapprentissage
- **Entraînement unique au démarrage** : le modèle n'est plus réentraîné à chaque itération
- Boucle de rafraîchissement toutes les 60 secondes
- Visualisation Plotly : prix passés (60 min) vs prédictions futures (60 min)

**Architecture** :
- Récupération de 1000 bougies minute via API Binance
- Échantillonnage régulier et interpolation des données manquantes
- LSTM 2 couches (100 neurones) + Dropout 0.2
- 50 epochs au démarrage, puis prédictions continues sans réentraînement

**Objectif** : Visualiser en temps réel les prédictions du modèle et observer son comportement face aux données réelles qui arrivent minute par minute.

## Limites communes aux deux versions

- Toujours un réentraînement complet (V2) ou au démarrage (V2.1), pas de sauvegarde/chargement du modèle
- Pas de validation robuste ni de métriques de performance
- Pas de stratégie pour gérer la dérive du modèle dans le temps (V2.1)
- Simplification extrême des features (uniquement le prix)