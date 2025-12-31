# Version 5 – Interface web avec Dash

La V5 introduit une interface web pour visualiser les prédictions et sépare l'entraînement de l'inférence.

## Changements par rapport à V4

- **Interface web** avec Dash : application accessible dans le navigateur
- **Séparation** de l'entraînement et de la prédiction en deux scripts distincts
- Mise à jour automatique du graphique toutes les 30s sans recharger la page
- Notifications en temps réel (prix actuel, prédiction +1h)

## V5.py – Application web interactive

**Architecture Dash** :
- `create_dash_app()` : configure l'interface (layout HTML + graphique + zone de notification)
- `dcc.Interval` : composant qui déclenche automatiquement les mises à jour
- Callback `update_graph()` : récupère les données → réentraîne → prédit → met à jour le graphique

**Fonctionnement** :
- Lancement du serveur local sur `http://127.0.0.1:8050/`
- Chargement du modèle au démarrage (ou création si inexistant)
- Boucle automatique : toutes les 30s, le callback est déclenché et met à jour le graphique
- Affichage des bandes de confiance comme dans V4.1

**Avantages** :
- Interface centralisée et accessible depuis n'importe quel navigateur
- Pas de gestion manuelle de fenêtres Plotly
- Possibilité d'ajouter des contrôles interactifs (boutons, sliders)

## train_model.py – Entraînement séparé

**Objectif** : Entraîner le modèle offline sur un dataset historique au format CSV.

**Fonctionnement** :
- `load_data()` : charge un CSV avec colonnes `timestamp` (index) et `close`
- `prepare_data()` : crée des séquences de 60 minutes pour prédire **le timestep suivant** (1 minute)
- `build_lstm_model()` : LSTM 2 couches de 64 neurones (architecture plus légère)
- `train_model()` : entraîne pendant 10 epochs et sauvegarde dans `btc_lstm_model.h5`

**Avantages** :
- Entraînement initial long et approfondi sans bloquer l'application web
- Le modèle pré-entraîné peut ensuite être chargé par V5.py pour du fine-tuning

## Problème identifié : incompatibilité d'architecture

**train_model.py** entraîne un modèle qui prédit **1 timestep** (sortie : Dense(1))  
**V5.py** attend un modèle qui prédit **60 timesteps** (sortie : Dense(60))

Les deux scripts ne sont pas compatibles. Le modèle sauvegardé par `train_model.py` ne peut pas être utilisé directement par `V5.py` sans modification.

**Solution** : Uniformiser les architectures (soit les deux prédisent 1 timestep, soit les deux prédisent 60 timesteps).

## Limites

- Incompatibilité entre train_model.py et V5.py
- Dash devient verbeux pour des interfaces complexes
- Réentraînement à chaque cycle toujours présent (comme V4)
- Pas de métriques de validation ni de backtest