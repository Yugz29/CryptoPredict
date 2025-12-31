# V3 – Ajout des indicateurs techniques

## Changements par rapport à V2

- Passage de **1 feature** (prix close) à **9 features** : prix OHLCV + indicateurs techniques
- Fenêtre d'observation : **48 minutes** (au lieu de 24)
- Modèle LSTM plus large : 128 → 64 neurones
- Librairie `ta` au lieu de `talib` (plus simple à installer)

## Indicateurs ajoutés

- **RSI (14)** : mesure le momentum (suracheté/survendu)
- **MACD** : détecte les tendances
- **EMA (20)** : moyenne mobile exponentielle
- **ATR (14)** : mesure la volatilité

L'idée : donner plus de contexte au modèle. Le prix seul ne raconte qu'une partie de l'histoire du marché.

## Ce que fait le code

**`get_historical_data()`** – Récupère les données OHLCV depuis Binance, calcule les indicateurs avec la librairie `ta`, échantillonne à la minute et interpole les valeurs manquantes.

**`prepare_data()`** – Normalise les 9 features ensemble, crée des séquences de 48 minutes, mais ne prédit que le prix de clôture (les autres features servent de contexte).

**`build_model()`** – LSTM à 2 couches (128 puis 64 neurones) avec dropout. Prend 48 minutes × 9 features en entrée, sort 60 prédictions.

**`predict_future()`** – Prédit les 60 prochaines minutes. Reconstruit une matrice complète pour dénormaliser correctement (toutes les features ont été normalisées ensemble).

**`live_predictions()`** – Entraîne une fois au démarrage, puis boucle : récupère les données → prédit → affiche → attend 60s.

## Limites

- Code plus complexe à cause de la normalisation multi-dimensionnelle
- Plus de features ne garantit pas de meilleures prédictions
- Pas de métriques ni de validation
- Le modèle ne se réadapte pas aux changements du marché