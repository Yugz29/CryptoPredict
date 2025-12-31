# Version 4 – Réentraînement continu

La V4 explore deux approches pour adapter le modèle aux données en temps réel.

## V4.py – Réentraînement complet à chaque itération

**Changements par rapport à V3** :
- Retour à **1 feature** (prix close uniquement)
- **Réentraînement complet** toutes les 30 secondes : reconstruction + entraînement
- Seulement 5 epochs par itération (au lieu de 50)
- Fenêtre réduite à 24 minutes (comme V1/V2)

**Objectif** : Adapter le modèle continuellement aux données les plus récentes en le réentraînant à chaque cycle.

**Ce que fait le code** :
- Boucle infinie : récupère données → construit nouveau modèle → entraîne → prédit → attend 30s
- Le modèle est **reconstruit de zéro** à chaque itération (poids réinitialisés)

**Problème majeur** :
- Extrêmement lent (peut prendre plus de 30s par cycle sur CPU)
- Le modèle ne garde aucune "mémoire" entre les itérations
- Inefficace : reconstruire l'architecture à chaque fois est inutile

---

## V4.1.py – Avec sauvegarde et bandes de confiance

**Améliorations par rapport à V4** :
- **Sauvegarde du modèle** après chaque entraînement (`lstm_model.h5`)
- **Chargement** du modèle existant au prochain cycle (pas de reconstruction)
- Modèle plus large : 128 → 128 neurones (au lieu de 100 → 100)
- **Huber Loss** au lieu de MSE (moins sensible aux outliers)
- Ajout de **bandes de confiance** calculées avec l'écart-type

**Ce que fait le code** :

**`build_or_load_model()`** – Tente de charger le modèle depuis `lstm_model.h5`. Si le fichier n'existe pas ou échoue, crée un nouveau modèle.

**Réentraînement incrémental** – Le modèle garde ses poids d'une itération à l'autre et continue d'apprendre avec les nouvelles données (5 epochs à chaque cycle).

**Bandes de confiance** – Calculées simplement avec `predictions ± écart-type` pour visualiser l'incertitude.

**Avantages** :
- Beaucoup plus rapide (pas de reconstruction)
- Le modèle accumule de l'apprentissage au fil du temps
- Visualisation de l'incertitude avec les bandes

**Limites** :
- Risque d'overfitting sur les données récentes
- Les bandes de confiance sont basiques (juste l'écart-type, pas une vraie incertitude bayésienne)
- Toujours pas de métriques de validation