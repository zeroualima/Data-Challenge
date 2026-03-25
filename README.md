# 🏆 Data Battle IA PAU 2026 — Prédiction probabiliste de fin d'alerte foudre

## 👥 Équipe

- **Nom de l'équipe :** L2ay2ay
- **Membres :**
  - ZEROUALI Mohammed Amine
  - BOUBEKRI Saad

---

## 🎯 Problématique

Les aéroports doivent suspendre certaines activités dès qu'un impact de foudre
survient dans un rayon de 20 km. Avec la méthode actuelle, l'alerte reste active
pendant **30 minutes fixes** après le dernier impact observé — indépendamment
de la dynamique réelle de l'orage.

L'objectif est de **réduire ce délai d'attente** en estimant, à chaque instant,
la probabilité que l'alerte soit encore active — permettant une reprise
d'activité plus rapide tout en maintenant un niveau de risque contrôlé.

---

## 💡 Solution proposée

Notre solution combine deux modèles complémentaires.

**Étape 1 — Processus de Hawkes (modèle probabiliste)**
Les impacts de foudre dans le disque de 20 km sont modélisés comme un processus
ponctuel auto-excitant. L'intensité conditionnelle
`λ*(t) = μ + α Σ exp(-β(t - τᵢ))` est estimée par maximum de vraisemblance
(méthode d'Ogata) sur l'historique des alertes. Ce modèle produit à chaque
instant l'indicateur `λ*(τₙ)` — le niveau d'excitation courant de l'orage.

**Étape 2 — LightGBM avec quantile regression (modèle prédictif)**
À partir des features temporelles (inter-arrivées, tendances) et de `λ*(τₙ)`,
treize modèles LightGBM prédisent les quantiles 5 % à 95 % du délai avant le
prochain impact. Ces quantiles construisent une **courbe de survie conditionnelle**
`S(u) = P(T > τₙ + u | Hₙ)` depuis laquelle on extrait le délai optimal :
```
u* = inf{ u ∈ [0, 30] | S(u) < ε }
```

Le paramètre ε (risque toléré) est contrôlé via le seuil de confiance θ,
permettant de trouver le meilleur compromis gain/risque.

**Résultats obtenus sur le jeu de validation :**
- Gain moyen : **~15 min** sur la méthode fixe de 30 min
- Calibration des quantiles : couverture réelle ≈ quantile théorique (q=0.80 → 0.799)
- Theta recommandé : **0.88** pour un risque < 2 %

---

## ⚙️ Stack technique

- **Langages :** Python 3.12
- **Frameworks :** LightGBM, NumPy, pandas, SciPy
- **Outils :** Git, Jupyter (évaluation), matplotlib
- **IA :** LightGBM (gradient boosting, quantile regression) + Processus de Hawkes (MLE)
- **Infrastructure :** 100 % local, aucune dépendance cloud

---

## 🚀 Installation & exécution

### Prérequis

- Python 3.10+
- pip

### Installation
```bash
git clone <url-du-depot>
cd "DATA CHALLENGE"
pip install pandas numpy scipy lightgbm scikit-learn matplotlib tqdm
```

Placer les données dans les dossiers suivants :
```
data_train_databattle2026/segment_alerts_all_airports_train.csv
dataset_test/dataset_set.csv
```

### Exécution

**Pipeline complet (entraînement + prédictions) :**
```bash
python3 main.py
```

Produit `dataset_test/predictions.csv` avec le format attendu par le jury.

**Génération des prédictions uniquement :**
```bash
python3 -m src.generate_predictions
```

**Évaluation locale (notebook du jury) :**
```bash
jupyter notebook dataset_test/Evaluation_databattle_meteorage.ipynb
```
Modifier `input_file` pour pointer vers `dataset_test/dataset_set.csv`.

---

## 📊 Explication du modèle et des résultats

Le fichier `Rapport/Data_Challenge.pdf` contient :
- La formulation mathématique rigoureuse du problème
- La dérivation de la courbe de survie conditionnelle
- Les résultats de validation (calibration des quantiles, feature importance)
- L'analyse des limites du modèle de Hawkes stationnaire
- La description complète du pipeline LightGBM

### Features les plus importantes (gain LightGBM)

| Feature | Description | Importance |
|---|---|---|
| `ia_max` | Inter-arrivée maximale observée | ★★★ |
| `ia_min` | Inter-arrivée minimale observée | ★★★ |
| `duree_ecoulee` | Durée depuis le début de l'alerte | ★★★ |
| `lambda_hawkes` | Intensité Hawkes courante λ*(τₙ) | ★★ |
| `ia_last` | Dernière inter-arrivée | ★★ |

---

## 🌱 Impact environnemental et social

**Green coding :**
- Code modulaire en 7 fichiers indépendants — maintenable et portable
- Aucune dépendance cloud ni GPU — fonctionne sur un laptop standard
- Entraînement LightGBM sur 92 000 exemples en < 5 min CPU
- Inférence vectorisée — 35 000 prédictions en < 30 secondes

**Impact métier :**
- Gain estimé de 10 à 15 min par alerte sur ~477 alertes test
- Impact direct : reprise plus rapide des opérations aéroportuaires
- Réduction du temps d'immobilisation des équipes au sol

**Effets rebonds identifiés :**
- Un modèle trop agressif (ε trop grand) augmente le risque d'impacts non
  couverts — le paramètre θ doit être calibré avec soin par Meteorage
- La déduplication à 10 secondes supprime les multi-détections capteurs
  mais pourrait éliminer des impacts légitimes dans des orages très denses

**Poursuite du projet :**
- Extension vers un Hawkes marqué intégrant amplitude et distance
- Application temps réel via une API légère (FastAPI) consommant les flux
  Meteorage en streaming
- Exploration d'un modèle non stationnaire capturant le déplacement spatial
  de l'orage (limitation identifiée du modèle actuel)