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
d'activité plus rapide tout en maintenant un niveau de risque contrôlé (R < 2 %).

---

## 💡 Solution proposée

Notre solution combine deux modèles complémentaires.

**Étape 1 — Processus de Hawkes (modèle probabiliste)**

Les impacts de foudre dans le disque de 20 km sont modélisés comme un processus
ponctuel auto-excitant. L'intensité conditionnelle
`λ*(t) = μ + α Σ exp(-β(t − τᵢ))` est estimée par maximum de vraisemblance
(méthode d'Ogata), avec une grille d'initialisations pour éviter les minima locaux.
Les paramètres sont ajustés **par aéroport** sur 10 ans de données d'entraînement :

| Aéroport | μ | α | β | α/β |
|---|---|---|---|---|
| Ajaccio  | 0.0557 | 0.2122 | 0.2411 | 0.880 |
| Bastia   | 0.0561 | 0.2146 | 0.2374 | 0.904 |
| Biarritz | 0.0540 | 0.2309 | 0.2527 | 0.914 |
| Nantes   | 0.0574 | 0.2310 | 0.2558 | 0.903 |
| Pise     | 0.0562 | 0.2041 | 0.2288 | 0.892 |

Ce modèle fournit à chaque instant l'indicateur `λ*(τₙ)` — le niveau d'excitation
courant de l'orage — utilisé comme feature clé pour la prédiction.

**Étape 2 — LightGBM avec régression par quantiles (modèle prédictif)**

À partir de 12 features temporelles (inter-arrivées, tendances, amplitude, `λ*(τₙ)`),
13 modèles LightGBM prédisent les quantiles 5 % à 95 % du délai avant le prochain
impact. Ces quantiles construisent une **courbe de survie conditionnelle** :

```
S(u | Hₙ) = P(T > τₙ + u | Hₙ)
u* = inf{ u ∈ [0, 30] | S(u) < ε }
```

Le paramètre ε est contrôlé via le seuil de confiance θ = 1 − ε, permettant de
trouver le meilleur compromis gain/risque selon le critère du jury.

**Résultats sur le jeu d'évaluation (612 alertes, 5 aéroports) :**

| Métrique | Valeur |
|---|---|
| Gain total (θ = 0.85) | **80 heures** |
| Risque R | **0.0182 < 0.02** ✅ |
| Alertes couvertes | 412 / 612 |
| Entraînement (92 377 exemples) | < 5 min CPU |

Détail par aéroport à θ = 0.85 :

| Aéroport | Alertes | Gain (h) | Manqués |
|---|---|---|---|
| Ajaccio  | 96  | 25.5 | 2 |
| Bastia   | 141 | 20.3 | 5 |
| Biarritz | 118 |  7.8 | 0 |
| Nantes   |  45 |  5.7 | 0 |
| Pise     | 146 | 21.3 | 0 |

---

## ⚙️ Stack technique

- **Langages :** Python 3.12
- **Frameworks :** LightGBM, NumPy, pandas, SciPy
- **Outils :** Git, Jupyter (évaluation), matplotlib
- **IA :** LightGBM (gradient boosting, régression par quantiles) + Processus de Hawkes (MLE d'Ogata)
- **Infrastructure :** 100 % local, aucune dépendance cloud ni GPU

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

Les données tronquées fournies par le jury sont présentes déjà dans `dataset_test/dataset_set.csv`, c'est à dire il suffit de les changer pour générer d'autres prédictions.


### Exécution

**Pipeline complet (entraînement sur toutes les données + génération des prédictions) :**

```bash
python3 -m src.generate_predictions
```

Produit `dataset_test/predictions.csv` pour les données de `dataset_test/dataset_set.csv` avec le format attendu par le jury
(colonnes : `airport`, `airport_alert_id`, `prediction_date`,
`predicted_date_end_alert`, `confidence`).

**Évaluation locale des prédictions :**

```bash
python3 evaluate_predictions.py \
  --data  ./dataset_test/segment_alerts_all_airports_eval.csv \
  --preds ./dataset_test/predictions.csv \
  --output ./figures/evaluation_curve.png
```

---

## 📁 Structure du projet

```
DATA CHALLENGE/
├── data_train_databattle2026/
│   └── segment_alerts_all_airports_train.csv   # données d'entraînement (10 ans)
├── dataset_test/
│   ├── dataset_set.csv                          # données test tronquées
│   ├── predictions.csv                          # prédictions générées (soumission)
│   └── Evaluation_databattle_meteorage.ipynb    # notebook jury
├── src/
│   ├── preprocessing.py   # segmentation des alertes
│   ├── hawkes.py          # estimation MLE du processus de Hawkes
│   ├── features.py        # construction des features
│   ├── lgbm_model.py      # entraînement LightGBM par quantile + u*
│   ├── simulation.py      # validation par simulation (thinning d'Ogata)
│   ├── validation.py      # goodness-of-fit
│   └── generate_predictions.py   # pipeline complet → predictions.csv
├── main.py                        # entraînement + évaluation locale (train only)
├── evaluate_predictions.py        # évaluation locale (courbe Pareto gain/risque)
├── Rapport/
│   └── rapport_final.pdf         # rapport complet
├── Presentation/
|   └── presentation.pdf
└── figures/                       # graphiques générés
```

---

## 📊 Modèle et résultats détaillés

Le fichier `Rapport/Data_Challenge.pdf` contient :
- La formulation mathématique rigoureuse du problème
- La dérivation de la courbe de survie conditionnelle
- Les résultats de calibration des quantiles
- La feature importance LightGBM
- L'analyse des limites du modèle de Hawkes stationnaire

### Features les plus importantes (gain LightGBM)

| Feature | Description |
|---|---|
| `ia_max` | Inter-arrivée maximale observée dans l'alerte |
| `ia_min` | Inter-arrivée minimale observée dans l'alerte |
| `duree_ecoulee` | Durée écoulée depuis le début de l'alerte |
| `lambda_hawkes` | Intensité Hawkes courante λ*(τₙ) |
| `ia_last` | Dernière inter-arrivée observée |

---

## 🌱 Impact environnemental et social

- Aucune dépendance cloud ni GPU — fonctionne sur un laptop standard
- Entraînement sur 92 377 exemples en moins de 5 minutes CPU
- Inférence vectorisée : 37 000 prédictions en moins de 30 secondes
- Gain estimé de **80 heures** sur ~412 alertes test → reprise plus rapide
  des opérations aéroportuaires avec un risque contrôlé (R = 1.82 %)