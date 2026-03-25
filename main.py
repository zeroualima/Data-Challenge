from pathlib import Path
import numpy as np
from src.preprocessing       import load_data
from src.hawkes               import fit_all_airports
from src.features             import build_features
from src.lgbm_model           import train_quantile_models, train_on_all_data
from src.generate_predictions import load_test_data, generate_predictions

Path('figures').mkdir(exist_ok=True)

TRAIN_PATH = './data_train_databattle2026/segment_alerts_all_airports_train.csv'
TEST_PATH  = './dataset_test/dataset_set.csv'

# 1. Chargement et Hawkes
print("=== Chargement données train ===")
df, df_inner, df_outer, alerts, alerts_abs, alerts_data, alerts_outer = \
    load_data(TRAIN_PATH)

print("\n=== Ajustement Hawkes ===")
all_params = fit_all_airports(alerts)

# 2. Features
print("\n=== Construction features ===")
df_features = build_features(
    alerts, all_params, df_inner,
    alerts_abs, alerts_data, alerts_outer,
)

# 3. Évaluation (train/test split) — pour mesurer la qualité
print("\n=== Évaluation du modèle ===")
models_eval, X_test, y_test = train_quantile_models(df_features)

# Gain moyen sur le test
from src.lgbm_model import predict_ustar_batch
import numpy as np
u_stars = predict_ustar_batch(models_eval, X_test, epsilons=[0.10])
gains   = 30.0 - u_stars[:, 0]
print(f"Gain moyen : {np.mean(gains):.1f} min")
print(f"Gain médian : {np.median(gains):.1f} min")

# 4. Entraînement final sur TOUTES les données
print("\n=== Entraînement final (toutes données) ===")
models_final = train_on_all_data(df_features)

# 5. Chargement données test tronquées
print("\n=== Chargement données test ===")
_, _, _, alerts_te, alerts_abs_te, alerts_data_te, _ = \
    load_test_data(TEST_PATH)

# 6. Génération predictions.csv
print("\n=== Génération des prédictions ===")
df_pred = generate_predictions(
    alerts_te, alerts_abs_te, alerts_data_te,
    all_params, models_final,
)
df_pred.to_csv('./dataset_test/predictions.csv', index=False)
print("✓ predictions.csv sauvegardé")
print(df_pred.head(6).to_string(index=False))