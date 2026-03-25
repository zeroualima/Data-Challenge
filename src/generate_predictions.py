"""
Génère predictions.csv pour les données de test tronquées.

Format de sortie :
    airport, airport_alert_id, prediction_date,
    predicted_date_end_alert, confidence

Logique :
    Pour chaque alerte et chaque instant τ_n (n ≥ 3 impacts observés),
    on calcule u*(ε) pour plusieurs valeurs de ε et on émet une ligne
    par (alerte, τ_n, ε).

    prediction_date          = timestamp absolu de τ_n
    predicted_date_end_alert = timestamp absolu de τ_n + u*(ε)
    confidence               = 1 − ε
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing import load_data, DELTA, GAP_DEDUP
from src.hawkes        import fit_all_airports
from src.features      import (build_features, _temporal_features,
                                _inner_features)
from src.lgbm_model    import (train_on_all_data, FEATURES)

EPSILONS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def _to_timestamp(t_min_abs: float) -> pd.Timestamp:
    """Minutes-since-epoch → ISO8601 string with microseconds (UTC)."""
    ts = pd.Timestamp(t_min_abs * 60, unit='s', tz='UTC')
    return ts.strftime('%Y-%m-%d %H:%M:%S.%f+00:00')


def _feature_vector(history, traj_data_slice, mu, alpha, beta) -> list:
    f_t = _temporal_features(history, mu, alpha, beta)
    f_i = _inner_features(traj_data_slice)
    return [
        f_t['n_impacts'],    f_t['duree_ecoulee'],
        f_t['ia_last'],      f_t['ia_mean3'],
        f_t['ia_mean5'],     f_t['ia_min'],
        f_t['ia_max'],       f_t['ia_slope'],
        f_t['lambda_hawkes'],
        f_i['amp_last'],     f_i['amp_mean'],
        f_i['prop_cloud'],
    ]


def load_test_data(path: str):
    """
    Charge les données de test tronquées.
    Même logique que load_data() mais on garde airport_alert_id
    fourni par le jury plutôt que de le reconstruire.
    """
    df = pd.read_csv(path)
    df.columns  = df.columns.str.strip()
    df['date']  = pd.to_datetime(df['date'], utc=True)
    df['t_min'] = (df['date'] - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds() / 60
    df = df.sort_values(['airport', 'date']).reset_index(drop=True)

    df_inner = df[df['dist'] <= 20].copy()
    df_outer = df[df['dist'] <= 50].copy()

    alerts       = {}
    alerts_abs   = {}
    alerts_data  = {}
    alerts_outer = {}

    # Dans load_test_data — remplacer la boucle de segmentation
    for airport, grp in df_inner.groupby('airport'):
        grp   = grp.sort_values('date').reset_index(drop=True)
        t_abs = grp['t_min'].values

        # Déduplication
        keep = [0]
        for i in range(1, len(t_abs)):
            if t_abs[i] - t_abs[keep[-1]] >= GAP_DEDUP:
                keep.append(i)
        grp   = grp.iloc[keep].reset_index(drop=True)
        t_abs = t_abs[keep]

        # Segmentation par silence de 30 min (même logique que load_data)
        trajs, trajs_abs, trajs_data = [], [], []
        cur_rel, cur_abs, cur_idx = [0.0], [t_abs[0]], [0]

        for i in range(1, len(t_abs)):
            if t_abs[i] - cur_abs[-1] >= DELTA:
                trajs.append(np.array(cur_rel))
                trajs_abs.append(np.array(cur_abs))
                trajs_data.append(grp.iloc[cur_idx].reset_index(drop=True))
                cur_rel = [0.0]
                cur_abs = [t_abs[i]]
                cur_idx = [i]
            else:
                dt = t_abs[i] - cur_abs[-1]
                cur_rel.append(cur_rel[-1] + dt)
                cur_abs.append(t_abs[i])
                cur_idx.append(i)

        trajs.append(np.array(cur_rel))
        trajs_abs.append(np.array(cur_abs))
        trajs_data.append(grp.iloc[cur_idx].reset_index(drop=True))

        valid = [(r, a, d) for r, a, d in zip(trajs, trajs_abs, trajs_data)
                if len(r) >= 2]

        alerts[airport]      = [v[0] for v in valid]
        alerts_abs[airport]  = [v[1] for v in valid]
        alerts_data[airport] = [v[2] for v in valid]
        print(f"  {airport:20s} : {len(alerts[airport])} alertes test")
    
        print(f"  Premier timestamp absolu : {t_abs[0]:.0f} min = "
        f"{pd.Timestamp(int(t_abs[0]*60), unit='s', tz='UTC')}")

    return df, df_inner, df_outer, alerts, alerts_abs, alerts_data, alerts_outer


def generate_predictions(alerts, alerts_abs, alerts_data, all_params, models, epsilons=EPSILONS) -> pd.DataFrame:
    from src.lgbm_model import predict_ustar_batch, FEATURES

    rows_meta = []   # (airport, alert_id, pred_date, tau_n_abs)
    X_all     = []   # vecteurs de features

    for airport, trajectories in alerts.items():
        p = all_params[airport]
        mu, alpha, beta = p['mu'], p['alpha'], p['beta']

        for k, (traj, traj_abs, traj_data) in enumerate(zip(
            trajectories,
            alerts_abs[airport],
            alerts_data[airport],
        )):
            valid_ids = traj_data['airport_alert_id'].dropna()
            alert_id  = valid_ids.iloc[0] if len(valid_ids) > 0 else k
            n_total   = len(traj)

            for n in range(3, n_total):
                tau_n_abs = traj_abs[n]
                history   = traj[:n+1]

                try:
                    f_n = _feature_vector(
                        history, traj_data.iloc[:n+1],
                        mu, alpha, beta,
                    )
                except Exception as e:
                    print(f"  [WARN] {airport} alerte {alert_id} n={n}: {e}")
                    continue

                X_all.append(f_n)
                rows_meta.append((airport, alert_id, tau_n_abs))

    if not X_all:
        print("Aucune prédiction générée.")
        return pd.DataFrame()

    X_all = np.array(X_all)  # (N, 12)
    print(f"  Calcul de u* pour {len(X_all)} points × {len(epsilons)} ε...")

    # Prédiction vectorisée — un seul passage
    u_stars = predict_ustar_batch(models, X_all, epsilons)  # (N, len(epsilons))

    # Construction du DataFrame
    rows = []
    for i, (airport, alert_id, tau_n_abs) in enumerate(rows_meta):
        pred_date = _to_timestamp(tau_n_abs)
        for j, eps in enumerate(epsilons):
            predicted_end = _to_timestamp(tau_n_abs + u_stars[i, j])
            rows.append({
                'airport'                  : airport,
                'airport_alert_id'         : int(alert_id),
                'prediction_date'          : pred_date,
                'predicted_date_end_alert' : predicted_end,
                'confidence'               : round(1.0 - eps, 4),
            })

    df_pred = pd.DataFrame(rows)
    print(f"\nPrédictions générées : {len(df_pred):,} lignes")
    print(f"  Aéroports  : {df_pred['airport'].nunique()}")
    print(f"  Alertes    : {df_pred['airport_alert_id'].nunique()}")
    print(f"  Confiances : {sorted(df_pred['confidence'].unique())}")
    return df_pred


if __name__ == '__main__':
    TRAIN_PATH  = './data_train_databattle2026/segment_alerts_all_airports_train.csv'
    TEST_PATH   = './dataset_test/dataset_set.csv'
    OUTPUT_PATH = './dataset_test/predictions.csv'
    Path('figures').mkdir(exist_ok=True)

    # 1. Charger données d'entraînement
    print("=== Chargement données train ===")
    (df_tr, df_inner_tr, df_outer_tr,
     alerts_tr, alerts_abs_tr, alerts_data_tr,
     alerts_outer_tr) = load_data(TRAIN_PATH)

    # 2. Hawkes sur toutes les données train
    print("\n=== Ajustement Hawkes ===")
    all_params = fit_all_airports(alerts_tr)

    # 3. Features sur toutes les données train
    print("\n=== Construction features ===")
    from src.features import build_features
    df_features = build_features(
        alerts_tr, all_params, df_inner_tr,
        alerts_abs_tr, alerts_data_tr, alerts_outer_tr,
    )

    # 4. Entraîner sur TOUTES les données (pas de split)
    print("\n=== Entraînement sur toutes les données ===")
    models = train_on_all_data(df_features)

    # 5. Charger données test tronquées
    print("\n=== Chargement données test ===")
    (df_te, df_inner_te, df_outer_te,
     alerts_te, alerts_abs_te, alerts_data_te,
     alerts_outer_te) = load_test_data(TEST_PATH)

    # 6. Générer prédictions
    print("\n=== Génération des prédictions ===")
    df_pred = generate_predictions(
        alerts_te, alerts_abs_te, alerts_data_te,
        all_params, models,
    )

    # 7. Sauvegarder
    # raw_dtype = pd.read_csv(TEST_PATH, nrows=1)['airport_alert_id'].dtype ###
    # df_pred['airport_alert_id'] = df_pred['airport_alert_id'].astype(raw_dtype) ###
    df_pred.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Sauvegardé dans {OUTPUT_PATH}")
    print("\nAperçu :")
    print(df_pred.head(6).to_string(index=False))