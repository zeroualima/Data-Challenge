"""
src/generate_predictions.py
----------------------------
Generates predictions.csv in the format expected by the evaluation notebook.

Key design:
    - Predict at EVERY τ_n (n ≥ 1), not just the last one.
    - confidence = 1 − ε  (NOT u*/30)
    - This produces a genuine Pareto curve when the evaluator sweeps θ:
        high θ → only conservative ε, late τ_n → low gain, low risk
        low  θ → aggressive ε, early τ_n → high gain, higher risk
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing import load_data, DELTA, GAP_DEDUP
from src.hawkes        import fit_all_airports
from src.features      import _temporal_features, _inner_features
from src.lgbm_model    import train_on_all_data, predict_ustar_batch, FEATURES

# ε grid — covers the full confidence range [0.70, 0.98] that θ will sweep
EPSILONS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_timestamp(t_min_abs: float) -> str:
    """Minutes-since-epoch → ISO8601 string, always with microseconds (UTC)."""
    ts = pd.Timestamp(t_min_abs * 60, unit='s', tz='UTC')
    return ts.strftime('%Y-%m-%d %H:%M:%S.%f+00:00')


def _feature_vector(history, traj_data_slice, mu, alpha, beta) -> list:
    """12-feature vector matching lgbm_model.FEATURES exactly."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Test data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(path: str):
    """
    Load truncated test data, segmenting by airport_alert_id from the jury
    (rather than re-segmenting by silence, which may differ).
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['date']  = pd.to_datetime(df['date'], utc=True)
    df['t_min'] = (df['date'] - pd.Timestamp('1970-01-01', tz='UTC')
                   ).dt.total_seconds() / 60
    df = df.sort_values(['airport', 'date']).reset_index(drop=True)

    df_inner = df[df['dist'] <= 20].copy()
    df_outer = df[df['dist'] <= 50].copy()

    alerts, alerts_abs, alerts_data, alerts_outer = {}, {}, {}, {}

    for airport, grp in df_inner.groupby('airport'):
        grp = grp.sort_values('date').reset_index(drop=True)

        trajs, trajs_abs, trajs_data = [], [], []

        # Segment by jury-provided airport_alert_id — authoritative
        for _, sub in grp.groupby('airport_alert_id', sort=True):
            sub   = sub.sort_values('date').reset_index(drop=True)
            t_abs = sub['t_min'].values

            # Deduplicate within 10 seconds
            keep = [0]
            for i in range(1, len(t_abs)):
                if t_abs[i] - t_abs[keep[-1]] >= GAP_DEDUP:
                    keep.append(i)
            sub   = sub.iloc[keep].reset_index(drop=True)
            t_abs = t_abs[keep]

            if len(t_abs) < 1:
                continue

            t_rel = t_abs - t_abs[0]
            trajs.append(t_rel)
            trajs_abs.append(t_abs)
            trajs_data.append(sub)

        alerts[airport]      = trajs
        alerts_abs[airport]  = trajs_abs
        alerts_data[airport] = trajs_data

        # Align outer ring data per alert
        df_out_ap  = df_outer[df_outer['airport'] == airport]
        outer_list = []
        for traj_abs_k in trajs_abs:
            t_start = traj_abs_k[0]
            t_end   = traj_abs_k[-1]
            chunk   = df_out_ap[
                (df_out_ap['t_min'] >= t_start - 0.1) &
                (df_out_ap['t_min'] <= t_end   + 0.1)
            ].reset_index(drop=True)
            outer_list.append(chunk)
        alerts_outer[airport] = outer_list

        print(f"  {airport:20s} : {len(trajs)} alertes test")

    return df, df_inner, df_outer, alerts, alerts_abs, alerts_data, alerts_outer


# ─────────────────────────────────────────────────────────────────────────────
# Prediction generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_predictions(
    alerts, alerts_abs, alerts_data,
    all_params, models,
    epsilons=EPSILONS,
) -> pd.DataFrame:
    """
    For each alert, emit one row per (τ_n, ε).
    confidence = 1 − ε so the evaluator's θ-sweep produces a Pareto curve.

    Predict at every τ_n with n ≥ 1 (need ≥ 1 inter-arrival for features).
    Single-strike alerts get a safe baseline row (u* = 30 min).
    """
    computable = []   # (airport, alert_id, tau_n_abs, f_n)
    fallback   = []   # (airport, alert_id, tau_last) → u* = 30

    for airport, trajectories in alerts.items():
        p = all_params[airport]
        mu, alpha, beta = p['mu'], p['alpha'], p['beta']

        for traj, traj_abs, traj_data in zip(
            trajectories,
            alerts_abs[airport],
            alerts_data[airport],
        ):
            alert_id = float(traj_data['airport_alert_id'].iloc[0])
            n_total  = len(traj)

            if n_total < 2:
                # Single-strike alert: emit safe baseline
                fallback.append((airport, alert_id, traj_abs[-1]))
                continue

            # Predict at every τ_n where we have ≥ 1 inter-arrival (n ≥ 1)
            for n in range(1, n_total):
                tau_n_abs = traj_abs[n]
                history   = traj[:n + 1]

                try:
                    f_n = _feature_vector(
                        history, traj_data.iloc[:n + 1],
                        mu, alpha, beta,
                    )
                    computable.append((airport, alert_id, tau_n_abs, f_n))
                except Exception as e:
                    print(f"  [WARN] {airport} alerte {alert_id} n={n}: {e}")
                    fallback.append((airport, alert_id, tau_n_abs))

    rows = []

    # ── Batch predict for all computable points ───────────────────────────────
    if computable:
        X_all   = np.array([c[3] for c in computable])
        print(f"  Calcul de u* pour {len(X_all)} points × {len(epsilons)} ε...")
        u_stars = predict_ustar_batch(models, X_all, epsilons)  # (N, len(ε))

        for i, (airport, alert_id, tau_n_abs, _) in enumerate(computable):
            pred_date = _to_timestamp(tau_n_abs)
            for j, eps in enumerate(epsilons):
                rows.append({
                    'airport'                  : airport,
                    'airport_alert_id'         : alert_id,
                    'prediction_date'          : pred_date,
                    'predicted_date_end_alert' : _to_timestamp(tau_n_abs + u_stars[i, j]),
                    'confidence'               : round(1.0 - eps, 4),  # ← correct
                })

    # ── Baseline rows for short/failed alerts ─────────────────────────────────
    for airport, alert_id, tau_last in fallback:
        pred_date = _to_timestamp(tau_last)
        for eps in epsilons:
            rows.append({
                'airport'                  : airport,
                'airport_alert_id'         : alert_id,
                'prediction_date'          : pred_date,
                'predicted_date_end_alert' : _to_timestamp(tau_last + 30.0),
                'confidence'               : round(1.0 - eps, 4),
            })

    df_pred = pd.DataFrame(rows)
    print(f"\nPrédictions générées : {len(df_pred):,} lignes")
    print(f"  Aéroports  : {df_pred['airport'].nunique()}")
    print(f"  Alertes    : {df_pred['airport_alert_id'].nunique()}")
    print(f"  Confiances : {sorted(df_pred['confidence'].unique())}")
    return df_pred


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    TRAIN_PATH  = './data_train_databattle2026/segment_alerts_all_airports_train.csv'
    TEST_PATH   = './dataset_test/dataset_set.csv'
    OUTPUT_PATH = './dataset_test/predictions.csv'
    Path('figures').mkdir(exist_ok=True)

    print("=== Chargement données train ===")
    (df_tr, df_inner_tr, df_outer_tr,
     alerts_tr, alerts_abs_tr, alerts_data_tr,
     alerts_outer_tr) = load_data(TRAIN_PATH)

    print("\n=== Ajustement Hawkes ===")
    all_params = fit_all_airports(alerts_tr)

    print("\n=== Construction features ===")
    from src.features import build_features
    df_features = build_features(
        alerts_tr, all_params, df_inner_tr,
        alerts_abs_tr, alerts_data_tr, alerts_outer_tr,
    )

    print("\n=== Entraînement sur toutes les données ===")
    models = train_on_all_data(df_features)

    print("\n=== Chargement données test ===")
    (df_te, df_inner_te, df_outer_te,
     alerts_te, alerts_abs_te, alerts_data_te,
     alerts_outer_te) = load_test_data(TEST_PATH)

    print("\n=== Génération des prédictions ===")
    df_pred = generate_predictions(
        alerts_te, alerts_abs_te, alerts_data_te,
        all_params, models,
    )

    # Match airport_alert_id dtype to the raw test CSV to avoid KeyError
    raw_dtype = pd.read_csv(TEST_PATH, nrows=1)['airport_alert_id'].dtype
    df_pred['airport_alert_id'] = df_pred['airport_alert_id'].astype(raw_dtype)

    df_pred.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Sauvegardé dans {OUTPUT_PATH}")
    print("\nAperçu :")
    print(df_pred.head(9).to_string(index=False))