"""
src/generate_predictions.py
----------------------------
Generates predictions.csv in the format expected by the evaluation notebook.

Root-cause fix vs previous version:
    The test CSV has airport_alert_id = NaN for the vast majority of rows.
    groupby('airport_alert_id') silently dropped those rows → only ~430/612
    alerts were processed → the 182 uncovered alerts inflated R above 0.02
    making it impossible to satisfy R < 0.02.

    Fix: segment by silence of 30 min (same as load_data in preprocessing.py),
    then recover airport_alert_id from whichever rows in the segment have one.
    Alerts that the jury hasn't labelled yet get id=-1 and still get predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing import load_data, DELTA, GAP_DEDUP
from src.hawkes        import fit_all_airports
from src.features      import _temporal_features, _inner_features
from src.lgbm_model    import train_on_all_data, predict_ustar_batch, FEATURES

EPSILONS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_timestamp(t_min_abs: float) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
# Test data loader  — silence-based segmentation (fixes NaN alert_id bug)
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(path: str):
    """
    Segments the test CSV using 30-min silence gaps (identical to load_data),
    then assigns airport_alert_id = first non-NaN id found in each segment,
    or -1 if the segment has no labelled id yet.

    This ensures ALL inner-zone strikes are processed regardless of whether
    the jury has assigned an alert id to them.
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
        grp   = grp.sort_values('date').reset_index(drop=True)
        t_abs = grp['t_min'].values

        # ── Step 1: deduplicate within 10 s ──────────────────────────────────
        keep = [0]
        for i in range(1, len(t_abs)):
            if t_abs[i] - t_abs[keep[-1]] >= GAP_DEDUP:
                keep.append(i)
        grp   = grp.iloc[keep].reset_index(drop=True)
        t_abs = t_abs[keep]

        # ── Step 2: segment by 30-min silence (same logic as preprocessing) ──
        trajs, trajs_abs, trajs_data = [], [], []
        cur_rel, cur_abs, cur_idx    = [0.0], [t_abs[0]], [0]

        for i in range(1, len(t_abs)):
            if t_abs[i] - cur_abs[-1] >= DELTA:
                # Close current segment
                trajs.append(np.array(cur_rel))
                trajs_abs.append(np.array(cur_abs))
                trajs_data.append(grp.iloc[cur_idx].reset_index(drop=True))
                cur_rel, cur_abs, cur_idx = [0.0], [t_abs[i]], [i]
            else:
                dt = t_abs[i] - cur_abs[-1]
                cur_rel.append(cur_rel[-1] + dt)
                cur_abs.append(t_abs[i])
                cur_idx.append(i)

        # Last segment
        trajs.append(np.array(cur_rel))
        trajs_abs.append(np.array(cur_abs))
        trajs_data.append(grp.iloc[cur_idx].reset_index(drop=True))

        # ── Step 3: assign alert_id — first non-NaN in segment, else -1 ─────
        # (segments with len < 1 are impossible here, but guard anyway)
        valid_trajs, valid_abs, valid_data = [], [], []
        for t_rel, t_a, t_d in zip(trajs, trajs_abs, trajs_data):
            if len(t_rel) < 1:
                continue

            # Recover alert id from whichever rows the jury labelled
            known_ids = t_d['airport_alert_id'].dropna()
            aid = float(known_ids.iloc[0]) if len(known_ids) > 0 else -1.0

            # Store alert_id back into the slice so generate_predictions can read it
            t_d = t_d.copy()
            t_d['airport_alert_id'] = aid

            valid_trajs.append(t_rel)
            valid_abs.append(t_a)
            valid_data.append(t_d)

        alerts[airport]      = valid_trajs
        alerts_abs[airport]  = valid_abs
        alerts_data[airport] = valid_data

        # ── Step 4: align outer ring data per alert ───────────────────────────
        df_out_ap  = df_outer[df_outer['airport'] == airport]
        outer_list = []
        for traj_abs_k in valid_abs:
            t_start = traj_abs_k[0]
            t_end   = traj_abs_k[-1]
            chunk   = df_out_ap[
                (df_out_ap['t_min'] >= t_start - 0.1) &
                (df_out_ap['t_min'] <= t_end   + 0.1)
            ].reset_index(drop=True)
            outer_list.append(chunk)
        alerts_outer[airport] = outer_list

        n_labelled   = sum(1 for d in valid_data if d['airport_alert_id'].iloc[0] != -1.0)
        n_unlabelled = len(valid_data) - n_labelled
        print(f"  {airport:20s} : {len(valid_trajs)} alertes  "
              f"({n_labelled} labelled, {n_unlabelled} unlabelled→id=-1)")

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
    Predict at every τ_n (n ≥ 1).  confidence = 1 − ε.
    Alerts with id=-1 are still predicted but won't be matched by the jury evaluator
    (they have no ground truth yet) — they add gain without adding risk.
    """
    computable = []   # (airport, alert_id, tau_n_abs, f_n)
    fallback   = []   # (airport, alert_id, tau_last) → u* = 30 min baseline

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
                fallback.append((airport, alert_id, traj_abs[-1]))
                continue

            for n in range(1, n_total):
                tau_n_abs = traj_abs[n]
                history   = traj[:n + 1]
                try:
                    f_n = _feature_vector(
                        history, traj_data.iloc[:n + 1], mu, alpha, beta)
                    computable.append((airport, alert_id, tau_n_abs, f_n))
                except Exception as e:
                    print(f"  [WARN] {airport} id={alert_id} n={n}: {e}")
                    fallback.append((airport, alert_id, tau_n_abs))

    rows = []

    if computable:
        X_all   = np.array([c[3] for c in computable])
        print(f"  Calcul u* : {len(X_all)} points × {len(epsilons)} ε ...")
        u_stars = predict_ustar_batch(models, X_all, epsilons)   # (N, len(ε))

        for i, (airport, alert_id, tau_n_abs, _) in enumerate(computable):
            pred_date = _to_timestamp(tau_n_abs)
            for j, eps in enumerate(epsilons):
                rows.append({
                    'airport'                  : airport,
                    'airport_alert_id'         : alert_id,
                    'prediction_date'          : pred_date,
                    'predicted_date_end_alert' : _to_timestamp(tau_n_abs + u_stars[i, j]),
                    'confidence'               : round(1.0 - eps, 4),
                })

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

    # Match airport_alert_id dtype to the raw eval CSV to avoid KeyError
    raw_dtype = pd.read_csv(TEST_PATH, nrows=1)['airport_alert_id'].dtype
    df_pred['airport_alert_id'] = df_pred['airport_alert_id'].astype(raw_dtype)

    # Drop unlabelled segments (id=-1) — no ground truth in eval dataset
    df_pred = df_pred[df_pred['airport_alert_id'] != -1.0]
    print(f"Après suppression id=-1 : {len(df_pred):,} lignes, "
        f"{df_pred['airport_alert_id'].nunique()} alertes")
        
    df_pred.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Sauvegardé dans {OUTPUT_PATH}")
    print("\nAperçu :")
    print(df_pred.head(9).to_string(index=False))