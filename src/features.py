import numpy as np
import pandas as pd

DELTA = 30


def _temporal_features(history: np.ndarray, mu, alpha, beta) -> dict:
    """Features temporelles et Hawkes depuis la séquence d'impacts."""
    inter    = np.diff(history)
    tau_n    = history[-1]
    n        = len(history)

    R          = sum(np.exp(-beta * (tau_n - s)) for s in history)
    lambda_now = mu + alpha * R

    ia_slope = (np.polyfit(range(min(5, len(inter))),
                inter[-min(5, len(inter)):], 1)[0]
                if len(inter) >= 3 else 0.0)

    return {
        'n_impacts'     : n,
        'duree_ecoulee' : tau_n,
        'ia_last'       : inter[-1],
        'ia_mean3'      : inter[-3:].mean(),
        'ia_mean5'      : inter[-min(5, len(inter)):].mean(),
        'ia_min'        : inter.min(),
        'ia_max'        : inter.max(),
        'ia_std'        : inter.std() if len(inter) >= 2 else 0.0,
        'ia_slope'      : ia_slope,
        'lambda_hawkes' : lambda_now,
    }


def _inner_features(recent: pd.DataFrame) -> dict:
    """Features depuis les impacts ≤ 20 km."""
    if len(recent) == 0:
        return {
            'amp_last': 0, 'amp_mean': 0, 'prop_cloud': 0.5,
            'dist_last': 20, 'dist_mean': 20, 'dist_trend': 0,
        }

    lons  = recent['lon'].values
    lats  = recent['lat'].values
    dists = recent['dist'].values

    dist_trend = (np.polyfit(range(min(10, len(dists))),
                  dists[-min(10, len(dists)):], 1)[0]
                  if len(dists) >= 3 else 0.0)

    return {
        'amp_last'   : abs(recent['amplitude'].iloc[-1]),
        'amp_mean'   : recent['amplitude'].abs().mean(),
        'prop_cloud' : recent['icloud'].mean(),
        'dist_last'  : dists[-1],
        'dist_mean'  : dists.mean(),
        'dist_trend' : dist_trend,   # + = s'éloigne, - = se rapproche
    }


def _outer_features(recent_outer: pd.DataFrame, tau_n_abs: float) -> dict:
    """
    Features depuis les impacts dans la couronne [20, 50] km.
    Fenêtre glissante : 30 dernières minutes avant tau_n.
    """
    empty = {
        'outer_count'       : 0,
        'outer_dist_mean'   : 50.0,
        'outer_dist_min'    : 50.0,
        'outer_bary_lon'    : 0.0,
        'outer_bary_lat'    : 0.0,
        'outer_bary_dist'   : 50.0,
        'outer_quad_N'      : 0.0,   # proportion d'impacts au Nord
        'outer_quad_S'      : 0.0,
        'outer_quad_E'      : 0.0,
        'outer_quad_W'      : 0.0,
        'outer_speed_kmpm'  : 0.0,   # vitesse déplacement barycentre
        'outer_bearing_sin' : 0.0,
        'outer_bearing_cos' : 1.0,
        'outer_density_trend': 0.0,  # densité croissante = orage qui arrive
    }

    if len(recent_outer) == 0:
        return empty

    # Fenêtre : 30 min glissantes avant tau_n_abs
    win = recent_outer[
        (recent_outer['t_min'] >= tau_n_abs - 30) &
        (recent_outer['t_min'] <= tau_n_abs)
    ]
    if len(win) == 0:
        return empty

    lons  = win['lon'].values
    lats  = win['lat'].values
    dists = win['dist'].values
    times = win['t_min'].values

    # Barycentre
    bary_lon  = lons.mean()
    bary_lat  = lats.mean()
    bary_dist = dists.mean()

    # Quadrants par rapport à l'aéroport
    # (approximation : lon > bary_aeroport = Est, lat > = Nord)
    # On utilise le signe par rapport à la médiane lon/lat de la fenêtre
    lon_med = np.median(lons)
    lat_med = np.median(lats)
    quad_N  = (lats > lat_med).mean()
    quad_S  = (lats <= lat_med).mean()
    quad_E  = (lons > lon_med).mean()
    quad_W  = (lons <= lon_med).mean()

    # Vitesse et direction du barycentre entre première et dernière moitié
    mid = len(win) // 2
    if mid >= 1:
        bary_lon_first = lons[:mid].mean()
        bary_lat_first = lats[:mid].mean()
        bary_lon_last  = lons[mid:].mean()
        bary_lat_last  = lats[mid:].mean()
        dt_win         = times[-1] - times[0]

        if dt_win > 0:
            dlat_km = (bary_lat_last - bary_lat_first) * 111.0
            dlon_km = ((bary_lon_last - bary_lon_first)
                       * 111.0 * np.cos(np.radians(bary_lat_last)))
            dist_moved    = np.sqrt(dlat_km**2 + dlon_km**2)
            speed_kmpm    = dist_moved / dt_win
            bearing       = np.degrees(np.arctan2(dlon_km, dlat_km))
            bearing_sin   = np.sin(np.radians(bearing))
            bearing_cos   = np.cos(np.radians(bearing))
        else:
            speed_kmpm = bearing_sin = 0.0
            bearing_cos = 1.0
    else:
        speed_kmpm = bearing_sin = 0.0
        bearing_cos = 1.0

    # Tendance de densité : plus d'impacts dans la 2e moitié = orage arrive
    if mid >= 1 and len(win) - mid > 0:
        density_trend = (len(win) - mid) / max(mid, 1) - 1.0
    else:
        density_trend = 0.0

    return {
        'outer_count'        : len(win),
        'outer_dist_mean'    : bary_dist,
        'outer_dist_min'     : dists.min(),
        'outer_bary_lon'     : bary_lon,
        'outer_bary_lat'     : bary_lat,
        'outer_bary_dist'    : bary_dist,
        'outer_quad_N'       : quad_N,
        'outer_quad_S'       : quad_S,
        'outer_quad_E'       : quad_E,
        'outer_quad_W'       : quad_W,
        'outer_speed_kmpm'   : min(speed_kmpm, 10.0),  # cap à 10 km/min
        'outer_bearing_sin'  : bearing_sin,
        'outer_bearing_cos'  : bearing_cos,
        'outer_density_trend': density_trend,
    }


def build_features(alerts, params_hawkes, df_inner,
                   alerts_abs, alerts_data, alerts_outer) -> pd.DataFrame:
    rows = []

    for airport, trajectories in alerts.items():
        p = params_hawkes[airport]
        mu, alpha, beta = p['mu'], p['alpha'], p['beta']

        for k, (traj, traj_data, traj_abs, outer_df) in enumerate(zip(
            trajectories,
            alerts_data[airport],
            alerts_abs[airport],
            alerts_outer[airport],
        )):
            T_end   = traj[-1] + DELTA
            n_total = len(traj)

            for n in range(3, n_total):
                tau_n     = traj[n]
                tau_n_abs = traj_abs[n]
                history   = traj[:n+1]

                f_temp  = _temporal_features(history, mu, alpha, beta)
                f_inner = _inner_features(traj_data.iloc[:n+1])
                f_outer = _outer_features(outer_df, tau_n_abs)

                # Temps jusqu'au prochain impact — ou DELTA si dernier impact
                if n < n_total - 1:
                    y = traj[n+1] - tau_n   # inter-arrivée suivante
                else:
                    y = DELTA                # tau_n est le dernier impact → 30 min


                rows.append({
                    **f_temp,
                    **f_inner,
                    **f_outer,
                    'airport'   : airport,
                    'y_minutes': y,
                    'is_last'  : int(n == n_total - 1),  # label utile
                })

    df = pd.DataFrame(rows)
    print(f"Dataset : {len(df)} lignes, {len(df.columns)-2} features")
    print(f"Cible y : min={df['y_minutes'].min():.1f}  "
          f"mean={df['y_minutes'].mean():.1f}  "
          f"max={df['y_minutes'].max():.1f} min")
    return df