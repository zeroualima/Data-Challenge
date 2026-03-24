import pandas as pd
import numpy as np

DELTA     = 30
GAP_DEDUP = 10 / 60

def _segment_airport(grp, t_abs):
    """Segmente les impacts d'un aéroport en alertes."""
    # Déduplication
    keep = [0]
    for i in range(1, len(t_abs)):
        if t_abs[i] - t_abs[keep[-1]] >= GAP_DEDUP:
            keep.append(i)
    grp   = grp.iloc[keep].reset_index(drop=True)
    t_abs = t_abs[keep]

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
    return valid


def load_data(path: str):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values(['airport', 'date']).reset_index(drop=True)

    # Colonne t_min sur tout le dataset
    df['t_min'] = df['date'].values.astype('int64') / 1e9 / 60

    df_inner = df[df['dist'] <= 20].copy()  # zone alerte
    df_outer = df[df['dist'] <= 50].copy()  # zone étendue

    alerts       = {}
    alerts_abs   = {}
    alerts_data  = {}   # impacts ≤ 20 km
    alerts_outer = {}   # impacts ≤ 50 km alignés par alerte

    for airport, grp in df_inner.groupby('airport'):
        grp   = grp.sort_values('date').reset_index(drop=True)
        t_abs = grp['t_min'].values

        valid = _segment_airport(grp, t_abs)

        alerts[airport]      = [v[0] for v in valid]
        alerts_abs[airport]  = [v[1] for v in valid]
        alerts_data[airport] = [v[2] for v in valid]

        # Pour chaque alerte, extraire les impacts outer correspondants
        df_out_ap = df_outer[df_outer['airport'] == airport].sort_values('t_min')
        outer_list = []
        for v in valid:
            t_start = v[1][0]   # timestamp absolu début alerte
            t_end   = v[1][-1]  # timestamp absolu fin alerte
            chunk = df_out_ap[
                (df_out_ap['t_min'] >= t_start - 0.1) &
                (df_out_ap['t_min'] <= t_end   + 0.1)
            ].reset_index(drop=True)
            outer_list.append(chunk)
        alerts_outer[airport] = outer_list

        print(f"{airport:20s} : {len(alerts[airport])} alertes "
              f"({sum(len(t) for t in alerts[airport])} impacts après dédup)")

    return df, df_inner, df_outer, alerts, alerts_abs, alerts_data, alerts_outer