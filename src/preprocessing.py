import pandas as pd
import numpy as np

DELTA = 30 * 60  # 30 min en secondes
R_ALERT = 20     # rayon d'alerte en km


### Chargement et prétraitement
def load_data(path: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values(['airport', 'date']).reset_index(drop=True)

    # Seuls les impacts dans le disque de 20 km comptent pour les alertes
    df_inner = df[df['dist'] <= R_ALERT].copy()

    # Reconstruction des alertes par aéroport
    # (au cas où airport_alert_id contient des NaN)
    alerts = {}
    for airport, grp in df_inner.groupby('airport'):
        grp = grp.sort_values('date')
        timestamps = grp['date'].values.astype('int64') / 1e9  # secondes UNIX

        trajectories = []
        current = [timestamps[0]]
        for t in timestamps[1:]:
            if t - current[-1] >= DELTA:
                trajectories.append(np.array(current) - current[0])
                current = [t]
            else:
                current.append(t)
        trajectories.append(np.array(current) - current[0])

        # On écarte les alertes à 1 seul impact (pas d'inter-arrivée)
        alerts[airport] = [traj for traj in trajectories if len(traj) >= 2]
        print(f"{airport:20s} : {len(alerts[airport])} alertes")

    return df, df_inner, alerts