import pandas as pd
from pathlib import Path

df = pd.read_csv('./data_train_databattle2026/segment_alerts_all_airports_train.csv')
out = Path('./data_by_airport')
out.mkdir(exist_ok=True)

for airport, grp in df.groupby('airport'):
    grp.to_csv(out / f'{airport.lower()}.csv', index=False)
    print(f"{airport:20s} : {len(grp):6d} lignes")