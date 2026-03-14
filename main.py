from src.preprocessing import load_data
from src.eda import eda
from src.hawkes import fit_all_airports
from src.validation import goodness_of_fit
from src.simulation import simulation_validation
from src.prediction import estimate_survival_and_ustar

from pathlib import Path
Path('figures').mkdir(exist_ok=True)

if __name__ == '__main__':
    # 1. Données
    df, df_inner, alerts = load_data('./data_train_databattle2026/segment_alerts_all_airports_train.csv')

    # 2. EDA
    # eda(alerts)

    # 3. Ajustement MLE
    all_params = fit_all_airports(alerts)
    # all_params = {
    #     "Ajaccio" : {'mu': 0.00376, 'alpha': 4.23484, 'beta': 4.33481, 'result': None},
    #     "Bastia" : {'mu': 0.00363, 'alpha': 4.96615, 'beta': 5.03144, 'result': None},
    #     "Biarritz" : {'mu': 0.00450, 'alpha': 5.52368, 'beta': 5.61222, 'result': None},
    #     "Nantes" : {'mu': 0.00240, 'alpha': 5.28938, 'beta': 5.37894, 'result': None},
    #     "Pise" : {'mu': 0.00591, 'alpha': 4.68142, 'beta': 4.76229, 'result': None},
    # }

    # 4 & 5. Validation par aéroport
    for airport, trajectories in alerts.items():
        p = all_params[airport]
        goodness_of_fit(p, trajectories, airport)
        simulation_validation(p, trajectories, airport, n_sim=300)

    # 6. Exemple de calcul de u* sur une alerte en cours
    airport_test = 'Ajaccio'
    exemple_alerte = alerts[airport_test][0]  # première alerte observée
    result = estimate_survival_and_ustar(
        all_params[airport_test],
        histoire=exemple_alerte,
        epsilon=0.10
    )