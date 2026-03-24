from pathlib import Path
from src.preprocessing import load_data
from src.hawkes        import fit_all_airports
from src.validation    import goodness_of_fit
from src.simulation    import simulation_validation
from src.features      import build_features
import numpy as np

Path('figures').mkdir(exist_ok=True)

# Hawkes (déjà fait)
df, df_inner, df_outer, alerts, alerts_abs, alerts_data, alerts_outer = load_data('./data_train_databattle2026/segment_alerts_all_airports_train.csv')
all_params = fit_all_airports(alerts)

# LightGBM
df_features = build_features(alerts, all_params, df_inner, alerts_abs, alerts_data, alerts_outer)




from src.lgbm_model import train_quantile_models, predict_ustar, plot_survival

models, X_test, y_test = train_quantile_models(df_features)

# # Exemple sur le premier point du test
# f_exemple = X_test[0]
# u_star, u_grid, survival = predict_ustar(models, f_exemple, epsilon=0.10)
# plot_survival(u_grid, survival, u_star, epsilon=0.10, title='Exemple de prédiction sur une alerte test')

gains = []
for i in range(len(X_test)):
    u_star, u_grid, survival = predict_ustar(models, X_test[i], epsilon=0.10)
    plot_survival(u_grid, survival, u_star, epsilon=0.10, title=f'Exemple de prédiction sur une alerte test {i}')
    gains.append(30.0 - u_star)

print(f"Gain moyen sur le test : {np.mean(gains):.1f} min")
print(f"Gain médian            : {np.median(gains):.1f} min")
print(f"% d'alertes avec gain>0: {np.mean(np.array(gains)>0)*100:.0f}%")