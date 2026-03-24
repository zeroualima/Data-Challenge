import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

FEATURES = [
    'n_impacts', 'duree_ecoulee',
    'ia_last', 'ia_mean3', 'ia_mean5', 'ia_min', 'ia_max', 'ia_slope',
    'lambda_hawkes',
    'amp_last', 'amp_mean', 'prop_cloud',
]

QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

def train_quantile_models(df: pd.DataFrame) -> dict:
    """
    Entraîne un modèle LightGBM par quantile.
    Retourne un dict {q: booster}.
    """
    np.random.seed(42)
    all_groups   = df.groupby(['airport', 'duree_ecoulee']).ngroup()
    unique_groups = all_groups.unique()
    np.random.shuffle(unique_groups)

    n_train      = int(0.8 * len(unique_groups))
    train_groups = set(unique_groups[:n_train])

    mask_train = all_groups.isin(train_groups)
    df_train   = df[mask_train]
    df_test    = df[~mask_train]

    X_train = df_train[FEATURES].values
    y_train = df_train['y_minutes'].values
    X_test  = df_test[FEATURES].values
    y_test  = df_test['y_minutes'].values

    print(f"Train : {len(df_train)}  |  Test : {len(df_test)}")

    base_params = {
        'metric'            : 'quantile',
        'num_leaves'        : 31,
        'learning_rate'     : 0.05,
        'n_estimators'      : 1000,
        'min_child_samples' : 10,
        'subsample'         : 0.8,
        'colsample_bytree'  : 0.8,
        'reg_alpha'         : 0.1,
        'reg_lambda'        : 0.1,
        'verbose'           : -1,
    }

    models = {}
    for q in QUANTILES:
        params = {**base_params, 'objective': 'quantile', 'alpha': q}
        train_data = lgb.Dataset(X_train, label=y_train,
                                 feature_name=FEATURES)
        val_data   = lgb.Dataset(X_test, label=y_test,
                                 feature_name=FEATURES,
                                 reference=train_data)
        model = lgb.train(
            params, train_data,
            valid_sets=[val_data],
            valid_names=['test'],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(500),
            ],
        )
        models[q] = model
        pred = model.predict(X_test)
        coverage = np.mean(y_test <= pred)
        print(f"  q={q:.2f}  couverture réelle={coverage:.3f}  "
              f"(idéal={q:.2f})")

    # Sauvegarder le modèle médian pour feature importance
    fig, ax = plt.subplots(figsize=(8, 6))
    lgb.plot_importance(models[0.50], ax=ax, max_num_features=12,
                        importance_type='gain',
                        title='Feature importance — modèle médian')
    plt.tight_layout()
    plt.savefig('figures/lgbm_quantile_importance.png', dpi=120)
    plt.close()

    # Évaluation MAE du modèle médian
    pred_median = models[0.50].predict(X_test)
    mae = np.mean(np.abs(y_test - pred_median))
    print(f"\nMAE médiane : {mae:.2f} min")

    return models, X_test, y_test
def predict_ustar(models, f_n, epsilon=0.10):
    """
    S(u) = P(prochain impact > u | f_n)
         = P(y > u | f_n)
    u* = inf{u : S(u) < epsilon}
    Gain = 30 - u*  (en minutes gagnées sur la méthode fixe)
    """
    f_n = np.array(f_n).reshape(1, -1)

    q_vals  = list(models.keys())
    q_preds = [models[q].predict(f_n)[0] for q in q_vals]

    # Grille dans [0, 30]
    u_grid   = np.linspace(0, 30, 300)
    survival = np.array([
        1.0 - np.interp(u, q_preds, q_vals, left=0.0, right=1.0)
        for u in u_grid
    ])

    below  = np.where(survival < epsilon)[0]
    u_star = u_grid[below[0]] if len(below) > 0 else 30.0

def predict_ustar(models, f_n, epsilon=0.10):
    f_n = np.array(f_n).reshape(1, -1)

    q_vals  = np.array(list(models.keys()))
    q_preds = np.array([models[q].predict(f_n)[0] for q in q_vals])

    # Trier pour garantir la monotonie de F
    sort_idx = np.argsort(q_preds)
    q_preds_sorted = q_preds[sort_idx]
    q_vals_sorted  = q_vals[sort_idx]

    # Ajouter les points limites :
    # u=0  → F(0) = 0  → S(0) = 1
    # u=30 → F(30) = 1 → S(30) = 0  (y <= 30 par construction)
    q_preds_full = np.concatenate([[0.0],  q_preds_sorted, [30.0]])
    q_vals_full  = np.concatenate([[0.0],  q_vals_sorted,  [1.0]])

    # S(u) = 1 - F(u)
    u_grid   = np.linspace(0, 30, 300)
    F_u      = np.interp(u_grid, q_preds_full, q_vals_full)
    survival = 1.0 - F_u

    # u* = premier u tel que S(u) < epsilon
    below  = np.where(survival < epsilon)[0]
    u_star = u_grid[below[0]] if len(below) > 0 else 30.0
    # gain   = 30.0 - u_star

    # print(f"u* = {u_star:.1f} min  →  Gain = {gain:.1f} min")
    return float(u_star), u_grid, survival

def plot_survival(u_grid, survival, u_star, epsilon, title=''):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(u_grid, survival, 'steelblue', lw=2,
            label='P(alerte active | Hₙ)')
    ax.axhline(epsilon, color='red', ls='--',
               label=f'ε = {epsilon:.0%}')
    ax.axvline(u_star, color='coral', ls='--',
               label=f'u* = {u_star:.1f} min')
    ax.fill_between(u_grid, survival, epsilon,
                    where=survival < epsilon,
                    alpha=0.15, color='green',
                    label='Zone de levée')
    ax.set_xlabel('u (min depuis début alerte)')
    ax.set_ylabel('P(T > u | Hₙ)')
    ax.set_title(title or 'Courbe de survie conditionnelle')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/survival_curve.png', dpi=120)
    plt.close()
    print(f"u* = {u_star:.1f} min  (ε={epsilon:.0%})")
    gain = 30.0 - u_star  # et non DELTA - u_star
    print(f"u* = {u_star:.1f} min  →  Gain = {gain:.1f} min")