from scipy.optimize import minimize
import numpy as np
from src.preprocessing import *

### Estimation MLE
def _neg_log_likelihood(params, trajectories):
    """Log-vraisemblance d'Ogata, sommée sur toutes les trajectoires."""
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0:
        return np.inf

    total = 0.0
    for traj in trajectories:
        n = len(traj)
        T = traj[-1] + DELTA  # fin d'observation = dernier impact + 30 min

        # Terme de compensation : -∫λ*(s)ds
        comp = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - traj)))

        # Terme de log-intensité (calcul récursif)
        log_sum = 0.0
        R = 0.0
        for i in range(n):
            if i > 0:
                R = np.exp(-beta * (traj[i] - traj[i-1])) * (1.0 + R)
            lam = mu + alpha * R
            if lam <= 0:
                return np.inf
            log_sum += np.log(lam)

        total += -(log_sum - comp)
    return total


def fit_hawkes(trajectories: list) -> dict:
    best = None
    # Grille d'initialisations pour éviter les minima locaux
    for mu0 in [1e-3, 5e-3, 1e-2]:
        for alpha0 in [0.1, 0.5, 0.9]:
            for beta0 in [0.02, 0.1, 0.5]:
                if alpha0 >= beta0:
                    continue
                res = minimize(
                    _neg_log_likelihood,
                    x0=[mu0, alpha0, beta0],
                    args=(trajectories,),
                    method='L-BFGS-B',
                    bounds=[(1e-6, None)] * 3
                )
                if best is None or res.fun < best.fun:
                    best = res

    mu, alpha, beta = best.x
    print(f"  μ={mu:.5f}  α={alpha:.5f}  β={beta:.5f}  "
          f"(ratio α/β={alpha/beta:.3f})")
    return {'mu': mu, 'alpha': alpha, 'beta': beta, 'result': best}


def fit_all_airports(alerts: dict) -> dict:
    params = {}
    for airport, trajectories in alerts.items():
        print(f"\n--- {airport} ---")
        params[airport] = fit_hawkes(trajectories)
    return params