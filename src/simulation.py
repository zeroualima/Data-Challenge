import numpy as np
import matplotlib.pyplot as plt

### Simulation de validation (algorithme de thinning d'Ogata)

DELTA = 30 * 60

def simulate_alert(mu, alpha, beta, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    events = [0.0]
    t = 0.0
    R = 0.0  # variable récursive : R = Σ exp(-β(t - τᵢ))

    while True:
        lam_bar = mu + alpha * R + alpha  # borne supérieure (juste après un event)
        dt = rng.exponential(1.0 / lam_bar)
        t_next = t + dt

        if dt >= DELTA:
            break

        # Mise à jour récursive de R à t_next (avant le potentiel nouvel event)
        R_next = np.exp(-beta * dt) * (1.0 + R)
        lam_next = mu + alpha * R_next

        if rng.uniform() <= lam_next / lam_bar:
            events.append(t_next)
            R = R_next  # on inclut le nouvel event dans R
        else:
            R = R_next  # pas d'event mais R se met à jour quand même

        t = t_next

    return np.array(events)


def simulate_hawkes(mu, alpha, beta, T_max, history=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    events = list(history) if history else [0.0]
    t = events[-1]

    # Initialiser R depuis l'historique
    R = sum(np.exp(-beta * (t - s)) for s in events[:-1])

    while t < T_max:
        lam_bar = mu + alpha * (R + 1) + 1e-10
        dt = rng.exponential(1.0 / lam_bar)
        t_next = t + dt

        if t_next > T_max:
            break

        R_next = np.exp(-beta * dt) * (1.0 + R)
        lam_next = mu + alpha * R_next

        if rng.uniform() <= lam_next / lam_bar:
            events.append(t_next)
            R = R_next
        else:
            R = R_next

        t = t_next

    return [e for e in events if not history or e > history[-1]]

def simulation_validation(params: dict, trajectories: list, airport: str,
                           n_sim: int = 500):
    mu, alpha, beta = params['mu'], params['alpha'], params['beta']
    rng = np.random.default_rng(42)

    # Propriétés observées
    obs_durations  = [(t[-1] + DELTA) / 60 for t in trajectories]
    obs_counts     = [len(t) for t in trajectories]
    obs_inter      = np.concatenate([np.diff(t) / 60 for t in trajectories])

    # Propriétés simulées
    sim_durations, sim_counts, sim_inter = [], [], []
    for i in range(n_sim):
        traj = simulate_alert(mu, alpha, beta, rng)
        sim_durations.append((traj[-1] + DELTA) / 60)
        sim_counts.append(len(traj))
        if len(traj) >= 2:
            sim_inter.extend(np.diff(traj) / 60)
        print(i) ##

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Validation par simulation — {airport}  '
                 f'(μ={mu:.4f}, α={alpha:.4f}, β={beta:.4f})')

    for ax, obs, sim, title, xlabel in zip(
        axes,
        [obs_durations, obs_counts, obs_inter],
        [sim_durations, sim_counts, sim_inter],
        ['Durées d\'alertes', 'Impacts par alerte', 'Inter-arrivées'],
        ['min', 'n', 'min']
    ):
        ax.hist(obs, bins=30, density=True, alpha=0.55,
                color='steelblue', label='Observé')
        ax.hist(sim, bins=30, density=True, alpha=0.55,
                color='coral',    label='Simulé')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'figures/{airport}_simulation.png', dpi=120, bbox_inches='tight')
    plt.close()