import numpy as np
import matplotlib.pyplot as plt

### Simulation de validation (algorithme de thinning d'Ogata)

DELTA = 30 # 30 min

def simulate_alert(mu, alpha, beta, rng=None, max_duration=100*60):
    if rng is None:
        rng = np.random.default_rng()

    events = [0.0]
    t = 0.0
    # R = Σ exp(-β*(t - τᵢ)) pour tous les événements passés
    # Après l'événement initial en t=0 : R = exp(-β*0) = 1
    R = 1.0

    while True:
        # Borne supérieure : intensité juste après le dernier événement
        # λ* est décroissante entre événements donc c'est un upper bound valide
        lam_bar = mu + alpha * R

        dt = rng.exponential(1.0 / lam_bar)
        t_next = t + dt

        if dt >= DELTA:
            break

        if t_next > max_duration:
            print("ooups")
            break
        
        # Décroissance de R vers t_next — PAS de +1 car aucun événement encore
        R_decayed = np.exp(-beta * dt) * R
        lam_next = mu + alpha * R_decayed

        if rng.uniform() <= lam_next / lam_bar:
            # Événement accepté : on ajoute sa contribution (+1) à R
            events.append(t_next)
            R = R_decayed + 1
        else:
            # Événement rejeté : R décroît seulement, pas de +1
            R = R_decayed

        t = t_next
    # print("nice")
    return np.array(events)


def simulate_hawkes(mu, alpha, beta, T_max, history=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    events = list(history) if history else [0.0]
    t = events[-1]

    # Initialiser R depuis l'historique
    R = sum(np.exp(-beta * (t - s)) for s in events[:-1])

    while t < T_max:
        lam_bar = mu + alpha * R + 1e-10
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

def simulation_validation(params: dict, trajectories: list, airport: str, n_sim: int = 500):
    print(f"################## Simulation for {airport} ##################")
    mu, alpha, beta = params['mu'], params['alpha'], params['beta']
    rng = np.random.default_rng(42)

    # Propriétés observées
    obs_durations  = [t[-1] + DELTA for t in trajectories]
    obs_counts     = [len(t) for t in trajectories]
    obs_inter      = np.concatenate([np.diff(t) for t in trajectories])
    # Propriétés simulées
    sim_durations, sim_counts, sim_inter = [], [], []
    for i in range(n_sim):
        traj = simulate_alert(mu, alpha, beta, rng)
        sim_durations.append(traj[-1] + DELTA)
        sim_counts.append(len(traj))
        if len(traj) >= 2:
            sim_inter.extend(np.diff(traj).tolist())
    print(f"  Durée moyenne OBSERVÉE : {np.mean(obs_durations):.1f} min")
    print(f"  Durée moyenne SIMULÉE  : {np.mean(sim_durations):.1f} min")
    print(f"  Impacts moyens OBSERVÉS : {np.mean(obs_counts):.1f}")
    print(f"  Impacts moyens SIMULÉS  : {np.mean(sim_counts):.1f}")

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