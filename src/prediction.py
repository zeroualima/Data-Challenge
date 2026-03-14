
from src.preprocessing import *
import matplotlib.pyplot as plt

### Calcul de u∗ en temps réel
def estimate_survival_and_ustar(params, history, epsilon=0.10, n_sim=3000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    mu, alpha, beta = params['mu'], params['alpha'], params['beta']

    # Initialiser R depuis l'historique observé
    def simulate_end(history):
        t = history[-1]
        R = sum(np.exp(-beta * (t - s)) for s in history)

        while True:
            lam_bar = mu + alpha * R + 1e-10
            dt = rng.exponential(1.0 / lam_bar)
            if dt >= DELTA:
                return dt
            t_next = t + dt
            R_next = np.exp(-beta * dt) * (1.0 + R)
            lam_next = mu + alpha * R_next
            if rng.uniform() <= lam_next / lam_bar:
                R = R_next
            else:
                R = R_next
            t = t_next

    end_delays = np.array([simulate_end(history) for _ in range(n_sim)])

    # Courbe de survie P(T > τₙ + u | H_n)
    u_grid = np.linspace(0, DELTA, 300)
    survival = np.array([np.mean(end_delays > u) for u in u_grid])

    # u* = premier u tel que P < ε
    idx = np.argmax(survival < epsilon)
    u_star = u_grid[idx] if survival[idx] < epsilon else DELTA

    print(f"u* = {u_star/60:.1f} min  (sous risque ε={epsilon:.0%})")
    print(f"Gain vs méthode fixe : {(DELTA - u_star)/60:.1f} min")

    # Visualisation
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(u_grid / 60, survival, 'steelblue', lw=2,
            label='P(alerte active | Hₙ)')
    ax.axhline(epsilon, color='red', ls='--', label=f'ε = {epsilon:.0%}')
    ax.axvline(u_star / 60, color='coral', ls='--',
               label=f'u* = {u_star/60:.1f} min')
    ax.fill_between(u_grid / 60, survival, epsilon,
                    where=survival < epsilon, alpha=0.15, color='green',
                    label='Zone de levée d\'alerte')
    ax.set_xlabel('u (min depuis dernier impact)')
    ax.set_ylabel('P(T > τₙ + u | Hₙ)')
    ax.set_title('Courbe de survie conditionnelle')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'figures/Ajaccio_ustar.png', dpi=120, bbox_inches='tight')
    plt.close()

    return {'u_star': u_star, 'u_grid': u_grid, 'survival': survival}