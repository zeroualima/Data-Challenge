from scipy.stats import kstest
from scipy.stats import probplot
import matplotlib.pyplot as plt
from src.preprocessing import *

###  Validation : théorème de rescaling temporel
def compute_residuals(params: dict, trajectories: list) -> np.ndarray:
    """
    Applique le théorème de rescaling temporel de Papangelou :
    si λ*(t) est bien spécifiée, les accroissements de la compensatrice
    Λ(τᵢ) = ∫₀^τᵢ λ*(s)ds entre impacts successifs ~ Exp(1).
    """
    mu, alpha, beta = params['mu'], params['alpha'], params['beta']
    all_residuals = []

    for traj in trajectories:
        cumulative = 0.0
        R = 0.0
        lambdas = [0.0]

        for i in range(1, len(traj)):
            dt = traj[i] - traj[i-1]
            # Intégrale analytique de λ*(s) entre traj[i-1] et traj[i]
            # ∫ (μ + α·R·e^{-β(s-tᵢ₋₁)}) ds = μ·dt + (α·R/β)·(1 - e^{-β·dt})
            delta_Lambda = mu * dt + (alpha * R / beta) * (1 - np.exp(-beta * dt))
            cumulative += delta_Lambda
            lambdas.append(cumulative)
            R = np.exp(-beta * dt) * (1.0 + R)

        residuals = np.diff(lambdas)
        all_residuals.extend(residuals[residuals > 0])

    return np.array(all_residuals)


def goodness_of_fit(params: dict, trajectories: list, airport: str): 
    print(f"################## Goodness of fit for {airport} ##################")
    residuals = compute_residuals(params, trajectories)
    stat, pvalue = kstest(residuals, 'expon', args=(0, 1))
    print(f"{airport} — KS stat={stat:.4f}  p-value={pvalue:.4f} "
          f"({'✓ non rejeté' if pvalue > 0.05 else '✗ rejeté'} à 5%)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f'Validation rescaling temporel — {airport}')

    # Histogramme vs Exp(1)
    axes[0].hist(residuals, bins=60, density=True, alpha=0.65,
                 color='steelblue', label='Résidus observés')
    x = np.linspace(0, np.percentile(residuals, 99), 300)
    axes[0].plot(x, np.exp(-x), 'r-', lw=2, label='Exp(1) théorique')
    axes[0].set_xlabel('Résidu Λ(τᵢ₊₁) − Λ(τᵢ)')
    axes[0].legend()

    # QQ-plot
    probplot(residuals, dist='expon', plot=axes[1])
    axes[1].set_title('QQ-plot vs Exp(1)')

    plt.tight_layout()
    plt.savefig(f'figures/{airport}_validation.png', dpi=120, bbox_inches='tight')
    plt.close()
    return stat, pvalue