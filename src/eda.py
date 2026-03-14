import matplotlib.pyplot as plt
from src.preprocessing import *

### Analyse exploratoire
def eda(alerts: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for airport, trajectories in alerts.items():
        # Inter-arrivées (en minutes)
        inter = [np.diff(traj) / 60 for traj in trajectories]
        inter_flat = np.concatenate(inter)

        # Durées d'alertes (en minutes)
        durations = [(traj[-1] + DELTA) / 60 for traj in trajectories]

        # Nombre d'impacts par alerte
        counts = [len(traj) for traj in trajectories]

        axes[0].hist(inter_flat, bins=50, alpha=0.5, density=True, label=airport)
        axes[1].hist(durations,  bins=30, alpha=0.5, density=True, label=airport)
        axes[2].hist(counts,     bins=30, alpha=0.5, density=True, label=airport)

    axes[0].set_title('Inter-arrivées (min)')
    axes[1].set_title('Durées d\'alertes (min)')
    axes[2].set_title('Impacts par alerte')
    for ax in axes:
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f'figures/exploration.png', dpi=120, bbox_inches='tight')
    plt.close()