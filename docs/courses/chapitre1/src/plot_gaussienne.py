import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration du style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def gaussian_pdf(x, mu, sigma2):
    """
    Calcule la densité de probabilité de la loi normale.

    Args:
        x: Valeurs où évaluer la densité
        mu: Moyenne
        sigma2: Variance

    Returns:
        Densité de probabilité
    """
    sigma = np.sqrt(sigma2)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def plot_gaussiennes_combinee():
    """
    Génère un graphique combiné avec plusieurs gaussiennes de paramètres variés.
    """
    x = np.linspace(-10, 10, 1000)

    # Différentes combinaisons de μ et σ²
    params = [
        (0, 1, 'Loi normale standard'),
        (0, 0.5, 'Faible variance'),
        (0, 2, 'Forte variance'),
        (2, 1, 'Moyenne décalée'),
    ]

    plt.figure(figsize=(12, 7))

    for mu, sigma2, label in params:
        y = gaussian_pdf(x, mu, sigma2)
        plt.plot(x, y, linewidth=2.5,
                label=rf'{label}: $\mu={mu}, \sigma^2={sigma2}$')

    plt.xlabel('$x$', fontsize=14)
    plt.ylabel(r'Densité de probabilité $f(x; \mu, \sigma^2)$', fontsize=14)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.tight_layout()

    # Sauvegarder
    output_path = Path(__file__).parent.parent / 'img' / 'gaussienne_combinee.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Image sauvegardée : {output_path}")
    plt.close()


def plot_mse_vs_n():
    """
    Génère un graphique montrant l'évolution de l'EQM en fonction de n.
    Pour la moyenne empirique, MSE = σ²/n
    """
    # Paramètres
    sigma2 = 4  # Variance de la population (exemple)
    n_values = np.arange(1, 201)  # Tailles d'échantillon de 1 à 200

    # Calcul de l'EQM pour chaque n
    mse_values = sigma2 / n_values

    plt.figure(figsize=(10, 6))

    # Graphique principal
    plt.plot(n_values, mse_values, linewidth=2.5, color='#2563eb', label=r'$\text{MSE}(\hat{\mu}_n) = \frac{\sigma^2}{n}$')

    # Points de référence
    reference_n = [5, 10, 20, 50, 100]
    reference_mse = [sigma2 / n for n in reference_n]
    plt.scatter(reference_n, reference_mse, s=80, color='#dc2626', zorder=5, label='Points de référence')

    # Annotations pour quelques points
    for n, mse in zip([10, 50, 100], [sigma2/10, sigma2/50, sigma2/100]):
        plt.annotate(f'n={n}\nMSE={mse:.2f}',
                    xy=(n, mse),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    plt.xlabel('Taille d\'échantillon $n$', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Sauvegarder
    output_path = Path(__file__).parent.parent / 'img' / 'mse_vs_n.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Image sauvegardée : {output_path}")
    plt.close()


def main():
    """Génère toutes les visualisations."""
    print("Génération des graphiques de la loi normale...\n")

    # Créer le dossier img s'il n'existe pas
    img_dir = Path(__file__).parent.parent / 'img'
    img_dir.mkdir(exist_ok=True)

    # Générer les graphiques
    plot_gaussiennes_combinee()
    plot_mse_vs_n()


if __name__ == "__main__":
    main()
