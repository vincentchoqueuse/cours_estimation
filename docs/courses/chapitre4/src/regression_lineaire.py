"""
Chapitre 3 : Régression Linéaire
Visualisations et exemples numériques
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

# Configuration globale
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def ols_estimator(A, x):
    """
    Calcule l'estimateur des moindres carrés ordinaires.

    Args:
        A: Matrice de design (m, p)
        x: Vecteur des observations (m,)

    Returns:
        s_hat: Estimateur OLS (p,)
        cov_s: Matrice de covariance estimée
        sigma2_hat: Estimation de la variance des erreurs
        x_hat: Valeurs ajustées
        residuals: Résidus
    """
    m, p = A.shape

    # Estimateur OLS
    AtA = A.T @ A
    s_hat = np.linalg.solve(AtA, A.T @ x)

    # Valeurs ajustées
    x_hat = A @ s_hat

    # Résidus
    residuals = x - x_hat

    # Estimation de sigma²
    sigma2_hat = np.sum(residuals**2) / (m - p)

    # Matrice de covariance de s_hat
    cov_s = sigma2_hat * np.linalg.inv(AtA)

    return s_hat, cov_s, sigma2_hat, x_hat, residuals


def compute_r2(x, x_hat):
    """
    Calcule le coefficient de détermination R².

    Args:
        x: Observations
        x_hat: Valeurs ajustées

    Returns:
        r2: Coefficient R²
        r2_adj: R² ajusté
    """
    m = len(x)
    p = 2  # À adapter selon le nombre de paramètres

    x_mean = np.mean(x)
    SST = np.sum((x - x_mean)**2)
    SSR = np.sum((x - x_hat)**2)
    SSE = np.sum((x_hat - x_mean)**2)

    r2 = SSE / SST
    r2_adj = 1 - (1 - r2) * (m - 1) / (m - p)

    return r2, r2_adj


def plot_geometric_interpretation():
    """
    Figure 1: Interprétation géométrique de l'OLS comme projection orthogonale.
    """
    # Création d'un exemple simple en 2D
    np.random.seed(42)

    # Générer des données
    m = 30
    t = np.linspace(0, 5, m)
    A = np.column_stack([np.ones(m), t])
    s_true = np.array([1.0, 0.8])

    x_clean = A @ s_true
    x = x_clean + np.random.normal(0, 0.3, m)

    # Estimation OLS
    s_hat, _, _, x_hat, residuals = ols_estimator(A, x)

    # Visualisation
    fig, ax = plt.subplots(figsize=(12, 7))

    # Points observés
    ax.scatter(t, x, color='blue', s=80, alpha=0.7, label='Observations $x_k$', zorder=3)

    # Droite de régression (valeurs ajustées)
    ax.plot(t, x_hat, 'r-', linewidth=2.5, label=r'Droite ajustée $\widehat{x} = A\widehat{s}$', zorder=2)

    # Résidus (projections orthogonales)
    for k in range(m):
        ax.plot([t[k], t[k]], [x[k], x_hat[k]], 'k--', linewidth=0.8, alpha=0.5, zorder=1)

    # Annoter quelques résidus
    for k in [5, 15, 25]:
        ax.annotate('', xy=(t[k], x_hat[k]), xytext=(t[k], x[k]),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
        ax.text(t[k] + 0.1, (x[k] + x_hat[k])/2, rf'$\widehat{{n}}_{{{k}}}$',
               fontsize=9, color='green')

    ax.set_xlabel(r'Variable explicative $t$', fontsize=12)
    ax.set_ylabel(r'Variable dépendante $x$', fontsize=12)
    ax.set_title('Interprétation géométrique : projection orthogonale sur l\'espace des colonnes de $A$', fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarde
    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'projection_geometrique.png', dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée: {output_dir / 'projection_geometrique.png'}")


def plot_r2_interpretation():
    """
    Figure 2: Décomposition de la variance et interprétation du R².
    """
    # Génération de données
    np.random.seed(42)
    m = 50
    t = np.linspace(0, 10, m)
    A = np.column_stack([np.ones(m), t])
    s_true = np.array([2.0, 1.5])

    sigma2 = 2.0
    x = A @ s_true + np.random.normal(0, np.sqrt(sigma2), m)

    # Estimation
    s_hat, _, _, x_hat, residuals = ols_estimator(A, x)

    # Calcul des sommes de carrés
    x_mean = np.mean(x)
    SST = np.sum((x - x_mean)**2)
    SSE = np.sum((x_hat - x_mean)**2)
    SSR = np.sum((x - x_hat)**2)
    r2 = SSE / SST

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Graphique 1: Décomposition visuelle
    ax = axes[0]
    ax.scatter(t, x, color='blue', s=60, alpha=0.7, label='Observations', zorder=3)
    ax.plot(t, x_hat, 'r-', linewidth=2.5, label='Valeurs ajustées', zorder=2)
    ax.axhline(x_mean, color='green', linestyle='--', linewidth=2, label=r'Moyenne $\bar{x}$', zorder=1)

    # Annoter la décomposition pour quelques points
    idx = [10, 25, 40]
    for k in idx:
        # SST: de la moyenne à l'observation
        ax.plot([t[k]-0.2, t[k]-0.2], [x_mean, x[k]], 'g-', linewidth=2, alpha=0.6)
        # SSE: de la moyenne à l'ajustée
        ax.plot([t[k], t[k]], [x_mean, x_hat[k]], 'orange', linewidth=2, alpha=0.6)
        # SSR: de l'ajustée à l'observation
        ax.plot([t[k]+0.2, t[k]+0.2], [x_hat[k], x[k]], 'purple', linewidth=2, alpha=0.6)

    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(r'$x$', fontsize=12)
    ax.set_title('Décomposition de la variance', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Graphique 2: Diagramme à barres
    ax = axes[1]
    components = ['SST\n(Totale)', 'SSE\n(Expliquée)', 'SSR\n(Résiduelle)']
    values = [SST, SSE, SSR]
    colors = ['green', 'orange', 'purple']

    bars = ax.bar(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Annoter les valeurs
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Annoter la relation SST = SSE + SSR
    ax.text(1, SST*0.5, rf'$R^2 = \frac{{SSE}}{{SST}} = {r2:.3f}$',
           fontsize=13, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_ylabel('Somme des carrés', fontsize=12)
    ax.set_title(rf'Décomposition: SST = SSE + SSR ($R^2={r2:.3f}$)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Sauvegarde
    output_dir = Path(__file__).parent.parent / 'img'
    plt.savefig(output_dir / 'r2_interpretation.png', dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée: {output_dir / 'r2_interpretation.png'}")


def plot_diagnostic_residuals():
    """
    Figure 3: Graphiques de diagnostic pour l'analyse des résidus.
    """
    # Génération de données avec un bon modèle
    np.random.seed(42)
    m = 100
    t = np.linspace(0, 10, m)
    A = np.column_stack([np.ones(m), t, t**2])
    s_true = np.array([1.0, 0.5, -0.05])

    sigma2 = 1.0
    x = A @ s_true + np.random.normal(0, np.sqrt(sigma2), m)

    # Estimation
    s_hat, _, sigma2_hat, x_hat, residuals = ols_estimator(A, x)

    # Standardisation des résidus
    # Calcul de P (matrice de projection)
    P = A @ np.linalg.inv(A.T @ A) @ A.T
    leverage = np.diag(P)
    residuals_std = residuals / (np.sqrt(sigma2_hat) * np.sqrt(1 - leverage))

    # Visualisation
    fig = plt.figure(figsize=(15, 10))

    # Graphique 1: Résidus vs Valeurs ajustées
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(x_hat, residuals, alpha=0.6, s=40)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Valeurs ajustées', fontsize=11)
    ax1.set_ylabel('Résidus', fontsize=11)
    ax1.set_title('Résidus vs Valeurs ajustées', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Graphique 2: Q-Q plot
    ax2 = fig.add_subplot(2, 2, 2)
    stats.probplot(residuals_std, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (normalité des résidus)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Graphique 3: Scale-Location (racine des résidus standardisés)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(x_hat, np.sqrt(np.abs(residuals_std)), alpha=0.6, s=40)
    ax3.set_xlabel('Valeurs ajustées', fontsize=11)
    ax3.set_ylabel(r'$\sqrt{|\text{Résidus standardisés}|}$', fontsize=11)
    ax3.set_title('Scale-Location (homoscédasticité)', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Graphique 4: Résidus vs Leverage
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(leverage, residuals_std, alpha=0.6, s=40)
    ax4.axhline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Leverage', fontsize=11)
    ax4.set_ylabel('Résidus standardisés', fontsize=11)
    ax4.set_title('Résidus vs Leverage (points influents)', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # Identifier les points influents (Cook's distance > 0.5)
    n_std = len(residuals_std)
    cooks_d = residuals_std**2 * leverage / (len(s_hat) * (1 - leverage))
    influential = np.where(cooks_d > 0.5)[0]
    if len(influential) > 0:
        ax4.scatter(leverage[influential], residuals_std[influential],
                   color='red', s=100, marker='x', linewidths=3, label='Influents')
        ax4.legend(fontsize=10)

    plt.tight_layout()

    # Sauvegarde
    output_dir = Path(__file__).parent.parent / 'img'
    plt.savefig(output_dir / 'diagnostic_residus.png', dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée: {output_dir / 'diagnostic_residus.png'}")


def demonstrate_gauss_markov():
    """
    Démonstration du théorème de Gauss-Markov par simulation Monte Carlo.
    """
    np.random.seed(42)

    # Paramètres
    m = 50
    n_sim = 1000

    t = np.linspace(0, 5, m)
    A = np.column_stack([np.ones(m), t])
    s_true = np.array([2.0, 1.0])
    sigma2 = 1.0

    # Stockage des estimations
    s_ols = np.zeros((n_sim, 2))
    s_biased = np.zeros((n_sim, 2))  # Estimateur linéaire biaisé (exemple)

    for sim in range(n_sim):
        # Génération des données
        x = A @ s_true + np.random.normal(0, np.sqrt(sigma2), m)

        # OLS
        s_ols[sim], _, _, _, _ = ols_estimator(A, x)

        # Estimateur linéaire simple (moyenne pour s0, pente naïve pour s1)
        s_biased[sim, 0] = np.mean(x)
        s_biased[sim, 1] = (x[-1] - x[0]) / (t[-1] - t[0])

    # Calcul des variances empiriques
    var_ols = np.var(s_ols, axis=0)
    var_biased = np.var(s_biased, axis=0)

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogrammes pour s_0
    ax = axes[0]
    ax.hist(s_ols[:, 0], bins=30, alpha=0.6, label=f'OLS (var={var_ols[0]:.4f})',
           color='blue', edgecolor='black')
    ax.hist(s_biased[:, 0], bins=30, alpha=0.6, label=f'Estimateur naïf (var={var_biased[0]:.4f})',
           color='red', edgecolor='black')
    ax.axvline(s_true[0], color='green', linestyle='--', linewidth=2, label='Vraie valeur')
    ax.set_xlabel(r'$\widehat{s}_0$', fontsize=12)
    ax.set_ylabel('Fréquence', fontsize=12)
    ax.set_title(rf'Distribution de $\widehat{{s}}_0$ ({n_sim} simulations)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Histogrammes pour s_1
    ax = axes[1]
    ax.hist(s_ols[:, 1], bins=30, alpha=0.6, label=f'OLS (var={var_ols[1]:.4f})',
           color='blue', edgecolor='black')
    ax.hist(s_biased[:, 1], bins=30, alpha=0.6, label=f'Estimateur naïf (var={var_biased[1]:.4f})',
           color='red', edgecolor='black')
    ax.axvline(s_true[1], color='green', linestyle='--', linewidth=2, label='Vraie valeur')
    ax.set_xlabel(r'$\widehat{s}_1$', fontsize=12)
    ax.set_ylabel('Fréquence', fontsize=12)
    ax.set_title(rf'Distribution de $\widehat{{s}}_1$ ({n_sim} simulations)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarde
    output_dir = Path(__file__).parent.parent / 'img'
    plt.savefig(output_dir / 'gauss_markov.png', dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée: {output_dir / 'gauss_markov.png'}")

    print(f"\n=== Théorème de Gauss-Markov ===")
    print(f"Variance OLS: s_0={var_ols[0]:.6f}, s_1={var_ols[1]:.6f}")
    print(f"Variance estimateur naïf: s_0={var_biased[0]:.6f}, s_1={var_biased[1]:.6f}")
    print(f"L'OLS a une variance minimale parmi les estimateurs linéaires sans biais!")


def compare_ridge_ols():
    """
    Comparaison entre OLS et régression ridge en présence de multicolinéarité.
    """
    np.random.seed(42)

    # Création de données avec multicolinéarité
    m = 50
    t1 = np.linspace(0, 5, m)
    t2 = t1 + np.random.normal(0, 0.1, m)  # t2 fortement corrélé à t1

    A = np.column_stack([np.ones(m), t1, t2])
    s_true = np.array([1.0, 2.0, -1.0])

    sigma2 = 0.5
    x = A @ s_true + np.random.normal(0, np.sqrt(sigma2), m)

    # OLS
    s_ols, _, _, _, _ = ols_estimator(A, x)

    # Ridge pour différentes valeurs de lambda
    lambdas = np.logspace(-3, 2, 50)
    s_ridge = np.zeros((len(lambdas), 3))

    for i, lam in enumerate(lambdas):
        AtA_ridge = A.T @ A + lam * np.eye(3)
        s_ridge[i] = np.linalg.solve(AtA_ridge, A.T @ x)

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Graphique 1: Évolution des coefficients
    ax = axes[0]
    for j in range(3):
        ax.semilogx(lambdas, s_ridge[:, j], linewidth=2, label=rf'$s_{j}$')
        ax.axhline(s_ols[j], color=f'C{j}', linestyle='--', alpha=0.5)

    ax.set_xlabel(r'Paramètre de régularisation $\lambda$', fontsize=12)
    ax.set_ylabel('Valeur des coefficients', fontsize=12)
    ax.set_title('Régression Ridge: évolution des coefficients', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Graphique 2: MSE en fonction de lambda
    ax = axes[1]
    mse_ridge = np.zeros(len(lambdas))
    for i, lam in enumerate(lambdas):
        x_hat_ridge = A @ s_ridge[i]
        mse_ridge[i] = np.mean((x - x_hat_ridge)**2)

    mse_ols = np.mean((x - A @ s_ols)**2)

    ax.semilogx(lambdas, mse_ridge, linewidth=2, label='Ridge MSE')
    ax.axhline(mse_ols, color='red', linestyle='--', linewidth=2, label='OLS MSE')
    ax.set_xlabel(r'Paramètre de régularisation $\lambda$', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Compromis biais-variance', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarde
    output_dir = Path(__file__).parent.parent / 'img'
    plt.savefig(output_dir / 'ridge_ols.png', dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée: {output_dir / 'ridge_ols.png'}")

    # Calcul du VIF
    print(f"\n=== Multicolinéarité ===")
    print(f"Corrélation entre t1 et t2: {np.corrcoef(t1, t2)[0, 1]:.4f}")
    print(f"Conditionnement de A^T A: {np.linalg.cond(A.T @ A):.2e}")


if __name__ == '__main__':
    print("=== Chapitre 3 : Régression Linéaire ===\n")

    # Génération des figures
    plot_geometric_interpretation()
    plot_r2_interpretation()
    plot_diagnostic_residuals()
    demonstrate_gauss_markov()
    compare_ridge_ols()

    print("\nToutes les figures ont été générées avec succès!")
