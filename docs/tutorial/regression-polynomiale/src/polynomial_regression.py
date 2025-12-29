"""
Tutoriel : Régression polynomiale
Estimation des coefficients d'un polynôme par moindres carrés
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration globale pour les graphiques
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def create_vandermonde_matrix(t, p):
    """
    Crée la matrice de Vandermonde pour un ensemble de points t et un degré p.

    Args:
        t: Vecteur des abscisses de taille m
        p: Degré du polynôme

    Returns:
        A: Matrice de Vandermonde de taille (m, p+1)
    """
    m = len(t)
    A = np.zeros((m, p + 1))

    for j in range(p + 1):
        A[:, j] = t**j

    return A


def polynomial_regression(t, x, p):
    """
    Estime les coefficients d'un polynôme de degré p par moindres carrés.

    Args:
        t: Vecteur des abscisses (taille m)
        x: Vecteur des observations (taille m)
        p: Degré du polynôme

    Returns:
        s_hat: Vecteur des coefficients estimés (taille p+1)
        A: Matrice de Vandermonde
    """
    # Construction de la matrice de design (Vandermonde)
    A = create_vandermonde_matrix(t, p)

    # Estimateur des moindres carrés: s_hat = (A^T A)^{-1} A^T x
    s_hat = np.linalg.solve(A.T @ A, A.T @ x)

    return s_hat, A


def evaluate_polynomial(s, t):
    """
    Évalue un polynôme aux points t.

    Args:
        s: Coefficients du polynôme [s_0, s_1, ..., s_p]
        t: Points d'évaluation

    Returns:
        y: Valeurs du polynôme aux points t
    """
    y = np.zeros_like(t)
    for j, coef in enumerate(s):
        y += coef * t**j
    return y


def generate_data(m, p_true, sigma2, t_min=0, t_max=5):
    """
    Génère des données polynomiales bruitées.

    Args:
        m: Nombre d'observations
        p_true: Degré du vrai polynôme
        sigma2: Variance du bruit
        t_min, t_max: Bornes de l'intervalle

    Returns:
        t: Abscisses
        x: Observations bruitées
        s_true: Vrais coefficients
    """
    # Génération des abscisses uniformément réparties
    t = np.linspace(t_min, t_max, m)

    # Coefficients arbitraires pour le polynôme de degré p_true
    # Exemple: pour p=3 -> s = [1, 2, -0.5, 0.1]
    s_true = np.array([1.0, 2.0, -0.5, 0.1][:p_true + 1])

    # Signal sans bruit
    x_clean = evaluate_polynomial(s_true, t)

    # Ajout de bruit gaussien
    n = np.random.normal(0, np.sqrt(sigma2), m)
    x = x_clean + n

    return t, x, s_true, x_clean


def plot_regression_example():
    """
    Génère la Figure 1: Ajustement polynomial de degré 3.
    """
    # Paramètres
    m = 20              # Nombre d'observations
    p_true = 3          # Degré du vrai polynôme
    sigma2 = 0.5**2     # Variance du bruit

    # Génération des données
    np.random.seed(42)
    t, x, s_true, x_clean = generate_data(m, p_true, sigma2)

    # Estimation avec le bon degré
    s_hat, A = polynomial_regression(t, x, p_true)

    # Évaluation du polynôme estimé sur une grille fine
    t_fine = np.linspace(0, 5, 200)
    x_hat = evaluate_polynomial(s_hat, t_fine)
    x_true = evaluate_polynomial(s_true, t_fine)

    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(t, x, color='blue', s=50, alpha=0.7, label='Observations bruitées', zorder=3)
    ax.plot(t_fine, x_true, 'g--', linewidth=2, label='Vrai polynôme', zorder=2)
    ax.plot(t_fine, x_hat, 'r-', linewidth=2, label=rf'Polynôme estimé (degré {p_true})', zorder=2)

    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(r'$x$', fontsize=12)
    ax.set_title(rf'Régression polynomiale de degré {p_true} ($m={m}$ observations)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarde
    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'polynomial_regression.png', dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée: {output_dir / 'polynomial_regression.png'}")

    # Affichage des coefficients
    print("\n=== Estimation des coefficients ===")
    print(f"Vrais coefficients:  {s_true}")
    print(f"Coefficients estimés: {s_hat}")
    print(f"Erreur absolue: {np.abs(s_hat - s_true)}")


def plot_order_comparison():
    """
    Génère la Figure 2: Comparaison de différents ordres de polynômes.
    """
    # Paramètres
    m = 20
    p_true = 3
    sigma2 = 0.5**2

    # Génération des données
    np.random.seed(42)
    t, x, s_true, x_clean = generate_data(m, p_true, sigma2)

    # Différents degrés à tester
    degrees = [1, 3, 8]

    # Grille fine pour l'évaluation
    t_fine = np.linspace(0, 5, 200)
    x_true = evaluate_polynomial(s_true, t_fine)

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, p in enumerate(degrees):
        ax = axes[idx]

        # Estimation
        s_hat, A = polynomial_regression(t, x, p)
        x_hat = evaluate_polynomial(s_hat, t_fine)

        # Calcul de l'EQM
        mse = np.mean((evaluate_polynomial(s_hat, t) - x_clean)**2)

        # Tracé
        ax.scatter(t, x, color='blue', s=40, alpha=0.7, label='Observations', zorder=3)
        ax.plot(t_fine, x_true, 'g--', linewidth=2, label='Vrai polynôme', zorder=2)
        ax.plot(t_fine, x_hat, 'r-', linewidth=2, label=rf'Degré $p={p}$', zorder=2)

        # Titre avec classification
        if p < p_true:
            titre_type = "Sous-ajustement"
        elif p == p_true:
            titre_type = "Bon ajustement"
        else:
            titre_type = "Sur-ajustement"

        ax.set_xlabel(r'$t$', fontsize=11)
        ax.set_ylabel(r'$x$', fontsize=11)
        ax.set_title(rf'{titre_type} ($p={p}$, EQM={mse:.3f})', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarde
    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'polynomial_orders.png', dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée: {output_dir / 'polynomial_orders.png'}")


def study_condition_number():
    """
    Étudie le conditionnement de A^T A en fonction du degré p.
    """
    m = 30
    t = np.linspace(0, 5, m)

    degrees = range(1, 16)
    cond_numbers = []

    for p in degrees:
        A = create_vandermonde_matrix(t, p)
        AtA = A.T @ A
        cond = np.linalg.cond(AtA)
        cond_numbers.append(cond)

    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(degrees, cond_numbers, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel(r'Degré du polynôme $p$', fontsize=12)
    ax.set_ylabel(r'Conditionnement de $\mathbf{A}^T\mathbf{A}$', fontsize=12)
    ax.set_title(rf'Évolution du conditionnement numérique ($m={m}$ observations)', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarde
    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'conditioning.png', dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée: {output_dir / 'conditioning.png'}")


if __name__ == '__main__':
    print("=== Tutoriel : Régression polynomiale ===\n")

    # Génération des figures
    plot_regression_example()
    plot_order_comparison()
    study_condition_number()

    print("\nToutes les figures ont été générées avec succès!")
