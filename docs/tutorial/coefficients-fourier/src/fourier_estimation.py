"""
Tutoriel : Estimation des coefficients de Fourier
Comparaison entre échantillonnage arbitraire et sur multiples de périodes
Notation complexe avec exponentielles
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration globale
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def create_fourier_matrix(t, f0, Nf):
    """
    Construit la matrice de design A pour l'estimation de Fourier en notation complexe.

    Args:
        t: Vecteur des instants d'échantillonnage (taille m)
        f0: Fréquence fondamentale
        Nf: Nombre d'harmoniques

    Returns:
        A: Matrice de design complexe de taille (m, 2*Nf+1)
           Colonnes: [e^(-j2πNf*f0*t), ..., e^(-j2πf0*t), 1, e^(j2πf0*t), ..., e^(j2πNf*f0*t)]
    """
    m = len(t)
    A = np.zeros((m, 2 * Nf + 1), dtype=complex)

    # Colonnes pour u = -Nf, ..., -1, 0, 1, ..., Nf
    for idx, u in enumerate(range(-Nf, Nf + 1)):
        A[:, idx] = np.exp(1j * 2 * np.pi * u * f0 * t)

    return A


def fourier_estimation(t, x, f0, Nf):
    """
    Estime les coefficients de Fourier complexes par moindres carrés.

    Args:
        t: Instants d'échantillonnage
        x: Observations (réelles)
        f0: Fréquence fondamentale
        Nf: Nombre d'harmoniques

    Returns:
        s_hat: Vecteur des coefficients complexes estimés [s_{-Nf}, ..., s_0, ..., s_{Nf}]
        A: Matrice de design
    """
    A = create_fourier_matrix(t, f0, Nf)

    # Estimateur des moindres carrés (avec conjuguée hermitienne)
    s_hat = np.linalg.solve(A.conj().T @ A, A.conj().T @ x)

    return s_hat, A


def evaluate_fourier_signal(s, t, f0):
    """
    Évalue un signal à partir de ses coefficients de Fourier complexes.

    Args:
        s: Coefficients complexes [s_{-Nf}, ..., s_0, ..., s_{Nf}]
        t: Points d'évaluation
        f0: Fréquence fondamentale

    Returns:
        x: Signal évalué aux points t (réel si symétrie hermitienne)
    """
    Nf = (len(s) - 1) // 2
    x = np.zeros_like(t, dtype=complex)

    for idx, u in enumerate(range(-Nf, Nf + 1)):
        x += s[idx] * np.exp(1j * 2 * np.pi * u * f0 * t)

    # Retourner la partie réelle (le signal devrait être réel si symétrie hermitienne)
    return np.real(x)


def generate_fourier_data(t, f0, Nf, s_true, sigma2):
    """
    Génère des observations bruitées d'un signal périodique.

    Args:
        t: Instants d'échantillonnage
        f0: Fréquence fondamentale
        Nf: Nombre d'harmoniques
        s_true: Vrais coefficients de Fourier complexes
        sigma2: Variance du bruit

    Returns:
        x: Observations bruitées
        x_clean: Signal sans bruit
    """
    x_clean = evaluate_fourier_signal(s_true, t, f0)
    n = np.random.normal(0, np.sqrt(sigma2), len(t))
    x = x_clean + n

    return x, x_clean


def plot_arbitrary_duration():
    """
    Figure 1: Estimation avec durée d'observation arbitraire (1.5*T0).
    """
    # Paramètres
    f0 = 1.0            # Fréquence fondamentale (Hz)
    T0 = 1 / f0         # Période
    Nf = 2              # Nombre d'harmoniques
    m = 20              # Nombre d'échantillons
    Tobs = 1.5 * T0     # Durée d'observation arbitraire
    sigma2 = 0.1**2     # Variance du bruit

    # Vrais coefficients complexes avec symétrie hermitienne
    # s_0 = 0.5 (moyenne, réelle)
    # s_1 = 0.4 - j*0.25, s_{-1} = conjugué(s_1) = 0.4 + j*0.25
    # s_2 = 0.15 (réel), s_{-2} = 0.15
    # Ordre: [s_{-2}, s_{-1}, s_0, s_1, s_2]
    s_true = np.array([0.15 + 0j, 0.4 + 0.25j, 0.5 + 0j, 0.4 - 0.25j, 0.15 + 0j])

    # Échantillonnage uniforme sur [0, Tobs]
    t = np.linspace(0, Tobs, m)

    # Génération des données
    np.random.seed(42)
    x, x_clean = generate_fourier_data(t, f0, Nf, s_true, sigma2)

    # Estimation
    s_hat, A = fourier_estimation(t, x, f0, Nf)

    # Évaluation sur grille fine
    t_fine = np.linspace(0, 3*T0, 300)
    x_true = evaluate_fourier_signal(s_true, t_fine, f0)
    x_hat = evaluate_fourier_signal(s_hat, t_fine, f0)

    # Calcul du conditionnement (matrice de Gram hermitienne)
    AhA = A.conj().T @ A
    cond = np.linalg.cond(AhA)

    # Visualisation
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(t, x, color='blue', s=50, alpha=0.7, label='Observations bruitées', zorder=3)
    ax.plot(t_fine, x_true, 'g--', linewidth=2, label='Vrai signal', zorder=2)
    ax.plot(t_fine, x_hat, 'r-', linewidth=2, label='Signal estimé', zorder=2)

    # Marquer la zone d'observation
    ax.axvspan(0, Tobs, alpha=0.1, color='gray', label=rf'Zone d\'observation ($T_{{obs}}={Tobs:.1f}T_0$)')

    ax.set_xlabel(r'Temps $t$ (s)', fontsize=12)
    ax.set_ylabel(r'Signal $x(t)$', fontsize=12)
    ax.set_title(rf'Cas 1: Échantillonnage sur durée arbitraire ($N_f={Nf}$, $m={m}$, cond=$10^{{{int(np.log10(cond))}}}$)', fontsize=13)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarde
    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'fourier_arbitrary.png', dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée: {output_dir / 'fourier_arbitrary.png'}")

    print(f"\n=== Cas 1: Durée arbitraire ===")
    print(f"Vrais coefficients (complexes):")
    for idx, u in enumerate(range(-Nf, Nf + 1)):
        print(f"  s_{{{u:+d}}} = {s_true[idx]:.3f}")
    print(f"\nCoefficients estimés:")
    for idx, u in enumerate(range(-Nf, Nf + 1)):
        print(f"  ŝ_{{{u:+d}}} = {s_hat[idx]:.3f}")
    print(f"\nConditionnement de A^H A: {cond:.2e}")


def plot_matrix_comparison():
    """
    Figure 2: Comparaison de la structure de A^H A.
    """
    # Paramètres communs
    f0 = 1.0
    T0 = 1 / f0
    Nf = 3
    m = 40

    # Cas 1: Durée arbitraire
    Tobs1 = 1.5 * T0
    t1 = np.linspace(0, Tobs1, m)
    A1 = create_fourier_matrix(t1, f0, Nf)
    AhA1 = A1.conj().T @ A1

    # Cas 2: Une période complète
    Tobs2 = T0
    t2 = np.linspace(0, Tobs2, m)
    A2 = create_fourier_matrix(t2, f0, Nf)
    AhA2 = A2.conj().T @ A2

    # Normalisation pour visualisation (valeur absolue des matrices complexes)
    AhA1_norm = np.abs(AhA1) / np.max(np.abs(AhA1))
    AhA2_norm = np.abs(AhA2) / np.max(np.abs(AhA2))

    # Calcul des conditionnements
    cond1 = np.linalg.cond(AhA1)
    cond2 = np.linalg.cond(AhA2)

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cas 1: Matrice pleine
    im1 = axes[0].imshow(AhA1_norm, cmap='viridis', aspect='auto', interpolation='nearest')
    axes[0].set_title(rf'Cas 1: $T_{{obs}}={Tobs1:.1f}T_0$ (arbitraire)' + '\n' + rf'Matrice pleine (cond=$10^{{{int(np.log10(cond1))}}}$)', fontsize=12)
    axes[0].set_xlabel('Indice de colonne', fontsize=11)
    axes[0].set_ylabel('Indice de ligne', fontsize=11)
    plt.colorbar(im1, ax=axes[0], label='Valeur normalisée')

    # Cas 2: Matrice diagonale
    im2 = axes[1].imshow(AhA2_norm, cmap='viridis', aspect='auto', interpolation='nearest')
    axes[1].set_title(rf'Cas 2: $T_{{obs}}=T_0$ (une période complète)' + '\n' + rf'Matrice diagonale (cond=$10^{{{int(np.log10(cond2))}}}$)', fontsize=12)
    axes[1].set_xlabel('Indice de colonne', fontsize=11)
    axes[1].set_ylabel('Indice de ligne', fontsize=11)
    plt.colorbar(im2, ax=axes[1], label='Valeur normalisée')

    plt.tight_layout()

    # Sauvegarde
    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'fourier_matrices.png', dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée: {output_dir / 'fourier_matrices.png'}")

    print(f"\n=== Comparaison des matrices ===")
    print(f"Cas 1 (arbitraire): conditionnement = {cond1:.2e}")
    print(f"Cas 2 (une période): conditionnement = {cond2:.2e}")
    print(f"Ratio: {cond1/cond2:.1f}x meilleur conditionnement pour le Cas 2")

    # Vérification de la diagonalité pour le Cas 2
    diag_dominance = np.sum(np.abs(np.diag(AhA2))) / np.sum(np.abs(AhA2))
    print(f"Cas 2: proportion sur la diagonale = {diag_dominance:.1%}")
    print(f"\nRemarque: Cette propriété d'orthogonalité est préservée pour tout nombre entier de périodes P ≥ 1")


def study_sampling_impact():
    """
    Étudie l'impact du nombre d'échantillons et de la durée d'observation.
    """
    f0 = 1.0
    T0 = 1 / f0
    Nf = 2

    # Coefficients complexes avec symétrie hermitienne
    s_true = np.array([0.15 + 0j, 0.4 + 0.25j, 0.5 + 0j, 0.4 - 0.25j, 0.15 + 0j])
    sigma2 = 0.1**2

    # Variation du nombre d'échantillons (à durée fixe)
    m_values = np.arange(10, 101, 5)
    Tobs = 3 * T0

    mse_values = []
    cond_values = []

    np.random.seed(42)

    for m in m_values:
        t = np.linspace(0, Tobs, m)
        x, _ = generate_fourier_data(t, f0, Nf, s_true, sigma2)
        s_hat, A = fourier_estimation(t, x, f0, Nf)

        # EQM sur les coefficients complexes
        mse = np.mean(np.abs(s_hat - s_true)**2)
        cond = np.linalg.cond(A.conj().T @ A)

        mse_values.append(mse)
        cond_values.append(cond)

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(m_values, mse_values, 'o-', linewidth=2, markersize=5)
    axes[0].set_xlabel(r'Nombre d\'échantillons $m$', fontsize=12)
    axes[0].set_ylabel(r'EQM', fontsize=12)
    axes[0].set_title(rf'Évolution de l\'EQM ($T_{{obs}}={int(Tobs/T0)}T_0$)', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(m_values, cond_values, 'o-', linewidth=2, markersize=5)
    axes[1].set_xlabel(r'Nombre d\'échantillons $m$', fontsize=12)
    axes[1].set_ylabel(r'Conditionnement', fontsize=12)
    axes[1].set_title(rf'Évolution du conditionnement ($T_{{obs}}={int(Tobs/T0)}T_0$)', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarde
    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'fourier_impact.png', dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée: {output_dir / 'fourier_impact.png'}")


def compare_fft_vs_direct():
    """
    Compare le temps de calcul FFT vs calcul direct et démontre l'orthogonalité.
    """
    f0 = 1.0
    T0 = 1 / f0
    Nf = 8
    P = 4
    m = 128  # Puissance de 2 pour FFT

    # Échantillonnage sur P périodes
    Tobs = P * T0
    t = np.linspace(0, Tobs, m)

    # Vrai signal avec symétrie hermitienne
    s_true_half = (np.random.randn(Nf) + 1j * np.random.randn(Nf)) / 2
    s_true = np.concatenate([np.conj(s_true_half[::-1]), [np.random.randn()], s_true_half])
    sigma2 = 0.1**2

    # Génération des données
    np.random.seed(42)
    x, _ = generate_fourier_data(t, f0, Nf, s_true, sigma2)

    # Méthode directe
    A = create_fourier_matrix(t, f0, Nf)
    s_hat_direct = np.linalg.solve(A.conj().T @ A, A.conj().T @ x)

    # Vérification de l'orthogonalité: A^H A devrait être ≈ m*I
    AhA = A.conj().T @ A
    identity_approx = AhA / m

    print(f"\n=== Comparaison FFT vs Direct ===")
    print(f"Échantillonnage sur {P} périodes avec m = {m} points")
    print(f"\nVérification orthogonalité: ||A^H A - m*I|| / m = {np.linalg.norm(AhA - m*np.eye(2*Nf+1)) / m:.2e}")
    print(f"Complexité directe: O(m * Nf) = O({m} * {Nf}) = O({m*Nf})")
    print(f"Complexité avec FFT: O(m log m) = O({m} * {int(np.log2(m))}) = O({int(m*np.log2(m))})")
    print(f"Gain de complexité pour m grand: {(m*Nf)/(m*np.log2(m)):.1f}x")


if __name__ == '__main__':
    print("=== Tutoriel : Estimation des coefficients de Fourier (notation complexe) ===\n")

    # Génération des figures
    plot_arbitrary_duration()
    plot_matrix_comparison()
    study_sampling_impact()
    compare_fft_vs_direct()

    print("\nToutes les figures ont été générées avec succès!")
