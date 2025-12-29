"""
Tutoriel : Déconvolution et préfixe cyclique
Cas 1 : Convolution classique
Cas 2 : Convolution circulaire
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import toeplitz, circulant
import time

# Configuration globale
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


# ============================================================================
# CAS 1 : DÉCONVOLUTION AVEC CONVOLUTION CLASSIQUE
# ============================================================================

def create_toeplitz_matrix(h, p):
    """
    Crée la matrice de convolution de Toeplitz.

    Args:
        h: Réponse impulsionnelle de longueur L
        p: Longueur du signal d'entrée

    Returns:
        A: Matrice de Toeplitz de taille (m, p) où m = p + L - 1
    """
    L = len(h)
    m = p + L - 1

    col = np.zeros(m)
    col[:L] = h

    row = np.zeros(p)
    row[0] = h[0]

    A = toeplitz(col, row)
    return A


def deconvolution_cas1(x, h):
    """
    Déconvolution - Cas 1 : convolution classique.

    Args:
        x: Observations de taille m
        h: Réponse impulsionnelle de longueur L

    Returns:
        s_hat: Signal estimé de longueur p = m - L + 1
        A: Matrice de design
    """
    m = len(x)
    L = len(h)
    p = m - L + 1

    A = create_toeplitz_matrix(h, p)

    # Estimateur des moindres carrés
    s_hat = np.linalg.solve(A.T @ A, A.T @ x)

    return s_hat, A


def generate_data_cas1(s, h, sigma2):
    """
    Génère des observations pour le Cas 1 (convolution classique).

    Args:
        s: Signal d'entrée de longueur p
        h: Réponse impulsionnelle de longueur L
        sigma2: Variance du bruit

    Returns:
        x: Observations bruitées de longueur m = p + L - 1
    """
    x_clean = np.convolve(s, h, mode='full')
    n = np.random.normal(0, np.sqrt(sigma2), len(x_clean))
    x = x_clean + n
    return x


def plot_cas1():
    """
    Figure 1: Illustration du Cas 1.
    """
    h = np.array([0.5, 0.3, 0.2])
    p = 5

    A = create_toeplitz_matrix(h, p)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(A, cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_xlabel('Indice de colonne (signal $s$)', fontsize=12)
    ax.set_ylabel('Indice de ligne (observations $x$)', fontsize=12)
    ax.set_title(rf'Cas 1 : Matrice de Toeplitz ($L={len(h)}$, $p={p}$)', fontsize=13)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f'{A[i, j]:.1f}',
                   ha="center", va="center",
                   color="white" if A[i,j] > 0.3 else "black",
                   fontsize=9)

    plt.colorbar(im, ax=ax, label='Valeur')
    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'deconv_toeplitz.png', dpi=150, bbox_inches='tight')
    print(f"✓ Figure sauvegardée: deconv_toeplitz.png")


# ============================================================================
# CAS 2 : DÉCONVOLUTION AVEC CONVOLUTION CIRCULAIRE
# ============================================================================

def create_circulant_matrix(h, p):
    """
    Crée la matrice de convolution circulante.

    Args:
        h: Réponse impulsionnelle de longueur L
        p: Longueur du signal d'entrée

    Returns:
        A_circ: Matrice circulante de taille (p, p)
    """
    L = len(h)

    col = np.zeros(p)
    col[0] = h[0]
    for l in range(1, L):
        col[p - l] = h[l]

    A_circ = circulant(col)
    return A_circ


def deconvolution_cas2(x, h):
    """
    Déconvolution - Cas 2 : convolution circulaire (méthode matricielle).

    Args:
        x: Observations de taille p
        h: Réponse impulsionnelle de longueur L

    Returns:
        s_hat: Signal estimé de longueur p
        A_circ: Matrice de design
    """
    p = len(x)
    A_circ = create_circulant_matrix(h, p)

    # Estimateur des moindres carrés
    s_hat = np.linalg.solve(A_circ.T @ A_circ, A_circ.T @ x)

    return s_hat, A_circ


def deconvolution_cas2_fft(x, h):
    """
    Déconvolution - Cas 2 : convolution circulaire (méthode FFT).

    Args:
        x: Observations de taille p
        h: Réponse impulsionnelle de longueur L

    Returns:
        s_hat: Signal estimé de longueur p
    """
    p = len(x)
    L = len(h)

    # Extension de h à la longueur p
    h_extended = np.zeros(p)
    h_extended[0] = h[0]
    for l in range(1, L):
        h_extended[p - l] = h[l]

    # Réponse fréquentielle du canal
    H = np.fft.fft(h_extended)

    # DFT des observations
    X = np.fft.fft(x)

    # Égalisation fréquentielle
    S_hat = X / H

    # Retour au domaine temporel
    s_hat = np.fft.ifft(S_hat).real

    return s_hat


def generate_data_cas2(s, h, sigma2):
    """
    Génère des observations pour le Cas 2 (convolution circulaire).
    Utilise le préfixe cyclique pour transformer la convolution linéaire en circulaire.

    Args:
        s: Signal d'entrée de longueur p
        h: Réponse impulsionnelle de longueur L
        sigma2: Variance du bruit

    Returns:
        x: Observations après suppression du CP (longueur p)
    """
    p = len(s)
    L = len(h)

    # Ajout du préfixe cyclique
    s_cp = np.concatenate([s[-(L-1):], s])

    # Convolution linéaire
    y_full = np.convolve(s_cp, h, mode='full')

    # Suppression du préfixe cyclique
    x_clean = y_full[L-1:L-1+p]

    # Ajout de bruit
    n = np.random.normal(0, np.sqrt(sigma2), p)
    x = x_clean + n

    return x


def plot_comparison():
    """
    Figure 2: Comparaison Cas 1 vs Cas 2.
    """
    h = np.array([0.8, 0.5, 0.3])
    p = 20

    # Cas 1
    A_cas1 = create_toeplitz_matrix(h, p)
    AtA_cas1 = A_cas1.T @ A_cas1

    # Cas 2
    A_cas2 = create_circulant_matrix(h, p)
    AtA_cas2 = A_cas2.T @ A_cas2

    # Normalisation
    AtA_cas1_norm = AtA_cas1 / np.max(np.abs(AtA_cas1))
    AtA_cas2_norm = AtA_cas2 / np.max(np.abs(AtA_cas2))

    # Conditionnement
    cond_cas1 = np.linalg.cond(AtA_cas1)
    cond_cas2 = np.linalg.cond(AtA_cas2)

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im1 = axes[0].imshow(np.abs(AtA_cas1_norm), cmap='viridis', aspect='auto', interpolation='nearest')
    axes[0].set_title(rf'Cas 1 : $\mathbf{{A}}^T\mathbf{{A}}$ (pleine)' + '\n' + rf'cond = $10^{{{int(np.log10(cond_cas1))}}}$', fontsize=12)
    axes[0].set_xlabel('Indice de colonne', fontsize=11)
    axes[0].set_ylabel('Indice de ligne', fontsize=11)
    plt.colorbar(im1, ax=axes[0], label='Valeur normalisée')

    im2 = axes[1].imshow(np.abs(AtA_cas2_norm), cmap='viridis', aspect='auto', interpolation='nearest')
    axes[1].set_title(rf'Cas 2 : $\mathbf{{A}}^T\mathbf{{A}}$ (diagonalisable par DFT)' + '\n' + rf'cond = $10^{{{int(np.log10(cond_cas2))}}}$', fontsize=12)
    axes[1].set_xlabel('Indice de colonne', fontsize=11)
    axes[1].set_ylabel('Indice de ligne', fontsize=11)
    plt.colorbar(im2, ax=axes[1], label='Valeur normalisée')

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'deconv_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Figure sauvegardée: deconv_comparison.png")

    print(f"\nComparaison des conditionnements:")
    print(f"  Cas 1: {cond_cas1:.2e}")
    print(f"  Cas 2: {cond_cas2:.2e}")
    print(f"  Amélioration: {cond_cas1/cond_cas2:.1f}x")


def compare_performance():
    """
    Figure 3: Comparaison des performances.
    """
    h = np.array([0.8, 0.5, 0.3])
    L = len(h)
    p = 64
    sigma2 = 0.01**2

    # Signal de test
    np.random.seed(42)
    s_true = np.random.randn(p)

    print(f"\nComparaison des performances (p={p}):")

    # Cas 1
    x_cas1 = generate_data_cas1(s_true, h, sigma2)
    t_start = time.time()
    s_hat_cas1, A_cas1 = deconvolution_cas1(x_cas1, h)
    t_cas1 = (time.time() - t_start) * 1000

    mse_cas1 = np.mean((s_hat_cas1 - s_true)**2)

    print(f"\n  Cas 1 (convolution classique):")
    print(f"    Temps: {t_cas1:.2f} ms")
    print(f"    EQM: {mse_cas1:.6f}")

    # Cas 2 (matrice)
    x_cas2 = generate_data_cas2(s_true, h, sigma2)
    t_start = time.time()
    s_hat_cas2, A_cas2 = deconvolution_cas2(x_cas2, h)
    t_cas2 = (time.time() - t_start) * 1000

    mse_cas2 = np.mean((s_hat_cas2 - s_true)**2)

    print(f"\n  Cas 2 (convolution circulaire - matrice):")
    print(f"    Temps: {t_cas2:.2f} ms")
    print(f"    EQM: {mse_cas2:.6f}")
    print(f"    Gain: {t_cas1/t_cas2:.1f}x plus rapide")

    # Cas 2 (FFT)
    t_start = time.time()
    s_hat_cas2_fft = deconvolution_cas2_fft(x_cas2, h)
    t_cas2_fft = (time.time() - t_start) * 1000

    mse_cas2_fft = np.mean((s_hat_cas2_fft - s_true)**2)

    print(f"\n  Cas 2 (convolution circulaire - FFT):")
    print(f"    Temps: {t_cas2_fft:.2f} ms")
    print(f"    EQM: {mse_cas2_fft:.6f}")
    print(f"    Gain: {t_cas1/t_cas2_fft:.1f}x plus rapide que Cas 1")

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(s_true, 'g-', linewidth=2, label='Signal vrai', alpha=0.8)
    axes[0].plot(s_hat_cas1, 'r--', linewidth=1.5, label='Estimé', alpha=0.8)
    axes[0].set_xlabel('Indice', fontsize=11)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('Cas 1 : Convolution classique', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(s_true, 'g-', linewidth=2, label='Signal vrai', alpha=0.8)
    axes[1].plot(s_hat_cas2, 'b--', linewidth=1.5, label='Estimé', alpha=0.8)
    axes[1].set_xlabel('Indice', fontsize=11)
    axes[1].set_ylabel('Amplitude', fontsize=11)
    axes[1].set_title('Cas 2 : Convolution circulaire (matrice)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(s_true, 'g-', linewidth=2, label='Signal vrai', alpha=0.8)
    axes[2].plot(s_hat_cas2_fft, 'm--', linewidth=1.5, label='Estimé', alpha=0.8)
    axes[2].set_xlabel('Indice', fontsize=11)
    axes[2].set_ylabel('Amplitude', fontsize=11)
    axes[2].set_title('Cas 2 : Convolution circulaire (FFT)', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'deconv_performance.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure sauvegardée: deconv_performance.png")


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  Tutoriel : Déconvolution et préfixe cyclique")
    print("=" * 70)

    print("\n" + "="*70)
    print("CAS 1 : DÉCONVOLUTION AVEC CONVOLUTION CLASSIQUE")
    print("="*70)
    plot_cas1()

    print("\n" + "="*70)
    print("CAS 2 : DÉCONVOLUTION AVEC CONVOLUTION CIRCULAIRE")
    print("="*70)
    plot_comparison()
    compare_performance()

    print("\n" + "="*70)
    print("  ✓ Toutes les figures ont été générées avec succès!")
    print("="*70)
    print("\nNote: Le Cas 2 ne se produit pas naturellement.")
    print("L'OFDM utilise un préfixe cyclique pour transformer")
    print("la convolution linéaire en convolution circulaire.")
