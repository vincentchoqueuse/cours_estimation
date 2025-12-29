"""
Tutoriel : Estimation d'un canal FIR
=====================================

Ce script illustre l'estimation d'un canal linéaire (FIR) à partir
d'observations entrée-sortie en utilisant l'estimateur des moindres carrés.

Contenu :
1. Construction de la matrice de Toeplitz
2. Estimation OLS de la réponse impulsionnelle
3. Impact du type de signal d'entrée (blanc, BPSK, périodique)
4. Performance en fonction du SNR
5. Performance en fonction du nombre d'échantillons

Auteur: Cours d'Estimation Statistique
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import os

# Configuration matplotlib
plt.rcParams.update({
    'font.size': 10,
    'text.usetex': False,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 12
})


def create_toeplitz_matrix(s, L):
    """
    Crée la matrice de Toeplitz S pour l'estimation de canal.

    La ligne k contient : [s[k], s[k-1], ..., s[k-L+1]]
    avec des zéros pour les indices négatifs.

    Parameters:
    -----------
    s : array (m,)
        Signal d'entrée
    L : int
        Longueur de la réponse impulsionnelle

    Returns:
    --------
    S : array (m, L)
        Matrice de Toeplitz
    """
    m = len(s)

    # Pad avec des zéros au début pour gérer les indices négatifs
    s_padded = np.concatenate([np.zeros(L-1), s])

    # Construire la matrice de Toeplitz
    S = np.zeros((m, L))
    for k in range(m):
        # Ligne k : [s[k], s[k-1], ..., s[k-L+1]]
        # Dans s_padded : indices [k+L-1, k+L-2, ..., k]
        S[k, :] = s_padded[k+L-1:k-1:-1] if k > 0 else s_padded[L-1::-1]

    return S


def estimate_channel_ols(s, x, L):
    """
    Estime la réponse impulsionnelle du canal par OLS.

    Parameters:
    -----------
    s : array (m,)
        Signal d'entrée (connu)
    x : array (m,)
        Signal de sortie (observé)
    L : int
        Longueur supposée de la réponse impulsionnelle

    Returns:
    --------
    h_hat : array (L,)
        Estimation de la réponse impulsionnelle
    """
    # Construire la matrice S
    S = create_toeplitz_matrix(s, L)

    # OLS : h_hat = (S^T S)^{-1} S^T x
    StS = S.T @ S
    h_hat = np.linalg.solve(StS, S.T @ x)

    return h_hat


def generate_channel(L, channel_type='multipath'):
    """
    Génère une réponse impulsionnelle de canal.

    Parameters:
    -----------
    L : int
        Longueur de la réponse impulsionnelle
    channel_type : str
        Type de canal ('multipath', 'simple', 'exponential')

    Returns:
    --------
    h : array (L,)
        Réponse impulsionnelle
    """
    if channel_type == 'multipath':
        # Canal à trajets multiples (typique en télécommunications)
        h = np.zeros(L)
        h[0] = 1.0          # Trajet direct
        h[2] = 0.5          # Réflexion
        h[5] = 0.3          # Réflexion retardée
        h[8] = 0.15         # Réflexion lointaine

    elif channel_type == 'simple':
        # Canal simple avec décroissance
        h = np.array([1.0, 0.5, 0.2])
        h = np.pad(h, (0, L - len(h)))

    elif channel_type == 'exponential':
        # Décroissance exponentielle
        alpha = 0.7
        h = alpha ** np.arange(L)
        h = h / np.linalg.norm(h)

    return h


def convolve_with_channel(s, h, noise_std=0.1):
    """
    Applique le canal à un signal d'entrée.

    Calcule x[k] = sum_l h[l] * s[k-l] + w[k]
    pour k = 0, ..., m-1, avec s[k] = 0 pour k < 0.

    Cette implémentation est cohérente avec la matrice de Toeplitz
    utilisée dans estimate_channel_ols.

    Parameters:
    -----------
    s : array (m,)
        Signal d'entrée
    h : array (L,)
        Réponse impulsionnelle du canal
    noise_std : float
        Écart-type du bruit additif

    Returns:
    --------
    x : array (m,)
        Signal de sortie bruité
    """
    m = len(s)
    L = len(h)

    # Pad avec des zéros au début (comme dans la matrice S)
    s_padded = np.concatenate([np.zeros(L-1), s])

    # Convolution manuelle : x[k] = sum_l h[l] * s[k-l]
    x_clean = np.zeros(m)
    for k in range(m):
        # s[k-l] pour l=0,...,L-1 correspond à s_padded[k+L-1-l]
        for l in range(L):
            x_clean[k] += h[l] * s_padded[k + L - 1 - l]

    # Ajout de bruit
    noise = np.random.randn(m) * noise_std
    x = x_clean + noise

    return x


def compute_mse(h_true, h_hat):
    """
    Calcule l'erreur quadratique moyenne entre la vraie réponse
    et l'estimation (alignées en longueur).
    """
    L = min(len(h_true), len(h_hat))
    return np.mean((h_true[:L] - h_hat[:L])**2)


def snr_to_noise_std(s, h, snr_db):
    """
    Convertit un SNR en dB en écart-type de bruit.

    SNR = 10 log10(P_signal / P_bruit)
    """
    # Puissance du signal (sortie sans bruit)
    # Utiliser la même convolution que dans convolve_with_channel
    x_clean = convolve_with_channel(s, h, noise_std=0.0)
    P_signal = np.mean(x_clean**2)

    # Puissance du bruit cible
    snr_linear = 10**(snr_db / 10)
    P_noise = P_signal / snr_linear

    return np.sqrt(P_noise)


def illustrate_channel_estimation():
    """
    Figure 1 : Estimation de canal avec différents types de signaux d'entrée.
    """
    print("\n" + "="*70)
    print("1. Estimation avec différents signaux d'entrée")
    print("="*70)

    # Paramètres
    m = 200
    L = 12
    noise_std = 0.05
    np.random.seed(42)

    # Génération du canal
    h_true = generate_channel(L, 'multipath')

    # Trois types de signaux d'entrée
    signals = {
        'Blanc gaussien': np.random.randn(m),
        'BPSK (binaire)': np.random.choice([-1, 1], size=m),
        'Sinusoïde': np.sin(2 * np.pi * 0.1 * np.arange(m))
    }

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    for idx, (name, s) in enumerate(signals.items()):
        # Normaliser le signal d'entrée
        s = s / np.std(s)

        # Appliquer le canal
        x = convolve_with_channel(s, h_true, noise_std)

        # Estimer le canal
        h_hat = estimate_channel_ols(s, x, L)

        # Calculer MSE
        mse = compute_mse(h_true, h_hat)

        # Conditionnement
        S = create_toeplitz_matrix(s, L)
        cond_number = np.linalg.cond(S.T @ S)

        print(f"\n{name}:")
        print(f"  MSE = {mse:.6f}")
        print(f"  Conditionnement de S^T S = {cond_number:.2f}")

        # Graphique signal d'entrée
        ax = axes[idx, 0]
        ax.plot(s[:100], linewidth=1)
        ax.set_xlabel('Échantillon n')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Signal d\'entrée : {name}')
        ax.grid(True, alpha=0.3)

        # Graphique estimation du canal
        ax = axes[idx, 1]
        ax.stem(h_true, linefmt='b-', markerfmt='bo', basefmt='b-',
                label='Canal vrai')
        ax.stem(h_hat, linefmt='r--', markerfmt='rx', basefmt='r--',
                label=f'Estimation (MSE={mse:.4f})')
        ax.set_xlabel('Coefficient l')
        ax.set_ylabel('Amplitude $h_l$')
        ax.set_title(f'Réponse impulsionnelle (cond={cond_number:.1f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarder
    output_dir = '../img'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/canal_estimation.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure sauvegardée : {output_dir}/canal_estimation.png")


def mse_vs_snr():
    """
    Figure 2 : Performance en fonction du SNR.
    """
    print("\n" + "="*70)
    print("2. Performance en fonction du SNR")
    print("="*70)

    # Paramètres
    m = 200
    L = 12
    snr_db_range = np.linspace(-5, 30, 20)
    n_trials = 50
    np.random.seed(42)

    # Génération du canal
    h_true = generate_channel(L, 'multipath')

    # Calculer MSE pour chaque SNR
    mse_mean = []
    mse_std = []

    for snr_db in snr_db_range:
        mse_trials = []

        for _ in range(n_trials):
            # Signal d'entrée (blanc gaussien)
            s = np.random.randn(m)
            s = s / np.std(s)

            # Convertir SNR en noise_std
            noise_std = snr_to_noise_std(s, h_true, snr_db)

            # Appliquer le canal
            x = convolve_with_channel(s, h_true, noise_std)

            # Estimer
            h_hat = estimate_channel_ols(s, x, L)

            # MSE
            mse = compute_mse(h_true, h_hat)
            mse_trials.append(mse)

        mse_mean.append(np.mean(mse_trials))
        mse_std.append(np.std(mse_trials))

    mse_mean = np.array(mse_mean)
    mse_std = np.array(mse_std)

    # Graphique
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(snr_db_range, mse_mean, 'b-', linewidth=2, label='MSE moyenne')
    ax.fill_between(snr_db_range, mse_mean - mse_std, mse_mean + mse_std,
                    alpha=0.3, color='blue', label='± 1 écart-type')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('MSE')
    ax.set_title('Erreur quadratique moyenne en fonction du SNR')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Sauvegarder
    output_dir = '../img'
    plt.savefig(f'{output_dir}/mse_vs_snr.png', dpi=300, bbox_inches='tight')
    print(f"✓ Figure sauvegardée : {output_dir}/mse_vs_snr.png")



def mse_vs_length():
    """
    Figure 3 : Performance en fonction du nombre d'échantillons.
    """
    print("\n" + "="*70)
    print("3. Performance en fonction du nombre d'échantillons")
    print("="*70)

    # Paramètres
    L = 12
    m_range = np.logspace(np.log10(50), np.log10(1000), 15).astype(int)
    n_trials = 50
    snr_db = 20
    np.random.seed(42)

    # Génération du canal
    h_true = generate_channel(L, 'multipath')

    # Calculer MSE pour chaque longueur
    mse_mean = []
    mse_std = []

    for m in m_range:
        mse_trials = []

        for _ in range(n_trials):
            # Signal d'entrée
            s = np.random.randn(m)
            s = s / np.std(s)

            # Convertir SNR
            noise_std = snr_to_noise_std(s, h_true, snr_db)

            # Appliquer le canal
            x = convolve_with_channel(s, h_true, noise_std)

            # Estimer
            h_hat = estimate_channel_ols(s, x, L)

            # MSE
            mse = compute_mse(h_true, h_hat)
            mse_trials.append(mse)

        mse_mean.append(np.mean(mse_trials))
        mse_std.append(np.std(mse_trials))

    mse_mean = np.array(mse_mean)
    mse_std = np.array(mse_std)

    # Graphique
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(m_range, mse_mean, 'b-', linewidth=2, marker='o', label='MSE empirique')
    ax.fill_between(m_range, mse_mean - mse_std, mse_mean + mse_std,
                    alpha=0.3, color='blue', label='± 1 écart-type')

    # Borne de Cramér-Rao (plancher de bruit théorique)
    # CRB = σ²_w * trace((S^T S)^{-1})
    # Calculer la CRB exacte pour chaque m
    crb = []
    for m in m_range:
        # Générer un signal typique
        s_test = np.random.randn(m)
        s_test = s_test / np.std(s_test)

        # Calculer σ²_w à partir du SNR
        noise_std_crb = snr_to_noise_std(s_test, h_true, snr_db)

        # Construire S et calculer trace((S^T S)^{-1})
        S = create_toeplitz_matrix(s_test, L)
        StS = S.T @ S
        StS_inv = np.linalg.inv(StS)
        trace_inv = np.trace(StS_inv)

        # CRB exacte pour la MSE moyenne (normalisée par L)
        # E[MSE] = E[(1/L)||h_hat - h||²] = (1/L) * σ²_w * trace((S^T S)^{-1})
        crb.append(noise_std_crb**2 * trace_inv / L)

    crb = np.array(crb)
    ax.plot(m_range, crb, 'g:', linewidth=2, label=f'Borne de Cramér-Rao')

    ax.set_xlabel('Nombre d\'échantillons m')
    ax.set_ylabel('MSE')
    ax.set_title(f'MSE en fonction de la longueur d\'observation (SNR = {snr_db} dB)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    print(f"\n  Pour m = {m_range[0]}: MSE = {mse_mean[0]:.6f}")
    print(f"  Pour m = {m_range[-1]}: MSE = {mse_mean[-1]:.6f}")
    print(f"  Réduction : facteur {mse_mean[0]/mse_mean[-1]:.1f}")

    plt.tight_layout()

    # Sauvegarder
    output_dir = '../img'
    plt.savefig(f'{output_dir}/mse_vs_length.png', dpi=300, bbox_inches='tight')
    print(f"✓ Figure sauvegardée : {output_dir}/mse_vs_length.png")


def demonstrate_conditioning():
    """
    Démonstration de l'impact du conditionnement.
    """
    print("\n" + "="*70)
    print("4. Impact du conditionnement de la matrice S^T S")
    print("="*70)

    m = 200
    L = 12

    # Différents signaux
    signals = {
        'Blanc gaussien': np.random.randn(m),
        'BPSK': np.random.choice([-1, 1], size=m),
        'Chirp': np.sin(2 * np.pi * np.linspace(0, 10, m) * np.arange(m) / m),
        'Sinusoïde pure': np.sin(2 * np.pi * 0.1 * np.arange(m))
    }

    print(f"\n{'Signal':<20} {'Conditionnement':<20} {'Interprétation'}")
    print("-" * 70)

    for name, s in signals.items():
        s = s / np.std(s)
        S = create_toeplitz_matrix(s, L)
        cond = np.linalg.cond(S.T @ S)

        if cond < 100:
            interp = "Excellent"
        elif cond < 1000:
            interp = "Bon"
        elif cond < 10000:
            interp = "Moyen"
        else:
            interp = "Mauvais (matrice mal conditionnée)"

        print(f"{name:<20} {cond:<20.2e} {interp}")


def compare_signal_designs():
    """
    Figure 4 : Comparaison de différents designs de signaux pour minimiser
    la trace de la covariance sous contrainte d'énergie.
    """
    print("\n" + "="*70)
    print("5. Design optimal du signal d'entrée")
    print("="*70)

    m = 200
    L = 12
    E = 100  # Budget énergétique
    np.random.seed(42)

    # Générer le canal
    h_true = generate_channel(L, 'multipath')

    # Générer une PRBS (Pseudo-Random Binary Sequence)
    prbs = np.random.choice([-1, 1], size=m)

    # Multi-sinusoïdes (5 fréquences bien espacées)
    freqs = [0.05, 0.15, 0.25, 0.35, 0.45]
    multi_sine = np.zeros(m)
    for f in freqs:
        multi_sine += np.sin(2 * np.pi * f * np.arange(m))

    # Différents signaux (normalisés à même énergie E)
    signals = {
        'Bruit blanc': np.random.randn(m),
        'PRBS': prbs,
        'Multi-sinusoïdes': multi_sine,
        'Sinusoïde pure': np.sin(2 * np.pi * 0.1 * np.arange(m))
    }

    # Normaliser tous les signaux à énergie E
    for name in signals:
        signals[name] = signals[name] / np.linalg.norm(signals[name]) * np.sqrt(E)

    # Calculer les métriques pour chaque signal
    results = {}
    for name, s in signals.items():
        # Construire S
        S = create_toeplitz_matrix(s, L)
        StS = S.T @ S

        # Métriques
        eigenvalues = np.linalg.eigvalsh(StS)
        cond = np.linalg.cond(StS)
        trace_inv = np.trace(np.linalg.inv(StS))

        results[name] = {
            'eigenvalues': eigenvalues,
            'cond': cond,
            'trace_inv': trace_inv,
            'signal': s
        }

        print(f"\n{name}:")
        print(f"  Conditionnement : {cond:.2e}")
        print(f"  trace((S^T S)^{-1}) : {trace_inv:.6f}")
        print(f"  λ_min / λ_max : {eigenvalues[0] / eigenvalues[-1]:.2e}")

    # Visualisation
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)

    # Pour chaque signal
    for idx, (name, s) in enumerate(signals.items()):
        res = results[name]

        # Signal temporel
        ax = fig.add_subplot(gs[idx, 0])
        ax.plot(s[:100], linewidth=1)
        ax.set_xlabel('Échantillon n')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{name}\n(Cond={res["cond"]:.1e})')
        ax.grid(True, alpha=0.3)

        # Spectre fréquentiel (FFT)
        ax = fig.add_subplot(gs[idx, 1])
        spectrum = np.abs(np.fft.fft(s))**2
        freqs_fft = np.fft.fftfreq(len(s))
        ax.plot(freqs_fft[:len(s)//2], spectrum[:len(s)//2], linewidth=1)
        ax.set_xlabel('Fréquence normalisée')
        ax.set_ylabel('Densité spectrale')
        ax.set_title('Spectre de puissance')
        ax.grid(True, alpha=0.3)

        # Valeurs propres de S^T S
        ax = fig.add_subplot(gs[idx, 2])
        ax.stem(res['eigenvalues'], linefmt='b-', markerfmt='bo', basefmt='b-')
        ax.set_xlabel('Indice')
        ax.set_ylabel('Valeur propre')
        ax.set_title(f'Valeurs propres de $S^T S$')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    # Sauvegarder
    output_dir = '../img'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/signal_design_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure sauvegardée : {output_dir}/signal_design_comparison.png")

    # Conclusion
    print("\n" + "-"*70)
    print("CONCLUSION :")
    print("  → Bruit blanc et PRBS : conditionnement optimal (≈1)")
    print("  → Multi-sinusoïdes : acceptable si bien espacées")
    print("  → Sinusoïde pure : très mauvais (matrice quasi-singulière)")
    print("  → Pour minimiser trace((S^T S)^{-1}), utiliser un signal à spectre plat")


def test_consistency():
    """
    Test de cohérence : vérifier que l'estimation fonctionne sans bruit.
    """
    print("\n" + "="*70)
    print("TEST DE COHÉRENCE")
    print("="*70)

    np.random.seed(42)
    m = 200
    L = 12

    # Générer un canal et un signal
    h_true = generate_channel(L, 'multipath')
    s = np.random.randn(m)
    s = s / np.std(s)

    # Appliquer le canal SANS bruit
    x = convolve_with_channel(s, h_true, noise_std=0.0)

    # Estimer le canal
    h_hat = estimate_channel_ols(s, x, L)

    # MSE doit être très proche de 0
    mse = compute_mse(h_true, h_hat)

    print(f"\nCanal vrai :    {h_true}")
    print(f"Canal estimé :  {h_hat}")
    print(f"Différence :    {h_true - h_hat}")
    print(f"MSE (sans bruit) : {mse:.2e}")

    if mse < 1e-10:
        print("✓ Test réussi : estimation parfaite sans bruit")
    else:
        print("✗ Test échoué : l'estimation n'est pas correcte même sans bruit!")
        print("  → Vérifier la cohérence entre convolution et matrice de Toeplitz")


def main():
    """Fonction principale."""

    print("="*70)
    print("Tutoriel : Estimation d'un canal FIR")
    print("="*70)

    # Test de cohérence
    test_consistency()

    # Figure 1 : Différents signaux d'entrée
    illustrate_channel_estimation()

    # Figure 2 : Performance vs SNR
    mse_vs_snr()

    # Figure 3 : Performance vs longueur
    mse_vs_length()

    # Figure 4 : Design optimal du signal
    compare_signal_designs()

    # Démonstration conditionnement
    demonstrate_conditioning()

    print("\n" + "="*70)
    print("Script terminé avec succès !")
    print("="*70)


if __name__ == '__main__':
    main()
