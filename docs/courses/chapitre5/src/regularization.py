"""
Chapitre 5 : Régularisation en Régression Linéaire
Illustration de Ridge, LASSO, Elastic Net et validation croisée
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuration globale
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Couleurs
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# ============================================================================
# GÉNÉRATION DE DONNÉES SYNTHÉTIQUES
# ============================================================================

def generate_synthetic_data(m=100, p=50, n_active=10, noise_std=1.0,
                           correlation=0.5, random_state=42):
    """
    Génère des données synthétiques pour la régression.

    Args:
        m: Nombre d'observations
        p: Nombre de variables
        n_active: Nombre de variables réellement actives
        noise_std: Écart-type du bruit
        correlation: Corrélation entre variables consécutives
        random_state: Seed aléatoire

    Returns:
        A: Matrice de design (m, p)
        x: Vecteur d'observations (m,)
        s_true: Vrai vecteur de coefficients (p,)
    """
    np.random.seed(random_state)

    # Génération de la matrice A avec corrélation
    A = np.random.randn(m, p)

    # Ajout de corrélation entre variables consécutives
    for j in range(1, p):
        A[:, j] = correlation * A[:, j-1] + np.sqrt(1 - correlation**2) * A[:, j]

    # Vrai vecteur de coefficients (sparse)
    s_true = np.zeros(p)
    active_idx = np.random.choice(p, n_active, replace=False)
    s_true[active_idx] = np.random.randn(n_active) * 3

    # Génération des observations
    x_clean = A @ s_true
    noise = np.random.randn(m) * noise_std
    x = x_clean + noise

    return A, x, s_true


# ============================================================================
# FIGURE 1 : COMPROMIS BIAIS-VARIANCE
# ============================================================================

def plot_bias_variance_tradeoff():
    """
    Figure 1: Illustration du compromis biais-variance.
    """
    complexity = np.linspace(0, 1, 100)

    # Courbes stylisées
    bias2 = 1.5 * (1 - complexity)**2
    variance = 1.5 * complexity**2
    noise = np.ones_like(complexity) * 0.3
    total_error = bias2 + variance + noise

    # Minimum
    min_idx = np.argmin(total_error)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(complexity, bias2, 'b-', linewidth=2.5, label='Biais²')
    ax.plot(complexity, variance, 'r-', linewidth=2.5, label='Variance')
    ax.plot(complexity, total_error, 'g-', linewidth=3, label='Erreur totale')
    ax.plot(complexity, noise, 'k--', linewidth=2, label='Bruit irréductible', alpha=0.5)

    # Marquer le minimum
    ax.axvline(complexity[min_idx], color='gray', linestyle=':', alpha=0.7)
    ax.plot(complexity[min_idx], total_error[min_idx], 'go', markersize=12,
            label=f'Optimum (complexité = {complexity[min_idx]:.2f})')

    ax.set_xlabel('Complexité du modèle (1/λ)', fontsize=12)
    ax.set_ylabel('Erreur', fontsize=12)
    ax.set_title('Compromis Biais-Variance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    # Annotations
    ax.text(0.15, 1.2, 'Sous-apprentissage\n(underfitting)',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(0.85, 1.2, 'Surapprentissage\n(overfitting)',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
    print("✓ Figure sauvegardée: bias_variance_tradeoff.png")


# ============================================================================
# FIGURE 2 & 3 : CHEMINS DE RÉGULARISATION (RIDGE & LASSO)
# ============================================================================

def plot_regularization_paths():
    """
    Figures 2 & 3: Chemins de régularisation Ridge et LASSO.
    """
    # Données
    m, p = 50, 20
    A, x, s_true = generate_synthetic_data(m=m, p=p, n_active=8,
                                          noise_std=0.5, correlation=0.3)

    # Normalisation
    scaler = StandardScaler()
    A_scaled = scaler.fit_transform(A)

    # Grille de lambda (log scale)
    n_lambdas = 100
    lambdas = np.logspace(-2, 3, n_lambdas)

    # Stockage des coefficients
    coefs_ridge = np.zeros((p, n_lambdas))
    coefs_lasso = np.zeros((p, n_lambdas))

    # Calcul des chemins
    for i, lambda_val in enumerate(lambdas):
        # Ridge
        ridge = Ridge(alpha=lambda_val, fit_intercept=False)
        ridge.fit(A_scaled, x)
        coefs_ridge[:, i] = ridge.coef_

        # LASSO
        lasso = Lasso(alpha=lambda_val/m, fit_intercept=False, max_iter=10000)
        lasso.fit(A_scaled, x)
        coefs_lasso[:, i] = lasso.coef_

    # --- FIGURE 2 : RIDGE ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for j in range(p):
        ax.plot(np.log10(lambdas), coefs_ridge[j, :], linewidth=2, alpha=0.7)

    ax.set_xlabel('log₁₀(λ)', fontsize=12)
    ax.set_ylabel('Coefficients θⱼ', fontsize=12)
    ax.set_title('Chemin de Régularisation Ridge (L2)', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'ridge_path.png', dpi=150, bbox_inches='tight')
    print("✓ Figure sauvegardée: ridge_path.png")

    # --- FIGURE 3 : LASSO ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for j in range(p):
        ax.plot(np.log10(lambdas), coefs_lasso[j, :], linewidth=2, alpha=0.7)

    ax.set_xlabel('log₁₀(λ)', fontsize=12)
    ax.set_ylabel('Coefficients θⱼ', fontsize=12)
    ax.set_title('Chemin de Régularisation LASSO (L1)', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Annotation pour montrer les mises à zéro
    ax.text(-1.5, np.max(coefs_lasso)*0.8,
            'Mises à zéro\nabruptes',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

    plt.tight_layout()
    plt.savefig(output_dir / 'lasso_path.png', dpi=150, bbox_inches='tight')
    print("✓ Figure sauvegardée: lasso_path.png")


# ============================================================================
# FIGURE 4 : NORMES Lp
# ============================================================================

def plot_lp_norms():
    """
    Figure 4: Visualisation des boules unités pour normes L0, L1, L2.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    theta = np.linspace(0, 2*np.pi, 1000)

    # --- L0 (pseudo-norme) ---
    ax = axes[0]
    # Boule L0: seulement les axes (4 coins en 2D)
    corners = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    ax.scatter(corners[:, 0], corners[:, 1], s=200, c='red', marker='o', zorder=3)
    ax.plot([-1, 1], [0, 0], 'r-', linewidth=3, alpha=0.5)
    ax.plot([0, 0], [-1, 1], 'r-', linewidth=3, alpha=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('θ₁', fontsize=12)
    ax.set_ylabel('θ₂', fontsize=12)
    ax.set_title('Norme L0\n‖θ‖₀ ≤ 1', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # --- L1 (Manhattan / LASSO) ---
    ax = axes[1]
    # Losange
    diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    ax.fill(diamond[:, 0], diamond[:, 1], color='orange', alpha=0.3)
    ax.plot(diamond[:, 0], diamond[:, 1], 'o-', color='orange', linewidth=3, markersize=8)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('θ₁', fontsize=12)
    ax.set_ylabel('θ₂', fontsize=12)
    ax.set_title('Norme L1 (LASSO)\n‖θ‖₁ ≤ 1', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # Annotation : coins favorisent la parcimonie
    ax.text(0.7, 0.7, 'Coins →\nparcimonie', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

    # --- L2 (Euclidienne / Ridge) ---
    ax = axes[2]
    # Cercle
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    ax.fill(x_circle, y_circle, color='blue', alpha=0.3)
    ax.plot(x_circle, y_circle, 'b-', linewidth=3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('θ₁', fontsize=12)
    ax.set_ylabel('θ₂', fontsize=12)
    ax.set_title('Norme L2 (Ridge)\n‖θ‖₂ ≤ 1', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'lp_norms.png', dpi=150, bbox_inches='tight')
    print("✓ Figure sauvegardée: lp_norms.png")


# ============================================================================
# FIGURE 5 : VALIDATION CROISÉE
# ============================================================================

def plot_cross_validation():
    """
    Figure 5: Erreur de validation croisée en fonction de λ.
    """
    # Données
    m, p = 80, 30
    A, x, s_true = generate_synthetic_data(m=m, p=p, n_active=10,
                                          noise_std=1.0, correlation=0.4)

    # Normalisation
    scaler = StandardScaler()
    A_scaled = scaler.fit_transform(A)

    # Grille de lambda
    n_lambdas = 50
    lambdas = np.logspace(-3, 2, n_lambdas)

    # Validation croisée
    cv_scores = []
    cv_stds = []

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for lambda_val in lambdas:
        lasso = Lasso(alpha=lambda_val/m, fit_intercept=False, max_iter=10000)
        scores = -cross_val_score(lasso, A_scaled, x, cv=kf,
                                  scoring='neg_mean_squared_error')
        cv_scores.append(np.mean(scores))
        cv_stds.append(np.std(scores))

    cv_scores = np.array(cv_scores)
    cv_stds = np.array(cv_stds)

    # Trouver le minimum et la règle "one SE"
    min_idx = np.argmin(cv_scores)
    lambda_min = lambdas[min_idx]
    se_threshold = cv_scores[min_idx] + cv_stds[min_idx]

    # Lambda "one SE": le plus grand lambda avec erreur <= min + SE
    one_se_idx = np.where(cv_scores <= se_threshold)[0][-1]
    lambda_1se = lambdas[one_se_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    log_lambdas = np.log10(lambdas)

    # Courbe avec intervalle de confiance
    ax.plot(log_lambdas, cv_scores, 'b-', linewidth=2.5, label='Erreur CV moyenne')
    ax.fill_between(log_lambdas,
                     cv_scores - cv_stds,
                     cv_scores + cv_stds,
                     alpha=0.3, color='blue', label='± 1 écart-type')

    # Lignes verticales pour λ_min et λ_1SE
    ax.axvline(np.log10(lambda_min), color='green', linestyle='--', linewidth=2,
               label=f'λ_min = {lambda_min:.3f}')
    ax.axvline(np.log10(lambda_1se), color='red', linestyle='--', linewidth=2,
               label=f'λ_1SE = {lambda_1se:.3f}')

    # Ligne horizontale pour le seuil "one SE"
    ax.axhline(se_threshold, color='gray', linestyle=':', alpha=0.7)

    ax.set_xlabel('log₁₀(λ)', fontsize=12)
    ax.set_ylabel('Erreur quadratique moyenne (CV)', fontsize=12)
    ax.set_title('Validation Croisée 10-fold pour LASSO', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'cv_error.png', dpi=150, bbox_inches='tight')
    print("✓ Figure sauvegardée: cv_error.png")

    print(f"\nRésultats de validation croisée:")
    print(f"  λ optimal (min CV): {lambda_min:.3f}")
    print(f"  λ one-SE (plus parcimonieux): {lambda_1se:.3f}")


# ============================================================================
# FIGURE 6 : COMPARAISON DES MÉTHODES
# ============================================================================

def plot_methods_comparison():
    """
    Figure 6: Comparaison Ridge, LASSO, Elastic Net sur données synthétiques.
    """
    # Données avec variables corrélées
    m, p = 100, 50
    A, x, s_true = generate_synthetic_data(m=m, p=p, n_active=10,
                                          noise_std=1.0, correlation=0.6)

    # Normalisation
    scaler = StandardScaler()
    A_scaled = scaler.fit_transform(A)

    # Split train/test
    split = int(0.7 * m)
    A_train, A_test = A_scaled[:split], A_scaled[split:]
    x_train, x_test = x[:split], x[split:]

    # Modèles avec λ choisi par CV
    models = {
        'OLS': LinearRegression(fit_intercept=False),
        'Ridge': Ridge(alpha=1.0, fit_intercept=False),
        'LASSO': Lasso(alpha=0.1, fit_intercept=False, max_iter=10000),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False, max_iter=10000)
    }

    # Entraînement et prédictions
    results = {}
    for name, model in models.items():
        model.fit(A_train, x_train)
        pred_test = model.predict(A_test)
        mse = mean_squared_error(x_test, pred_test)
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)
        results[name] = {
            'coef': model.coef_,
            'mse': mse,
            'n_nonzero': n_nonzero
        }

    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    idx = np.arange(p)
    bar_width = 0.8

    for ax, (name, model) in zip(axes.flat, models.items()):
        coefs = results[name]['coef']
        mse = results[name]['mse']
        n_nz = results[name]['n_nonzero']

        # Barres colorées : rouge pour vrais actifs, bleu pour autres
        colors = ['red' if s_true[j] != 0 else 'blue' for j in range(p)]

        ax.bar(idx, coefs, width=bar_width, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.bar(idx, s_true, width=bar_width, color='none', edgecolor='green',
               linewidth=2, linestyle='--', label='Vrais coefficients')

        ax.set_xlabel('Indice de variable j', fontsize=11)
        ax.set_ylabel('Coefficient θⱼ', fontsize=11)
        ax.set_title(f'{name}\nMSE = {mse:.3f} | Variables actives: {n_nz}/{p}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(-1, p)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'img'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'regularization_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Figure sauvegardée: regularization_comparison.png")

    print(f"\n{'='*70}")
    print("Comparaison des méthodes:")
    print(f"{'='*70}")
    for name in models.keys():
        print(f"\n{name}:")
        print(f"  MSE test: {results[name]['mse']:.3f}")
        print(f"  Variables actives: {results[name]['n_nonzero']}/{p}")

        # Vérifier la sélection correcte
        true_active = set(np.where(s_true != 0)[0])
        pred_active = set(np.where(np.abs(results[name]['coef']) > 1e-5)[0])
        correct = len(true_active & pred_active)
        print(f"  Variables correctement sélectionnées: {correct}/{len(true_active)}")


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  Chapitre 5 : Régularisation en Régression Linéaire")
    print("=" * 70)

    print("\nGénération des figures...\n")

    print("Figure 1: Compromis biais-variance")
    plot_bias_variance_tradeoff()

    print("\nFigures 2 & 3: Chemins de régularisation")
    plot_regularization_paths()

    print("\nFigure 4: Normes Lp")
    plot_lp_norms()

    print("\nFigure 5: Validation croisée")
    plot_cross_validation()

    print("\nFigure 6: Comparaison des méthodes")
    plot_methods_comparison()

    print("\n" + "=" * 70)
    print("  ✓ Toutes les figures ont été générées avec succès!")
    print("=" * 70)
