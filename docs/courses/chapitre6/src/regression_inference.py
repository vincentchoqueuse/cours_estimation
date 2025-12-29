"""
Chapitre 6 : Régression Linéaire - Inférence et Diagnostic
===========================================================

Ce script illustre :
1. Intervalles de confiance et tests d'hypothèses (t-tests, F-test)
2. Graphiques de diagnostic des résidus (4 graphiques standards)
3. Détection de la multicolinéarité (VIF, conditionnement)
4. Détection de l'hétéroscédasticité
5. Détection de points aberrants et influents (Cook's distance, leverage)
6. Comparaison OLS vs Ridge vs LASSO vs WLS

Auteur: Cours d'Estimation Statistique
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import os

# Configuration matplotlib pour rendu LaTeX
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


def generate_regression_data(n=100, p=5, noise_std=1.0, seed=42):
    """
    Génère des données de régression synthétiques.

    Parameters:
    -----------
    n : int
        Nombre d'observations
    p : int
        Nombre de variables explicatives
    noise_std : float
        Écart-type du bruit
    seed : int
        Graine aléatoire

    Returns:
    --------
    X : array (n, p)
        Matrice de design
    y : array (n,)
        Variable dépendante
    true_coef : array (p,)
        Vrais coefficients
    """
    np.random.seed(seed)

    # Génération de X avec corrélation faible
    X = np.random.randn(n, p)

    # Vrais coefficients (alternance de valeurs significatives et nulles)
    true_coef = np.array([2.0, -1.5, 0.0, 3.0, 0.0])[:p]

    # Variable dépendante avec bruit gaussien
    y = X @ true_coef + np.random.randn(n) * noise_std

    return X, y, true_coef


def compute_ols_statistics(X, y):
    """
    Calcule les statistiques OLS complètes : coefficients, erreurs-types,
    statistiques t, p-valeurs, intervalles de confiance, R², F-stat.

    Parameters:
    -----------
    X : array (n, p)
        Matrice de design
    y : array (n,)
        Variable dépendante

    Returns:
    --------
    dict contenant toutes les statistiques
    """
    n, p = X.shape

    # Estimation OLS
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y

    # Prédictions et résidus
    y_hat = X @ beta_hat
    residuals = y - y_hat

    # Variance estimée
    dof = n - p  # Degrés de liberté
    sigma2_hat = np.sum(residuals**2) / dof

    # Erreurs-types des coefficients
    se_beta = np.sqrt(np.diag(XtX_inv) * sigma2_hat)

    # Statistiques t
    t_stats = beta_hat / se_beta

    # p-valeurs (test bilatéral)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

    # Intervalles de confiance à 95%
    t_critical = stats.t.ppf(0.975, dof)
    ci_lower = beta_hat - t_critical * se_beta
    ci_upper = beta_hat + t_critical * se_beta

    # R² et R² ajusté
    SS_tot = np.sum((y - np.mean(y))**2)
    SS_res = np.sum(residuals**2)
    SS_exp = SS_tot - SS_res
    R2 = SS_exp / SS_tot
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - p)

    # Test F global
    F_stat = (SS_exp / (p - 1)) / (SS_res / dof) if p > 1 else np.nan
    F_pvalue = 1 - stats.f.cdf(F_stat, p - 1, dof) if p > 1 else np.nan

    return {
        'beta_hat': beta_hat,
        'se_beta': se_beta,
        't_stats': t_stats,
        'p_values': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'residuals': residuals,
        'y_hat': y_hat,
        'sigma2_hat': sigma2_hat,
        'R2': R2,
        'R2_adj': R2_adj,
        'F_stat': F_stat,
        'F_pvalue': F_pvalue,
        'dof': dof
    }


def compute_vif(X):
    """
    Calcule le Variance Inflation Factor (VIF) pour chaque variable.

    VIF_j = 1 / (1 - R²_j) où R²_j est le R² de la régression de X_j sur les autres variables.

    Parameters:
    -----------
    X : array (n, p)
        Matrice de design

    Returns:
    --------
    vif : array (p,)
        VIF pour chaque variable
    """
    n, p = X.shape
    vif = np.zeros(p)

    for j in range(p):
        # Régression de X[:, j] sur toutes les autres variables
        X_j = X[:, j]
        X_others = np.delete(X, j, axis=1)

        if X_others.shape[1] > 0:
            # OLS
            beta = np.linalg.lstsq(X_others, X_j, rcond=None)[0]
            y_pred = X_others @ beta

            # R²
            SS_tot = np.sum((X_j - np.mean(X_j))**2)
            SS_res = np.sum((X_j - y_pred)**2)
            R2_j = 1 - SS_res / SS_tot

            # VIF
            vif[j] = 1 / (1 - R2_j) if R2_j < 0.9999 else np.inf
        else:
            vif[j] = 1.0

    return vif


def compute_leverage_and_cooks_distance(X, y, residuals):
    """
    Calcule le leverage (h_k) et la distance de Cook (D_k) pour chaque observation.

    Parameters:
    -----------
    X : array (n, p)
        Matrice de design
    y : array (n,)
        Variable dépendante
    residuals : array (n,)
        Résidus

    Returns:
    --------
    leverage : array (n,)
        Leverage de chaque observation
    cooks_d : array (n,)
        Distance de Cook de chaque observation
    """
    n, p = X.shape

    # Matrice de projection (hat matrix)
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    leverage = np.diag(H)

    # Résidus standardisés
    sigma2_hat = np.sum(residuals**2) / (n - p)
    std_residuals = residuals / (np.sqrt(sigma2_hat * (1 - leverage)))

    # Distance de Cook
    cooks_d = (std_residuals**2 / p) * (leverage / (1 - leverage))

    return leverage, cooks_d, std_residuals


def plot_diagnostic_residuals(X, y, stats_dict):
    """
    Génère les 4 graphiques de diagnostic standards pour l'analyse des résidus.

    1. Residuals vs Fitted Values (détection hétéroscédasticité et non-linéarité)
    2. Q-Q Plot (vérification normalité)
    3. Scale-Location (vérification homoscédasticité)
    4. Residuals vs Leverage (détection points influents avec Cook's distance)
    """
    residuals = stats_dict['residuals']
    y_hat = stats_dict['y_hat']

    # Calcul leverage et Cook's distance
    leverage, cooks_d, std_residuals = compute_leverage_and_cooks_distance(X, y, residuals)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Residuals vs Fitted Values
    ax = axes[0, 0]
    ax.scatter(y_hat, residuals, alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='y=0')
    ax.set_xlabel('Valeurs ajustées $\\hat{x}_k$')
    ax.set_ylabel('Résidus $\\hat{n}_k$')
    ax.set_title('Résidus vs Valeurs ajustées')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Q-Q Plot
    ax = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normalité)')
    ax.grid(True, alpha=0.3)

    # 3. Scale-Location (sqrt des résidus standardisés en valeur absolue)
    ax = axes[1, 0]
    sqrt_std_resid = np.sqrt(np.abs(std_residuals))
    ax.scatter(y_hat, sqrt_std_resid, alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
    # Ligne de tendance
    z = np.polyfit(y_hat, sqrt_std_resid, 1)
    p = np.poly1d(z)
    ax.plot(sorted(y_hat), p(sorted(y_hat)), "r--", linewidth=1.5, label='Tendance')
    ax.set_xlabel('Valeurs ajustées $\\hat{x}_k$')
    ax.set_ylabel('$\\sqrt{|r_k^*|}$')
    ax.set_title('Scale-Location (Homoscédasticité)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Residuals vs Leverage avec Cook's distance
    ax = axes[1, 1]
    ax.scatter(leverage, std_residuals, alpha=0.6, s=40, edgecolors='k', linewidth=0.5)

    # Lignes de référence pour Cook's distance
    n, p = X.shape
    x_range = np.linspace(0.001, max(leverage), 100)

    # Contours pour D = 0.5 et D = 1
    for D_threshold, ls, label in [(0.5, '--', "D = 0.5"), (1.0, '-.', "D = 1.0")]:
        # D = (r*²/p) * (h/(1-h))
        # => r*² = D * p * (1-h) / h
        # => r* = ±sqrt(D * p * (1-h) / h)
        y_upper = np.sqrt(D_threshold * p * (1 - x_range) / x_range)
        y_lower = -y_upper
        ax.plot(x_range, y_upper, 'r', linestyle=ls, linewidth=1.5, alpha=0.7)
        ax.plot(x_range, y_lower, 'r', linestyle=ls, linewidth=1.5, alpha=0.7, label=label)

    # Marquer les points avec Cook's distance > 0.5
    influential = cooks_d > 0.5
    if np.any(influential):
        ax.scatter(leverage[influential], std_residuals[influential],
                  s=100, facecolors='none', edgecolors='red', linewidth=2,
                  label=f'Points influents (D>0.5): {np.sum(influential)}')

    ax.set_xlabel('Leverage $h_k$')
    ax.set_ylabel('Résidus standardisés $r_k^*$')
    ax.set_title('Résidus vs Leverage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarder
    output_dir = '../img'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/diagnostic_residus.png', dpi=300, bbox_inches='tight')
    print(f"✓ Figure sauvegardée : {output_dir}/diagnostic_residus.png")


def demonstrate_multicollinearity():
    """
    Démontre la détection de la multicolinéarité avec VIF et conditionnement.
    """
    print("\n" + "="*70)
    print("DÉMONSTRATION : Multicolinéarité")
    print("="*70)

    n = 100
    np.random.seed(42)

    # Cas 1 : Pas de multicolinéarité
    X1 = np.random.randn(n, 3)
    y1 = X1 @ np.array([1.0, 2.0, -1.0]) + np.random.randn(n) * 0.5

    vif1 = compute_vif(X1)
    cond1 = np.linalg.cond(X1.T @ X1)

    print("\nCas 1 : Variables indépendantes")
    print(f"  VIF : {vif1}")
    print(f"  Conditionnement de A^T A : {cond1:.2f}")
    print(f"  → Pas de multicolinéarité (VIF < 5)")

    # Cas 2 : Multicolinéarité sévère
    X2 = np.random.randn(n, 3)
    # Créer une forte corrélation : X2[:, 2] ≈ 0.95 * X2[:, 0] + 0.05 * X2[:, 1]
    X2[:, 2] = 0.95 * X2[:, 0] + 0.05 * X2[:, 1] + 0.05 * np.random.randn(n)
    y2 = X2 @ np.array([1.0, 2.0, -1.0]) + np.random.randn(n) * 0.5

    vif2 = compute_vif(X2)
    cond2 = np.linalg.cond(X2.T @ X2)

    print("\nCas 2 : Multicolinéarité sévère (X3 ≈ 0.95*X1 + 0.05*X2)")
    print(f"  VIF : {vif2}")
    print(f"  Conditionnement de A^T A : {cond2:.2f}")
    print(f"  → Multicolinéarité sévère (VIF > 10)")

    # Matrice de corrélation
    corr_matrix = np.corrcoef(X2.T)
    print(f"\n  Matrice de corrélation :")
    print(f"  {corr_matrix}")


def demonstrate_heteroscedasticity():
    """
    Démontre la détection de l'hétéroscédasticité.
    """
    print("\n" + "="*70)
    print("DÉMONSTRATION : Hétéroscédasticité")
    print("="*70)

    n = 200
    np.random.seed(42)

    # Variable explicative
    X = np.random.uniform(0, 10, (n, 1))

    # Cas 1 : Homoscédasticité (variance constante)
    y_homo = 2 + 3 * X[:, 0] + np.random.randn(n) * 2.0

    # Cas 2 : Hétéroscédasticité (variance proportionnelle à X)
    y_hetero = 2 + 3 * X[:, 0] + np.random.randn(n) * (0.5 * X[:, 0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Homoscédasticité
    ax = axes[0]
    stats_homo = compute_ols_statistics(X, y_homo)
    ax.scatter(stats_homo['y_hat'], stats_homo['residuals'], alpha=0.6, s=30)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Valeurs ajustées $\\hat{x}_k$')
    ax.set_ylabel('Résidus $\\hat{n}_k$')
    ax.set_title('Homoscédasticité (variance constante)')
    ax.grid(True, alpha=0.3)

    # Hétéroscédasticité
    ax = axes[1]
    stats_hetero = compute_ols_statistics(X, y_hetero)
    ax.scatter(stats_hetero['y_hat'], stats_hetero['residuals'], alpha=0.6, s=30)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Valeurs ajustées $\\hat{x}_k$')
    ax.set_ylabel('Résidus $\\hat{n}_k$')
    ax.set_title('Hétéroscédasticité (variance croissante)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    print("\n  → Cas 1 : Résidus dispersés uniformément (homoscédasticité)")
    print("  → Cas 2 : Résidus en forme d'entonnoir (hétéroscédasticité)")


def demonstrate_outliers_and_influence():
    """
    Démontre la détection de points aberrants et influents.
    """
    print("\n" + "="*70)
    print("DÉMONSTRATION : Points aberrants et influents")
    print("="*70)

    n = 50
    np.random.seed(42)

    # Données normales
    X = np.random.randn(n, 1)
    y = 2 + 3 * X[:, 0] + np.random.randn(n) * 0.5

    # Ajouter un point aberrant (outlier) : résidu élevé mais pas de leverage élevé
    X_outlier = np.vstack([X, [[0.0]]])
    y_outlier = np.append(y, [10.0])  # Valeur aberrante

    # Ajouter un point influent (leverage élevé)
    X_influential = np.vstack([X, [[5.0]]])  # Loin des autres X
    y_influential = np.append(y, [2 + 3 * 5.0])  # Suit le modèle

    # Ajouter un point à la fois aberrant ET influent
    X_both = np.vstack([X, [[5.0]]])
    y_both = np.append(y, [0.0])  # Loin du modèle ET leverage élevé

    # Calculs
    stats_outlier = compute_ols_statistics(X_outlier, y_outlier)
    leverage_outlier, cooks_outlier, _ = compute_leverage_and_cooks_distance(
        X_outlier, y_outlier, stats_outlier['residuals'])

    stats_influential = compute_ols_statistics(X_influential, y_influential)
    leverage_influential, cooks_influential, _ = compute_leverage_and_cooks_distance(
        X_influential, y_influential, stats_influential['residuals'])

    stats_both = compute_ols_statistics(X_both, y_both)
    leverage_both, cooks_both, _ = compute_leverage_and_cooks_distance(
        X_both, y_both, stats_both['residuals'])

    print(f"\nPoint aberrant (résidu élevé, leverage faible) :")
    print(f"  Leverage : {leverage_outlier[-1]:.4f}")
    print(f"  Distance de Cook : {cooks_outlier[-1]:.4f}")
    print(f"  → Aberrant mais peu influent")

    print(f"\nPoint avec leverage élevé (suit le modèle) :")
    print(f"  Leverage : {leverage_influential[-1]:.4f}")
    print(f"  Distance de Cook : {cooks_influential[-1]:.4f}")
    print(f"  → Leverage élevé mais pas aberrant (Cook faible)")

    print(f"\nPoint aberrant ET influent :")
    print(f"  Leverage : {leverage_both[-1]:.4f}")
    print(f"  Distance de Cook : {cooks_both[-1]:.4f}")
    print(f"  → TRÈS INFLUENT (Cook > 0.5)")


def compare_methods():
    """
    Compare OLS, Ridge, LASSO et WLS.
    """
    print("\n" + "="*70)
    print("COMPARAISON : OLS vs Ridge vs LASSO vs WLS")
    print("="*70)

    # Données avec multicolinéarité
    n, p = 100, 8
    np.random.seed(42)

    X = np.random.randn(n, p)
    # Créer multicolinéarité
    X[:, 3] = 0.9 * X[:, 0] + 0.1 * np.random.randn(n)
    X[:, 7] = 0.9 * X[:, 1] + 0.1 * np.random.randn(n)

    true_coef = np.array([2.0, -1.5, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0])
    y = X @ true_coef + np.random.randn(n) * 1.0

    # OLS
    stats_ols = compute_ols_statistics(X, y)
    beta_ols = stats_ols['beta_hat']

    # Ridge
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    beta_ridge = ridge.coef_

    # LASSO
    lasso = Lasso(alpha=0.5)
    lasso.fit(X_scaled, y)
    beta_lasso = lasso.coef_

    print(f"\nVrais coefficients : {true_coef}")
    print(f"\nOLS    : {beta_ols}")
    print(f"Ridge  : {beta_ridge}")
    print(f"LASSO  : {beta_lasso}")
    print(f"\nNombre de coefficients non nuls :")
    print(f"  OLS    : {np.sum(np.abs(beta_ols) > 0.01)} / {p}")
    print(f"  Ridge  : {np.sum(np.abs(beta_ridge) > 0.01)} / {p}")
    print(f"  LASSO  : {np.sum(np.abs(beta_lasso) > 0.01)} / {p}")


def print_inference_results(X, y, true_coef):
    """
    Affiche un tableau complet des résultats d'inférence.
    """
    print("\n" + "="*70)
    print("RÉSULTATS D'INFÉRENCE (Intervalles de confiance et tests)")
    print("="*70)

    stats = compute_ols_statistics(X, y)

    print(f"\nR² = {stats['R2']:.4f}  |  R² ajusté = {stats['R2_adj']:.4f}")
    print(f"Test F : F = {stats['F_stat']:.4f}, p-valeur = {stats['F_pvalue']:.6f}")

    if stats['F_pvalue'] < 0.05:
        print("→ Le modèle est globalement significatif (p < 0.05)")
    else:
        print("→ Le modèle n'est pas globalement significatif (p ≥ 0.05)")

    print(f"\n{'Var':<6} {'Vrai':<8} {'Estimé':<10} {'SE':<8} {'t-stat':<8} {'p-value':<10} {'IC 95%':<25} {'Signif.'}")
    print("-" * 100)

    for j in range(len(stats['beta_hat'])):
        signif = "***" if stats['p_values'][j] < 0.001 else \
                 "**" if stats['p_values'][j] < 0.01 else \
                 "*" if stats['p_values'][j] < 0.05 else ""

        true_val = true_coef[j] if j < len(true_coef) else 0.0

        print(f"X{j:<5} {true_val:<8.2f} {stats['beta_hat'][j]:<10.4f} "
              f"{stats['se_beta'][j]:<8.4f} {stats['t_stats'][j]:<8.4f} "
              f"{stats['p_values'][j]:<10.6f} "
              f"[{stats['ci_lower'][j]:6.3f}, {stats['ci_upper'][j]:6.3f}]  {signif}")

    print("\nSignif. codes: *** p<0.001, ** p<0.01, * p<0.05")


def main():
    """Fonction principale."""

    print("="*70)
    print("Chapitre 6 : Régression Linéaire - Inférence et Diagnostic")
    print("="*70)

    # 1. Générer des données
    print("\n1. Génération de données synthétiques...")
    X, y, true_coef = generate_regression_data(n=100, p=5, noise_std=1.0)
    print(f"   Dimensions : X = {X.shape}, y = {y.shape}")

    # 2. Inférence statistique
    print("\n2. Inférence statistique (intervalles de confiance et tests)")
    print_inference_results(X, y, true_coef)

    # 3. Graphiques de diagnostic
    print("\n3. Génération des graphiques de diagnostic des résidus...")
    stats = compute_ols_statistics(X, y)
    plot_diagnostic_residuals(X, y, stats)

    # 4. Multicolinéarité
    demonstrate_multicollinearity()

    # 5. Hétéroscédasticité
    demonstrate_heteroscedasticity()

    # 6. Points aberrants et influents
    demonstrate_outliers_and_influence()

    # 7. Comparaison des méthodes
    compare_methods()

    print("\n" + "="*70)
    print("Script terminé avec succès !")
    print("="*70)


if __name__ == '__main__':
    main()
