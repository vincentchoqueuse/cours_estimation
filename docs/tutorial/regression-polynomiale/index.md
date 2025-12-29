# Tutoriel : Régression polynomiale

## Introduction

La **régression polynomiale** est une application directe du modèle linéaire présenté au [Chapitre 2](/courses/chapitre2/#exemple-modele-lineaire-et-moindres-carres). L'objectif est d'approximer des données par un polynôme de degré fixé.

## Problématique

Soit $m$ observations $(t_1, x_1), \ldots, (t_m, x_m)$ où $t_k$ sont des abscisses (par exemple des temps) et $x_k$ sont des ordonnées bruités. Nous cherchons à modéliser la relation entre $t$ et $x$ par un polynôme de degré $p$ :

$$
x(t) = s_0 + s_1 t + s_2 t^2 + \cdots + s_p t^p
$$

## Formulation en modèle linéaire

### Construction de la matrice de design

Bien que nous cherchions un polynôme (non linéaire en $t$), le problème est **linéaire en les coefficients** $\mathbf{s} = [s_0, s_1, \ldots, s_p]^T$. Nous pouvons écrire :

$$
\mathbf{x} = \mathbf{A}\mathbf{s} + \mathbf{n}
$$

où :
- $\mathbf{x} = [x_1, \ldots, x_m]^T$ est le vecteur des observations
- $\mathbf{s} = [s_0, s_1, \ldots, s_p]^T$ est le vecteur des coefficients du polynôme ($(p+1)$ paramètres)
- $\mathbf{n}$ est le bruit gaussien
- $\mathbf{A} \in \mathbb{R}^{m \times (p+1)}$ est la **matrice de Vandermonde** :

$$
\mathbf{A} = \begin{bmatrix}
1 & t_1 & t_1^2 & \cdots & t_1^p \\
1 & t_2 & t_2^2 & \cdots & t_2^p \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & t_m & t_m^2 & \cdots & t_m^p
\end{bmatrix}
$$

**Remarque** : Chaque ligne $k$ de $\mathbf{A}$ contient les puissances successives de $t_k$.

### Estimateur des moindres carrés

D'après le chapitre 2, l'estimateur du MLE (qui coïncide avec les moindres carrés) est :

$$
\widehat{\mathbf{s}}_{MLE} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{x}
$$

Le polynôme ajusté est alors :

$$
\widehat{x}(t) = \widehat{s}_0 + \widehat{s}_1 t + \widehat{s}_2 t^2 + \cdots + \widehat{s}_p t^p
$$

## Exemple numérique

### Données

Considérons $m = 20$ observations générées à partir d'un polynôme de degré 3 bruité :

$$
x(t_k) = 1 + 2t_k - 0.5t_k^2 + 0.1t_k^3 + n_k, \quad t_k \in [0, 5]
$$

avec $n_k \sim \mathcal{N}(0, 0.5^2)$.

### Ajustement

Nous cherchons à estimer les coefficients avec un polynôme de degré $p = 3$ (ordre correct).

**Matrice de design** (premières lignes) :

$$
\mathbf{A} = \begin{bmatrix}
1 & 0.00 & 0.00 & 0.00 \\
1 & 0.26 & 0.07 & 0.02 \\
1 & 0.53 & 0.28 & 0.15 \\
\vdots & \vdots & \vdots & \vdots
\end{bmatrix}
$$

**Résultats** : Le script Python ci-dessous génère les données et estime les coefficients.

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/polynomial_regression.png" alt="Régression polynomiale" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 1: Ajustement polynomial de degré 3</p>
</div>

## Choix de l'ordre du polynôme

### Sur-ajustement et sous-ajustement

- **Sous-ajustement** ($p$ trop petit) : Le modèle est trop simple, l'erreur est élevée
- **Bon ajustement** ($p$ correct) : Le modèle capture la tendance des données
- **Sur-ajustement** ($p$ trop grand) : Le modèle colle trop aux données bruitées, mauvaise généralisation

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/polynomial_orders.png" alt="Comparaison des ordres" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 2: Comparaison de différents ordres de polynômes</p>
</div>

### Critère de sélection

Pour choisir $p$, on peut utiliser :
- **Validation croisée** : Séparer données d'entraînement/test
- **Critères d'information** : AIC, BIC qui pénalisent la complexité

## Propriétés de l'estimateur

### Biais et variance

- **Biais** : $\mathbb{E}[\widehat{\mathbf{s}}_{MLE}] = \mathbf{s}$ (sans biais)
- **Matrice de covariance** : $\text{Cov}(\widehat{\mathbf{s}}_{MLE}) = \sigma^2(\mathbf{A}^T\mathbf{A})^{-1}$

**Remarque** : La matrice $\mathbf{A}^T\mathbf{A}$ peut être mal conditionnée pour $p$ grand, rendant l'inversion numérique instable.

### Intervalle de confiance

Pour chaque coefficient $s_j$, un intervalle de confiance asymptotique à 95% est :

$$
IC_{0.95}(s_j) = \left[\widehat{s}_j - 1.96 \sqrt{[\text{Cov}(\widehat{\mathbf{s}})]_{jj}}, \widehat{s}_j + 1.96 \sqrt{[\text{Cov}(\widehat{\mathbf{s}})]_{jj}}\right]
$$

## Exercices

1. Générer des données polynomiales de degré 2 et ajuster des polynômes de degrés 1, 2, 3, 5
2. Comparer l'erreur quadratique moyenne pour chaque ordre
3. Étudier l'effet du nombre d'observations $m$ sur la qualité de l'estimation
4. Investiguer le conditionnement de $\mathbf{A}^T\mathbf{A}$ en fonction de $p$

## Code Python

Le script complet est disponible dans `src/polynomial_regression.py`.
