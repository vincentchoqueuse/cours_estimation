# Chapitre 4 : Régression Linéaire - Fondements

## Introduction

La **régression linéaire** est l'une des méthodes statistiques les plus fondamentales et les plus utilisées <Cite refKey="kay1993" /> <Cite refKey="casella2002" short />. Elle permet de modéliser la relation entre une variable dépendante (ou variable à expliquer) et une ou plusieurs variables indépendantes (ou variables explicatives). Nous avons vu au [Chapitre 2](/courses/chapitre2/#exemple-modele-parametrique-et-moindres-carres) que sous l'hypothèse de bruit gaussien, l'estimateur du maximum de vraisemblance coïncide avec l'estimateur des moindres carrés. Dans ce chapitre, nous approfondissons l'étude de ce modèle en analysant ses propriétés, ses applications et ses extensions.

## Modèle de régression linéaire

### Formulation générale

Considérons un ensemble de $m$ observations. Le **modèle de régression linéaire** s'écrit :

$$
\mathbf{x} = \mathbf{A}\mathbf{s} + \mathbf{n}
$$

où :

- $\mathbf{x} = [x_1, \ldots, x_m]^T \in \mathbb{R}^m$ est le vecteur des **observations** (variable dépendante)
- $\mathbf{A} \in \mathbb{R}^{m \times p}$ est la **matrice de design** (variables explicatives)
- $\mathbf{s} = [s_1, \ldots, s_p]^T \in \mathbb{R}^p$ est le vecteur des **paramètres** (coefficients de régression)
- $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_m)$ est le vecteur des **erreurs** (bruit gaussien)

**Hypothèses du modèle** :

1. **Linéarité** : La relation entre $\mathbf{x}$ et $\mathbf{A}$ est linéaire en $\mathbf{s}$
2. **Indépendance** : Les erreurs $n_k$ sont indépendantes
3. **Homoscédasticité** : Les erreurs ont toutes la même variance $\sigma^2$
4. **Normalité** : Les erreurs suivent une loi normale $\mathcal{N}(0, \sigma^2)$
5. **Non-colinéarité** : Les colonnes de $\mathbf{A}$ sont linéairement indépendantes (rang plein)

### Interprétation des composantes

Pour chaque observation $k = 1, \ldots, m$, nous avons :

$$
x_k = \sum_{j=1}^{p} A_{kj} s_j + n_k
$$

où $A_{kj}$ est la valeur de la $j$-ième variable explicative pour l'observation $k$.

## Estimateur des moindres carrés

### Dérivation

L'**estimateur des moindres carrés ordinaires** (Ordinary Least Squares, OLS) minimise la somme des carrés des résidus :

$$
\widehat{\mathbf{s}}_{OLS} = \arg\min_{\mathbf{s}} \|\mathbf{x} - \mathbf{A}\mathbf{s}\|^2_2 = \arg\min_{\mathbf{s}} \sum_{k=1}^{m} (x_k - \sum_{j=1}^{p} A_{kj} s_j)^2
$$

En développant le critère et en annulant le gradient, nous obtenons les **équations normales** :

$$
\mathbf{A}^T\mathbf{A}\mathbf{s} = \mathbf{A}^T\mathbf{x}
$$

Si $\mathbf{A}$ est de rang plein (i.e., $\text{rang}(\mathbf{A}) = p$ avec $m \geq p$), alors $\mathbf{A}^T\mathbf{A}$ est inversible et la solution unique est :

$$
\widehat{\mathbf{s}}_{OLS} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{x} = \mathbf{A}^\dagger\mathbf{x}
$$

où $\mathbf{A}^\dagger = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$ est la **pseudo-inverse de Moore-Penrose** de $\mathbf{A}$.

### Interprétation géométrique

L'estimateur OLS projette orthogonalement le vecteur des observations $\mathbf{x}$ sur le sous-espace engendré par les colonnes de $\mathbf{A}$. Le vecteur des **valeurs ajustées** (fitted values) est :

$$
\widehat{\mathbf{x}} = \mathbf{A}\widehat{\mathbf{s}}_{OLS} = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{x} = \mathbf{P}\mathbf{x}
$$

où $\mathbf{P} = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$ est la **matrice de projection** sur l'espace des colonnes de $\mathbf{A}$.

Le vecteur des **résidus** est :

$$
\widehat{\mathbf{n}} = \mathbf{x} - \widehat{\mathbf{x}} = \mathbf{x} - \mathbf{P}\mathbf{x} = (\mathbf{I} - \mathbf{P})\mathbf{x}
$$

**Propriété remarquable** : Les résidus sont orthogonaux aux valeurs ajustées : $\widehat{\mathbf{n}}^T\widehat{\mathbf{x}} = 0$.

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/projection_geometrique.png" alt="Interprétation géométrique OLS" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 1: Interprétation géométrique de l'estimateur OLS comme projection orthogonale</p>
</div>

## Propriétés de l'estimateur OLS

### Sans biais

L'estimateur OLS est **sans biais** :

$$
\mathbb{E}[\widehat{\mathbf{s}}_{OLS}] = \mathbb{E}[(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{x}] = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbb{E}[\mathbf{x}]
$$

Puisque $\mathbb{E}[\mathbf{x}] = \mathbf{A}\mathbf{s}$ (car $\mathbb{E}[\mathbf{n}] = \mathbf{0}$), nous obtenons :

$$
\mathbb{E}[\widehat{\mathbf{s}}_{OLS}] = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{A}\mathbf{s} = \mathbf{s}
$$

### Matrice de covariance

La **matrice de covariance** de l'estimateur OLS est :

$$
\text{Cov}(\widehat{\mathbf{s}}_{OLS}) = \sigma^2(\mathbf{A}^T\mathbf{A})^{-1}
$$

**Démonstration** :

$$
\begin{align}
\text{Cov}(\widehat{\mathbf{s}}_{OLS}) &= \text{Cov}((\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{x}) \\
&= (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T \text{Cov}(\mathbf{x}) \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1} \\
&= (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T (\sigma^2 \mathbf{I}) \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1} \\
&= \sigma^2(\mathbf{A}^T\mathbf{A})^{-1}
\end{align}
$$

**Conséquence** : La variance du $j$-ième coefficient est :

$$
\text{var}(\widehat{s}_j) = \sigma^2 [(\mathbf{A}^T\mathbf{A})^{-1}]_{jj}
$$

### Théorème de Gauss-Markov

Le **théorème de Gauss-Markov** <Cite refKey="casella2002" /> établit que, parmi tous les estimateurs **linéaires** et **sans biais**, l'estimateur OLS a la **variance minimale**.

**Énoncé** : Soit $\widetilde{\mathbf{s}}$ un estimateur linéaire sans biais quelconque de $\mathbf{s}$. Alors :

$$
\text{Cov}(\widetilde{\mathbf{s}}) - \text{Cov}(\widehat{\mathbf{s}}_{OLS}) \succeq 0
$$

où $\mathbf{B} \succeq 0$ signifie que $\mathbf{B}$ est semi-définie positive.

En d'autres termes, l'estimateur OLS est **BLUE** (Best Linear Unbiased Estimator) : le meilleur estimateur linéaire sans biais.

::: tip Remarque importante
Le théorème de Gauss-Markov ne nécessite **pas** l'hypothèse de normalité des erreurs. Il suffit que les erreurs soient non corrélées et de variance constante.
:::

### Loi de l'estimateur

Sous l'hypothèse de normalité des erreurs $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$, l'estimateur OLS suit une **loi normale multivariée** :

$$
\widehat{\mathbf{s}}_{OLS} \sim \mathcal{N}\left(\mathbf{s}, \sigma^2(\mathbf{A}^T\mathbf{A})^{-1}\right)
$$

Par conséquent, chaque coefficient $\widehat{s}_j$ suit une loi normale :

$$
\widehat{s}_j \sim \mathcal{N}\left(s_j, \sigma^2 [(\mathbf{A}^T\mathbf{A})^{-1}]_{jj}\right)
$$

## Estimation de la variance des erreurs

### Estimateur non biaisé de $\sigma^2$

La variance des erreurs $\sigma^2$ est généralement inconnue. Nous l'estimons par :

$$
\widehat{\sigma}^2 = \frac{1}{m-p} \sum_{k=1}^{m} \widehat{n}_k^2 = \frac{1}{m-p} \|\mathbf{x} - \mathbf{A}\widehat{\mathbf{s}}_{OLS}\|^2
$$

où $m - p$ est le nombre de **degrés de liberté** (nombre d'observations moins nombre de paramètres).

**Propriété** : Cet estimateur est sans biais : $\mathbb{E}[\widehat{\sigma}^2] = \sigma^2$.

### Somme des carrés et décomposition

La somme totale des carrés peut se décomposer :

$$
\underbrace{\sum_{k=1}^{m} (x_k - \bar{x})^2}_{\text{SST (Total)}} = \underbrace{\sum_{k=1}^{m} (\widehat{x}_k - \bar{x})^2}_{\text{SSE (Expliquée)}} + \underbrace{\sum_{k=1}^{m} (x_k - \widehat{x}_k)^2}_{\text{SSR (Résiduelle)}}
$$

où $\bar{x} = \frac{1}{m}\sum_{k=1}^{m} x_k$ est la moyenne empirique des observations.

## Coefficient de détermination $R^2$

### Définition

Le **coefficient de détermination** $R^2$ mesure la proportion de la variance de $\mathbf{x}$ expliquée par le modèle :

$$
R^2 = \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\sum_{k=1}^{m} (x_k - \widehat{x}_k)^2}{\sum_{k=1}^{m} (x_k - \bar{x})^2}
$$

**Interprétation** :

- $R^2 = 1$ : Le modèle explique parfaitement les données (ajustement parfait)
- $R^2 = 0$ : Le modèle n'explique aucune variance (pas mieux qu'une simple moyenne)
- $0 < R^2 < 1$ : Le modèle explique partiellement les données

### Coefficient de détermination ajusté

Le $R^2$ ajusté pénalise l'ajout de variables explicatives :

$$
R^2_{adj} = 1 - \frac{(1-R^2)(m-1)}{m-p}
$$

Cette version corrige le fait que $R^2$ augmente mécaniquement avec le nombre de paramètres, même si les variables ajoutées n'apportent pas d'information significative.

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/r2_interpretation.png" alt="Interprétation du R²" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 2: Décomposition de la variance et interprétation du R²</p>
</div>

<Bibliography :keys="['kay1993', 'casella2002', 'wasserman2004']" />
