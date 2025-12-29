# Chapitre 5 : Régularisation en Régression Linéaire

## Introduction

Dans le [Chapitre 4](/courses/chapitre4/), nous avons étudié l'estimateur des moindres carrés ordinaires (OLS) pour la régression linéaire. Bien que l'OLS possède d'excellentes propriétés théoriques (BLUE, sans biais), il présente des **limitations pratiques** importantes :

1. **Surapprentissage** : Lorsque le nombre de variables $p$ est grand par rapport au nombre d'observations $m$, l'OLS peut ajuster parfaitement les données d'entraînement mais mal généraliser
2. **Instabilité** : Quand les variables sont corrélées (multicolinéarité), les coefficients estimés deviennent instables
3. **Variance élevée** : Pour $p$ proche de $m$, la variance des estimateurs explose

La **régularisation** est une famille de techniques qui ajoutent une **pénalité** au critère des moindres carrés pour contrôler la complexité du modèle et améliorer ses performances de généralisation.

### Problème général de régularisation

Au lieu de minimiser simplement la somme des carrés des résidus, nous minimisons :

$$
\widehat{\boldsymbol\theta}_{\text{reg}} = \arg\min_{\boldsymbol\theta} \left[\|\mathbf{x} - \mathbf{A}\boldsymbol\theta\|^2_2 + \lambda \Omega(\boldsymbol\theta)\right]
$$

où :

- $\|\mathbf{x} - \mathbf{A}\boldsymbol\theta\|^2_2$ est le **terme d'attache aux données** (fidélité)
- $\Omega(\boldsymbol\theta)$ est le **terme de régularisation** (pénalité)
- $\lambda \geq 0$ est le **paramètre de régularisation** qui contrôle le compromis

**Interprétation** : La régularisation favorise les solutions qui :

- Ajustent bien les données (premier terme)
- Respectent certaines contraintes de simplicité (second terme)

## Compromis biais-variance

### Décomposition

L'erreur de prédiction d'un modèle se décompose en trois termes :

$$
\mathbb{E}\left[(y - \widehat{y})^2\right] = \text{Biais}^2 + \text{Variance} + \text{Bruit irréductible}
$$

où :

- **Biais** : Écart systématique entre les prédictions moyennes et les vraies valeurs
- **Variance** : Sensibilité du modèle aux variations dans les données d'entraînement
- **Bruit irréductible** : Variance du bruit $\sigma^2$

### Compromis

- **Modèle simple** (forte régularisation) : Biais élevé, variance faible
- **Modèle complexe** (faible régularisation) : Biais faible, variance élevée
- **Objectif** : Trouver le bon équilibre pour minimiser l'erreur totale

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/bias_variance_tradeoff.png" alt="Compromis biais-variance" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 1: Compromis biais-variance en fonction de la complexité du modèle</p>
</div>

::: tip Intuition
La régularisation **augmente légèrement le biais** mais **réduit fortement la variance**, ce qui améliore les performances de généralisation.
:::

## Régularisation L2 (Ridge)

### Définition

La **régression ridge** ajoute une pénalité sur la **norme L2** des coefficients :

$$
\widehat{\boldsymbol\theta}_{\text{ridge}} = \arg\min_{\boldsymbol\theta} \left[\|\mathbf{x} - \mathbf{A}\boldsymbol\theta\|^2_2 + \lambda \|\boldsymbol\theta\|^2_2\right]
$$

avec $\|\boldsymbol\theta\|^2_2 = \sum_{j=1}^{p} \theta_j^2$.

**Effet** : Pénalise les coefficients de grande amplitude, favorisant des solutions avec des coefficients plus petits et plus stables.

### Solution analytique

La régression ridge admet une **solution explicite** :

$$
\widehat{\boldsymbol\theta}_{\text{ridge}} = (\mathbf{A}^T\mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^T \mathbf{x}
$$

**Remarques** :

- Pour $\lambda = 0$ : on retrouve l'OLS $(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{x}$
- Pour $\lambda > 0$ : la matrice $\mathbf{A}^T\mathbf{A} + \lambda \mathbf{I}$ est toujours inversible (même si $\mathbf{A}^T\mathbf{A}$ ne l'est pas)
- Le terme $\lambda \mathbf{I}$ **stabilise** l'inversion en ajoutant une valeur sur la diagonale

### Interprétation bayésienne

Comme vu au [Chapitre 3](/courses/chapitre3/#cas-lineaire-regularisation-de-tikhonov), la régression ridge correspond à un estimateur MAP avec une loi _a priori_ gaussienne :

$$
\boldsymbol\theta \sim \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})
$$

avec $\lambda = \sigma^2/\tau^2$. La régularisation L2 traduit donc une **croyance a priori** que les coefficients sont centrés autour de zéro.

### Propriétés

1. **Réduction de la variance** : Ridge diminue la variance des estimateurs au prix d'un léger biais
2. **Stabilité** : Les coefficients sont plus stables face à de petites perturbations des données
3. **Pas de sélection de variables** : Ridge ne met jamais de coefficients exactement à zéro
4. **Coefficients groupés** : Les variables corrélées tendent à avoir des coefficients similaires

### Chemin de régularisation

L'évolution des coefficients $\widehat{\theta}_j(\lambda)$ en fonction de $\lambda$ est appelée **chemin de régularisation** (regularization path). Pour ridge :

- $\lambda = 0$ : coefficients OLS (potentiellement grands)
- $\lambda \to \infty$ : tous les coefficients tendent vers zéro
- Trajectoires **continues** et monotones

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/ridge_path.png" alt="Chemin Ridge" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 2: Trajectoire des coefficients ridge en fonction de log(λ)</p>
</div>

## Régularisation L1 (LASSO)

### Définition

Le **LASSO** (Least Absolute Shrinkage and Selection Operator) utilise une pénalité **L1** :

$$
\widehat{\boldsymbol\theta}_{\text{LASSO}} = \arg\min_{\boldsymbol\theta} \left[\|\mathbf{x} - \mathbf{A}\boldsymbol\theta\|^2_2 + \lambda \|\boldsymbol\theta\|_1\right]
$$

avec $\|\boldsymbol\theta\|_1 = \sum_{j=1}^{p} |\theta_j|$.

**Effet** : Met certains coefficients **exactement à zéro**, réalisant ainsi une **sélection automatique de variables**.

### Pas de solution analytique

Contrairement à ridge, le LASSO n'a **pas de solution explicite** en raison de la non-différentiabilité de $|\theta_j|$ en $\theta_j = 0$. On utilise des algorithmes d'optimisation :

- **Coordinate descent** : optimise séquentiellement chaque coefficient
- **LARS** (Least Angle Regression) : construit le chemin de régularisation efficacement
- **Proximal gradient** : méthodes d'optimisation convexe avec opérateur proximal

### Interprétation bayésienne

Le LASSO correspond à un estimateur MAP avec une loi _a priori_ de **Laplace** (double exponentielle) :

$$
p(\theta_j) \propto \exp\left(-\frac{|\theta_j|}{\tau}\right)
$$

Cette loi a des queues plus lourdes que la gaussienne, favorisant des coefficients nuls ou grands (sparse).

### Propriétés

1. **Sélection de variables** : Met automatiquement certains coefficients à zéro
2. **Parcimonie** : Produit des modèles **parcimonieux** (sparse) avec peu de variables actives
3. **Instabilité de sélection** : En présence de variables corrélées, LASSO peut arbitrairement en choisir une
4. **Limitation $p > m$** : LASSO sélectionne au maximum $m$ variables (pas plus que d'observations)

### Chemin de régularisation

Pour LASSO :

- $\lambda = 0$ : solution OLS (si $p < m$)
- $\lambda$ croissant : coefficients mis progressivement à zéro
- Trajectoires **linéaires par morceaux** avec des points de rupture où des variables entrent/sortent

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/lasso_path.png" alt="Chemin LASSO" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 3: Trajectoire des coefficients LASSO - notez les mises à zéro abruptes</p>
</div>

### Soft-thresholding

L'opérateur de mise à jour pour coordinate descent est le **soft-thresholding** :

$$
\widehat{\theta}_j = \text{sign}(\widetilde{\theta}_j) \max\left(|\widetilde{\theta}_j| - \lambda, 0\right)
$$

où $\widetilde{\theta}_j$ est la solution OLS partielle. Cet opérateur "pousse" les coefficients vers zéro.

## Régularisation L0

### Définition

La **régularisation L0** pénalise le **nombre de coefficients non nuls** :

$$
\widehat{\boldsymbol\theta}_{\text{L0}} = \arg\min_{\boldsymbol\theta} \left[\|\mathbf{x} - \mathbf{A}\boldsymbol\theta\|^2_2 + \lambda \|\boldsymbol\theta\|_0\right]
$$

avec $\|\boldsymbol\theta\|_0 = \#\{j : \theta_j \neq 0\}$ (nombre de composantes non nulles).

**Objectif** : Trouver le sous-ensemble optimal de variables qui minimise l'erreur.

### Problème NP-difficile

La régularisation L0 est un **problème combinatoire NP-difficile** :

- Il faut tester toutes les combinaisons de variables (il y en a $2^p$)
- Infaisable pour $p$ grand (typiquement $p > 30$)

### Algorithmes approchés

#### 1. Recherche exhaustive

Pour $p$ petit, on peut énumérer tous les sous-ensembles de taille $k$ :

$$
\binom{p}{k} = \frac{p!}{k!(p-k)!}
$$

et choisir celui qui minimise l'erreur (critère AIC, BIC, validation croisée).

#### 2. Forward selection (sélection séquentielle avant)

**Algorithme glouton** :

1. Commencer avec un modèle vide
2. À chaque étape, ajouter la variable qui améliore le plus le critère
3. Arrêter selon un critère (AIC, BIC, validation)

**Complexité** : $O(p^2)$ pour sélectionner $k$ variables

#### 3. Backward elimination (élimination séquentielle arrière)

**Algorithme glouton** :

1. Commencer avec toutes les variables
2. À chaque étape, retirer la variable qui dégrade le moins le critère
3. Arrêter selon un critère

#### 4. Stepwise selection

Combinaison de forward et backward : à chaque étape, on peut ajouter ou retirer une variable.

### Critères de sélection

Pour choisir le nombre de variables, on utilise des critères qui pénalisent la complexité :

#### AIC (Akaike Information Criterion)

$$
\text{AIC} = m \log\left(\frac{\text{RSS}}{m}\right) + 2k
$$

où RSS est la somme des carrés des résidus et $k$ est le nombre de paramètres.

#### BIC (Bayesian Information Criterion)

$$
\text{BIC} = m \log\left(\frac{\text{RSS}}{m}\right) + k \log(m)
$$

**Différence** : BIC pénalise plus fortement la complexité que AIC ($\log(m) > 2$ pour $m > 8$).

### Relaxation convexe : du L0 au L1

La norme L0 est **non convexe** (discontinue). LASSO est la **relaxation convexe** la plus proche :

$$
\|\boldsymbol\theta\|_0 \approx \|\boldsymbol\theta\|_1
$$

Pour $\lambda$ suffisamment grand, LASSO peut trouver la solution L0 optimale.

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/lp_norms.png" alt="Normes Lp" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 4: Boules unités pour les normes L0, L1, L2</p>
</div>

## Elastic Net

### Définition

**Elastic Net** combine les pénalités L1 et L2 :

$$
\widehat{\boldsymbol\theta}_{\text{EN}} = \arg\min_{\boldsymbol\theta} \left[\|\mathbf{x} - \mathbf{A}\boldsymbol\theta\|^2_2 + \lambda_1 \|\boldsymbol\theta\|_1 + \lambda_2 \|\boldsymbol\theta\|^2_2\right]
$$

Souvent reparamétrisé avec un paramètre de mélange $\alpha \in [0,1]$ :

$$
\widehat{\boldsymbol\theta}_{\text{EN}} = \arg\min_{\boldsymbol\theta} \left[\|\mathbf{x} - \mathbf{A}\boldsymbol\theta\|^2_2 + \lambda \left(\alpha \|\boldsymbol\theta\|_1 + (1-\alpha) \|\boldsymbol\theta\|^2_2\right)\right]
$$

avec :

- $\alpha = 1$ : LASSO pur
- $\alpha = 0$ : Ridge pur
- $0 < \alpha < 1$ : combinaison

### Motivation

Elastic Net hérite des **avantages des deux méthodes** :

- **Sélection de variables** (grâce à L1)
- **Stabilité et groupement** (grâce à L2)

Particulièrement utile quand :

- $p \gg m$ : LASSO est limité à $m$ variables, Elastic Net non
- Variables corrélées : LASSO en choisit arbitrairement une, Elastic Net les groupe

### Propriété de groupement

Quand deux variables $j$ et $k$ sont fortement corrélées, Elastic Net tend à leur donner des coefficients similaires :

$$
|\widehat{\theta}_j - \widehat{\theta}_k| \leq C \cdot (1-\alpha)
$$

Cette propriété est absente de LASSO.

## Validation croisée et choix de λ

### Problème

Comment choisir le paramètre de régularisation $\lambda$ ?

**Objectif** : Minimiser l'**erreur de généralisation** (pas l'erreur d'entraînement).

### Validation croisée k-fold

**Procédure** :

1. Diviser les données en $K$ plis (folds) de taille égale
2. Pour chaque valeur de $\lambda$ candidate :
   - Pour chaque pli $k = 1, \ldots, K$ :
     - Entraîner le modèle sur $K-1$ plis
     - Prédire sur le pli $k$ restant
     - Calculer l'erreur $e_k(\lambda)$
   - Calculer l'erreur moyenne : $\text{CV}(\lambda) = \frac{1}{K}\sum_{k=1}^{K} e_k(\lambda)$
3. Choisir $\widehat{\lambda} = \arg\min_{\lambda} \text{CV}(\lambda)$

**Valeurs typiques** : $K = 5$ ou $K = 10$

### Règle du "one standard error"

Plutôt que de choisir le $\lambda$ qui minimise exactement l'erreur CV, on peut choisir le **plus grand** $\lambda$ tel que :

$$
\text{CV}(\lambda) \leq \min_{\lambda'} \text{CV}(\lambda') + \text{SE}
$$

où SE est l'erreur standard de l'erreur CV.

**Avantage** : Modèle plus parcimonieux (moins de variables) avec une erreur similaire.

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/cv_error.png" alt="Erreur de validation croisée" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 5: Erreur CV en fonction de log(λ) avec intervalles de confiance</p>
</div>

## Comparaison des méthodes

| Méthode         | Sélection | Groupement | Solution   | Convexe | $p \gg m$ |
| --------------- | --------- | ---------- | ---------- | ------- | --------- |
| **OLS**         | Non       | Oui        | Analytique | Oui     | Non       |
| **Ridge (L2)**  | Non       | Oui        | Analytique | Oui     | Oui       |
| **LASSO (L1)**  | Oui       | Non        | Numérique  | Oui     | Limité    |
| **Elastic Net** | Oui       | Oui        | Numérique  | Oui     | Oui       |
| **L0**          | Oui       | -          | NP-dur     | Non     | Oui       |

### Guide pratique

**Utiliser Ridge quand** :

- Toutes les variables sont potentiellement pertinentes
- On veut stabiliser les coefficients (multicolinéarité)
- On a besoin d'une solution analytique rapide

**Utiliser LASSO quand** :

- On veut sélectionner automatiquement des variables
- On suspecte que seules quelques variables sont importantes
- On veut un modèle interprétable (parcimonieux)

**Utiliser Elastic Net quand** :

- On veut sélectionner des variables ET gérer la corrélation
- $p \gg m$ (plus de variables que d'observations)
- On a des groupes de variables corrélées

**Utiliser L0 (greedy) quand** :

- $p$ est petit ($p < 30$)
- On veut exactement $k$ variables
- On peut se permettre le coût computationnel

## Implémentation pratique

### Normalisation des variables

**Crucial** : Avant d'appliquer ridge ou LASSO, normaliser les variables :

$$
A_{kj} \leftarrow \frac{A_{kj} - \bar{A}_j}{\sigma_j}
$$

**Raison** : La pénalité doit être invariante à l'échelle des variables. Sans normalisation, une variable mesurée en km serait pénalisée différemment que la même mesurée en m.

### Grille de λ

Tester une grille logarithmique :

$$
\lambda_k = \lambda_{\max} \cdot \left(\frac{\lambda_{\min}}{\lambda_{\max}}\right)^{(k-1)/(K-1)}, \quad k = 1, \ldots, K
$$

avec typiquement $K = 100$ valeurs entre $\lambda_{\min}$ et $\lambda_{\max}$.

Pour LASSO : $\lambda_{\max} = \max_j |\mathbf{A}_j^T \mathbf{x}|$ (tous les coefficients sont nuls).

### Warm start

Pour calculer le chemin de régularisation, utiliser le **warm start** :

- Initialiser avec la solution pour $\lambda_{k-1}$ lors de l'optimisation pour $\lambda_k$
- Accélère considérablement le calcul (10-100×)

## Exemple numérique

### Données synthétiques

Générons des données avec :

- $m = 100$ observations
- $p = 50$ variables
- Seulement 10 variables vraiment actives
- Corrélation entre certaines variables

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/regularization_comparison.png" alt="Comparaison des méthodes" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 6: Comparaison Ridge, LASSO, Elastic Net sur données synthétiques</p>
</div>

**Résultats** :

- **Ridge** : tous les coefficients non nuls mais petits
- **LASSO** : 12 coefficients non nuls (sélection correcte + 2 faux positifs)
- **Elastic Net** : 11 coefficients non nuls, meilleur compromis
