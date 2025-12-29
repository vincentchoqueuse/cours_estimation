# Chapitre 3 : Estimation Bayésienne

## Introduction

L'**approche bayésienne** de l'estimation statistique <Cite refKey="casella2002" /> repose sur une vision fondamentalement différente de celle présentée au [Chapitre 2](/courses/chapitre2/). Alors que l'approche **fréquentiste** considère les paramètres comme des quantités fixes (mais inconnues), l'approche **bayésienne** traite les paramètres comme des **variables aléatoires** dont on cherche à inférer la distribution à partir des données observées.

### Comparaison des paradigmes

| Aspect                 | Approche Fréquentiste           | Approche Bayésienne                                     |
| ---------------------- | ------------------------------- | ------------------------------------------------------- |
| Paramètre $\theta$     | Quantité fixe inconnue          | Variable aléatoire                                      |
| Données $\mathbf{x}$   | Variables aléatoires            | Observées (fixes une fois collectées)                   |
| Estimation             | Point estimé $\widehat{\theta}$ | Distribution _a posteriori_ $p(\theta \mid \mathbf{x})$ |
| Incertitude            | Intervalles de confiance        | Intervalles de crédibilité                              |
| Information _a priori_ | Non utilisée                    | Incorporée via la loi _a priori_                        |

::: tip Philosophie bayésienne
L'approche bayésienne permet d'incorporer des **connaissances a priori** sur les paramètres (expertise du domaine, résultats d'études antérieures) et de les combiner avec les données observées pour obtenir une distribution **a posteriori** qui quantifie l'incertitude sur les paramètres.
:::

## Théorème de Bayes

### Formulation générale

Le **théorème de Bayes** est le fondement mathématique de l'inférence bayésienne. Pour un paramètre $\theta$ et des observations $\mathbf{x}$, il s'énonce :

$$
p(\theta \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \theta) p(\theta)}{p(\mathbf{x})}
$$

où :

- $p(\theta)$ est la **loi a priori** : notre connaissance sur $\theta$ **avant** d'observer les données
- $p(\mathbf{x} \mid \theta)$ est la **vraisemblance** : probabilité d'observer $\mathbf{x}$ sachant $\theta$
- $p(\mathbf{x})$ est l'**évidence** (ou vraisemblance marginale) : $p(\mathbf{x}) = \int p(\mathbf{x} \mid \theta) p(\theta) d\theta$
- $p(\theta \mid \mathbf{x})$ est la **loi a posteriori** : notre connaissance sur $\theta$ **après** avoir observé les données

### Forme proportionnelle

En pratique, on utilise souvent la forme proportionnelle :

$$
p(\theta \mid \mathbf{x}) \propto p(\mathbf{x} \mid \theta) p(\theta)
$$

car l'évidence $p(\mathbf{x})$ est une constante de normalisation (indépendante de $\theta$).

**Interprétation** : La loi _a posteriori_ combine la vraisemblance (information des données) et la loi _a priori_ (information préalable).

## Lois a priori

Le choix de la loi _a priori_ est crucial en statistique bayésienne. Il existe plusieurs types de lois _a priori_ selon le niveau d'information disponible.

### Lois a priori informatives

Les **lois a priori informatives** reflètent une connaissance substantielle sur le paramètre avant d'observer les données. Par exemple :

- Si l'on sait que $\theta$ est proche de 5, on peut choisir $\theta \sim \mathcal{N}(5, 1)$
- Elles incorporent expertise du domaine ou résultats d'études antérieures

### Lois a priori faiblement informatives

Les **lois a priori faiblement informatives** expriment une certaine connaissance mais restent vagues. Par exemple :

- $\theta \sim \mathcal{N}(0, 100)$ pour un paramètre sans information précise
- Elles laissent les données dominer l'inférence

### Lois a priori non informatives

Les **lois a priori non informatives** (ou vagues) traduisent l'absence totale de connaissance _a priori_. Par exemple :

- **Loi uniforme** : $p(\theta) \propto 1$ (constante) sur un intervalle
- **Loi de Jeffreys** : $p(\theta) \propto \sqrt{I(\theta)}$ où $I(\theta)$ est l'information de Fisher

**Remarque** : Avec une loi _a priori_ non informative, l'estimateur bayésien converge souvent vers l'estimateur du maximum de vraisemblance.

## Estimateurs bayésiens

L'approche bayésienne fournit la **loi _a posteriori_** $p(\theta \mid \mathbf{x})$, qui décrit toute l'information disponible sur le paramètre $\theta$ après observation des données. Cependant, en pratique, il est souvent nécessaire de fournir une **estimation ponctuelle** (une valeur unique) du paramètre plutôt qu'une distribution complète.

Plusieurs estimateurs ponctuels peuvent être dérivés de la loi _a posteriori_, chacun optimisant un critère différent.

### Estimateur du maximum a posteriori (MAP)

L'**estimateur MAP** (Maximum A Posteriori) est la valeur de $\theta$ qui maximise la loi _a posteriori_ :

$$
\widehat{\theta}_{MAP} = \arg\max_{\theta} p(\theta \mid \mathbf{x}) = \arg\max_{\theta} \left[p(\mathbf{x} \mid \theta) p(\theta)\right]
$$

En prenant le logarithme :

$$
\widehat{\theta}_{MAP} = \arg\max_{\theta} \left[\log p(\mathbf{x} \mid \theta) + \log p(\theta)\right]
$$

**Lien avec le MLE** : Avec une loi _a priori_ uniforme ($p(\theta) \propto 1$), l'estimateur MAP coïncide avec le MLE :

$$
\widehat{\theta}_{MAP} = \widehat{\theta}_{MLE} \quad \text{si } p(\theta) \propto 1
$$

### Estimateur de l'espérance a posteriori (EAP)

L'**estimateur EAP** (Expected A Posteriori) est l'espérance de la loi _a posteriori_ :

$$
\widehat{\theta}_{EAP} = \mathbb{E}[\theta \mid \mathbf{x}] = \int \theta \, p(\theta \mid \mathbf{x}) \, d\theta
$$

**Propriété** : Cet estimateur minimise l'erreur quadratique moyenne _a posteriori_ :

$$
\widehat{\theta}_{EAP} = \arg\min_{\widehat{\theta}} \mathbb{E}\left[(\theta - \widehat{\theta})^2 \mid \mathbf{x}\right]
$$

### Estimateur médian a posteriori

L'**estimateur médian** est la médiane de la loi _a posteriori_, qui minimise l'erreur absolue moyenne :

$$
\widehat{\theta}_{med} = \arg\min_{\widehat{\theta}} \mathbb{E}\left[|\theta - \widehat{\theta}| \mid \mathbf{x}\right]
$$

## Exemples

### Exemple 1 : Moyenne d'une loi normale avec variance connue

Soit $\mathbf{x} = [x_1, \ldots, x_n]$ un échantillon i.i.d. avec $x_k \sim \mathcal{N}(\mu, \sigma^2)$ où $\sigma^2$ est **connue**.

**Loi a priori** : Choisissons une loi normale $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

**Vraisemblance** :

$$
p(\mathbf{x} \mid \mu) = \prod_{k=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_k - \mu)^2}{2\sigma^2}\right) \propto \exp\left(-\frac{n(\bar{x} - \mu)^2}{2\sigma^2}\right)
$$

où $\bar{x} = \frac{1}{n}\sum_{k=1}^{n} x_k$

**Loi a posteriori** : La loi normale est conjuguée, donc $\mu \mid \mathbf{x} \sim \mathcal{N}(\mu_n, \sigma_n^2)$ avec :

$$
\begin{align}
\mu_n &= \frac{\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}} = \frac{\sigma^2 \mu_0 + n\sigma_0^2 \bar{x}}{\sigma^2 + n\sigma_0^2} \\
\sigma_n^2 &= \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}} = \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n\sigma_0^2}
\end{align}
$$

**Interprétation** :

- $\mu_n$ est une **moyenne pondérée** entre la moyenne _a priori_ $\mu_0$ et la moyenne empirique $\bar{x}$
- Quand $n \to \infty$ : $\mu_n \to \bar{x}$ (les données dominent)
- Quand $\sigma_0^2 \to \infty$ (loi _a priori_ non informative) : $\mu_n \to \bar{x}$

**Estimateurs** :

- MAP : $\widehat{\mu}_{MAP} = \mu_n$ (mode de la gaussienne)
- EAP : $\widehat{\mu}_{EAP} = \mu_n$ (espérance de la gaussienne)

**Intervalle de crédibilité à 95%** :

$$
IC_{95\%} = \left[\mu_n - 1.96\sigma_n, \mu_n + 1.96\sigma_n\right]
$$

### Exemple 2 : Proportion d'une loi de Bernoulli

Soit $\mathbf{x} = [x_1, \ldots, x_n]$ avec $x_k \sim \text{Bernoulli}(\theta)$ (succès/échecs).

**Loi a priori** : Loi Bêta $\theta \sim \text{Beta}(\alpha, \beta)$ (conjuguée)

$$
p(\theta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha-1} (1-\theta)^{\beta-1}
$$

**Vraisemblance** : Avec $s = \sum_{k=1}^{n} x_k$ (nombre de succès) :

$$
p(\mathbf{x} \mid \theta) = \theta^s (1-\theta)^{n-s}
$$

**Loi a posteriori** : $\theta \mid \mathbf{x} \sim \text{Beta}(\alpha + s, \beta + n - s)$

**Estimateurs** :

- MAP : $\widehat{\theta}_{MAP} = \frac{\alpha + s - 1}{\alpha + \beta + n - 2}$ (pour $\alpha, \beta > 1$)
- EAP : $\widehat{\theta}_{EAP} = \frac{\alpha + s}{\alpha + \beta + n}$

**Cas particulier** : Avec une loi _a priori_ uniforme $\text{Beta}(1, 1)$ :

$$
\widehat{\theta}_{EAP} = \frac{s + 1}{n + 2}, \quad \widehat{\theta}_{MAP} = \frac{s}{n} = \widehat{\theta}_{MLE}
$$

### Exemple 3 : Modèle paramétrique général avec loi a priori gaussienne

Considérons le modèle paramétrique général introduit au [Chapitre 2](/courses/chapitre2/#exemple-modele-parametrique-et-moindres-carres) :

$$
\mathbf{x} = \mathbf{s}(\boldsymbol\theta) + \mathbf{n}, \quad \mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_m)
$$

**Loi a priori** : Supposons $\boldsymbol\theta \sim \mathcal{N}(\boldsymbol\mu_0, \boldsymbol\Sigma_0)$

**Log-vraisemblance** :

$$
\log p(\mathbf{x} \mid \boldsymbol\theta) \propto -\frac{1}{2\sigma^2} \|\mathbf{x} - \mathbf{s}(\boldsymbol\theta)\|^2
$$

**Log-probabilité a priori** :

$$
\log p(\boldsymbol\theta) \propto -\frac{1}{2}(\boldsymbol\theta - \boldsymbol\mu_0)^T \boldsymbol\Sigma_0^{-1} (\boldsymbol\theta - \boldsymbol\mu_0)
$$

**Estimateur MAP** : Maximiser la log-probabilité _a posteriori_ :

$$
\begin{align}
\widehat{\boldsymbol\theta}_{MAP} &= \arg\max_{\boldsymbol\theta} \left[\log p(\mathbf{x} \mid \boldsymbol\theta) + \log p(\boldsymbol\theta)\right] \\
&= \arg\min_{\boldsymbol\theta} \left[\|\mathbf{x} - \mathbf{s}(\boldsymbol\theta)\|^2 + \frac{\sigma^2}{1}(\boldsymbol\theta - \boldsymbol\mu_0)^T \boldsymbol\Sigma_0^{-1} (\boldsymbol\theta - \boldsymbol\mu_0)\right]
\end{align}
$$

**Interprétation** : Le terme $(\boldsymbol\theta - \boldsymbol\mu_0)^T \boldsymbol\Sigma_0^{-1} (\boldsymbol\theta - \boldsymbol\mu_0)$ agit comme un **terme de régularisation** qui pénalise les valeurs de $\boldsymbol\theta$ éloignées de $\boldsymbol\mu_0$.

#### Cas linéaire : Régularisation de Tikhonov

Pour le modèle linéaire $\mathbf{s}(\boldsymbol\theta) = \mathbf{A}\boldsymbol\theta$ avec une loi _a priori_ $\boldsymbol\theta \sim \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})$ (centrée) :

$$
\widehat{\boldsymbol\theta}_{MAP} = \arg\min_{\boldsymbol\theta} \left[\|\mathbf{x} - \mathbf{A}\boldsymbol\theta\|^2 + \lambda \|\boldsymbol\theta\|^2\right]
$$

avec $\lambda = \sigma^2/\tau^2$. C'est la **régularisation de Tikhonov** (ou régression ridge), qui a une solution analytique :

$$
\widehat{\boldsymbol\theta}_{MAP} = (\mathbf{A}^T\mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^T \mathbf{x}
$$

::: tip Lien avec la régularisation
L'approche bayésienne fournit une **justification probabiliste** aux méthodes de régularisation :

- **Ridge** ($L^2$) = loi _a priori_ gaussienne $\mathcal{N}(0, \tau^2 I)$
- **LASSO** ($L^1$) = loi _a priori_ de Laplace

Ces méthodes sont développées au [Chapitre 5](/courses/chapitre5/#regularisation) dans le contexte de la régression linéaire.
:::

## Méthodes de calcul

Pour des modèles complexes, la loi _a posteriori_ n'a pas de forme analytique simple. On utilise alors des **méthodes numériques**.

### Approximation de Laplace

L'**approximation de Laplace** consiste à approcher la loi _a posteriori_ par une gaussienne centrée en $\widehat{\theta}_{MAP}$ :

$$
p(\theta \mid \mathbf{x}) \approx \mathcal{N}\left(\widehat{\theta}_{MAP}, \left[-\nabla^2 \log p(\theta \mid \mathbf{x})\Big|_{\theta=\widehat{\theta}_{MAP}}\right]^{-1}\right)
$$

### Méthodes de Monte Carlo par chaînes de Markov (MCMC)

Les **méthodes MCMC** (Markov Chain Monte Carlo) permettent d'échantillonner la loi _a posteriori_ sans la calculer explicitement :

- **Algorithme de Metropolis-Hastings** : génère une chaîne de Markov qui converge vers la loi _a posteriori_
- **Échantillonneur de Gibbs** : cas particulier pour les lois conditionnelles connues
- **Hamiltonian Monte Carlo (HMC)** : utilise le gradient pour explorer efficacement l'espace

**Outils modernes** : Des logiciels comme **Stan**, **PyMC**, **JAGS** facilitent l'inférence bayésienne avec MCMC.

::: tip Remarque
Les méthodes MCMC sont devenues incontournables en statistique bayésienne moderne, permettant de traiter des modèles très complexes (modèles hiérarchiques, apprentissage profond bayésien, etc.).
:::

## Avantages et inconvénients de l'approche bayésienne

### Avantages

1. **Incorporation de connaissance a priori** : Permet d'intégrer expertise et études antérieures
2. **Interprétation probabiliste naturelle** : La loi _a posteriori_ quantifie directement l'incertitude sur $\theta$
3. **Flexibilité** : S'applique à des modèles très complexes (hiérarchiques, non paramétriques)
4. **Prédiction naturelle** : Distribution prédictive _a posteriori_ pour de nouvelles observations
5. **Régularisation** : Les lois _a priori_ agissent comme régularisateurs, évitant le surapprentissage

### Inconvénients

1. **Choix de la loi a priori** : Subjectif et peut influencer les résultats (surtout avec peu de données)
2. **Coût de calcul** : Nécessite souvent des méthodes numériques (MCMC) coûteuses
3. **Interprétation** : Nécessite d'adopter le paradigme bayésien (probabilité subjective)
4. **Convergence** : Les méthodes MCMC nécessitent de vérifier la convergence des chaînes

<Bibliography :keys="['casella2002', 'kay1993']" />
