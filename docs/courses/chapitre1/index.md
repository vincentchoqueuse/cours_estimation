# Chapitre 1 : Concepts de base

## Introduction

L'estimation statistique vise à déterminer la valeur de paramètres inconnus à partir des données observées d'un échantillon <Cite refKey="kay1993" short /> <Cite refKey="casella2002" short />. Pour déterminer la valeur des paramètres, les observations sont décrites à l'aide d'un **modèle statistique**, c'est-à-dire d'une loi de probabilité dépendant des paramètres d'intérêt.
L'estimation consiste alors à exploiter conjointement les données et ce modèle paramétré afin de fournir une approximation des paramètres inconnus.

Dans ce chapitre, nous considérons en particulier un exemple très simple: l'estimation de la moyenne d'un échantillon issu d'une loi gaussienne.

### Modèle statistique

Un **modèle statistique** est un ensemble de distributions de probabilité $\mathcal{P} = \{P_\theta : \theta \in \Theta\}$ indexé par un paramètre $\theta$ appartenant à un espace de paramètres $\Theta$.

#### Exemple : Loi normale (gaussienne)

**Définition** : Une variable aléatoire $X$ suit une **loi normale** (ou gaussienne) $\mathcal{N}(\mu, \sigma^2)$ de paramètres $\mu \in \mathbb{R}$ (moyenne) et $\sigma^2 > 0$ (variance) si elle admet pour densité de probabilité :

$$
f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad x \in \mathbb{R}
$$

**Propriétés** :

- $\mathbb{E}[X] = \mu$ (moyenne ou espérance)
- $\text{var}(X) = \sigma^2$ (variance)

**Visualisations** :

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/gaussienne_combinee.png" alt="Loi normale avec différents paramètres" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 1: Loi normale avec différents paramètres μ et σ²</p>
</div>

Soit $\mathbf{x}=[x_1, \ldots, x_n]$ un échantillon i.i.d. composé de $n$ éléments issus d'une loi normale $\mathcal{N}(\mu, \sigma^2)$. Le modèle statistique est :

$$
\mathcal{P} = \{\mathcal{N}(\mu, \sigma^2) : (\mu, \sigma^2) \in \mathbb{R} \times \mathbb{R}_+^*\}
$$

### Estimateur

Un **estimateur** de $\theta$ est une statistique

$$\widehat{\theta}_n = g(\mathbf{x})$$

qui approxime le paramètre $\theta$ à partir de l'échantillon $\mathbf{x}$. Par convention, l'estimateur est indiqué avec un $\widehat{.}$

## Propriétés des estimateurs

Pour évaluer la qualité d'un estimateur, nous étudions plusieurs propriétés fondamentales. Ces propriétés permettent de comparer différents estimateurs et de choisir le plus approprié selon le contexte.

### Biais

Le **biais** d'un estimateur $\widehat{\theta}_n$ mesure l'écart entre sa valeur moyenne et la vraie valeur du paramètre. Il est défini par :

$$
\text{Biais}(\widehat{\theta}_n) = \mathbb{E}[\widehat{\theta}_n] - \theta
$$

Un estimateur est dit **sans biais** si $\text{Biais}(\widehat{\theta}_n) = 0$ pour tout $\theta$, c'est-à-dire si $\mathbb{E}[\widehat{\theta}_n] = \theta$. Cela signifie qu'en moyenne, l'estimateur "vise juste".

### Variance

La **variance** d'un estimateur $\widehat{\theta}_n$ mesure sa variabilité autour de sa moyenne :

$$
\text{var}(\widehat{\theta}_n) = \mathbb{E}\left[(\widehat{\theta}_n - \mathbb{E}[\widehat{\theta}_n])^2\right]
$$

La variance quantifie la dispersion des estimations d'un échantillon à l'autre. Un estimateur peut être sans biais mais avoir une variance élevée (peu fiable), ou avoir un biais non nul mais une variance faible (précis mais biaisé).

### Efficacité

Parmi les estimateurs sans biais, celui de **variance minimale** est dit **efficace**.

La **borne de Cramér-Rao** <Cite refKey="cramer1946" short /> établit une limite inférieure pour la variance de tout estimateur sans biais <Cite refKey="kay1993" /> : sous certaines conditions de régularité,

$$
\text{var}(\widehat{\theta}_n) \geq \frac{1}{nI(\theta)}
$$

où $I(\theta)$ est l'information de Fisher. Un estimateur qui atteint cette borne est **optimal** au sens de la variance <Cite refKey="lehmann1998" short />.

### Convergence

Un estimateur est dit **convergent** s'il se rapproche du paramètre vrai lorsque la taille de l'échantillon augmente. Formellement, $\widehat{\theta}_n$ **converge en probabilité** vers $\theta$ si :

$$
\forall \varepsilon > 0, \quad \lim_{n \to \infty} P(|\widehat{\theta}_n - \theta| > \varepsilon) = 0
$$

On note : $\widehat{\theta}_n \xrightarrow{P} \theta$

Pour un échantillon suffisamment grand, la probabilité que l'estimateur soit éloigné de la vraie valeur devient arbitrairement petite. C'est une propriété **asymptotique** essentielle.

### Erreur quadratique moyenne (EQM)

Le biais et la variance évaluent des aspects distincts de la qualité d'un estimateur. L'**erreur quadratique moyenne** (EQM ou MSE en anglais) est un critère **global** qui combine ces deux aspects :

$$
\text{MSE}(\widehat{\theta}_n) = \mathbb{E}\left[(\widehat{\theta}_n - \theta)^2\right]
$$

**Décomposition biais-variance** : Cette quantité peut s'exprimer de manière remarquable comme :

$$
\text{MSE}(\widehat{\theta}_n) = \text{var}(\widehat{\theta}_n) + \left[\text{Biais}(\widehat{\theta}_n)\right]^2
$$

L'EQM représente l'erreur moyenne au carré entre l'estimateur et la vraie valeur. Cette décomposition montre qu'il existe un **compromis biais-variance** :

- Un estimateur peut avoir un biais non nul mais une variance faible
- À l'inverse, un estimateur sans biais peut avoir une variance élevée
- L'objectif est de **minimiser l'EQM totale**, ce qui peut parfois justifier d'accepter un petit biais pour réduire significativement la variance

### Intervalles de confiance

Les estimateurs ponctuels (comme $\widehat{\theta}_n$) fournissent une seule valeur pour approximer le paramètre inconnu. Cependant, du fait de la variabilité aléatoire des données, cette estimation est incertaine. Les **intervalles de confiance** permettent de quantifier cette incertitude en fournissant un intervalle qui contient le paramètre avec une probabilité contrôlée.

Un **intervalle de confiance** de niveau $1-\alpha$ pour $\theta$ est un intervalle aléatoire $[L_n, U_n]$ (qui dépend des données) tel que :

$$
P_\theta(L_n \leq \theta \leq U_n) \geq 1 - \alpha, \quad \forall \theta \in \Theta
$$

**Interprétation** : Le nombre $1-\alpha$ (typiquement 90%, 95% ou 99%) est appelé **niveau de confiance**. Cela signifie que si nous répétons l'expérience un grand nombre de fois, environ $(1-\alpha) \times 100\%$ des intervalles construits contiendront la vraie valeur de $\theta$.

## Exemple : Estimation de la moyenne

Pour un échantillon $\mathbf{x}=[x_1, \ldots, x_n]$ i.i.d. de moyenne $\mu$, un estimateur naturel de $\mu$ est la moyenne empirique définie par :

$$
\widehat{\mu}_n = g(\mathbf{x})= \frac{1}{n} \sum_{k=1}^{n} x_k
$$

**Propriétés** :

- Sans biais : $\mathbb{E}[\widehat{\mu}_n] = \mu$, donc $\text{Biais}(\widehat{\mu}_n) = 0$
- Variance : $\text{var}(\widehat{\mu}_n) = \frac{\sigma^2}{n}$
- Erreur quadratique moyenne :
  $$
  \text{MSE}(\widehat{\mu}_n) = \text{var}(\widehat{\mu}_n) + [\text{Biais}(\widehat{\mu}_n)]^2 = \frac{\sigma^2}{n} + 0 = \frac{\sigma^2}{n}
  $$
- Convergence : $\widehat{\mu}_n \xrightarrow{P} \mu$ (loi des grands nombres)

**Interprétation** : L'EQM décroît en $1/n$, ce qui signifie que la qualité de l'estimation s'améliore lorsque la taille de l'échantillon augmente. La figure suivante présente l'évolution de l'EQM en fonction de $n$ lorsque $\sigma^2=4$.
Nous observons que l'EQM diminue rapidement au début puis plus lentement. Pour diviser l'EQM par 2, il faut multiplier la taille d'échantillon par 4.

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/mse_vs_n.png" alt="Évolution de l'EQM en fonction de n" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 2: Évolution de l'erreur quadratique moyenne en fonction de la taille d'échantillon n</p>
</div>

<Bibliography :keys="['kay1993', 'lehmann1998', 'casella2002', 'cramer1946']" />
