# Chapitre 2 : Estimateurs Ponctuels

Dans ce chapitre, nous nous interessons au estimateur ponctuel <Cite refKey="kay1993" short /> <Cite refKey="casella2002" short />. Un **estimateur ponctuel** est une **règle de décision** qui associe aux données observées **une seule valeur** pour estimer un paramètre inconnu.

### Définition formelle

Soit un vecteur de $l$ paramètres inconnus $\boldsymbol\theta$ et un vecteur d’observations $\mathbf{x}=[x_1, \cdots, x_n]$ aléatoire.
Un **estimateur ponctuel** est une variable aléatoire

$$
\widehat{\boldsymbol \theta} = \mathbf{g}(\mathbf{x})
$$

où $\mathbf{g}(\cdot): \mathbb{R}^n \to \mathbb{R}^l$ est une fonction des données.

Une fois les données observées $\mathbf{x}$, l'estimateur prend une **valeur numérique unique** $\widehat{\boldsymbol \theta}$. Dans ce chapitre, nous proposons plusieurs techniques pour construire la fonction $\mathbf{g}(\cdot)$.

## Méthode des moments

La méthode des moments est l'une des techniques les plus anciennes et les plus simples pour construire des estimateurs <Cite refKey="casella2002" />. Elle consiste à égaler les **moments théoriques** de la distribution aux **moments empiriques** calculés à partir des données.

### Principe

Soit $\boldsymbol\theta = [\theta_1, \ldots, \theta_l]$ un vecteur de $l$ paramètres inconnus à estimer. Notons $m_u(\boldsymbol\theta)$ le **moment théorique** d'ordre $u$ :

$$
m_u(\boldsymbol\theta) = \mathbb{E}[x^u]
$$

où l'espérance dépend des paramètres $\boldsymbol\theta$.

Le **moment empirique** d'ordre $u$ est calculé à partir des observations $\mathbf{x} = [x_1, \ldots, x_n]$ :

$$
\widehat{m}_u = \frac{1}{n}\sum_{k=1}^{n} x_k^u
$$

### Estimateur de la méthode des moments

Pour estimer $l$ paramètres, nous résolvons le système d'équations suivant :

$$
\begin{cases}
m_1(\boldsymbol\theta) = \widehat{m}_1 \\
m_2(\boldsymbol\theta) = \widehat{m}_2 \\
\vdots \\
m_l(\boldsymbol\theta) = \widehat{m}_l
\end{cases}
$$

L'**estimateur de la méthode des moments** $\widehat{\boldsymbol\theta}_{MM}$ est la solution de ce système.

### Exemple : Loi normale

Considérons l'estimation des paramètres $\boldsymbol\theta = [\mu, \sigma^2]$ d'une loi normale $\mathcal{N}(\mu, \sigma^2)$ à partir d'un échantillon i.i.d. $\mathbf{x} = [x_1, \ldots, x_n]$.

**Moments théoriques** :

- Premier moment : $m_1(\mu, \sigma^2) = \mathbb{E}[x] = \mu$
- Deuxième moment : $m_2(\mu, \sigma^2) = \mathbb{E}[x^2] = \mu^2 + \sigma^2$

**Moments empiriques** :

- $\widehat{m}_1 = \frac{1}{n}\sum_{k=1}^{n} x_k$
- $\widehat{m}_2 = \frac{1}{n}\sum_{k=1}^{n} x_k^2$

**Système d'équations** :

$$
\begin{cases}
\mu = \widehat{m}_1 \\
\mu^2 + \sigma^2 = \widehat{m}_2
\end{cases}
$$

**Solution** : Les estimateurs de la méthode des moments sont :

$$
\begin{align}
\widehat{\mu}_{MM} &= \frac{1}{n}\sum_{k=1}^{n} x_k \\
\widehat{\sigma}^2_{MM} &= \frac{1}{n}\sum_{k=1}^{n} x_k^2 - \left(\frac{1}{n}\sum_{k=1}^{n} x_k\right)^2 = \frac{1}{n}\sum_{k=1}^{n} (x_k - \widehat{\mu}_{MM})^2
\end{align}
$$

où la dernière expression s'obtient en simplifiant le developpement du carré.

### Avantages et inconvénients

**Avantages** :

- Méthode simple et intuitive
- Ne nécessite pas de connaître la forme complète de la distribution
- Toujours applicable dès que les moments existent

**Inconvénients** :

- Peut donner des estimateurs biaisés
- N'exploite pas toute l'information disponible dans les données
- Peut être moins efficace que d'autres méthodes (comme le MLE)

## Méthode du maximum de vraisemblance

La méthode du maximum de vraisemblance (maximum likelihood) <Cite refKey="kay1993" /> <Cite refKey="lehmann1998" short /> consiste à construire un estimateur le plus vraisemblable au sens des données observées $\mathbf{x}$.

### Principe

Pour un échantillon $\mathbf{x}=[x_1, \ldots, x_n]$ i.i.d. de densité de probabilité $f(x; \boldsymbol \theta)$, l'**estimateur du maximum de vraisemblance** $\widehat{\boldsymbol \theta}_{MLE}$ maximise la fonction de vraisemblance :

$$
\widehat{\boldsymbol \theta}_{MLE} = \arg\max_{\theta \in \Theta} L(\boldsymbol \theta)
$$

où la **vraisemblance** est définie par :

$$
L(\boldsymbol \theta; x_1, \ldots, x_n) = \prod_{k=1}^{n} f(x_k; \boldsymbol \theta)
$$

#### Remarques

- En pratique, sans perte de généralité, au lieu de maximiser directement la fonction de vraisemblance, il est souvent préferable de maximiser la **log-vraisemblance**. Dans le cas d'un grand nombre de densité de probabilité (notamment la loi gaussienne), l'expression du log-vraisemblance est nettement plus simple à manipuler. La log-vraisemblance est définie par

$$
\ell(\boldsymbol \theta) = \log L(\boldsymbol \theta) = \sum_{k=1}^{n} \log f(x_k; \boldsymbol \theta)
$$

- Pour trouver le maximum de la log-vraisemblance, on cherche les valeurs de $\boldsymbol \theta$ qui annulent le **gradient** :

$$
\nabla_{\boldsymbol \theta} \ell(\boldsymbol \theta) = \begin{bmatrix} \frac{\partial \ell}{\partial \theta_1} \\ \vdots \\ \frac{\partial \ell}{\partial \theta_l} \end{bmatrix} = \mathbf{0}
$$

Dans le cas d'un paramètre **scalaire** $\theta$, on résout simplement $\frac{d\ell(\theta)}{d\theta} = 0$ (dérivée ordinaire).

Dans certains cas, il est possible de trouver une solution analytique. Dans la majorité des cas, il est nécessaire de recourir à des algorithmes d'optimisation numérique.

### Exemple : Loi normale

Considérons l'estimation des paramètres $\boldsymbol \theta=[\mu, \sigma^2]$ à partir d'un échantillon i.i.d. $\mathbf{x}=[x_1, \ldots, x_n]$ avec $x_n \sim \mathcal{N}(\mu, \sigma^2)$. Pour estimer les paramètres, nous allons calculer la log-vraisemblance. Dans ce cas de figure, la log vraisemblance s'exprime sous la forme

$$
\ell(\mu, \sigma^2) = -\frac{n}{2} \log(2\pi) - \frac{n}{2} \log(\sigma^2) - \frac{1}{2\sigma^2} \sum_{k=1}^{n} (x_k - \mu)^2
$$

Les MLE des paramètrees $\boldsymbol \theta=[\mu, \sigma^2]$ sont :

$$
\begin{align}
\widehat{\mu}_{MLE}&= \frac{1}{n} \sum_{k=1}^{n} x_k, \\
\widehat{\sigma}^2_{MLE} &= \frac{1}{n} \sum_{k=1}^{n} (x_k - \widehat{\mu}_{MLE})^2
\end{align}
$$

### Exemple : Modèle paramétrique et moindres carrés

Considérons un **modèle paramétrique** où l'on observe un vecteur $\mathbf{x} = [x_1, \ldots, x_m]^T \in \mathbb{R}^m$ généré par :

$$
x_k = s_k(\boldsymbol\theta) + n_k, \quad k = 1, \ldots, m
$$

ou de façon vectorielle :

$$
\mathbf{x} = \mathbf{s}(\boldsymbol\theta) + \mathbf{n}
$$

où :

- $\mathbf{s}(\boldsymbol\theta) = [s_1(\boldsymbol\theta), \ldots, s_m(\boldsymbol\theta)]^T$ est un vecteur de **fonctions paramétriques** (connues)
- $\boldsymbol\theta \in \mathbb{R}^p$ est le vecteur de paramètres à estimer (inconnu)
- $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_m)$ est un bruit gaussien de moyenne nulle et de matrice de covariance $\sigma^2 \mathbf{I}_m$

Sous ce modèle, les observations suivent une loi normale multivariée $\mathbf{x} \sim \mathcal{N}(\mathbf{s}(\boldsymbol\theta), \sigma^2 \mathbf{I}_m)$ de densité :

$$
f(\mathbf{x}; \boldsymbol\theta, \sigma^2) = \frac{1}{(2\pi\sigma^2)^{m/2}} \exp\left(-\frac{1}{2\sigma^2} \|\mathbf{x} - \mathbf{s}(\boldsymbol\theta)\|^2\right)
$$

**Log-vraisemblance** :

$$
\ell(\boldsymbol\theta, \sigma^2) = -\frac{m}{2}\log(2\pi) - \frac{m}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\|\mathbf{x} - \mathbf{s}(\boldsymbol\theta)\|^2_2
$$

Sous l'hypothèse de bruit gaussien, maximiser la log-vraisemblance par rapport à $\boldsymbol\theta$ revient à minimiser :

$$
\widehat{\boldsymbol\theta}_{MLE} = \arg\min_{\boldsymbol\theta} \|\mathbf{x} - \mathbf{s}(\boldsymbol\theta)\|^2_2 = \arg\min_{\boldsymbol\theta} \sum_{k=1}^{m} (x_k - s_k(\boldsymbol\theta))^2
$$

Ce critère correspond aux **moindres carrés non linéaires**. En général, il n'existe pas de solution analytique et il faut recourir à des **algorithmes d'optimisation** (gradient, Newton, Levenberg-Marquardt, etc.).

#### Cas particulier : modèle linéaire

Lorsque $\mathbf{s}(\boldsymbol\theta)$ est une fonction **linéaire** des paramètres, c'est-à-dire :

$$
\mathbf{s}(\boldsymbol\theta) = \mathbf{A}\boldsymbol\theta
$$

où $\mathbf{A} \in \mathbb{R}^{m \times p}$ est une matrice de design connue.Ce problème admet une **solution analytique** :

$$
\widehat{\boldsymbol\theta}_{MLE} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{x} = \mathbf{A}^\dagger\mathbf{x}
$$

où $\mathbf{A}^\dagger$ est la pseudo-inverse de $\mathbf{A}$. Cette solution est appelée **estimateur des moindres carrés ordinaires** (Ordinary Least Squares, OLS).

::: tip Approfondissement
Le cas particulier du modèle linéaire, ses propriétés, son analyse détaillée et ses extensions sont présentés dans le [Chapitre 4](/courses/chapitre4/) consacré à la régression linéaire.
:::

### Propriétés

L'estimateur du maximum de vraisemblance possède des propriétés remarquables qui en font l'une des méthodes d'estimation les plus utilisées en statistique. Ces propriétés sont principalement **asymptotiques**, c'est-à-dire qu'elles se manifestent lorsque la taille de l'échantillon $n$ tend vers l'infini.

#### Consistance

Sous certaines conditions de régularité, l'estimateur du MLE est **consistant** : il converge en probabilité vers la vraie valeur du paramètre lorsque $n \to \infty$ :

$$
\widehat{\boldsymbol \theta}_{MLE} \xrightarrow{P} \boldsymbol \theta
$$

**Interprétation** : Avec un échantillon suffisamment grand, l'estimateur du MLE s'approche arbitrairement de la vraie valeur du paramètre.

#### Normalité asymptotique

L'estimateur du MLE suit asymptotiquement une **loi normale** :

$$
\sqrt{n}(\widehat{\boldsymbol \theta}_{MLE} - \boldsymbol \theta) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \mathbf{I}^{-1}(\boldsymbol \theta))
$$

où $\mathbf{I}(\boldsymbol \theta)$ est la **matrice d'information de Fisher** définie par :

$$
\mathbf{I}(\boldsymbol \theta) = -\mathbb{E}\left[\nabla^2_{\boldsymbol \theta} \ell(\boldsymbol \theta)\right] = \mathbb{E}\left[\left(\nabla_{\boldsymbol \theta} \ell(\boldsymbol \theta)\right)\left(\nabla_{\boldsymbol \theta} \ell(\boldsymbol \theta)\right)^T\right]
$$

Dans le cas **scalaire** ($\theta$ unidimensionnel), l'information de Fisher est :

$$
I(\theta) = -\mathbb{E}\left[\frac{d^2 \ell(\theta)}{d\theta^2}\right] = \mathbb{E}\left[\left(\frac{d \ell(\theta)}{d\theta}\right)^2\right]
$$

**Interprétation** : Pour $n$ grand, l'estimateur du MLE se comporte approximativement comme une variable normale de moyenne $\boldsymbol \theta$ et de variance $\frac{1}{n}\mathbf{I}^{-1}(\boldsymbol \theta)$.

#### Efficacité asymptotique

L'estimateur du MLE atteint asymptotiquement la **borne de Cramér-Rao** <Cite refKey="cramer1946" short /> : parmi tous les estimateurs sans biais, le MLE a asymptotiquement la variance minimale <Cite refKey="kay1993" />.

$$
\text{var}(\widehat{\boldsymbol \theta}_{MLE}) \approx \frac{1}{n\mathbf{I}(\boldsymbol \theta)}
$$

**Interprétation** : Le MLE est asymptotiquement **optimal** au sens de la variance : aucun autre estimateur sans biais ne peut avoir une variance plus faible pour un échantillon de grande taille.

#### Invariance par reparamétrisation

Si $\widehat{\boldsymbol \theta}_{MLE}$ est l'estimateur du MLE de $\boldsymbol \theta$, alors pour toute fonction $g$ (suffisamment régulière), $g(\widehat{\boldsymbol \theta}_{MLE})$ est l'estimateur du MLE de $g(\boldsymbol \theta)$ :

$$
\widehat{g(\boldsymbol \theta)}_{MLE} = g(\widehat{\boldsymbol \theta}_{MLE})
$$

**Exemple** : Si $\widehat{\sigma}^2_{MLE}$ est le MLE de $\sigma^2$, alors $\sqrt{\widehat{\sigma}^2_{MLE}}$ est le MLE de $\sigma$.

**Interprétation** : Cette propriété est très pratique car elle permet d'estimer facilement des transformations de paramètres sans avoir à recalculer le MLE dans le nouvel espace paramétrique.

::: tip Remarque
Ces propriétés asymptotiques font du MLE une méthode d'estimation très puissante, particulièrement pour les grands échantillons. Cependant, pour de petits échantillons, le MLE peut être biaisé ou avoir une variance élevée.
:::

### Intervalles de confiance

Grâce à la propriété de normalité asymptotique, nous pouvons construire des **intervalles de confiance** pour les paramètres estimés par MLE.

#### Cas scalaire

Pour un paramètre **scalaire** $\theta$, nous avons asymptotiquement :

$$
\widehat{\theta}_{MLE} \sim \mathcal{N}\left(\theta, \frac{1}{nI(\theta)}\right)
$$

En standardisant, nous obtenons :

$$
\frac{\widehat{\theta}_{MLE} - \theta}{\sqrt{\frac{1}{nI(\theta)}}} \sim \mathcal{N}(0, 1)
$$

Un **intervalle de confiance asymptotique** de niveau $1-\alpha$ pour $\theta$ est :

$$
IC_{1-\alpha}(\theta) = \left[\widehat{\theta}_{MLE} - z_{\alpha/2}\sqrt{\frac{1}{nI(\widehat{\theta}_{MLE})}}, \widehat{\theta}_{MLE} + z_{\alpha/2}\sqrt{\frac{1}{nI(\widehat{\theta}_{MLE})}}\right]
$$

où $z_{\alpha/2}$ est le quantile d'ordre $1-\alpha/2$ de la loi normale centrée réduite $\mathcal{N}(0,1)$.

**Remarque** : En pratique, nous remplaçons $I(\theta)$ par $I(\widehat{\theta}_{MLE})$ car $\theta$ est inconnu. C'est ce qu'on appelle l'approche **plug-in**.

#### Cas multidimensionnel

Pour un vecteur de paramètres $\boldsymbol \theta = [\theta_1, \ldots, \theta_l]^T$, un intervalle de confiance pour le $j$-ième coefficient est :

$$
IC_{1-\alpha}(\theta_j) = \left[\widehat{\theta}_j - z_{\alpha/2}\sqrt{[\mathbf{I}^{-1}(\widehat{\boldsymbol \theta}_{MLE})]_{jj}/n}, \widehat{\theta}_j + z_{\alpha/2}\sqrt{[\mathbf{I}^{-1}(\widehat{\boldsymbol \theta}_{MLE})]_{jj}/n}\right]
$$

où $[\mathbf{I}^{-1}]_{jj}$ désigne l'élément diagonal $(j,j)$ de la matrice d'information inverse.

### Avantages et inconvénients

**Avantages** :

- Exploite toute l'information disponible dans les données (efficacité)
- Propriétés asymptotiques optimales : convergence, normalité asymptotique, efficacité asymptotique
- Invariant par reparamétrisation : si $\widehat{\boldsymbol\theta}_{MLE}$ est le MLE de $\boldsymbol\theta$, alors $g(\widehat{\boldsymbol\theta}_{MLE})$ est le MLE de $g(\boldsymbol\theta)$
- Fournit une approche systématique pour construire des estimateurs

**Inconvénients** :

- Nécessite de connaître la forme complète de la distribution (densité de probabilité)
- Peut être difficile à calculer (pas toujours de solution analytique)
- Peut donner des estimateurs biaisés pour de petits échantillons

<Bibliography :keys="['kay1993', 'lehmann1998', 'casella2002', 'cramer1946']" />
