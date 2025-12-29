# Aide-mémoire : Formules essentielles

## Estimateurs classiques

### Moyenne empirique

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^{n} X_i
$$

**Propriétés** :
- $\mathbb{E}[\bar{X}_n] = \mu$ (sans biais)
- $\text{Var}(\bar{X}_n) = \frac{\sigma^2}{n}$
- $\bar{X}_n \xrightarrow{P} \mu$ (loi des grands nombres)
- $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{\mathcal{L}} \mathcal{N}(0, \sigma^2)$ (TCL)

### Variance empirique

$$
S_n^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X}_n)^2
$$

**Propriétés** :
- $\mathbb{E}[S_n^2] = \sigma^2$ (sans biais)
- $S_n^2 \xrightarrow{P} \sigma^2$

---

## Méthodes d'estimation

### Méthode des moments

Égaler les moments théoriques et empiriques :

$$
\mathbb{E}[X^k] = \frac{1}{n} \sum_{i=1}^{n} X_i^k, \quad k = 1, 2, \ldots
$$

### Maximum de vraisemblance

**Vraisemblance** :
$$
L(\theta) = \prod_{i=1}^{n} f(X_i; \theta)
$$

**Log-vraisemblance** :
$$
\ell(\theta) = \sum_{i=1}^{n} \log f(X_i; \theta)
$$

**MLE** :
$$
\hat{\theta}_{MLE} = \arg\max_{\theta} \ell(\theta)
$$

Résoudre : $\frac{\partial \ell(\theta)}{\partial \theta} = 0$

---

## Propriétés des estimateurs

### Biais

$$
\text{Biais}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta
$$

### Erreur quadratique moyenne (MSE)

$$
\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2] = \text{Var}(\hat{\theta}) + [\text{Biais}(\hat{\theta})]^2
$$

### Information de Fisher

$$
I(\theta) = \mathbb{E}\left[\left(\frac{\partial \log f(X; \theta)}{\partial \theta}\right)^2\right] = -\mathbb{E}\left[\frac{\partial^2 \log f(X; \theta)}{\partial \theta^2}\right]
$$

### Borne de Cramér-Rao

Pour tout estimateur sans biais $\hat{\theta}$ :

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{n I(\theta)}
$$

---

## Intervalles de confiance

### IC pour la moyenne (variance connue)

Si $X_i \sim \mathcal{N}(\mu, \sigma^2)$ avec $\sigma^2$ connue :

$$
IC_{1-\alpha}(\mu) = \left[\bar{X}_n - z_{\alpha/2} \frac{\sigma}{\sqrt{n}}, \bar{X}_n + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}\right]
$$

### IC pour la moyenne (variance inconnue)

Si $X_i \sim \mathcal{N}(\mu, \sigma^2)$ avec $\sigma^2$ inconnue :

$$
IC_{1-\alpha}(\mu) = \left[\bar{X}_n - t_{n-1, \alpha/2} \frac{S_n}{\sqrt{n}}, \bar{X}_n + t_{n-1, \alpha/2} \frac{S_n}{\sqrt{n}}\right]
$$

### IC asymptotique (général)

Si $\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{\mathcal{L}} \mathcal{N}(0, \sigma^2(\theta))$ :

$$
IC_{1-\alpha}(\theta) = \left[\hat{\theta}_n - z_{\alpha/2} \frac{\hat{\sigma}_n}{\sqrt{n}}, \hat{\theta}_n + z_{\alpha/2} \frac{\hat{\sigma}_n}{\sqrt{n}}\right]
$$

---

## Lois de probabilité usuelles

| Loi | Densité/Masse | $\mathbb{E}[X]$ | $\text{Var}(X)$ |
|-----|---------------|-----------------|-----------------|
| $\mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| $\mathcal{E}(\lambda)$ | $\lambda e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |
| $\text{Poisson}(\lambda)$ | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ |
| $\text{Bernoulli}(p)$ | $p^x (1-p)^{1-x}$ | $p$ | $p(1-p)$ |
| $\text{Binomial}(n,p)$ | $\binom{n}{k} p^k (1-p)^{n-k}$ | $np$ | $np(1-p)$ |

---

## Quantiles usuels

### Loi normale $\mathcal{N}(0,1)$

| Niveau | $\alpha$ | $z_{\alpha/2}$ |
|--------|----------|----------------|
| 90%    | 0.10     | 1.645          |
| 95%    | 0.05     | 1.960          |
| 99%    | 0.01     | 2.576          |

### Loi de Student (approx. pour $n \geq 30$)

Pour $n \geq 30$, $t_{n-1, \alpha/2} \approx z_{\alpha/2}$
