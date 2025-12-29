# Exercices pratiques

## Exercice 1 : Estimation de la moyenne d'une population normale

### Énoncé

On mesure la taille (en cm) de $n = 20$ individus tirés au hasard d'une population. On suppose que la taille suit une loi normale $\mathcal{N}(\mu, \sigma^2)$ avec $\sigma = 10$ cm.

Les données observées sont :

```
165, 172, 158, 180, 175, 168, 162, 177, 169, 173,
171, 166, 179, 164, 170, 176, 167, 174, 163, 178
```

### Questions

1. Calculer la moyenne empirique $\bar{X}_n$ des observations
2. Calculer un intervalle de confiance à 95% pour $\mu$
3. Que peut-on conclure ?

### Solution

**1. Moyenne empirique**

$$
\bar{X}_n = \frac{1}{20} \sum_{i=1}^{20} X_i = 170.3 \text{ cm}
$$

**2. Intervalle de confiance**

Pour $\alpha = 0.05$, on a $z_{\alpha/2} = z_{0.025} = 1.96$.

L'intervalle de confiance est :

$$
IC_{0.95}(\mu) = \left[170.3 - 1.96 \times \frac{10}{\sqrt{20}}, 170.3 + 1.96 \times \frac{10}{\sqrt{20}}\right]
$$

$$
IC_{0.95}(\mu) = [170.3 - 4.38, 170.3 + 4.38] = [165.92, 174.68]
$$

**3. Conclusion**

Avec 95% de confiance, la taille moyenne de la population se situe entre 165.92 cm et 174.68 cm.

---

## Exercice 2 : Méthode du maximum de vraisemblance

### Énoncé

On observe $n$ réalisations $X_1, \ldots, X_n$ i.i.d. de loi exponentielle $\mathcal{E}(\lambda)$ de densité :

$$
f(x; \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

### Questions

1. Écrire la fonction de vraisemblance $L(\lambda)$
2. Calculer la log-vraisemblance $\ell(\lambda)$
3. Trouver l'estimateur du maximum de vraisemblance $\hat{\lambda}_{MLE}$
4. Montrer que $\hat{\lambda}_{MLE}$ est sans biais

### Solution

**1. Vraisemblance**

$$
L(\lambda) = \prod_{i=1}^{n} \lambda e^{-\lambda x_i} = \lambda^n e^{-\lambda \sum_{i=1}^{n} x_i}
$$

**2. Log-vraisemblance**

$$
\ell(\lambda) = n \log \lambda - \lambda \sum_{i=1}^{n} x_i
$$

**3. MLE**

On dérive par rapport à $\lambda$ :

$$
\frac{\partial \ell}{\partial \lambda} = \frac{n}{\lambda} - \sum_{i=1}^{n} x_i = 0
$$

D'où :

$$
\hat{\lambda}_{MLE} = \frac{n}{\sum_{i=1}^{n} X_i} = \frac{1}{\bar{X}_n}
$$

**4. Biais**

On sait que $\mathbb{E}[X_i] = \frac{1}{\lambda}$, donc $\mathbb{E}[\bar{X}_n] = \frac{1}{\lambda}$.

Par l'inégalité de Jensen (car $x \mapsto 1/x$ est convexe) :

$$
\mathbb{E}\left[\frac{1}{\bar{X}_n}\right] \geq \frac{1}{\mathbb{E}[\bar{X}_n]} = \lambda
$$

Donc $\hat{\lambda}_{MLE}$ est **biaisé** (positivement).

Pour trouver l'espérance exacte, on peut utiliser le fait que $2\lambda \sum X_i \sim \chi^2_{2n}$.
