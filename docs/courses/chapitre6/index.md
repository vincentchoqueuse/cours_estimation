# Chapitre 6 : Régression Linéaire - Inférence et Diagnostic

## Introduction

Dans le [Chapitre 4](/courses/chapitre4/), nous avons étudié le modèle de régression linéaire, l'estimateur des moindres carrés (OLS) et ses propriétés fondamentales. Le [Chapitre 5](/courses/chapitre5/) a présenté les techniques de régularisation (Ridge, LASSO, Elastic Net) pour traiter les problèmes de surapprentissage et de multicolinéarité.

Dans ce chapitre, nous approfondissons l'**inférence statistique** et le **diagnostic** pour la régression linéaire :

- Construction d'**intervalles de confiance** pour les coefficients
- **Tests d'hypothèses** (tests individuels et test global)
- **Diagnostic des résidus** pour vérifier les hypothèses du modèle
- **Problèmes courants** (multicolinéarité, hétéroscédasticité, points aberrants) et leurs solutions
- **Extensions** du modèle OLS (ridge, LASSO, moindres carrés pondérés)

## Intervalles de confiance et tests d'hypothèses

### Intervalle de confiance pour un coefficient

Pour construire un intervalle de confiance pour $s_j$, nous utilisons la loi de Student. Sous $H_0 : s_j = s_j^{(0)}$, la statistique :

$$
T_j = \frac{\widehat{s}_j - s_j}{\widehat{\sigma}\sqrt{[(\mathbf{A}^T\mathbf{A})^{-1}]_{jj}}} \sim t_{m-p}
$$

suit une loi de Student à $m-p$ degrés de liberté.

Un **intervalle de confiance** de niveau $1-\alpha$ pour $s_j$ est :

$$
IC_{1-\alpha}(s_j) = \left[\widehat{s}_j - t_{m-p, \alpha/2} \cdot \widehat{\sigma}\sqrt{[(\mathbf{A}^T\mathbf{A})^{-1}]_{jj}}, \widehat{s}_j + t_{m-p, \alpha/2} \cdot \widehat{\sigma}\sqrt{[(\mathbf{A}^T\mathbf{A})^{-1}]_{jj}}\right]
$$

où $t_{m-p, \alpha/2}$ est le quantile d'ordre $1-\alpha/2$ de la loi $t_{m-p}$.

**Interprétation** : Nous sommes confiants à $(1-\alpha) \times 100\%$ que le vrai coefficient $s_j$ se trouve dans cet intervalle.

### Test de significativité d'un coefficient

Pour tester l'hypothèse nulle $H_0 : s_j = 0$ (le coefficient n'a pas d'effet) contre $H_1 : s_j \neq 0$ :

1. Calculer la statistique de test :

   $$
   T_j = \frac{\widehat{s}_j}{\widehat{\sigma}\sqrt{[(\mathbf{A}^T\mathbf{A})^{-1}]_{jj}}}
   $$

2. Comparer $|T_j|$ à $t_{m-p, \alpha/2}$

3. **Rejeter $H_0$** si $|T_j| > t_{m-p, \alpha/2}$ (coefficient significativement différent de 0)

La **p-valeur** associée est :

$$
p = 2P(t_{m-p} > |T_j|)
$$

**Interprétation** : La p-valeur représente la probabilité d'observer une valeur aussi extrême (ou plus) sous l'hypothèse nulle. Si $p < \alpha$ (généralement $\alpha = 0.05$), on rejette $H_0$ et on conclut que la variable $j$ a un effet significatif.

### Test global du modèle (test de Fisher)

Pour tester si **au moins une** variable explicative a un effet, nous testons :

$$
H_0 : s_1 = s_2 = \cdots = s_p = 0 \quad \text{vs} \quad H_1 : \exists j, s_j \neq 0
$$

La statistique de test de Fisher est :

$$
F = \frac{\text{SSE}/(p-1)}{\text{SSR}/(m-p)} = \frac{R^2/(p-1)}{(1-R^2)/(m-p)} \sim F_{p-1, m-p}
$$

sous $H_0$, où $F_{p-1, m-p}$ est la loi de Fisher à $(p-1, m-p)$ degrés de liberté.

**Rejeter $H_0$** si $F > F_{p-1, m-p, \alpha}$ (le modèle est globalement significatif).

**Interprétation** : Ce test permet de vérifier si le modèle dans son ensemble apporte une information significative par rapport à un modèle réduit à la seule moyenne.

::: tip Remarque
Le test de Fisher est un test **global** : il teste si au moins un coefficient est non nul. Les tests t individuels testent la significativité de **chaque** coefficient séparément.
:::

## Analyse des résidus

### Importance des résidus

L'analyse des **résidus** $\widehat{\mathbf{n}} = \mathbf{x} - \mathbf{A}\widehat{\mathbf{s}}_{OLS}$ permet de vérifier les hypothèses du modèle :

1. **Normalité** : Les résidus doivent suivre une loi normale
2. **Homoscédasticité** : La variance des résidus doit être constante
3. **Indépendance** : Les résidus ne doivent pas présenter de corrélation
4. **Linéarité** : Pas de tendance systématique dans les résidus

Si ces hypothèses ne sont pas respectées, les intervalles de confiance et les tests peuvent être invalides, même si l'estimateur OLS reste sans biais (sous certaines conditions).

### Graphiques de diagnostic

**Graphique 1: Résidus vs valeurs ajustées**

Ce graphique permet de détecter :

- **Hétéroscédasticité** : Si les résidus forment une forme d'entonnoir (variance croissante ou décroissante)
- **Non-linéarité** : Si les résidus montrent une tendance systématique (parabole, etc.)

**Comportement attendu** : Les résidus doivent être dispersés aléatoirement autour de zéro, sans structure apparente.

**Graphique 2: Q-Q plot (Quantile-Quantile)**

Compare la distribution des résidus à une loi normale théorique pour vérifier la **normalité**.

**Comportement attendu** : Les points doivent être approximativement alignés sur la bissectrice.

**Graphique 3: Scale-Location**

Affiche $\sqrt{|r_k^*|}$ (racine carrée des résidus standardisés en valeur absolue) en fonction des valeurs ajustées.

**Comportement attendu** : Ligne horizontale, indiquant une variance constante (homoscédasticité).

**Graphique 4: Résidus vs Leverage**

Identifie les **points influents** qui ont un fort impact sur l'estimation (leverage élevé et résidus importants).

**Comportement attendu** : Pas de points avec à la fois un leverage élevé et un résidu important (distance de Cook élevée).

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/diagnostic_residus.png" alt="Graphiques de diagnostic des résidus" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 1: Graphiques de diagnostic pour l'analyse des résidus</p>
</div>

### Résidus standardisés

Les **résidus standardisés** permettent de comparer les résidus sur une échelle commune :

$$
r_k^* = \frac{\widehat{n}_k}{\widehat{\sigma}\sqrt{1-P_{kk}}}
$$

où $P_{kk}$ est l'élément diagonal de la matrice de projection $\mathbf{P} = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$.

**Interprétation** : Un résidu standardisé $|r_k^*| > 3$ indique un point potentiellement aberrant.

## Problèmes courants et solutions

### Multicolinéarité

La **multicolinéarité** survient lorsque certaines variables explicatives sont fortement corrélées entre elles.

**Conséquences** :

- La matrice $\mathbf{A}^T\mathbf{A}$ devient mal conditionnée
- Les variances des estimateurs deviennent très élevées : $\text{var}(\widehat{s}_j) = \sigma^2 [(\mathbf{A}^T\mathbf{A})^{-1}]_{jj} \uparrow$
- Les coefficients deviennent instables (petites variations dans les données entraînent de grandes variations dans les estimations)
- Les intervalles de confiance deviennent très larges
- Les tests de significativité perdent en puissance

**Détection** :

1. **Matrice de corrélation** : Examiner les corrélations entre variables explicatives. Une corrélation $|\rho| > 0.8$ est problématique.

2. **Facteur d'inflation de la variance (VIF)** :

   $$
   \text{VIF}_j = \frac{1}{1 - R_j^2}
   $$

   où $R_j^2$ est le $R^2$ de la régression de la $j$-ième variable sur toutes les autres.

   - $\text{VIF}_j < 5$ : Pas de problème
   - $5 < \text{VIF}_j < 10$ : Multicolinéarité modérée
   - $\text{VIF}_j > 10$ : Multicolinéarité sévère

3. **Conditionnement de $\mathbf{A}^T\mathbf{A}$** : Si $\kappa(\mathbf{A}^T\mathbf{A}) > 10^3$, la matrice est mal conditionnée.

**Solutions** :

1. **Supprimer une des variables corrélées** : Identifier les paires de variables fortement corrélées et en retirer une.

2. **Régression ridge** : Voir section Extensions ci-dessous.

3. **Analyse en composantes principales (PCA)** : Transformer les variables en composantes non corrélées.

4. **Augmenter la taille de l'échantillon** : Plus de données peuvent réduire les variances.

### Hétéroscédasticité

L'**hétéroscédasticité** signifie que la variance des erreurs n'est pas constante : $\text{var}(n_k) = \sigma_k^2$ (dépend de $k$).

**Conséquences** :

- L'estimateur OLS reste **sans biais** : $\mathbb{E}[\widehat{\mathbf{s}}_{OLS}] = \mathbf{s}$
- Mais il n'est **plus efficace** (n'a plus la variance minimale)
- Les formules de variance sont **incorrectes** : $\text{Cov}(\widehat{\mathbf{s}}_{OLS}) \neq \sigma^2(\mathbf{A}^T\mathbf{A})^{-1}$
- Les **intervalles de confiance et tests sont invalides**

**Détection** :

1. **Observation visuelle** : Graphique résidus vs valeurs ajustées montre une forme d'entonnoir.

2. **Test de Breusch-Pagan** : Teste si la variance des erreurs dépend des variables explicatives.

3. **Test de White** : Version plus générale qui teste l'hétéroscédasticité sans hypothèse sur sa forme.

**Solutions** :

1. **Transformation de la variable dépendante** :

   - Logarithme : $\log(x)$ au lieu de $x$ (réduit l'impact des grandes valeurs)
   - Racine carrée : $\sqrt{x}$

2. **Moindres carrés pondérés (WLS)** : Voir section Extensions ci-dessous.

3. **Erreurs-types robustes (estimateur sandwich)** : Utiliser des estimateurs de variance robustes à l'hétéroscédasticité (Huber-White).

### Points aberrants et points influents

**Point aberrant (outlier)** : Observation avec un résidu très élevé (écart important entre observation et prédiction).

**Point influent (leverage point)** : Observation qui a un impact important sur l'estimation des coefficients. Un point a un leverage élevé s'il est éloigné des autres observations dans l'espace des variables explicatives.

**Distinction importante** :

- Un point peut être aberrant sans être influent (hors du modèle mais dans une zone dense)
- Un point peut être influent sans être aberrant (dans le modèle mais dans une zone isolée)
- Les points les plus problématiques sont à la fois aberrants **et** influents

**Mesures** :

1. **Résidus standardisés** : $r_k^* = \frac{\widehat{n}_k}{\widehat{\sigma}\sqrt{1-P_{kk}}}$

   - $|r_k^*| > 3$ : Point aberrant

2. **Leverage** : $h_k = P_{kk}$ (élément diagonal de la matrice de projection)

   - $h_k > \frac{2p}{m}$ : Leverage élevé

3. **Distance de Cook** : Mesure combinée qui quantifie l'influence globale de l'observation $k$ :
   $$
   D_k = \frac{(r_k^*)^2}{p} \cdot \frac{h_k}{1-h_k}
   $$
   - $D_k > 0.5$ : Point influent
   - $D_k > 1$ : Point très influent

**Solutions** :

1. **Vérifier les données** : Erreur de saisie, erreur de mesure ?

2. **Examiner le contexte** : Le point aberrant est-il légitime ou exceptionnel ?

3. **Estimation robuste** : Utiliser des méthodes robustes aux outliers (M-estimateurs, régression quantile).

4. **Supprimer avec précaution** : Ne supprimer que si justifié (après documentation).
