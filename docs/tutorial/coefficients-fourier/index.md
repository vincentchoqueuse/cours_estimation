# Tutoriel : Estimation des coefficients de Fourier

## Introduction

L'estimation des **coefficients de Fourier** permet de représenter un signal périodique comme une somme d'exponentielles complexes. Nous montrons ici comment formuler ce problème comme un modèle linéaire et comment la structure d'échantillonnage impacte la matrice de design.

::: tip Notation complexe
Nous utilisons la **notation exponentielle complexe** qui est plus compacte et fait directement le lien avec la transformée de Fourier discrète (DFT). L'orthogonalité des exponentielles complexes se démontre élégamment par des sommes géométriques.
:::

## Rappels : Série de Fourier

Un signal réel périodique $x(t)$ de période $T_0$ peut être décomposé en **série de Fourier complexe** :

$$
x(t) = \sum_{u=-N_f}^{N_f} s_u e^{j2\pi u f_0 t}
$$

où :

- $f_0 = 1/T_0$ est la fréquence fondamentale
- $N_f$ est le nombre d'harmoniques considérées
- $s_u \in \mathbb{C}$ sont les coefficients de Fourier complexes
- Pour un signal réel : $s_{-u} = \overline{s_u}$ (symétrie hermitienne)

**Lien avec la forme trigonométrique** : En posant $s_u = \frac{1}{2}(a_u - jb_u)$ pour $u > 0$ et $s_0 = a_0$ (réel), on retrouve :

$$
x(t) = a_0 + \sum_{u=1}^{N_f} \left[a_u \cos(2\pi u f_0 t) + b_u \sin(2\pi u f_0 t)\right]
$$

## Formulation en modèle linéaire

### Vecteur de paramètres

Nous cherchons à estimer le vecteur de coefficients complexes :

$$
\mathbf{s} = [s_{-N_f}, s_{-N_f+1}, \ldots, s_{-1}, s_0, s_1, \ldots, s_{N_f}]^T \in \mathbb{C}^{2N_f+1}
$$

### Observations bruitées

Soit $m$ échantillons $x_1, \ldots, x_m$ observés aux instants $t_1, \ldots, t_m$. Le modèle linéaire est :

$$
\mathbf{x} = \mathbf{A}\mathbf{s} + \mathbf{n}
$$

où la matrice de design $\mathbf{A} \in \mathbb{C}^{m \times (2N_f+1)}$ est :

$$
\mathbf{A} = \begin{bmatrix}
e^{-j2\pi N_f f_0 t_1} & \cdots & e^{-j2\pi f_0 t_1} & 1 & e^{j2\pi f_0 t_1} & \cdots & e^{j2\pi N_f f_0 t_1} \\
e^{-j2\pi N_f f_0 t_2} & \cdots & e^{-j2\pi f_0 t_2} & 1 & e^{j2\pi f_0 t_2} & \cdots & e^{j2\pi N_f f_0 t_2} \\
\vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots
\end{bmatrix}
$$

Chaque ligne $k$ correspond à l'évaluation des exponentielles complexes aux fréquences $-N_f f_0, \ldots, 0, \ldots, N_f f_0$ à l'instant $t_k$.

## Cas 1 : Échantillonnage sur une durée quelconque

### Configuration

Nous observons le signal sur une durée $T_{obs}$ quelconque (pas nécessairement un multiple de $T_0$) avec $m$ échantillons uniformément répartis :

$$
t_k = \frac{(k-1)T_{obs}}{m-1}, \quad k = 1, \ldots, m
$$

### Propriétés de la matrice $\mathbf{A}^T\mathbf{A}$

Dans ce cas général, la matrice $\mathbf{A}^T\mathbf{A}$ n'a **pas de structure particulière**. Elle est :

- **Pleine** : tous les éléments sont non nuls
- **Non diagonale** : les colonnes de $\mathbf{A}$ ne sont pas orthogonales

**Conséquence** : Le calcul de $(\mathbf{A}^T\mathbf{A})^{-1}$ nécessite une inversion matricielle complète (coût $O((2N_f+1)^3)$).

### Exemple

Soit un signal composé de 2 harmoniques ($N_f = 2$) observé sur $T_{obs} = 1.5 T_0$ avec $m = 20$ échantillons.

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/fourier_arbitrary.png" alt="Fourier durée quelconque" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 1: Estimation avec durée d'observation arbitraire</p>
</div>

## Cas 2 : Échantillonnage sur une période complète

### Configuration

Nous observons le signal sur **exactement une période** : $T_{obs} = T_0$ avec $m$ échantillons uniformément répartis :

$$
t_k = \frac{(k-1) T_0}{m-1}, \quad k = 1, \ldots, m
$$

### Propriétés remarquables de $\mathbf{A}^H\mathbf{A}$

Dans ce cas, la matrice de Gram $\mathbf{A}^H\mathbf{A}$ (où $\mathbf{A}^H$ est la transposée conjuguée) présente une **structure particulière** :

$$
\mathbf{A}^H\mathbf{A} = m \mathbf{I}_{2N_f+1}
$$

**Orthogonalité par somme géométrique** : Les colonnes de $\mathbf{A}$ sont exactement orthogonales lorsque nous échantillonnons sur une période complète. Pour deux fréquences $u$ et $v$ :

$$
\sum_{k=0}^{m-1} e^{j2\pi u f_0 t_k} e^{-j2\pi v f_0 t_k} = \sum_{k=0}^{m-1} e^{j2\pi (u-v) k/m}
$$

Cette somme est une **suite géométrique** de raison $q = e^{j2\pi (u-v)/m}$ :

$$
\sum_{k=0}^{m-1} q^k = \begin{cases}
m & \text{si } u = v \text{ (i.e., } q = 1) \\
\frac{1 - q^m}{1 - q} = 0 & \text{si } u \neq v \text{ et } q^m = 1
\end{cases}
$$

La condition $q^m = e^{j2\pi (u-v)} = 1$ est satisfaite si $(u-v)$ est entier, ce qui est toujours vrai pour $u, v \in \{-N_f, \ldots, N_f\}$ distincts.

**Conséquence** : L'inversion devient triviale :

$$
(\mathbf{A}^H\mathbf{A})^{-1} = \frac{1}{m}\mathbf{I}
$$

et l'estimateur se simplifie en une **transformée de Fourier discrète (DFT)** :

$$
\widehat{\mathbf{s}} = \frac{1}{m}\mathbf{A}^H\mathbf{x}
$$

Sous forme scalaire, nous obtenons :

$$\widehat{s}_u = \frac{1}{m}\sum_{k=0}^{m-1} x_k e^{-j2\pi u f_0 t_k}$$

### Comparaison des matrices

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/fourier_matrices.png" alt="Comparaison des matrices" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 2: Structure de $\mathbf{A}^T\mathbf{A}$ selon le type d'échantillonnage</p>
</div>

**Gauche** : $T_{obs}$ quelconque → matrice $\mathbf{A}^H\mathbf{A}$ pleine
**Droite** : $T_{obs} = T_0$ → matrice $\mathbf{A}^H\mathbf{A}$ diagonale

### Lien avec la transformée de Fourier rapide (FFT)

Lorsque $m = 2^p$ et que nous échantillonnons sur un nombre entier de périodes ($T_{obs} = P \cdot T_0$), le calcul de $\mathbf{A}^H\mathbf{x}$ peut être effectué efficacement avec la **FFT** :

- Complexité directe : $O(m \cdot N_f)$
- Complexité avec FFT : $O(m \log m)$

**Remarque** : La propriété d'orthogonalité $\mathbf{A}^H\mathbf{A} = m \mathbf{I}$ est préservée pour tout nombre entier de périodes $P \geq 1$. La DFT standard calcule exactement $\widehat{s}_u = \frac{1}{m}\sum_{k=0}^{m-1} x_k e^{-j2\pi uk/m}$ pour $u = 0, \ldots, m-1$.

## Exemple numérique complet

### Signal de test

Considérons un signal composé de 2 harmoniques en **notation complexe** :

$$
x(t) = s_0 + s_1 e^{j2\pi f_0 t} + s_{-1} e^{-j2\pi f_0 t} + s_2 e^{j4\pi f_0 t} + s_{-2} e^{-j4\pi f_0 t}
$$

avec $s_0 = 0.5$ (moyenne réelle), $s_1 = 0.4 - j0.25$, $s_{-1} = \overline{s_1} = 0.4 + j0.25$ (symétrie hermitienne pour signal réel), $s_2 = 0.15$, $s_{-2} = 0.15$ (réel), $f_0 = 1$ Hz et bruit $n \sim \mathcal{N}(0, 0.1^2)$.

**Équivalent trigonométrique** : $x(t) = 0.5 + 0.8\cos(2\pi f_0 t) + 0.5\sin(2\pi f_0 t) + 0.3\cos(4\pi f_0 t)$

### Résultats

Le tableau ci-dessous compare les deux approches :

| Méthode | $T_{obs}$ | Structure de $\mathbf{A}^H\mathbf{A}$ | Conditionnement         | EQM   |
| ------- | --------- | ------------------------------------- | ----------------------- | ----- |
| Cas 1   | $1.5 T_0$ | Pleine                                | Moyen ($\approx 10^2$)  | 0.015 |
| Cas 2   | $T_0$     | Diagonale                             | Excellent ($\approx 1$) | 0.008 |

::: tip Observation
L'échantillonnage sur une période complète (ou plus généralement sur un **nombre entier de périodes** $T_{obs} = P \cdot T_0$) améliore significativement le conditionnement numérique et réduit l'EQM !
:::
