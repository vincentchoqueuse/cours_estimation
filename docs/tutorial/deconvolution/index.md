---
prev:
  text: 'Coefficients de Fourier'
  link: '/tutorial/coefficients-fourier'
next:
  text: 'Estimation de canal FIR'
  link: '/tutorial/estimation-canal'
---

# Tutoriel : Déconvolution

## Introduction

La **déconvolution** consiste à estimer un signal d'entrée $\mathbf{s}$ à partir d'observations $\mathbf{x}$ obtenues après convolution avec une réponse impulsionnelle $h[n]$ connue. Ce tutoriel compare deux approches selon le type de convolution utilisé.

### Modèle général

Considérons un système linéaire invariant dans le temps (LTI) de réponse impulsionnelle $h[n]$ de longueur $L$. Le signal observé s'écrit :

$$
x[n] = \sum_{l=0}^{L-1} h[l] s[n-l] + w[n]
$$

où $w[n]$ est un bruit additif. Notre objectif est d'estimer $\mathbf{s}$ à partir de $\mathbf{x}$ et $h$.

## Cas 1 : Déconvolution avec convolution classique

### Formulation

La convolution classique (ou linéaire) entre un signal $\mathbf{s}$ de longueur $p$ et $h$ de longueur $L$ produit un signal de longueur $m = p + L - 1$.

Le modèle linéaire s'écrit :

$$
\mathbf{x} = \mathbf{A}\mathbf{s} + \mathbf{n}
$$

où $\mathbf{A} \in \mathbb{R}^{m \times p}$ est une **matrice de Toeplitz** :

$$
\mathbf{A} = \begin{bmatrix}
h_1 & 0 & 0 & \cdots & 0 \\
h_2 & h_1 & 0 & \cdots & 0 \\
h_3 & h_2 & h_1 & \cdots & 0 \\
\vdots & h_3 & h_2 & \ddots & \vdots \\
h_L & \vdots & h_3 & \ddots & 0 \\
0 & h_L & \vdots & \ddots & h_1 \\
\vdots & \ddots & h_L & \ddots & \vdots \\
0 & \cdots & 0 & h_L & h_{L-1}
\end{bmatrix}
$$

Une matrice de Toeplitz est constante le long de ses diagonales.

### Estimateur des moindres carrés

L'estimateur du maximum de vraisemblance est :

$$
\widehat{\mathbf{s}} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{x}
$$

### Exemple

Pour $h = [0.5, 0.3, 0.2]$ ($L = 3$) et $p = 5$, on obtient une matrice $7 \times 5$ :

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/deconv_toeplitz.png" alt="Matrice de Toeplitz" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 1: Matrice de Toeplitz pour la convolution classique</p>
</div>

### Problèmes

- **Complexité** : L'inversion de $\mathbf{A}^T\mathbf{A}$ coûte $O(p^3)$ opérations
- **Conditionnement** : La matrice $\mathbf{A}^T\mathbf{A}$ est pleine et peut être mal conditionnée
- **Pas de structure exploitable** : Impossible d'utiliser des algorithmes rapides comme la FFT

---

## Cas 2 : Déconvolution avec convolution circulaire

### Hypothèse

Supposons maintenant que le signal reçu $\mathbf{x}$ de longueur $p$ soit lié au signal $\mathbf{s}$ (également de longueur $p$) par une **convolution circulaire** au lieu d'une convolution linéaire :

$$
x[n] = \sum_{l=0}^{L-1} h[l] s[(n-l) \bmod p] + w[n]
$$

où $(n-l) \bmod p$ indique que les indices "bouclent" de manière périodique.

### Formulation matricielle

Le modèle s'écrit :

$$
\mathbf{x} = \mathbf{A}_{circ}\mathbf{s} + \mathbf{n}
$$

où $\mathbf{A}_{circ} \in \mathbb{R}^{p \times p}$ est une **matrice circulante** :

$$
\mathbf{A}_{circ} = \begin{bmatrix}
h_1 & h_L & h_{L-1} & \cdots & h_2 \\
h_2 & h_1 & h_L & \cdots & h_3 \\
h_3 & h_2 & h_1 & \cdots & h_4 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
h_L & h_{L-1} & h_{L-2} & \cdots & h_1 \\
0 & h_L & h_{L-1} & \cdots & h_2 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & h_L & h_{L-1}
\end{bmatrix}
$$

Chaque ligne est une rotation circulaire de la précédente.

### Propriété fondamentale

Une matrice circulante est **diagonalisable par la DFT** :

$$
\mathbf{A}_{circ} = \mathbf{F}^H \mathbf{\Lambda} \mathbf{F}
$$

où $\mathbf{F}$ est la matrice DFT et $\mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_p)$ avec $\lambda_k = (\mathbf{F}\mathbf{h})_k$.

### Estimateur avec FFT

Grâce à cette propriété, l'estimateur se calcule efficacement dans le domaine fréquentiel :

1. DFT des observations : $\mathbf{X} = \text{FFT}(\mathbf{x})$
2. Réponse fréquentielle du canal : $\mathbf{H} = \text{FFT}(\mathbf{h})$
3. Égalisation fréquentielle : $\widehat{S}_k = \frac{X_k}{H_k}$ pour $k = 0, \ldots, p-1$
4. IFFT : $\widehat{\mathbf{s}} = \text{IFFT}(\widehat{\mathbf{S}})$

**Complexité** : $O(p \log p)$ au lieu de $O(p^3)$

## Focus : OFDM et préfixe cyclique

Dans la pratique, le **Cas 2** (convolution circulaire) ne se produit pas naturellement. Un canal physique génère toujours une convolution linéaire (Cas 1).

L'**OFDM** (Orthogonal Frequency-Division Multiplexing) est une technique qui permet de **transformer artificiellement** la convolution linéaire en convolution circulaire en ajoutant un **préfixe cyclique** : on répète les $L-1$ derniers échantillons du signal au début de la transmission.

**Signal transmis** :

$$
\mathbf{s}_{CP} = [s_{p-L+2}, \ldots, s_p, s_1, s_2, \ldots, s_p]
$$

Après convolution avec le canal et suppression du préfixe cyclique, le signal reçu correspond exactement au **Cas 2** (convolution circulaire).

### Gain de l'OFDM

Dans un canal sélectif en fréquence (réponse impulsionnelle longue), cette technique offre un avantage majeur :

- **Sans préfixe cyclique** : Égalisation par inversion matricielle en $O(p^3)$
- **Avec préfixe cyclique (OFDM)** : Égalisation par simple division complexe dans le domaine fréquentiel en $O(p \log p)$

Le coût est un **overhead** de $\frac{L-1}{p}$ échantillons supplémentaires transmis, généralement faible (5-25%) par rapport au gain algorithmique.

Cette technique est utilisée dans **WiFi** (802.11a/g/n/ac/ax), **4G LTE**, **5G**, et **DVB-T**.

## Code Python

Le script complet est disponible dans `src/deconvolution.py`.
