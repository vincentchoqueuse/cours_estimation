---
prev:
  text: "Déconvolution"
  link: "/tutorial/deconvolution"
next: false
---

# Tutoriel : Estimation d'un canal FIR

## Introduction

L'**estimation de canal** est un problème fondamental en traitement du signal et télécommunications. Un **canal FIR** (Finite Impulse Response) est un système linéaire caractérisé par sa réponse impulsionnelle $\mathbf{h} = [h_0, h_1, \ldots, h_{L-1}]^T$ de longueur $L$.

Dans ce tutoriel, nous considérons le problème d'**identification de canal** : estimer les coefficients $\mathbf{h}$ à partir d'observations entrée-sortie.

### Modèle du canal

Le signal de sortie $x[n]$ est obtenu par convolution du signal d'entrée $s[n]$ avec la réponse impulsionnelle du canal :

$$
x[n] = \sum_{l=0}^{L-1} h_l \, s[n-l] + w[n]
$$

où $w[n]$ est un bruit additif gaussien.

**Objectif** : Estimer $\mathbf{h}$ à partir de :

- Signal d'entrée connu : $\mathbf{s} = [s_1, \ldots, s_m]^T$
- Signal de sortie observé : $\mathbf{x} = [x_1, \ldots, x_m]^T$

## Formulation en modèle linéaire

### Construction de la matrice de design

Pour $m$ échantillons temporels, le modèle s'écrit sous forme vectorielle :

$$
\mathbf{x} = \mathbf{S}\mathbf{h} + \mathbf{w}
$$

où $\mathbf{S} \in \mathbb{R}^{m \times L}$ est une **matrice de Toeplitz** construite à partir du signal d'entrée $\mathbf{s}$ :

$$
\mathbf{S} = \begin{bmatrix}
s_1 & 0 & 0 & \cdots & 0 \\
s_2 & s_1 & 0 & \cdots & 0 \\
s_3 & s_2 & s_1 & \cdots & 0 \\
\vdots & s_3 & s_2 & \ddots & \vdots \\
s_L & \vdots & s_3 & \ddots & 0 \\
s_{L+1} & s_L & \vdots & \ddots & s_1 \\
\vdots & \ddots & s_L & \ddots & \vdots \\
s_m & \cdots & \cdots & s_{m-L+2} & s_{m-L+1}
\end{bmatrix}
$$

Chaque ligne $k$ de $\mathbf{S}$ contient les $L$ dernières valeurs du signal d'entrée avant l'instant $k$ : $[s_k, s_{k-1}, \ldots, s_{k-L+1}]$.

::: tip Remarque
Ce problème est **dual** de la déconvolution :

- **Déconvolution** : $h$ connu, estimer $s$ à partir de $x = s * h$
- **Estimation de canal** : $s$ connu, estimer $h$ à partir de $x = s * h$
  :::

## Estimateur des moindres carrés

### Solution analytique

L'estimateur OLS minimise la somme des carrés des erreurs de prédiction :

$$
\widehat{\mathbf{h}}_{OLS} = \arg\min_{\mathbf{h}} \|\mathbf{x} - \mathbf{S}\mathbf{h}\|^2
$$

Si $\mathbf{S}$ est de rang plein ($m \geq L$ et $\mathbf{S}$ de rang $L$), la solution unique est :

$$
\widehat{\mathbf{h}}_{OLS} = (\mathbf{S}^T\mathbf{S})^{-1}\mathbf{S}^T\mathbf{x} = \mathbf{S}^\dagger\mathbf{x}
$$

où $\mathbf{S}^\dagger$ est la pseudo-inverse de Moore-Penrose.

### Propriétés

Sous les hypothèses du modèle linéaire gaussien :

1. **Sans biais** : $\mathbb{E}[\widehat{\mathbf{h}}_{OLS}] = \mathbf{h}$

2. **Matrice de covariance** : $\text{Cov}(\widehat{\mathbf{h}}_{OLS}) = \sigma_w^2(\mathbf{S}^T\mathbf{S})^{-1}$

3. **Variance minimale** : BLUE (Best Linear Unbiased Estimator) par le théorème de Gauss-Markov

### Conditionnement et qualité d'estimation

La qualité de l'estimation dépend du **conditionnement** de $\mathbf{S}^T\mathbf{S}$ :

- Si $\text{cond}(\mathbf{S}^T\mathbf{S})$ est faible : estimation robuste
- Si $\text{cond}(\mathbf{S}^T\mathbf{S})$ est élevé : estimation sensible au bruit

Le conditionnement dépend du **signal d'entrée** $\mathbf{s}$ :

- **Signal blanc** (ex: bruit gaussien) : bon conditionnement, matrice $\mathbf{S}^T\mathbf{S} \approx \sigma_s^2 \mathbf{I}$
- **Signal périodique** (ex: sinusoïde pure) : mauvais conditionnement, matrice singulière
- **Séquence d'apprentissage** optimale : signaux pseudo-aléatoires (PN sequences, PRBS)

## Visualisations

Le script Python génère trois figures illustrant le problème :

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/canal_estimation.png" alt="Estimation de canal FIR" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 1: Estimation de canal avec différents types de signaux d'entrée</p>
</div>

**Figure 1** montre trois scénarios :

1. **Signal blanc gaussien** : Excellente estimation (bon conditionnement)
2. **Signal BPSK** : Bonne estimation (signal binaire aléatoire)
3. **Signal périodique** : Mauvaise estimation (mauvais conditionnement)

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/mse_vs_snr.png" alt="Performance en fonction du SNR" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 2: Erreur quadratique moyenne (MSE) en fonction du rapport signal/bruit (SNR)</p>
</div>

**Figure 2** illustre l'impact du bruit sur la qualité d'estimation :

- À SNR élevé (peu de bruit) : MSE faible, estimation précise
- À SNR faible (bruit important) : MSE élevée, estimation imprécise
- La pente montre la sensibilité de l'estimateur au bruit

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/mse_vs_length.png" alt="Performance en fonction de la longueur d'observation" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 3: MSE en fonction du nombre d'échantillons d'observation</p>
</div>

**Figure 3** montre l'effet de la longueur de la séquence d'apprentissage :

- Plus d'échantillons → MSE diminue (loi en $1/m$)
- Convergence asymptotique vers la vraie réponse impulsionnelle

## Design optimal du signal d'entrée

Connaissant la matrice de covariance de l'estimateur, il peut être interessant de se focaliser sur le design du signal d'entrée en optimisation les performances d'estimation.

### Formulation du problème

Pour la contrainte $E = \|\mathbf{s}\|^2$, quel signal d'entrée $\mathbf{s}$ minimise la variance d'estimation ?

La matrice de covariance de l'estimateur OLS est :

$$
\text{Cov}(\widehat{\mathbf{h}}) = \sigma_w^2 (\mathbf{S}^T\mathbf{S})^{-1}
$$

La **trace** de cette matrice représente la variance totale :

$$
\text{trace}(\text{Cov}(\widehat{\mathbf{h}})) = \sigma_w^2 \text{trace}\left((\mathbf{S}^T\mathbf{S})^{-1}\right)
$$

**Problème d'optimisation** (D-optimal design) :

$$
\begin{align}
\min_{\mathbf{s}} \quad & \text{trace}\left((\mathbf{S}^T\mathbf{S})^{-1}\right) \\
\text{s.c.} \quad & \|\mathbf{s}\|^2 \leq E
\end{align}
$$

Équivalent à :

$$
\max_{\mathbf{s}} \quad \text{trace}(\mathbf{S}^T\mathbf{S}) \quad \text{ou} \quad \max_{\mathbf{s}} \quad \lambda_{\min}(\mathbf{S}^T\mathbf{S})
$$

### Solution optimale

Le signal optimal est un **bruit blanc** (ou signal à spectre plat) :

**Propriété** : Pour un signal blanc $\mathbf{s}$ de variance $\sigma_s^2 = E/m$ :

$$
\mathbf{S}^T\mathbf{S} \approx m \sigma_s^2 \mathbf{I}_L = E \cdot \mathbf{I}_L
$$

La matrice $\mathbf{S}^T\mathbf{S}$ est **diagonale** et bien conditionnée.

**Conséquence** :

$$
\text{trace}\left((\mathbf{S}^T\mathbf{S})^{-1}\right) \approx \frac{L}{E}
$$

Ce qui donne la variance minimale possible.

### Comparaison de différents signaux

<div style="text-align: center; margin: 2rem 0;">
  <img src="./img/signal_design_comparison.png" alt="Comparaison de signaux d'entrée" style="max-width: 100%; height: auto;">
  <p style="font-style: italic; color: #666; margin-top: 0.5rem;">Figure 4: Comparaison de différents designs de signaux d'entrée</p>
</div>

**Figure 4** compare 4 types de signaux à énergie constante :

1. **Bruit blanc gaussien** : $\text{cond}(\mathbf{S}^T\mathbf{S}) \approx 1$ (optimal)
2. **PRBS (Pseudo-Random Binary Sequence)** : $\text{cond}(\mathbf{S}^T\mathbf{S}) \approx 1$ (quasi-optimal)
3. **Multi-sinusoïdes** : $\text{cond}(\mathbf{S}^T\mathbf{S}) \approx 10-100$ (acceptable si bien espacées)
4. **Sinusoïde pure** : $\text{cond}(\mathbf{S}^T\mathbf{S}) \to \infty$ (mauvais, matrice singulière)

### Spectre fréquentiel et valeurs propres

Le conditionnement de $\mathbf{S}^T\mathbf{S}$ est directement lié au **spectre** du signal d'entrée :

- **Spectre plat** (bruit blanc) → Toutes les valeurs propres égales → Conditionnement optimal
- **Spectre avec trous** → Certaines valeurs propres faibles → Mauvais conditionnement
- **Ligne spectrale unique** → Une seule valeur propre non nulle → Matrice singulière

**Recommandation pratique** : Utiliser des séquences PRBS ou du bruit blanc gaussien filtré pour l'estimation de canal.

## Code Python

Le script complet est disponible dans `src/canal_estimation.py`.
