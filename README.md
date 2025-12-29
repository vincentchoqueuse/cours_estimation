# Cours d'Estimation Statistique

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![VitePress](https://img.shields.io/badge/VitePress-1.x-brightgreen.svg)](https://vitepress.dev/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

Site de cours interactif sur l'estimation statistique construit avec VitePress, incluant des visualisations Python et des démonstrations mathématiques.

## 📚 Contenu du cours

### Partie 1 : Fondamentaux de l'estimation

- **Chapitre 1** : Concepts de base (estimateurs, biais, variance, borne de Cramér-Rao)
- **Chapitre 2** : Estimateurs ponctuels (méthode des moments, maximum de vraisemblance)
- **Chapitre 3** : Estimation bayésienne (MAP, EAP, lois a priori)

### Partie 2 : Régression linéaire

- **Chapitre 4** : Fondements (OLS, propriétés, théorème de Gauss-Markov, R²)
- **Chapitre 5** : Régularisation (Ridge, LASSO, Elastic Net, validation croisée)
- **Chapitre 6** : Inférence et diagnostic (tests, intervalles de confiance, résidus)

### Tutoriels pratiques

- **Régression polynomiale** : Approximation de données par polynômes
- **Coefficients de Fourier** : Estimation de séries de Fourier
- **Déconvolution** : Convolution classique vs circulaire, applications OFDM
- **Estimation de canal FIR** : Identification de systèmes, design optimal de signaux

## 🚀 Démarrage rapide

### Prérequis

- Node.js (v18 ou supérieur)
- Python 3.8+ (pour générer les figures)
- npm ou yarn

### Installation

```bash
# Cloner le dépôt
git clone <repository-url>
cd estimation

# Installer les dépendances Node.js
npm install

# Créer un environnement virtuel Python (optionnel)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances Python
pip install -r requirements.txt
```

### Développement

```bash
# Lancer le serveur de développement VitePress
npm run docs:dev
# ou
make dev

# Le site sera accessible sur http://localhost:5173
```

### Générer les figures

```bash
# Exécuter tous les scripts Python pour générer les visualisations
make run-scripts

# Ou exécuter un script spécifique
cd docs/courses/chapitre1/src
python plot_gaussienne.py
```

### Build pour production

```bash
# Construire le site statique
npm run docs:build
# ou
make build

# Prévisualiser le build de production
npm run docs:preview
# ou
make preview
```

## 📁 Structure du projet

```
estimation/
├── docs/                          # Sources du site
│   ├── .vitepress/
│   │   ├── config.js             # Configuration VitePress
│   │   ├── theme/
│   │   │   ├── index.js          # Thème personnalisé
│   │   │   ├── components/       # Composants Vue (Cite, Bibliography)
│   │   │   └── custom.css        # Styles personnalisés
│   │   └── data/
│   │       └── references.js     # Base de données bibliographique
│   ├── courses/                  # Chapitres de cours
│   │   ├── chapitre1/
│   │   │   ├── index.md
│   │   │   ├── img/              # Figures générées
│   │   │   └── src/              # Scripts Python
│   │   ├── chapitre2/...
│   │   └── ...
│   ├── tutorial/                 # Tutoriels pratiques
│   │   ├── regression-polynomiale/
│   │   ├── coefficients-fourier/
│   │   ├── deconvolution/
│   │   └── estimation-canal/
│   ├── cheatsheet/               # Aide-mémoire
│   ├── index.md                  # Page d'accueil
│   └── Makefile                  # Commandes utiles
├── package.json
├── requirements.txt              # Dépendances Python
├── LICENSE
└── README.md
```

## ✨ Fonctionnalités

### Mathématiques

- **Support LaTeX complet** avec MathJax 3
- Formules inline : `$\bar{x} = \frac{1}{n}\sum x_i$`
- Formules display : `$$\text{MSE} = E[(\hat{\theta} - \theta)^2]$$`

### Références bibliographiques

- Système de citations structuré avec composant `<Cite>`
- Bibliographies automatiques avec `<Bibliography>`
- Base centralisée dans `.vitepress/data/references.js`

### Visualisations interactives

- **Scripts Python** pour générer toutes les figures
- Visualisations matplotlib haute qualité (300 DPI)
- Automatisation avec Makefile

### Navigation

- Sidebar hiérarchique
- Table des matières contextuelle
- Liens prev/next entre chapitres
- Recherche intégrée

## 🔧 Commandes Makefile

```bash
make dev              # Lance le serveur de développement
make build            # Build le site pour production
make preview          # Prévisualise le build
make run-scripts      # Exécute tous les scripts Python
make clean            # Nettoie les fichiers de build
```

## 📖 Utilisation

### Ajouter une formule mathématique

```markdown
La moyenne empirique est $\bar{X}_n = \frac{1}{n} \sum_{i=1}^{n} X_i$.

L'estimateur du maximum de vraisemblance satisfait :

$$
\frac{\partial \ell(\theta)}{\partial \theta} = 0
$$
```

### Citer une référence

```markdown
Selon <Cite refKey="kay1993" />, la borne de Cramér-Rao établit...

Version courte : <Cite refKey="casella2002" short />

## Références

<Bibliography :keys="['kay1993', 'casella2002', 'lehmann1998']" />
```

### Ajouter un nouveau chapitre

1. Créer le dossier `docs/courses/chapitre-n/`
2. Ajouter `index.md`, `src/`, `img/`
3. Mettre à jour `.vitepress/config.js` (sidebar)
4. Ajouter le script Python au Makefile

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

- Signaler des erreurs ou typos
- Proposer des améliorations de contenu
- Ajouter des exercices ou tutoriels
- Améliorer les visualisations

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- [VitePress](https://vitepress.dev/) pour le framework de documentation
- [MathJax](https://www.mathjax.org/) pour le rendu LaTeX
- Communauté Python scientifique (NumPy, Matplotlib, SciPy, scikit-learn)

## 📧 Contact

Pour toute question ou suggestion, ouvrez une issue sur GitHub.
