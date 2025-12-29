import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'
import footnote from 'markdown-it-footnote'

export default defineConfig({
  title: 'Estimation Statistique',
  description: 'Cours sur l\'estimation statistique',
  base: '/cours_estimation/',

  themeConfig: {
    editLink: {
      pattern: 'https://github.com/vincentchoqueuse/cours_estimation/issues',
      text: 'Suggest any modification on GitLab'
    },
    outline: {
      level: [2, 3], // Afficher les titres de niveau 2 (##) et 3 (###)
      label: 'Sur cette page'
    },

    nav: [
      { text: 'Accueil', link: '/' },
      { text: 'Cours', link: '/courses/', activeMatch: '/courses/' },
      { text: 'Tutoriels', link: '/tutorial/', activeMatch: '/tutorial/' },
      { text: 'Cheatsheet', link: '/cheatsheet/', activeMatch: '/cheatsheet/' }
    ],

    sidebar: {
      '/courses/': [
        {
          text: 'Estimation Statistique',
          items: [
            { text: 'Introduction', link: '/courses/' },
            { text: 'Chapitre 1: Concepts de base', link: '/courses/chapitre1' },
            { text: 'Chapitre 2: Estimateurs ponctuels', link: '/courses/chapitre2' },
            { text: 'Chapitre 3: Estimation bayésienne', link: '/courses/chapitre3' }
          ]
        },
        {
          text: 'Régression Linéaire',
          items: [
            { text: 'Chapitre 4: Fondements', link: '/courses/chapitre4' },
            { text: 'Chapitre 5: Régularisation', link: '/courses/chapitre5' },
            { text: 'Chapitre 6: Inférence et diagnostic', link: '/courses/chapitre6' }
          ]
        }
      ],
      '/tutorial/': [
        {
          text: 'Tutoriels',
          items: [
            { text: 'Introduction', link: '/tutorial/' },
            { text: 'Régression polynomiale', link: '/tutorial/regression-polynomiale' },
            { text: 'Coefficients de Fourier', link: '/tutorial/coefficients-fourier' },
            { text: 'Déconvolution', link: '/tutorial/deconvolution' },
            { text: 'Estimation de canal FIR', link: '/tutorial/estimation-canal' }
          ]
        }
      ],
      '/cheatsheet/': [
        {
          text: 'Aide-mémoire',
          items: [
            { text: 'Formules essentielles', link: '/cheatsheet/' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com' }
    ]
  },

  markdown: {
    config: (md) => {
      md.use(mathjax3)
      md.use(footnote)
    }
  },

  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => tag.includes('mjx-')
      }
    }
  }
})
