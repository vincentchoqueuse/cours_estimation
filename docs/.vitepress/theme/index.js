import DefaultTheme from 'vitepress/theme'
import Bibliography from './components/Bibliography.vue'
import Cite from './components/Cite.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // Enregistrer les composants globalement
    app.component('Bibliography', Bibliography)
    app.component('Cite', Cite)
  }
}
