<script setup>
import references from '../../data/references.js'

const props = defineProps({
  // Clé de la référence (ex: 'lehmann1998')
  refKey: {
    type: String,
    required: true
  },
  // Afficher seulement l'année (ex: [1])
  short: {
    type: Boolean,
    default: false
  }
})

const ref = references[props.refKey]

function formatCitation() {
  if (!ref) {
    return `[${props.refKey}?]`
  }

  if (props.short) {
    // Format court avec lien vers la référence
    return `[${ref.id}]`
  } else {
    // Format long: Auteur (Année)
    const authors = ref.authors.split(',')[0].split('&')[0].trim()
    return `${authors} (${ref.year})`
  }
}
</script>

<template>
  <a
    :href="`#ref-${refKey}`"
    class="citation"
    :title="ref ? ref.title : 'Référence non trouvée'"
  >
    {{ formatCitation() }}
  </a>
</template>

<style scoped>
.citation {
  color: var(--vp-c-brand);
  text-decoration: none;
  font-weight: 500;
}

.citation:hover {
  text-decoration: underline;
}
</style>
