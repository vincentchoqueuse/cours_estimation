<script setup>
import { computed } from "vue";
import references from "../../data/references.js";

const props = defineProps({
  // Liste des clés de références à afficher (ex: ['lehmann1998', 'casella2002'])
  keys: {
    type: Array,
    default: () => [],
  },
  // Afficher toutes les références si true
  all: {
    type: Boolean,
    default: false,
  },
});

const displayedRefs = computed(() => {
  if (props.all) {
    return Object.values(references);
  }
  return props.keys.map((key) => references[key]).filter(Boolean);
});

function formatReference(ref) {
  let formatted = `${ref.authors} (${ref.year}). `;

  if (ref.type === "book") {
    formatted += `<em>${ref.title}</em>`;
    if (ref.edition) {
      formatted += `, ${ref.edition}`;
    }
    if (ref.publisher) {
      formatted += `. ${ref.publisher}`;
    }
  } else if (ref.type === "article") {
    formatted += `"${ref.title}". `;
    if (ref.journal) {
      formatted += `<em>${ref.journal}</em>`;
    }
    if (ref.volume) {
      formatted += `, ${ref.volume}`;
    }
    if (ref.pages) {
      formatted += `, pp. ${ref.pages}`;
    }
  } else {
    formatted += ref.title;
    if (ref.publisher) {
      formatted += `. ${ref.publisher}`;
    }
  }

  return formatted + ".";
}
</script>

<template>
  <div class="bibliography">
    <h2>Références</h2>
    <ol class="reference-list">
      <li
        v-for="ref in displayedRefs"
        :key="ref.id"
        :id="`ref-${ref.id}`"
        class="reference-item"
      >
        <span v-html="formatReference(ref)"></span>
      </li>
    </ol>
  </div>
</template>

<style scoped>
.bibliography {
  padding-top: 2rem;
}

.bibliography h2 {
  font-size: 1.5rem;
  margin-bottom: 1rem;
  color: var(--vp-c-text-1);
}

.reference-list {
  list-style: none;
  counter-reset: reference-counter;
  padding-left: 0;
}

.reference-item {
  counter-increment: reference-counter;
  margin-bottom: 0.75rem;
  padding-left: 2.5rem;
  position: relative;
  line-height: 1.6;
  color: var(--vp-c-text-2);
}

.reference-item::before {
  content: "[" counter(reference-counter) "]";
  position: absolute;
  left: 0;
  font-weight: 600;
  color: var(--vp-c-brand);
}

.reference-item em {
  font-style: italic;
}
</style>
