# Makefile pour le projet d'estimation statistique
# Usage: make [target]

# Variables
PYTHON = python
NPM = npm

# Couleurs pour l'affichage
BLUE = \033[0;34m
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

# Scripts Python
SCRIPTS = \
	courses/chapitre1/src/plot_gaussienne.py \
	tutorial/regression-polynomiale/src/polynomial_regression.py \
	tutorial/deconvolution/src/deconvolution.py \
	tutorial/estimation-canal/src/canal_estimation.py \
	courses/chapitre4/src/regression_lineaire.py \
	courses/chapitre5/src/regularization.py \
	courses/chapitre6/src/regression_inference.py \
	tutorial/coefficients-fourier/src/fourier_estimation.py

# Cibles principales
.PHONY: all help run-scripts build dev clean

# Cible par défaut
all: help

## help: Affiche cette aide
help:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Makefile - Projet Estimation Statistique$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(GREEN)Cibles disponibles :$(NC)"
	@echo ""
	@echo "  $(YELLOW)make run-scripts$(NC)      - Exécute tous les scripts Python"
	@echo ""
	@echo "  $(YELLOW)make dev$(NC)              - Lance le serveur de développement VitePress"
	@echo "  $(YELLOW)make build$(NC)            - Build le site VitePress pour production"
	@echo "  $(YELLOW)make preview$(NC)          - Prévisualise le build de production"
	@echo ""
	@echo "  $(YELLOW)make clean$(NC)            - Nettoie les fichiers de build"
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"

## run-scripts: Exécute tous les scripts Python
run-scripts:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Exécution de tous les scripts Python$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@for script in $(SCRIPTS); do \
		if [ -f "$$script" ]; then \
			echo "$(BLUE)► Exécution de $$script...$(NC)"; \
			dir=$$(dirname "$$script"); \
			base=$$(basename "$$script"); \
			(cd "$$dir" && $(PYTHON) "$$base") && \
				echo "$(GREEN)✓ $$script terminé$(NC)" || \
				echo "$(RED)✗ $$script a échoué$(NC)"; \
			echo ""; \
		else \
			echo "$(YELLOW)⚠ $$script non trouvé$(NC)"; \
			echo ""; \
		fi; \
	done
	@echo "$(GREEN)✓ Tous les scripts ont été exécutés$(NC)"

## build: Build le site VitePress pour production
build:
	@echo "$(BLUE)► Build du site VitePress...$(NC)"
	@$(NPM) run docs:build
	@echo "$(GREEN)✓ Site buildé avec succès → .vitepress/dist/$(NC)"

## dev: Lance le serveur de développement VitePress
dev:
	@echo "$(BLUE)► Lancement du serveur de développement...$(NC)"
	@echo "$(YELLOW)  Le site sera accessible sur http://localhost:5173$(NC)"
	@$(NPM) run docs:dev

## preview: Prévisualise le build de production
preview: build
	@echo "$(BLUE)► Prévisualisation du site buildé...$(NC)"
	@$(NPM) run docs:preview

## clean: Nettoie les fichiers de build
clean:
	@echo "$(BLUE)► Nettoyage des fichiers de build...$(NC)"
	@rm -rf .vitepress/dist .vitepress/cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Fichiers de build nettoyés$(NC)"

# Raccourcis utiles
.PHONY: server s b c r

## server (ou s): Raccourci pour 'make dev'
server: dev
s: dev

## b: Raccourci pour 'make build'
b: build

## c: Raccourci pour 'make clean'
c: clean

## r: Raccourci pour 'make run-scripts'
r: run-scripts
