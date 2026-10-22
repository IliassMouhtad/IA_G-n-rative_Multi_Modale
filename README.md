# 📘 Travail à Faire : Bases de PyTorch & Deep Learning

Ce dépôt présente la réalisation complète de l'activité pratique / devoir sur les **Bases de PyTorch**, couvrant les fondamentaux du calcul tensoriel, la différentiation automatique, ainsi que la conception, l'entraînement et l'évaluation de modèles de Deep Learning (MLP et CNN).

---

## 👨‍💻 Auteur
- **Étudiant :** Iliass Mouhtad
- **Cadre :** Module d'Intelligence Artificielle / Deep Learning & IA Multimodale

---

## 🎯 Objectifs du TP
1. **Maîtriser les tenseurs PyTorch** : manipulation des formes, types de données et exécution sur CPU / GPU.
2. **Comprendre le mécanisme d'Autograd** : construction dynamique du graphe de calcul et rétropropagation du gradient (`backward()`).
3. **Implémenter la descente de gradient** : régression linéaire simple et multi-variables avec calcul des pertes et optimisation via `torch.optim`.
4. **Construire des réseaux de neurones denses (MLP)** :
   - Classification multi-classes sur le jeu de données **Iris**.
   - Classification d'images sur **MNIST** (avec suivi d'époques, matrice de confusion et rapport de métriques).
5. **Concevoir un réseau de neurones convolutif (CNN)** :
   - Empilement de couches `Conv2d`, fonctions d'activation `ReLU` et pooling `MaxPool2d`.
   - Régularisation par `Dropout`.
   - Analyse comparative des performances et visualisation des erreurs de classification.
   - Sauvegarde et rechargement des poids du modèle (`state_dict`).

---

## 📁 Structure du Dépôt

| Fichier | Description |
|---|---|
| **[`Activité pratique Bases de Pytorch.ipynb`](Activité%20pratique%20Bases%20de%20Pytorch.ipynb)** | Notebook Jupyter principal contenant l'intégralité du code complété, commenté et exécuté avec tous les résultats et tracés graphiques. |
| **[`graphe_operations.png`](graphe_operations.png)** | Schéma du graphe computationnel illustrant les opérations avant et la propagation arrière des gradients. |
| **[`README.md`](README.md)** | Rapport synthétique du travail effectué et guide d'exécution. |

---

## 📊 Résumé des Travaux & Résultats Obtenus

### 1. Tenseurs & Autograd
- Manipulation des tenseurs, vérification de la disponibilité de l'accélération matérielle.
- Modélisation du calcul de gradient sur une fonction analytique :
  $$y = w \cdot x + b, \quad \mathcal{L} = (y - y_{\text{cible}})^2$$
- Calcul et vérification mathématique des dérivées partielles $\frac{\partial \mathcal{L}}{\partial w}$ et $\frac{\partial \mathcal{L}}{\partial b}$ (illustré dans [`graphe_operations.png`](graphe_operations.png)).

### 2. Régression Linéaire Multi-variables (Prédiction de Salaires)
- Modélisation du salaire à partir de l'ancienneté et du niveau d'expertise ($X \in \mathbb{R}^{300 \times 2}$).
- Entraînement par descente de gradient avec `nn.MSELoss` et `torch.optim.Adam`.
- Convergence de la perte et identification des coefficients réels avec une erreur absolue moyenne (**MAE**) de **381.64 DH**.

### 3. Classification Multi-classes (Dataset Iris)
- Normalisation des attributs (`StandardScaler`) et séparation entraînement/test stratifiée (80% / 20%).
- Architecture MLP ($4 \rightarrow 16 \rightarrow 3$) avec activation ReLU et fonction de coût `CrossEntropyLoss`.
- **Résultats :**
  - **Accuracy sur le jeu de test : 100.0%** (contre 90.0% pour un modèle linéaire direct sans couche cachée).
  - Matrice de confusion sans aucune confusion.
  - Inférence validée sur un nouvel échantillon avec probabilités Softmax.

### 4. Classification d'Images MNIST (Perceptron Multicouche - MLP)
- Chargement des données normalisées via `DataLoader` (lots de 64 pour l'entraînement, 256 pour le test).
- Modèle : aplatissement (`Flatten`) $\rightarrow$ linéaire ($784 \rightarrow 128$) $\rightarrow$ ReLU $\rightarrow$ linéaire ($128 \rightarrow 10$).
- Entraînement sur 5 époques.
- **Performances sur le jeu de test (10 000 images) :**
  - **Accuracy globale : 97.58%**
  - **Précision macro : 97.57%**
  - **Rappel macro : 97.55%**
  - **F1-Score macro : 97.56%**
- Visualisation de la matrice de confusion $10 \times 10$ et identification des images de chiffres ambigus mal classés.
- Sauvegarde du modèle et vérification de rechargement validée.

### 5. Classification d'Images MNIST (Réseau Convolutif - CNN)
- **Architecture :**
  - Bloc 1 : `Conv2d(1, 16, kernel=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)`
  - Bloc 2 : `Conv2d(16, 32, kernel=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)`
  - Classifieur : `Flatten` $\rightarrow$ `Linear(1568, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(p=0.5)` $\rightarrow$ `Linear(128, 10)`
- Entraînement sur 3 époques (perte finale réduite à $0.0351$).
- **Performances :**
  - **Accuracy sur le jeu de test : 98.88%** (nette amélioration par rapport au MLP).
  - Tracé de la courbe de perte.
  - Affichage détaillé d'échantillons mal classés pour interprétation des erreurs.
  - Sauvegarde et rechargement des poids du réseau convolutionnel avec validation de consistance.

---

## ⚙️ Instructions d'Exécution

### Prérequis
- Python 3.10 ou supérieur
- Jupyter Notebook / JupyterLab ou VS Code

### Installation des bibliothèques nécessaires
```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

### Lancement du Notebook
```bash
# Cloner le dépôt
git clone https://github.com/IliassMouhtad/IA_G-n-rative_Multi_Modale.git
cd IA_G-n-rative_Multi_Modale

# Lancer Jupyter Notebook
jupyter notebook "Activité pratique Bases de Pytorch.ipynb"
```
