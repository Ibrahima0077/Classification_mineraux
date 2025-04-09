# Mineral Classifier: CNN + Tabular Model

Ce projet est un modèle de deep learning combinant des **images de minéraux** et des **propriétés physiques (dureté, densité)** pour prédire le type de minéral.  
Il s’inspire d’une recherche dans le cadre du concours PARC AI League 2025.

## Objectif

Créer un modèle capable d’identifier automatiquement un minéral à partir de :
- son image (traitée via CNN – DenseNet121)
- ses propriétés tabulaires (dureté, densité)

## Technologies utilisées

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy, Matplotlib

## Architecture du modèle

- **Entrée 1** : image redimensionnée en 224x224x3, traitée avec un CNN pré-entraîné DenseNet121 (ImageNet)
- **Entrée 2** : données tabulaires (dureté, densité), passées par un MLP
- **Fusion** des deux sources d’information via `Concatenate()`
- **Sortie** : `Dense(10, activation='softmax')` pour prédire le minéral

## Résultats

- Entraînement sur un jeu de données de 10 minéraux.
- Précision atteinte après 5 epochs : ~49% en train, ~30% en validation (à améliorer avec plus de données et d’epochs).


## À venir

- Entraînement sur plus d’epochs
- Augmentation du dataset

## Auteur

Projet réalisé par [IBRAHIMA DIAKITE] dans le cadre de mon apprentissage en Deep Learning.


