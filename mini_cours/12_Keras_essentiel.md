# 12 — Un petit réseau de neurones avec Keras : l'essentiel

> **Utilisé dans** : M1-B2 (tâche 5, puis 8 si le réseau est mis en service), puis le module 6.
> **Prérequis** : `11_Embeddings_essentiel.md`, et la présentation « Entraîner un modèle » (perte, descente de gradient, surapprentissage).


## 1. Un réseau de neurones, c'est quoi ?

Tu connais déjà la régression logistique : une somme pondérée des entrées, passée dans une fonction qui la ramène entre 0 et 1. Un réseau de neurones **empile** ce principe :

```
384 nombres          couche cachée              couche de sortie
(l'embedding)  ──▶  128 « neurones »     ──▶   9 neurones (un par motif)
                    chacun : somme pondérée     softmax : 9 probabilités
                    + fonction ReLU             qui somment à 1
```

- Chaque **neurone** calcule une somme pondérée de ses entrées, puis applique une fonction non linéaire (**ReLU** : garde les valeurs positives, met les négatives à zéro).
- La **couche cachée** permet au réseau de combiner les entrées de façon plus riche qu'une régression logistique.
- La **couche de sortie** a un neurone par classe ; la fonction **softmax** transforme ses 9 valeurs en 9 probabilités.

L'entraînement est celui que tu connais : **minimiser la log loss** (ici sa version multi-classes) par **descente de gradient**, en ajustant tous les poids du réseau.

## 2. Keras, avec PyTorch comme moteur

**Keras** est une bibliothèque qui décrit un réseau en quelques lignes. Depuis sa version 3, elle peut s'appuyer sur plusieurs moteurs de calcul : TensorFlow, JAX ou **PyTorch**. Ici, PyTorch est déjà installé pour les embeddings : on l'utilise aussi pour Keras, et on évite d'installer TensorFlow.

Le moteur se choisit par une variable d'environnement, **avant le premier `import keras`** :

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
```

## 3. Décrire le réseau

```python
import keras

model = keras.Sequential([
    keras.Input(shape=(384,)),                        # la taille d'un vecteur d'entrée
    keras.layers.Dense(64, activation="relu"),        # une couche cachée
    keras.layers.Dense(5, activation="softmax"),      # une sortie par classe (ici 5 classes)
])
model.summary()   # affiche les couches et le nombre de paramètres
```

`Dense` est une couche « complète » : chaque neurone est relié à toutes les entrées. Une couche `Dropout(0.3)` placée après une couche cachée « éteint » au hasard 30 % des neurones à chaque pas d'entraînement : une forme de **régularisation** contre le surapprentissage.

## 4. Compiler : la perte, l'optimiseur, les métriques

```python
model.compile(
    optimizer="adam",                            # une descente de gradient améliorée
    loss="sparse_categorical_crossentropy",      # la log loss multi-classes, cibles = entiers
    metrics=["accuracy"],
)
```

Keras attend des **classes numérotées** (0, 1, 2…), pas des textes. `LabelEncoder` de scikit-learn fait la conversion dans les deux sens :

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder().fit(y_train)          # y_train : les motifs en texte
y_train_num = le.transform(y_train)       # "bruit" → 1, etc.
le.inverse_transform([1])                 # → array(['bruit'])
```

## 5. Entraîner : époques, lots, validation

```python
history = model.fit(
    X_train, y_train_num,
    validation_split=0.2,     # 20 % de l'entraînement mis de côté pour surveiller
    epochs=50,                # nombre maximal de passages sur les données
    batch_size=32,            # nombre d'exemples par pas de descente
)
```

- Une **époque** = un passage complet sur les données d'entraînement.
- À chaque époque, Keras affiche la perte sur l'entraînement (`loss`) et sur la validation (`val_loss`). `history.history` les garde en mémoire pour tracer les courbes.
- ⚠️ `validation_split` prend **les dernières lignes** du tableau, sans les mélanger. Si tes données sont triées (par motif, par date…), la validation ne ressemblera pas à l'entraînement : mélange-les avant, ou utilise un découpage stratifié et `validation_data=(X_val, y_val)`.

## 6. Savoir s'arrêter : les courbes et l'arrêt précoce

```
perte
  │ ╲
  │  ╲___                    entraînement : baisse toujours
  │   ╲  ‾‾‾‾‾‾‾‾‾╲___
  │    ╲_____________________
  │      ╲____ ‿‿‿‾‾‾‾‾‾‾‾‾‾‾  validation : baisse, puis remonte
  │            ↑
  └────────────┼──────────────▶ époques
           meilleur moment
```

C'est la courbe du surapprentissage, vue époque par époque. Plutôt que de deviner le bon nombre d'époques, on laisse Keras **s'arrêter tout seul** quand la perte de validation ne s'améliore plus :

```python
stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model.fit(..., callbacks=[stop])
```

- `patience=5` : on attend 5 époques sans amélioration avant de s'arrêter.
- `restore_best_weights=True` : on garde les poids de la meilleure époque, pas ceux de la dernière.

## 7. Prédire, sauvegarder, recharger

```python
probas = model.predict(X_test)               # une ligne de 9 probabilités par avis
classes = probas.argmax(axis=1)              # l'indice de la plus probable
motifs = le.inverse_transform(classes)       # retour aux noms des motifs

model.save("models/motifs.keras")            # tout le réseau dans un seul fichier
model = keras.models.load_model("models/motifs.keras")
```

Pour mettre le réseau en service, l'API a besoin de Keras, de PyTorch, de la variable `KERAS_BACKEND=torch`, **et** de l'ordre des classes du `LabelEncoder` (à sauvegarder à côté, par exemple dans un JSON). Ce sont autant de dépendances supplémentaires par rapport à une régression logistique : c'est un critère de choix à peser.

## 8. Exercice guidé — un premier réseau sur un jeu jouet

Pour comprendre les courbes sans attendre l'encodage des avis, on part d'un jeu généré :

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=2000, n_features=50, n_informative=10,
                           n_classes=4, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)
```

**Étape 1 — Décrire et compiler** un réseau à une couche cachée de 64 neurones, sortie à 4 classes, comme en sections 3 et 4. Affiche `model.summary()`.

✅ *Le résumé indique 3 524 paramètres entraînables : (50 × 64 + 64) + (64 × 4 + 4).*

**Étape 2 — Entraîner 100 époques sans arrêt précoce**, avec `validation_split=0.2`, puis trace `loss` et `val_loss` avec matplotlib.

✅ *La perte d'entraînement baisse jusqu'au bout (vers 0,01) ; la perte de validation atteint son minimum vers la 30e époque, puis remonte nettement : c'est le surapprentissage.*

**Étape 3 — Ajouter l'arrêt précoce** et recommencer (en reconstruisant le modèle). À quelle époque s'arrête-t-il ?

**Étape 4 — Ajouter un `Dropout(0.3)`** après la couche cachée. L'écart entre les deux courbes change-t-il ?

Puis applique la même démarche à tes embeddings d'avis (tâche 5 du brief).

## 9. Pièges fréquents

| Symptôme | Cause probable |
|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | `KERAS_BACKEND` réglé après `import keras`, ou pas réglé du tout : redémarrer le noyau, régler la variable en premier |
| `ValueError` sur la perte, ou cibles refusées | Motifs passés en texte : les convertir en entiers avec `LabelEncoder` |
| La sortie ne somme pas à 1, ou une seule probabilité | Mauvaise activation de sortie : `softmax` avec un neurone par classe |
| `val_loss` remonte dès les premières époques | Surapprentissage : réseau trop gros, pas de dropout, ou validation qui ne ressemble pas à l'entraînement |
| La validation est bien pire que prévu, dès le départ | Données triées avant `validation_split` : mélanger, ou fournir `validation_data` |
| Résultats un peu différents à chaque exécution | Initialisation aléatoire des poids : `keras.utils.set_random_seed(42)` pour les fixer |
| Ré-entraîner dans la même cellule continue l'ancien modèle | `fit` reprend où il s'était arrêté : reconstruire le modèle avant chaque essai |
| L'API renvoie des motifs décalés | Ordre des classes perdu : sauvegarder `le.classes_` avec le modèle |

## Pour aller plus loin

- Le guide d'introduction de Keras : https://keras.io/getting_started/
- Les couches disponibles : https://keras.io/api/layers/
