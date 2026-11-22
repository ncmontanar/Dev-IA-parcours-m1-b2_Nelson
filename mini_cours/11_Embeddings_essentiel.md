# 11 — Les embeddings : l'essentiel

> **Utilisé dans** : M1-B2 (tâches 1, 4 et 8), puis le module 3 (recherche par similarité) et l'IA générative.
> **Prérequis** : `05_HuggingFace_essentiel.md`, `09_Pipeline_sklearn_essentiel.md`.

## 1. Le problème : un modèle ne lit pas du texte

Un classifieur scikit-learn attend des **nombres**. Pour classer des avis, il faut d'abord transformer chaque texte en un vecteur de nombres. Deux grandes façons de faire :

| | Compter les mots (TF-IDF) | Embeddings |
|---|---|---|
| Principe | une colonne par mot du vocabulaire, la valeur dit si le mot est présent et à quel point il est rare | un modèle pré-entraîné résume le **sens** du texte en quelques centaines de nombres |
| Taille du vecteur | des milliers de colonnes, presque toutes à zéro | fixe, par exemple 384 |
| « matelas défoncé » et « sommier affaissé » | aucun mot commun : vecteurs sans rapport | sens proche : vecteurs proches |
| Coût | instantané, aucun téléchargement | un modèle de plusieurs centaines de Mo, plus lent |

TF-IDF est une excellente référence, rapide et simple. Sa limite : il ne connaît que les mots qu'il a vus à l'entraînement.

## 2. Ce qu'est un embedding

Un modèle d'embeddings a été entraîné, sur des millions de phrases, à placer les textes **de sens proche** à des endroits proches d'un espace à plusieurs centaines de dimensions.

```
                   « le lit était inconfortable »  ●
                       « matelas défoncé »  ●       ← proches
espace à 384
dimensions
                                  « wifi en panne »  ●   ← loin
```

On ne lit pas les 384 nombres un par un : ils n'ont pas de sens isolément. Ce qui compte, ce sont les **distances** entre vecteurs. La mesure la plus utilisée est la **similarité cosinus** : 1 pour deux vecteurs dans la même direction, autour de 0 pour des textes sans rapport.

C'est de l'**apprentissage par transfert** : on réutilise ce qu'un modèle a appris sur une tâche générale (comprendre des phrases) pour une tâche à nous (classer des avis), sans le ré-entraîner.

## 3. En pratique avec `sentence-transformers`

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vecteurs = encoder.encode(["Le lit était très inconfortable.", "Le matelas était défoncé."])
vecteurs.shape      # (2, 384) : un vecteur de 384 nombres par phrase
```

- Au premier appel, le modèle est **téléchargé** (environ 470 Mo) puis gardé en cache.
- `encode` accepte une liste de textes et renvoie un tableau numpy : une ligne par texte.
- `batch_size=64` encode les textes par paquets : plus rapide sur un gros volume.
- Similarité entre deux lots de vecteurs : `from sentence_transformers.util import cos_sim`.

Comme pour tout modèle pré-entraîné, **lis sa fiche** (cf. mini-cours 05) : langues couvertes, données d'entraînement, et **longueur maximale**. Ce modèle tronque silencieusement les textes au-delà d'environ 128 tokens : un avis très long n'est résumé que par son début.

## 4. Encoder une fois, réutiliser souvent

Encoder un millier d'avis prend de quelques secondes à une minute selon la machine. Les vecteurs ne changent pas tant que les textes et le modèle ne changent pas : **on les calcule une fois et on les sauvegarde**.

```python
from pathlib import Path
import numpy as np

cache = Path("data/cache/avis.npy")
if cache.exists():
    X = np.load(cache)
else:
    X = encoder.encode(avis["texte"].tolist(), batch_size=64, show_progress_bar=True)
    np.save(cache, X)
```

⚠️ Les lignes du tableau sont dans **le même ordre** que les textes encodés. Si tu filtres ou mélanges le `DataFrame` ensuite, applique la même sélection aux vecteurs (par exemple avec les mêmes index), sinon chaque vecteur se retrouve associé au mauvais motif.

## 5. Des vecteurs à un classifieur

Une fois les textes transformés en vecteurs, on revient en terrain connu : ce sont des **données tabulaires**, 384 colonnes numériques. N'importe quel classifieur de scikit-learn s'utilise comme d'habitude :

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=3000).fit(X_train, y_train)
clf.predict(X_test)
```

Pas besoin de mettre à l'échelle ni d'encoder : les vecteurs sont déjà des nombres de même ordre de grandeur.

**En production**, la chaîne complète est : texte → modèle d'embeddings → vecteur → classifieur → motif. Les deux modèles doivent être chargés au démarrage de l'API, et **le même modèle d'embeddings** doit servir à l'entraînement et à la prédiction : un classifieur entraîné sur les vecteurs d'un modèle ne comprend pas ceux d'un autre.

## 6. Évaluer un classifieur à plusieurs classes

Avec 9 motifs, la matrice de confusion devient un tableau 9 × 9, et chaque motif a sa propre précision, son rappel et son F1 (`classification_report` les affiche tous). Pour résumer en un chiffre :

| Moyenne | Calcul | Quand l'utiliser |
|---|---|---|
| `average="macro"` | moyenne simple des F1 des motifs | chaque motif compte autant, même rare |
| `average="weighted"` | moyenne pondérée par le nombre d'avis de chaque motif | les motifs fréquents comptent plus |

Le service qualité veut repérer **tous** les motifs, y compris les rares (l'emplacement, par exemple) : le F1 macro est le plus adapté.

## 7. Exercice guidé — prendre en main les embeddings

**Étape 1 — Sentir la similarité.** Encode ces phrases et affiche la matrice de similarité cosinus :

```python
phrases = [
    "Le matelas était défoncé.",
    "Le sommier s'affaissait au milieu.",
    "Le wifi ne fonctionnait pas.",
    "Connexion internet inexistante.",
]
```

✅ *Les deux phrases sur la literie sont très proches (environ 0,7), sans un seul mot en commun. Les deux sur la connexion sont proches aussi (environ 0,5), et bien plus que « matelas » et « connexion » (environ 0,1). Les valeurs exactes dépendent du modèle : c'est l'ordre qui compte.*

**Étape 2 — Encoder le corpus et mettre en cache.** Encode les avis de `avis_motifs.csv` et de `avis_recents.csv`, sauvegarde les deux tableaux dans `data/cache/`, puis relance la cellule.

✅ *Le deuxième lancement est instantané. Les tableaux ont autant de lignes que de textes et 384 colonnes.*

**Étape 3 — Un premier classifieur.** Découpe les vecteurs **avec le même découpage** que celui de ta référence TF-IDF (mêmes index), entraîne une régression logistique, et calcule le F1 macro.

**Étape 4 — Chercher des voisins.** Pour un avis du test, trouve les 3 avis d'entraînement les plus proches (similarité cosinus) et lis-les. Ont-ils le même motif ?

## 8. Pièges fréquents

| Symptôme | Cause probable |
|---|---|
| Chaque essai prend plusieurs minutes | Les textes sont ré-encodés à chaque fois : mettre les vecteurs en cache |
| Scores proches du hasard | Vecteurs et étiquettes désalignés après un filtre ou un mélange |
| `ValueError: X has 384 features, but ... is expecting 1500 features` | Un classifieur entraîné sur TF-IDF reçoit des embeddings, ou l'inverse |
| Prédictions absurdes en production, bonnes en test | Pas le même modèle d'embeddings à l'entraînement et dans l'API |
| Le premier lancement semble bloqué | Téléchargement du modèle (environ 470 Mo) |
| `OSError: We couldn't connect to 'https://huggingface.co'` | Pas de connexion au premier lancement, ou nom du modèle mal orthographié |
| Les avis longs sont mal classés | Texte tronqué au-delà de la longueur maximale du modèle |
| Résultats différents d'un poste à l'autre, très légèrement | Normal : petites différences de calcul selon le processeur |

## Pour aller plus loin

- Documentation de `sentence-transformers` : https://sbert.net
- La fiche du modèle utilisé : https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
