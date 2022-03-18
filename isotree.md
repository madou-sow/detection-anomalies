# Détecter une valeur aberrante évidente

Référence https://github.com/david-cortes/isotree

Isolation Forest est un algorithme développé à l'origine pour la détection de valeurs aberrantes qui consiste à diviser au hasard des sous-échantillons de données en fonction d'attributs/caractéristiques/colonnes. L'idée est que, plus l'observation est rare, plus il est probable qu'une répartition uniforme aléatoire sur une caractéristique placerait des valeurs aberrantes seules dans une branche, et moins il faudra de divisions pour isoler une observation aberrante comme celle-ci. Le concept est étendu à la division des hyperplans dans le modèle étendu (c'est-à-dire la division par plus d'une colonne à la fois) et aux divisions guidées (pas entièrement aléatoires) dans le modèle SCiForest qui visent à isoler plus rapidement les valeurs aberrantes et à trouver des valeurs aberrantes groupées.

# L'algorithme en Langage C++

1. Données aléatoires d'une distribution normale standard (100 points générés aléatoirement, plus 1 valeur aberrante ajoutée manuellement) La bibliothèque suppose qu'il est passé en tant que pointeur unidimensionnel, suivant l'ordre des colonnes majeures (comme Fortran)

2. Ajoutez maintenant un point aberrant évident (3,3)

3. Ajuster un petit modèle de forêt d'isolement

4. Vérifiez quelle ligne a le score de valeur aberrante le plus élevé

5. Les modèles peuvent être sérialisés et désérialisés de manière très idiomatique
