# Anomalie-Detection

Dans la continuation de notre travail avec Kimura, nous allons maintenant passer au data stream anomaly detection problem.

Il y a en gros 3 techniques pour a détection d’anomalies, qui sont parfaitement résumées dans le papier suivant qui attaque la question du « online » (comme nous pour le clustering) : https://hal.archives-ouvertes.fr/hal-02874869v2/file/IForestASD_ScikitMultiflow_Version2_PourHAL.pdf

Il faudrait faire une implémentation en C de ce qui est proposé dans cet article. Sans doute en réduisant la dimension des données en entrée qui pourraient être des données 2D comme on a fait pour le clustering.

Notre fond de commerce c’est de pouvoir exécuter sur des devices comme celui de Terasic mentionné ci-après. Tu vas reconnaître des SDK pour ce type de board.

L’article sur HAL, lui, suppose que nous travaillons dans l’écosystème Python (voir la communauté Tinyml (https://www.tinyml.org/) , voir micropython (https://micropython.org/), voir river cité dans l’article https://github.com/online-ml/river/ et https://scikit-multiflow.github.io/). C’est la vision un peu orthogonale à la notre. Nous n’allons pas aller dans cette direction.

# Resultats

A ce propos nous nous sommes orienté vers quelques programmes 

  1. [ISOTREE](https://github.com/david-cortes/isotree)
  2. [LIBISOLATIONFOREST](https://github.com/msimms/LibIsolationForest)
  3. [RANGER](https://github.com/imbs-hl/ranger)
  4. [Machine Learning From Scratch with C++](https://github.com/magikerwin1993/ML-From-Scratch-With-CPP)
     - 4.1 [K-Means](https://github.com/magikerwin1993/ML-From-Scratch-With-CPP/tree/main/k-means)
     - 4.2 [K-NN](https://github.com/magikerwin1993/ML-From-Scratch-With-CPP/tree/main/k-nn)
     - 4.3 [REGRESSIONTREES](https://github.com/magikerwin1993/ML-From-Scratch-With-CPP/tree/main/RegressionTrees)
    
  
