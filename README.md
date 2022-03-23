# Anomalie-Detection

Dans la continuation de notre travail avec Kimura, nous allons maintenant passer au data stream anomaly detection problem.

Il y a en gros 3 techniques pour a détection d’anomalies, qui sont parfaitement résumées dans le papier suivant qui attaque la question du « online » (comme nous pour le clustering) : https://hal.archives-ouvertes.fr/hal-02874869v2/file/IForestASD_ScikitMultiflow_Version2_PourHAL.pdf

Il faudrait faire une implémentation en C de ce qui est proposé dans cet article. Sans doute en réduisant la dimension des données en entrée qui pourraient être des données 2D comme on a fait pour le clustering.

Notre fond de commerce c’est de pouvoir exécuter sur des devices comme celui de Terasic mentionné ci-après. Tu vas reconnaître des SDK pour ce type de board.

L’article sur HAL, lui, suppose que nous travaillons dans l’écosystème Python (voir la communauté Tinyml (https://www.tinyml.org/) , voir micropython (https://micropython.org/), voir river cité dans l’article https://github.com/online-ml/river/ et https://scikit-multiflow.github.io/). C’est la vision un peu orthogonale à la notre. Nous n’allons pas aller dans cette direction.

# Résultats de nos RECHERCHES

Différentes approches ont été conçues pour détecter anomalies : 
- basées sur les statistiques, 
- basées sur le clustering,
- basées sur l'isolement 

# Classification des approches et des méthodes

Généralement, les méthodes de détection d'anomalies sont basées sur le fait que les anomalies sont rares et ont un comportement différent par rapport aux données normales. Ces caractéristiques sont vraies pour les ensembles de données statiques et également pour les flux de données. Les approches de détection d'anomalies les plus utilisées sont les statistiques, le clustering, les plus proches voisins que nous allons
présente ci-dessous. Nous privilégierons une approche basée sur l'isolement : Isolation Forest.

**Basé sur les statistiques** : Les approches basées sur les statistiques établissent généralement un modèle qui caractérise le comportement normal basé sur l'ensemble de données. Les nouvelles données entrantes qui ne correspondent pas au modèle ou qui ont une très faible probabilité de correspondre au modèle sont considéré comme anormal. Certaines méthodes attribuent un score aux données en fonction de l'écarttion du modèle. Les méthodes basées sur les statistiques peuvent être paramétriques dans auquel cas ils doivent avoir une connaissance préalable de la distribution de l'ensemble de données.
Ils peuvent être non paramétriques où ils apprennent de l'ensemble de données donné pour en déduire la distribution sous-jacente. 

**Basé sur le clustering et basé sur les voisins les plus proches** : Clustering et plus proche voisin sont basées sur la proximité entre les observations. les méthodes de cette catégorie sont basées soit sur la distance (fondée sur la distance) soit la densité (basée sur la densité). Les méthodes de clustering divisent l'ensemble de données en différents clusters selon la similarité entre les observations. Le plus éloigné cluster ou le cluster qui a la plus petite densité peut être considéré comme un groupe d'anomalies. Les méthodes des plus proches voisins déterminent les voisins d'une observation en calculant la distance entre toutes les observations de la base de données. L'observation qui est éloignée de ses k plus proches voisins peut être con-
considérée comme une anomalie. Elle est aussi caractérisée comme l'observation qui a le moins de voisins dans un rayon r (un paramètre fixe). Ces approches ont besoin de calculer la distance ou la densité entre toutes les observations dans le ensemble de données ou ils doivent avoir des connaissances préalables sur l'ensemble de données. Afin qu'ils peut souffrir d'une forte consommation de CPU, de temps et de mémoire ou d'un manque d'informations.

**Isolation-based** : le principe de l'isolation-based approche consiste à isoler les observations anormales de l'ensemble de données. Données d'anomalies
sont censés être très différents des normaux. Ils sont également censés représentent une très petite proportion de l'ensemble des données. Ainsi, ils sont susceptibles être rapidement isolé. Certaines méthodes sont basées sur l'isolement. Les méthodes basées sur l'isolement sont différentes des autres statistiques, clustering ou plus proches approches voisines car elles ne calculent pas une distance ou une densité à partir de l'ensemble de données. Par conséquent, ils ont une complexité moindre et sont plus évolutifs. Ils ne souffrent pas du problème de CPU, de mémoire ou de consommation de temps. Ainsi, 
les méthodes basées sur la lation sont adaptées au contexte du flux de données.

# Méthodes  expérimentales de détection d’anomalies très utilisées dans la littérature

Méthodes se basant à la fois sur le type de jeux de données (flux, séries temporelles, graphes, etc.), le domaine d’application et l’approche considérée (statistique, classification, clustering, etc.). Trois algorithmes : 
- **LOF**, Local Outlier Factor (LOF) est une méthode phare de la détection d’anomalies locales basée sur la densité de l’observation en question par rapport à la densité de ses plus proches voisins. Proposée par Breunig et al. (2000), LOF est une méthode non supervisée qui donne un score représentant le degré d’aberrance de l’observation. Les observations dont le degré d’aberrance est largement supérieur à 1 sont considérées comme anomalies. 
- **OC-SVM**, One-Class Support Vector Machine (OC-SVM) est une méthode de détection d’anomalies qui applique des algorithmes de SVM au problème de One class classification (OCC) proposée par Schölkopf et al. (2000, 2001). Le séparateur à vaste marge (SVM) appelé aussi machine à vecteurs de support est très utilisé pour l’apprentissage automatique du fait de sa puissance et de sa polyvalence (classification linéaire, non-linéaire, régression). OCC est une approche de classification semi-supervisée qui consiste à repérer toutes les observations appartenant à une classe précise connue pendant l’apprentissage, dans tout le jeu de données. L’idée clé de cette méthode est de trouver un hyperplan dans un espace de grande dimension qui sépare les anomalies des données normales.
- **Isolation Forest**, IForest est une méthode basée sur les arbres de décision et les forêts aléatoires(Liu et al. (2008, 2012)). Elle utilise l’isolation d’observations à partir de la construction de plusieurs arbres aléatoires. Quand une forêt d’arbres aléatoires et indépendants produit collectivement un chemin d’accès court pour atteindre une observation depuis la racine, celle-ci a une forte probabilité d’être une anomalie. Le nombre d’arbres utilisés est donc un important paramètre pour IForest. Le seuil de la détection est aussi un paramètre clé, il est donné par le score calculé pour chaque observation relativement aux autres observations.


À ce propos vérifions si ces programmes en langage C++ ne traitent pas déjà la question

  1. [ISOTREE](https://github.com/antaresatlantide/anomalie-detection/blob/main/isotree.md)
  2. [LIBISOLATIONFOREST](https://github.com/antaresatlantide/anomalie-detection/blob/main/LibIsolationForest.md)
  3. [RANGER](https://github.com/antaresatlantide/anomalie-detection/blob/main/ranger.md)
  4. [Machine Learning From Scratch with C++](https://github.com/antaresatlantide/anomalie-detection/blob/main/MLfromcrashcpp.md)
     - 4.1 [K-Means](https://github.com/antaresatlantide/anomalie-detection/blob/main/MLfromcrashcpp.md)
     - 4.2 [K-NN](https://github.com/antaresatlantide/anomalie-detection/blob/main/MLfromcrashcpp.md)
     - 4.3 [REGRESSIONTREES](https://github.com/antaresatlantide/anomalie-detection/blob/main/MLfromcrashcpp.md)
    
  
