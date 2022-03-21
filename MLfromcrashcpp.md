
 ## Machine Learning From Scratch with C++
 
 [Référence](https://github.com/eriklindernoren/ML-From-Scratch)

Implementations of Classic Machine Learning Algorithm with C++
(some visualization works are implemented with python)

Most of the machine learning implementations are based on python, and I hope to provide some reference for C++ enthusiasts through this repository.


### [k-means](https://github.com/magikerwin1993/ML-From-Scatch-With-CPP/tree/main/k-means)

J'ai une histoire un peu compliquée en ce qui concerne C++. Quand j'avais 15 ans et que j'apprenais à coder, je n'arrivais pas à choisir entre python et C++ et j'ai donc essayé d'apprendre les deux en même temps. L'un de mes premiers projets non triviaux était un programme C++ pour calculer des orbites - en y repensant maintenant, je peux voir que ce que je faisais réellement était une implémentation (horriblement inefficace) de la méthode d'Euler. Je ne pouvais tout simplement pas comprendre les tableaux de taille fixe (sans parler des pointeurs !). Dans tous les cas, j'ai vite réalisé que jongler avec C++ et python était intenable - non seulement j'étais nouveau dans les concepts (tels que les systèmes de types et la POO), mais je devais apprendre deux ensembles de syntaxe en plus de deux saveurs de ces concepts. J'ai décidé de m'engager dans Python et je n'ai pas vraiment regardé en arrière depuis.

Maintenant, presque 6 ans plus tard (tempus fugit !), après avoir terminé le cours d'informatique de première année à Cambridge, j'ai l'impression d'être dans un bien meilleur endroit pour m'initier au C++. Ma motivation est aidée par le fait que tous les travaux pratiques de calcul de deuxième année pour la physique sont effectués en C++, sans compter que le C++ est incroyablement utile en finance quantitative (ce qui m'intéresse profondément).

À cette fin, j'ai décidé de me lancer directement et d'implémenter un algorithme d'apprentissage automatique à partir de zéro. J'ai choisi k-means en raison de sa signification personnelle pour moi : lorsque j'ai découvert ML pour la première fois, k-means a été l'un des premiers algorithmes que j'ai pleinement étudié et j'ai passé un bon moment à expérimenter différentes modifications et implémentations en python. De plus, étant donné que l'objectif principal de cet article est d'apprendre le C++, il est logique d'utiliser un algorithme que je comprends relativement bien.

S'il vous plaît, permettez-moi d'ajouter l'avertissement que cela ne sera certainement pas une solution optimale - ce message est vraiment un exercice d'apprentissage pour moi et je serais plus qu'heureux de recevoir des critiques constructives. Comme toujours, tout le code de ce projet se trouve sur GitHub.

### Qu'est-ce que le clustering k-means ?

J'ai décidé de donner quatre brèves explications avec une rigueur croissante. Rien au-delà de la première explication n'est vraiment essentiel pour la suite de cet article, alors n'hésitez pas à vous arrêter à tout moment.

https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/

1. Le clustering k-means nous permet de trouver des groupes de points similaires dans un ensemble de données.
2. Le clustering k-means consiste à trouver des groupes de points dans un ensemble de données de sorte que la variance totale au sein des groupes soit minimisée.
3. Le clustering k-means est la tâche de partitionner l'espace des caractéristiques en k sous-ensembles afin de minimiser les écarts de somme des carrés (WCSS) intra-cluster, qui est la somme des distances quadra euclidiennes entre chaque point de données et le centroïde.
4. Formellement, le clustering k-means consiste à trouver une partition S={S1,S2,…Sk} où S satisfait :
argmin∑i=1k∑x∈Si∥x−μi∥2

### L'algorithme des k-moyennes
Le problème de clustering k-means est en fait incroyablement difficile à résoudre. Disons que nous avons juste N = 120 et k = 5, c'est-à-dire que nous avons 120 points de données que nous voulons regrouper en 5 clusters. Le nombre de partitions possibles est supérieur au nombre d'atomes dans l'univers (5120≈1083) – pour chacune, il faut alors calculer le WCSS (lire : variance) et choisir la meilleure partition.

De toute évidence, tout type de solutions de force brute est insoluble (pour être précis, le problème a une complexité exponentielle). Il faut donc se tourner vers des solutions approchées. L'algorithme approximatif le plus célèbre est l'algorithme de Lloyd, qui est souvent appelé "l'algorithme des k-moyennes". Dans cet article, je vais faire taire mon pédant intérieur et utiliser de manière interchangeable les termes algorithme k-means et clustering k-means, mais il convient de rappeler qu'ils sont légèrement distincts. Cela mis à part, l'algorithme de Lloyd est incroyablement simple :

1. Initialiser les clusters

L'algorithme doit commencer quelque part, nous devons donc trouver une manière grossière de regrouper les points. Pour ce faire, nous sélectionnons au hasard k points qui deviennent des «marqueurs», puis attribuons chaque point de données à son point marqueur le plus proche. Le résultat est k clusters. Bien qu'il s'agisse d'une méthode d'initialisation naïve, elle possède de belles propriétés - les régions plus densément peuplées sont plus susceptibles de contenir des centroïdes (ce qui est logique).

2. Calculer le centroïde de chaque cluster

Techniquement, l'algorithme de Lloyd calcule le centroïde de chaque partition de l'espace 3D via l'intégration, mais nous utilisons l'approximation raisonnable du calcul du centre de masse des points dans une partition donnée. Le rationnel derrière cela est que le centre de gravité d'un cluster « caractérise » le cluster dans un certain sens.

3. Attribuez chaque point au centroïde le plus proche et redéfinissez le cluster

Si un point actuellement dans le cluster 1 est en fait plus proche du centre de gravité du cluster 2, il est sûrement plus logique qu'il appartienne au cluster 2 ? C'est exactement ce que nous faisons, en boucle sur tous les points et en les attribuant à des clusters en fonction du centroïde le plus proche.

4. Répétez les étapes 2 et 3

Nous recalculons ensuite à plusieurs reprises les centroïdes et réattribuons les points au centroïde le plus proche. Il existe en fait une preuve très nette que cela converge : essentiellement, il n'y a qu'un nombre fini (bien que massif) de partitions possibles, et chaque mise à jour de k-means améliore au moins le WCSS. L'algorithme doit donc converger.

**Programme**


    /* 
    mamadou@port-lipn12:~/big-data/cerin24022022/ML-From-Scratch-With-CPP/k-means$ cat main.cpp
    k-means clustering is the task of finding groups of 
    pointrs in a dataset such that the total variance within
    groups is minimized.

    --> find argmin(sum(xi - ci)^2)

    algorithm:

    1. init the clusters

    iterations {
        2. assign each point to the nearest centroid
        3. redefine the cluster
    }

    */

    #include <ctime>    // for a random seed
    #include <fstream>  // for file reading
    #include <sstream>
    #include <iostream>
    #include <vector>
    #include <cmath>    // for pow()
    #include <cfloat>   // for DBL_MAX

    using namespace std;

    struct Point {
        int x, y;
        int cluster;
        double minDistance;

        Point(int _x, int _y) {
            x = _x;
            y = _y;
            cluster = -1;
            minDistance = DBL_MAX;
        }

        double distance(Point p) {
            return pow((this->x - p.x), 2) + pow(this->y - p.y, 2);
        }
    };

    vector<Point> readCSV(string path) {
        vector<Point> points;
        string line;
        ifstream file(path);

        getline(file, line); // pop header
        while (getline(file, line)) {
            stringstream lineStream(line);

            double x, y;
            string bit;
            getline(lineStream, bit, ',');
            getline(lineStream, bit, ',');
            getline(lineStream, bit, ',');
            getline(lineStream, bit, ',');
            x = stof(bit);
            getline(lineStream, bit, '\n');
            y = stof(bit);

            points.push_back(Point(x, y));
        }

        file.close();
        return points;
    }

    void kMeansClustering(vector<Point> &points, int epochs, int k) {

        // 1. init centroids
        vector<Point> centroids;
        srand(time(0)); // need to set the random seed
        int numOfPoints = points.size();
        for (int i=0; i<k; i++) {
            //int pointIdx = rand() % numOfPoints;
            int pointIdx = i;
            centroids.push_back(points.at(pointIdx));
            centroids.back().cluster = i;
        }

        // do some iterations
        for (int e=0; e<epochs; e++) {

            // 2. assign points to a cluster
            for (auto &point : points) {
                point.minDistance = DBL_MAX;
                for (int c=0; c<centroids.size(); c++) {
                    double distance = point.distance(centroids[c]);
                    if (distance < point.minDistance) {
                        point.minDistance = distance;
                        point.cluster = c;
                    }
                }
            }

            // 3. redefine centroids
            vector<int> sizeOfEachCluster(k, 0);
            vector<double> sumXOfEachCluster(k, 0);
            vector<double> sumYOfEachCluster(k, 0);
            for (auto point : points) {
                sizeOfEachCluster[point.cluster] += 1;
                sumXOfEachCluster[point.cluster] += point.x;
                sumYOfEachCluster[point.cluster] += point.y;
            }
            for (int i=0; i<centroids.size(); i++) {
                centroids[i].x = (sizeOfEachCluster[i] == 0) ? 0 : sumXOfEachCluster[i] / sizeOfEachCluster[i];
                centroids[i].y = (sizeOfEachCluster[i] == 0) ? 0 : sumYOfEachCluster[i] / sizeOfEachCluster[i];
            }

            // 4. write to a file
            ofstream file1;
            file1.open("points_iter_" + to_string(e) + ".csv");
            file1 << "x,y,clusterIdx" << endl;
            for (auto point : points) {
                file1 << point.x << "," << point.y << "," << point.cluster << endl;
            }
            file1.close();

            ofstream file2;
            file2.open("centroids_iter_" + to_string(e) + ".csv");
            file2 << "x,y,clusterIdx" << endl;
            for (auto centroid : centroids) {
                file2 << centroid.x << "," << centroid.y << "," << centroid.cluster << endl;
            }
            file2.close();

        }

    }

    int main() {
        // [option 1] load csv
        vector<Point> points = readCSV("./mall_customers.csv");
        // [option 2] 
        // vector<Point> points = {
        //     {12, 39}, {20, 36}, {28, 30}, {18, 52}, {29, 54}, {33, 46}, {24, 55}, {45, 59}, {60, 35}, {52, 70},
        //     {51, 66}, {52, 63}, {55, 58}, {53, 23}, {55, 58}, {53, 23}, {55, 14}, {61, 8}, {64, 19}, {69, 7}, {72, 24}
        // };

        kMeansClustering(points, 5, 6);

**Éxécution**


mamadou@port-lipn12:~/big-data/cerin24022022/ML-From-Scratch-With-CPP/k-means$ cat mall_customers.csv 
CustomerID,Genre,Age,Annual Income (k$),Spending Score (1-100)
0001,Male,19,15,39
0002,Male,21,15,81
0003,Female,20,16,6
0004,Female,23,16,77
0005,Female,31,17,40
0006,Female,22,17,76
0007,Female,35,18,6
0008,Female,23,18,94
0009,Male,64,19,3
0010,Female,30,19,72
0011,Male,67,19,14
0012,Female,35,19,99
0013,Female,58,20,15
0014,Female,24,20,77
0015,Male,37,20,13
0016,Male,22,20,79
0017,Female,35,21,35
0018,Male,20,21,66
0019,Male,52,23,29
0020,Female,35,23,98
0021,Male,35,24,35
0022,Male,25,24,73

mamadou@port-lipn12:~/big-data/cerin24022022/ML-From-Scratch-With-CPP/k-means$ cat points_iter_0.csv
x,y,clusterIdx
15,39,0
15,81,1
16,6,2
16,77,3
17,40,4
17,76,5
18,6,2
18,94,1
19,3,2

mamadou@port-lipn12:~/big-data/cerin24022022/ML-From-Scratch-With-CPP/k-means$ cat points_iter_1.csv
x,y,clusterIdx
15,39,0
15,81,3
16,6,0
16,77,3
17,40,0
17,76,3
18,6,0
18,94,1
19,3,0

mamadou@port-lipn12:~/big-data/cerin24022022/ML-From-Scratch-With-CPP/k-means$ cat points_iter_2.csv
x,y,clusterIdx
15,39,0
15,81,3
16,6,0
16,77,3
17,40,0
17,76,3
18,6,0
18,94,1
19,3,0
19,72,3
19,14,0
19,99,1
20,15,0

        mamadou@port-lipn12:~/big-data/cerin24022022/ML-From-Scratch-With-CPP/k-means$ cat points_iter_3.csv
        x,y,clusterIdx
        15,39,0
        15,81,3
        16,6,0
        16,77,3
        17,40,0
        17,76,3
        18,6,0
        18,94,1
        19,3,0
        19,72,3


        mamadou@port-lipn12:~/big-data/cerin24022022/ML-From-Scratch-With-CPP/k-means$ cat centroids_iter_0.csv
        x,y,clusterIdx
        15,39,0
        25,91,1
        66,10,2
        16,77,3
        57,44,4
        67,76,5
        mamadou@port-lipn12:~/big-data/cerin24022022/ML-From-Scratch-With-CPP/k-means$ cat centroids_iter_1.csv 
        x,y,clusterIdx
        23,20,0
        28,88,1
        85,14,2
        23,72,3
        55,48,4
        86,81,5
        mamadou@port-lipn12:~/big-data/cerin24022022/ML-From-Scratch-With-CPP/k-means$ cat centroids_iter_2.csv 
        x,y,clusterIdx
        25,20,0
        27,90,1
        88,17,2
        25,72,3
        55,49,4
        86,82,5
        mamadou@port-lipn12:~/big-data/cerin24022022/ML-From-Scratch-With-CPP/k-means$ cat centroids_iter_3.csv 
        x,y,clusterIdx
        25,20,0
        27,90,1
        88,17,2
        25,72,3
        55,49,4
        86,82,5


### [K-NN (k nearest neighbors)](https://github.com/magikerwin1993/ML-From-Scatch-With-CPP/tree/main/k-nn)

En intelligence artificielle, plus précisément en apprentissage automatique, la méthode des k plus proches voisins est une méthode d’apprentissage supervisé. En abrégé k-NN ou KNN, de l anglais k-nearest neighbors.

Dans ce cadre, on dispose d’une base de données d'apprentissage constituée de N couples « entrée-sortie ». Pour estimer la sortie associée à une nouvelle entrée x, la méthode des k plus proches voisins consiste à prendre en compte (de façon identique) les k échantillons d'apprentissage dont l’entrée est la plus proche de la nouvelle entrée x, selon une distance à définir. Puisque cet algorithme est basé sur la distance, la normalisation peut améliorer sa précision

En reconnaissance de forme, l'algorithme des k plus proches voisins (k-NN) est une méthode non paramétrique utilisée pour la classification et la régression. Dans les deux cas, il s'agit de classer l'entrée dans la catégorie à laquelle appartient les k plus proches voisins dans l'espace des caractéristiques identifiées par apprentissage. Le résultat dépend si l'algorithme est utilisé à des fins de classification ou de régression :

1. en classification k-NN, le résultat est une classe d'appartenance. Un objet d'entrée est classifié selon le résultat majoritaire des statistiques de classes d'appartenance de ses k plus proches voisins, (k est un nombre entier positif généralement petit). Si k = 1, alors l'objet est affecté à la classe d'appartenance de son proche voisin.
2. en régression k-NN, le résultat est la valeur pour cet objet. Cette valeur est la moyenne des valeurs des k plus proches voisins.


La méthode k-NN est basée sur l'apprentissage préalable, ou l'apprentissage faible, où la fonction est évaluée localement, le calcul définitif étant effectué à l'issue de la classification. L'algorithme k-NN est parmi les plus simples des algorithmes de machines learning.
