
# Détecter une valeur aberrante évidente

Référence https://github.com/david-cortes/isotree

Isolation Forest est un algorithme développé à l'origine pour la détection de valeurs aberrantes qui consiste à diviser au hasard des sous-échantillons de données en fonction d'attributs/caractéristiques/colonnes. L'idée est que, plus l'observation est rare, plus il est probable qu'une répartition uniforme aléatoire sur une caractéristique placerait des valeurs aberrantes seules dans une branche, et moins il faudra de divisions pour isoler une observation aberrante comme celle-ci. Le concept est étendu à la division des hyperplans dans le modèle étendu (c'est-à-dire la division par plus d'une colonne à la fois) et aux divisions guidées (pas entièrement aléatoires) dans le modèle SCiForest qui visent à isoler plus rapidement les valeurs aberrantes et à trouver des valeurs aberrantes groupées.

# L'algorithme en Langage C++

1. Données aléatoires d'une distribution normale standard (100 points générés aléatoirement, plus 1 valeur aberrante ajoutée manuellement) La bibliothèque suppose qu'il est passé en tant que pointeur unidimensionnel, suivant l'ordre des colonnes majeures (comme Fortran)

2. Ajoutez maintenant un point aberrant évident (3,3)

3. Ajuster un petit modèle de forêt d'isolement

4. Vérifiez quelle ligne a le score de valeur aberrante le plus élevé

5. Les modèles peuvent être sérialisés et désérialisés de manière très idiomatique

\*
## /home/mamadou/big-data/cerin24022022/cpp-isotree/example/isotree_cpp_oop_ex.cpp

#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include "isotree_oop.hpp"

/*  To compile this example, first build the package through the cmake system:
      mkdir build
      cd build
      cmake ..
      make
      sudo make install
      sudo ldconfig
   Then compile this single file and link to the shared library:
     g++ -o test isotree_cpp_oop_ex.cpp -lisotree -std=c++11

   Or to link against it without a system install, assuming the cmake system
   has already built the library under ./build and this command is called from
   the root folder:
     g++ -o test example/isotree_cpp_oop_ex.cpp -std=c++11 -I./include -l:libisotree.so -L./build -Wl,-rpath,./build

   Then run with './test'
*/

int which_max(std::vector<double> &v)
{
    auto loc_max_el = std::max_element(v.begin(), v.end());
    return std::distance(v.begin(), loc_max_el);
}


int main()
{
    /* Random data from a standard normal distribution
       (100 points generated randomly, plus 1 outlier added manually)
       Library assumes it is passed as a single-dimensional pointer,
       following column-major order (like Fortran) */
    int nrow = 101;
    int ncol = 2;
    std::vector<double> X( nrow * ncol );
    std::default_random_engine rng(1);
    std::normal_distribution<double> rnorm(0, 1);
    #define get_ix(row, col) (row + col*nrow)
    for (int col = 0; col < ncol; col++)
        for (int row = 0; row < 100; row++)
            X[get_ix(row, col)] = rnorm(rng);

    /* Now add obvious outlier point (3,3) */
    X[get_ix(100, 0)] = 3.;
    X[get_ix(100, 1)] = 3.;

    /* Fit a small isolation forest model
       (see 'fit_model.cpp' for the documentation) */
    isotree::IsolationForest iso = isotree::IsolationForest();
    iso.fit(X.data(), nrow, ncol);

    /* Check which row has the highest outlier score
       (see file 'predict.cpp' for the documentation) */
    std::vector<double> outlier_scores = iso.predict(X.data(), nrow, true);

    int row_highest = which_max(outlier_scores);
    std::cout << "Point with highest outlier score: [";
    std::cout << X[get_ix(row_highest, 0)] << ", ";
    std::cout << X[get_ix(row_highest, 1)] << "]" << std::endl;

    /* Models can be serialized and de-serialized very idiomatically */
    std::stringstream ss;
    ss << iso; /* <- serialize */
    ss >> iso; /* <- deserialize */

    return EXIT_SUCCESS;
}
 
\*
