 ## ranger : A Fast Implementation of Random Forests

Marvin N. Wright

https://github.com/imbs-hl/ranger

ranger est une implémentation rapide des forêts aléatoires (Breiman 2001) ou du partitionnement récursif, 
particulièrement adapté aux données de grande dimension. Les forêts de classification, de régression et de survie sont prises en charge. 
Les forêts de classification et de régression sont mises en œuvre comme dans la forêt aléatoire originale (Breiman 2001), 
les forêts de survie comme dans les forêts de survie aléatoires (Ishwaran et al. 2008). 
Comprend des implémentations d'arbres extrêmement aléatoires (Geurts et al. 2006) et des forêts de régression quantiles (Meinshausen 2006).

ranger est écrit en C++, mais une version pour R est également disponible. 
Nous recommandons d'utiliser la version R. Il est facile à installer et à utiliser et les résultats sont facilement disponibles pour 
une analyse plus approfondie. La version R est aussi rapide que la version C++ autonome.

**Programme en C++**

         /*-------------------------------------------------------------------------------

         mamadou@port-lipn12:~/big-data/cerin24022022/cpp-mnwright/ranger/cpp_version/src$ cat main.cpp 

          This file is part of ranger.

          Copyright (c) [2014-2018] [Marvin N. Wright]

          This software may be modified and distributed under the terms of the MIT license.          
          Please note that the C++ core of ranger is distributed under MIT license and the
          R package "ranger" under GPL3 license.
          #-------------------------------------------------------------------------------*/

         #include <iostream>
         #include <fstream>
         #include <stdexcept>
         #include <string>
         #include <memory>

         #include "globals.h"
         #include "ArgumentHandler.h"
         #include "ForestClassification.h"
         #include "ForestRegression.h"
         #include "ForestSurvival.h"
         #include "ForestProbability.h"
         #include "utility.h"

         using namespace ranger;

         void run_ranger(const ArgumentHandler& arg_handler, std::ostream& verbose_out) {
           verbose_out << "Starting Ranger." << std::endl;

           // Create forest object
           std::unique_ptr<Forest> forest { };
           switch (arg_handler.treetype) {
           case TREE_CLASSIFICATION:
             if (arg_handler.probability) {
               forest = make_unique<ForestProbability>();
             } else {
               forest = make_unique<ForestClassification>();
             }
             break;
           case TREE_REGRESSION:
             forest = make_unique<ForestRegression>();
             break;
           case TREE_SURVIVAL:
             forest = make_unique<ForestSurvival>();
             break;
           case TREE_PROBABILITY:
             forest = make_unique<ForestProbability>();
             break;
           }

           // Call Ranger
           forest->initCpp(arg_handler.depvarname, arg_handler.memmode, arg_handler.file, arg_handler.mtry,
               arg_handler.outprefix, arg_handler.ntree, &verbose_out, arg_handler.seed, arg_handler.nthreads,
               arg_handler.predict, arg_handler.impmeasure, arg_handler.targetpartitionsize, arg_handler.splitweights,
               arg_handler.alwayssplitvars, arg_handler.statusvarname, arg_handler.replace, arg_handler.catvars,
               arg_handler.savemem, arg_handler.splitrule, arg_handler.caseweights, arg_handler.predall, arg_handler.fraction,
               arg_handler.alpha, arg_handler.minprop, arg_handler.holdout, arg_handler.predictiontype,
               arg_handler.randomsplits, arg_handler.maxdepth, arg_handler.regcoef, arg_handler.usedepth);

           forest->run(true, !arg_handler.skipoob);
           if (arg_handler.write) {
             forest->saveToFile();
           }
           forest->writeOutput();
           verbose_out << "Finished Ranger." << std::endl;
         }

         int main(int argc, char **argv) {

           try {
             // Handle command line arguments
             ArgumentHandler arg_handler(argc, argv);
             if (arg_handler.processArguments() != 0) {
               return 0;
             }
             arg_handler.checkArguments();

             if (arg_handler.verbose) {
               run_ranger(arg_handler, std::cout);
             } else {
               std::ofstream logfile { arg_handler.outprefix + ".log" };
               if (!logfile.good()) {
                 throw std::runtime_error("Could not write to logfile.");
               }
               run_ranger(arg_handler, logfile);
             }
           } catch (std::exception& e) {
             std::cerr << "Error: " << e.what() << " Ranger will EXIT now." << std::endl;
             return -1;
           }

           return 0;
         }



  **ForestClassification.h**


         /*-------------------------------------------------------------------------------

     mamadou@port-lipn12:~/big-data/cerin24022022/cpp-mnwright/ranger/cpp_version/src/Forest$ cat  ForestClassification.h


      This file is part of ranger.

      Copyright (c) [2014-2018] [Marvin N. Wright]

      This software may be modified and distributed under the terms of the MIT license.

      Please note that the C++ core of ranger is distributed under MIT license and the
      R package "ranger" under GPL3 license.
      #-------------------------------------------------------------------------------*/

     #ifndef FORESTCLASSIFICATION_H_
     #define FORESTCLASSIFICATION_H_

     #include <iostream>
     #include <map>
     #include <utility>
     #include <vector>

     #include "globals.h"
     #include "Forest.h"

     namespace ranger {

     class ForestClassification: public Forest {
     public:
       ForestClassification() = default;

       ForestClassification(const ForestClassification&) = delete;
       ForestClassification& operator=(const ForestClassification&) = delete;

       virtual ~ForestClassification() override = default;

       void loadForest(size_t num_trees, std::vector<std::vector<std::vector<size_t>> >& forest_child_nodeIDs,
           std::vector<std::vector<size_t>>& forest_split_varIDs, std::vector<std::vector<double>>& forest_split_values,
           std::vector<double>& class_values, std::vector<bool>& is_ordered_variable);

       const std::vector<double>& getClassValues() const {
         return class_values;
       }

       void setClassWeights(std::vector<double>& class_weights) {
         this->class_weights = class_weights;
       }

     protected:
       void initInternal() override;
       void growInternal() override;
       void allocatePredictMemory() override;
       void predictInternal(size_t sample_idx) override;
       void computePredictionErrorInternal() override;
       void writeOutputInternal() override;
       void writeConfusionFile() override;
       void writePredictionFile() override;
       void saveToFileInternal(std::ofstream& outfile) override;
       void loadFromFileInternal(std::ifstream& infile) override;

       // Classes of the dependent variable and classIDs for responses
       std::vector<double> class_values;
       std::vector<uint> response_classIDs;
       std::vector<std::vector<size_t>> sampleIDs_per_class;

       // Splitting weights
       std::vector<double> class_weights;

       // Table with classifications and true classes
       std::map<std::pair<double, double>, size_t> classification_table;

     private:
       double getTreePrediction(size_t tree_idx, size_t sample_idx) const;
       size_t getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const;
     };

     } // namespace ranger

     #endif /* FORESTCLASSIFICATION_H_ */
     
 **Exécution**

      mamadou@port-lipn12:~/big-data/cerin24022022/cpp-mnwright/ranger/cpp_version/build$ cat iristest.dat
      SepalLength SepalWidth PetalLength PetalWidth Species
      6 5.4 3.9 1.7 0.4 setosa
      9 4.4 2.9 1.4 0.2 setosa
      13 4.8 3 1.4 0.1 setosa
      15 5.8 4 1.2 0.2 setosa
      17 5.4 3.9 1.3 0.4 setosa
      19 5.7 3.8 1.7 0.3 setosa
      22 5.1 3.7 1.5 0.4 setosa
      23 4.6 3.6 1 0.2 setosa
      27 5 3.4 1.6 0.4 setosa
      
     mamadou@port-lipn12:~/big-data/cerin24022022/cpp-mnwright/ranger/cpp_version/build$ ./ranger --verbose --file iristest.dat --depvarname Species --treetype 1 --ntree 6 --nthreads 5
     Starting Ranger.
     Loading input file: iristest.dat.
     Growing trees ..
     Computing prediction error ..

     Tree type:                         Classification
     Dependent variable name:           Species
     Number of trees:                   6
     Sample size:                       50
     Number of independent variables:   4
     Mtry:                              2
     Target node size:                  1
     Variable importance mode:          0
     Memory mode:                       0
     Seed:                              0
     Number of threads:                 5

     Overall OOB prediction error:      0.851064

     Saved confusion matrix to file ranger_out.confusion.
     Finished Ranger.



     mamadou@port-lipn12:~/big-data/cerin24022022/cpp-mnwright/ranger/cpp_version/build$ cat ranger_out.confusion
     Overall OOB prediction error (Fraction missclassified): 0.851064

     Class specific prediction errors:
                     0.4     0.2     0.1     0.3     1.4     1.3     1.5     1     1.2     1.1     1.6     1.9     2.1     1.8     2.2     2.3     2
     predicted 0.4     1     4     2     1     0     0     0     0     0     0     0     0     0     0     0     0     0     
     predicted 0.2     2     2     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     
     predicted 0.1     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     
     predicted 0.3     1     0     0     0     0     0     0     0     0     1     0     0     0     0     0     0     0     
     predicted 1.4     0     0     0     0     0     1     1     0     0     0     1     0     0     0     0     0     0     
     predicted 1.3     0     0     0     0     3     2     1     2     1     0     0     0     0     0     0     0     0     
     predicted 1.5     0     0     0     0     0     1     0     0     0     0     0     0     0     0     0     0     0     
     predicted 1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     
     predicted 1.2     0     0     0     0     0     1     0     0     0     0     0     0     0     1     0     0     0     
     predicted 1.1     0     0     0     0     0     2     0     0     0     0     0     0     0     0     0     0     0     
     predicted 1.6     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     0     1     
     predicted 1.9     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     
     predicted 2.1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     
     predicted 1.8     0     0     0     0     0     0     0     0     0     0     1     0     1     2     0     3     0     
     predicted 2.2     0     0     0     0     0     0     0     0     0     0     0     0     0     2     0     1     0     
     predicted 2.3     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     
     predicted 2     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     


### Regression Trees

**CART (arbres de classification et de régression)**

Régression :
La fonction de coût qui est minimisée pour choisir les points de fractionnement est la somme de l'erreur quadratique sur tous les échantillons d'apprentissage compris dans le rectangle.

Classification :
La fonction de coût de Gini est utilisée pour fournir une indication de la pureté du nœud, où la pureté du nœud fait référence à la mixité des données d'apprentissage attribuées à chaque nœud.


1. Indice de Gini (fonction de coût pour évaluer les divisions dans l'ensemble de données)
2. Créer une division
3. Construire un arbre

    3.1 Nœuds terminaux (profondeur maximale de l'arborescence, nombre minimal d'enregistrements de nœuds)
    
    3.2 Découpage récursif
    
    3.3 Construire un arbre
    
    
4. Faites une prédiction
