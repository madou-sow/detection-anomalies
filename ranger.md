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

**Programme**

  /*
mamadou@port-lipn12:~/big-data/cerin24022022/ML-From-Scratch-With-CPP/RegressionTrees$ cat ClassificationRegressionTrees.cpp

CART (Classification and Regression Trees)

Regression:
The cost function that is minimized to choose split points is the sum squared error across all training samples that fall within the rectangle.

Classification:
The Gini cost function is used which provides an indication of how pure the node are, where node purity refers to how mixed the training data assigned to each node is.


Régression:
La fonction de coût qui est minimisée pour choisir les points de partage est la somme de l'erreur quadratique sur tous les échantillons d'apprentissage compris dans le rectangle.

Classification:
La fonction de coût de Gini est utilisée et fournit une indication de la pureté du nœud, où la pureté du nœud fait référence à la mixité des données d'apprentissage attribuées à chaque nœud.


1. Gini Index (cost function to evaluate splits in the dataset)
2. Create Split
3. Build a Tree
    3.1 Terminal Nodes (Maximum Tree Depth, Minimum Node Records)
    3.2 Recursive Splitting
    3.3 Building a Tree
4. Make a Prediction

*/

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <climits>
#include <cfloat>
#include <algorithm>

using namespace std;

double giniIndex(const vector<vector<vector<double>>> &groups, vector<double> classes) {
    
    // count all samples as split point
    int numOfInstances = 0;
    for (auto group : groups) {
        numOfInstances += group.size();
    }
    
    // sum weighted Gini index for each group
    double gini = 0;
    for (auto group : groups) {
        double size = group.size();
        if (size == 0) continue; // avoid divide by zero
        
        double score = 1.0; // 1 - p_1^2 - p_2^2 - ... - - p_N^2
        for (int classIdx : classes) {
            double count = 0;
            for (auto instance : group) {
                double label = instance[instance.size() - 1];
                if (label == classIdx) count++;
            }
            double p = count / size;
            score -= p * p;
        }
        gini += (size / numOfInstances) * score;
    }
    
    return gini;
}

vector<vector<vector<double>>> splitGroups(
    int featureIdx, double value, 
    const vector<vector<double>> &dataset) {
    
    vector<vector<double>> lefts;
    vector<vector<double>> rights;
    for (auto data : dataset) {
        if (data[featureIdx] < value) {
            lefts.push_back(data);
        } else {
            rights.push_back(data);
        }
    }
    
    return {lefts, rights};
}

struct Node {
    int featureIdx;
    double featureValue;
    double gini;
    vector<vector<vector<double>>> groups;
    
    Node* left = nullptr;
    Node* right = nullptr;
    double label = -1;
};

Node* getSplit(const vector<vector<double>> &dataset) {
    int numOfFeatures = dataset[0].size() - 1;
    
    // get labels lists
    unordered_set<double> bucket;
    for (auto data : dataset) {
        double classIdx = data[numOfFeatures];
        bucket.insert(classIdx);
    }
    vector<double> labels;
    for (auto label : bucket) labels.push_back(label);
    sort(labels.begin(), labels.end());
    
    // split groups by min gini
    double minGini = DBL_MAX;
    Node* info = new Node;
    for (int featureIdx=0; featureIdx<numOfFeatures; featureIdx++) {
        for (auto data : dataset) {
            auto groups = splitGroups(featureIdx, data[featureIdx], dataset);
            auto gini = giniIndex(groups, labels);
            // cout << "X1 < " << data[featureIdx] << ", gini = " << gini << endl;
            if (gini < minGini) {
                minGini = gini;
                info->featureIdx = featureIdx;
                info->featureValue = data[featureIdx];
                info->gini = gini;
                info->groups = groups;
            }
        }
    }
    return info;
}

// Create a terminal node value, and it will return most common output value
double toTerminal(const vector<vector<double>> &group) {
    unordered_map<double, int> counter;
    for (auto data : group) {
        double label = data[data.size()-1];
        if (counter.count(label) == 0) {
            counter[label] = 1;
        } else {
            counter[label] += 1;
        }
    }
    
    int maxCount = 0;
    double targetLabel;
    for (auto item : counter) {
        if (item.second > maxCount) {
            maxCount = item.second;
            targetLabel = item.first;
        }
    }
    return targetLabel;
}

// Create child splits for a node or make terminal
void split(Node* currNode, int maxDepth, int minSize, int depth) {
    auto leftGroup = currNode->groups[0];
    auto rightGroup = currNode->groups[1];
    currNode->groups.clear();
    
    // check for a no split
    if (leftGroup.empty() || rightGroup.empty()) {
        if (leftGroup.empty()) {
            currNode->right = new Node;
            currNode->right->label = toTerminal(rightGroup);
        } else {
            currNode->left = new Node;
            currNode->left->label = toTerminal(leftGroup);
        }
        return;
    }
    // check for max depth
    if (depth >= maxDepth) {
        currNode->left = new Node;
        currNode->left->label = toTerminal(leftGroup);
        currNode->right = new Node;
        currNode->right->label = toTerminal(rightGroup);
        return;
    }
    // process left child
    if (leftGroup.size() <= minSize) {
        currNode->left = new Node;
        currNode->left->label = toTerminal(leftGroup);
    } else {
        currNode->left = getSplit(leftGroup);
        split(currNode->left, maxDepth, minSize, depth+1);
    }
    // process right child
    if (rightGroup.size() <= minSize) {
        currNode->right = new Node;
        currNode->right->label = toTerminal(rightGroup);
    } else {
        currNode->right = getSplit(rightGroup);
        split(currNode->right, maxDepth, minSize, depth+1);
    }
}

Node* buildTree(
    const vector<vector<double>> &dataset, 
    int maxDepth, int minSize) {
    
    Node* root = getSplit(dataset);
    split(root, maxDepth, minSize, 1);
    return root;
}

void printTree(Node* root, int depth) {
    if (root == nullptr) return;
    
    if (root->label != -1) {
        cout << "depth: " << depth
            << ", label: " << root->label << endl;
    } else {
        cout << "depth: " << depth
            << ", featureIdx: " << root->featureIdx 
            << ", featureValue: " << root->featureValue << endl;
    }
    
    printTree(root->left, depth+1);
    printTree(root->right, depth+1);
}

double predict(Node* currNode, vector<double> data) {
    
    if (currNode->label != -1) return currNode->label;
    
    double featureValue = data[currNode->featureIdx];
    if (featureValue < currNode->featureValue) {
        if (currNode->left != nullptr) {
            return predict(currNode->left, data);
        }
    } else {
        if (currNode->right != nullptr) {
            return predict(currNode->right, data);
        }
    }
    return -1;
}

int main() {
    
    vector<vector<double>> dataset = {
        {2.771244718,1.784783929,0},
        {1.728571309,1.169761413,0},
        {3.678319846,2.81281357,0},
        {3.961043357,2.61995032,0},
        {2.999208922,2.209014212,0},
        {7.497545867,3.162953546,1},
        {9.00220326,3.339047188,1},
        {7.444542326,0.476683375,1},
        {10.12493903,3.234550982,1},
        {6.642287351,3.319983761,1}
    };
    
    Node* root = buildTree(dataset, 1, 1);
    
    printTree(root, 0);
    
    for (auto data : dataset) {
        double pred = predict(root, data);
        cout << "pred: " << pred << ", gt: " << data[data.size()-1] << endl;
    }

