## LibIsolationForest

https://github.com/msimms/LibIsolationForest

### Description

Ce projet contient des implémentations C++ et python de l'algorithme Isolation Forest. Isolation Forest est un algorithme de détection d'anomalies basé sur une collection d'arbres de décision générés aléatoirement.

## IsolationForest.cpp,  IsolationForest.h, main.cpp et Makefile

Un exemple d'utilisation de la version C++ de la bibliothèque se trouve dans main.cpp. Au fur et à mesure que la bibliothèque mûrit, j'ajouterai d'autres exemples de test à ce fichier.

    #~/big-data/cerin24022022/cpp-LibIsolationForest/cpp/Makefile

    CFLAGS=-Wall -O3 -std=c++11

    .PHONY: all clean 

    all : isolationforestsimms

    isolationforestsimms : main.o IsolationForest.o 
      ${CXX} ${CFLAGS} $^ -o $@ ${LDFLAGS} 

    main.o : main.cpp
      ${CXX} ${CFLAGS} -c $^ -o $@

    IsolationForest.o : IsolationForest.cpp
      ${CXX} ${CFLAGS} -c $^ -o $@

    clean : 
      -rm -f *.o 

**IsolationForest implementation(main.cpp)**

1. Ce fichier contient des tests et des exemples de code pour l'utilisation de l'implémentation C++ IsolationForest

2. Créer des exemples d'apprentissage

3. Échantillons de test (similaires aux échantillons d'apprentissage)

4. Exécutez un test avec l'échantillon qui ne contient pas de valeurs aberrantes.

5. Échantillons aberrants (différents des échantillons d'apprentissage).

        //#~/big-data/cerin24022022/cpp-LibIsolationForest/cpp/main.cpp
        //	MIT License
        //
        //  Copyright © 2017 Michael J Simms. All rights reserved.
        //
        //	Permission is hereby granted, free of charge, to any person obtaining a copy
        //	of this software and associated documentation files (the "Software"), to deal
        //	in the Software without restriction, including without limitation the rights
        //	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        //	copies of the Software, and to permit persons to whom the Software is
        //	furnished to do so, subject to the following conditions:
        //
        //	The above copyright notice and this permission notice shall be included in all
        //	copies or substantial portions of the Software.
        //
        //	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        //	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        //	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        //	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        //	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        //	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        //	SOFTWARE.

        // This file contains test and example code for using the C++ IsolationForest implementation.

        #include "IsolationForest.h"
        #include <stdlib.h>
        #include <inttypes.h>
        #include <iostream>
        #include <fstream>
        #include <chrono>

        // sow debut
        #include <cstdio>
        #include <cstring>
        #include <string>
        #include <vector>
        #include <iomanip>
        #include <pthread.h>
        #include <cmath>
        // sow fin

        using namespace IsolationForest;

        void test(std::ofstream& outStream, size_t numTrainingSamples, size_t numTestSamples, uint32_t numTrees, uint32_t subSamplingSize, bool dump)
        {
            Forest forest(numTrees, subSamplingSize);

            std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

            // Create some training samples.
            for (size_t i = 0; i < numTrainingSamples; ++i)
            {
                Sample sample("training");
                FeaturePtrList features;

                uint32_t x = rand() % 25;
                uint32_t y = rand() % 25;

                features.push_back(new Feature("x", x));
                features.push_back(new Feature("y", y));

                sample.AddFeatures(features);
                forest.AddSample(sample);

                if (outStream.is_open())
                {
                    outStream << "training," << x << "," << y << std::endl;
                }
            }

            // Create the isolation forest.
            forest.Create();

            // Test samples (similar to training samples).
            double avgControlScore = (double)0.0;
            double avgControlNormalizedScore = (double)0.0;
            for (size_t i = 0; i < numTestSamples; ++i)
            {
                Sample sample("control sample");
                FeaturePtrList features;

                uint32_t x = rand() % 25;
                uint32_t y = rand() % 25;

                features.push_back(new Feature("x", x));
                features.push_back(new Feature("y", y));
                sample.AddFeatures(features);

                // Run a test with the sample that doesn't contain outliers.
                double score = forest.Score(sample);
                double normalizedScore = forest.NormalizedScore(sample);
                avgControlScore += score;
                avgControlNormalizedScore += normalizedScore;

                if (outStream.is_open())
                {
                    outStream << "control," << x << "," << y << std::endl;
                }
            }
            avgControlScore /= numTestSamples;
            avgControlNormalizedScore /= numTestSamples;

            // Outlier samples (different from training samples).
            double avgOutlierScore = (double)0.0;
            double avgOutlierNormalizedScore = (double)0.0;
            for (size_t i = 0; i < numTestSamples; ++i)
            {
                Sample sample("outlier sample");
                FeaturePtrList features;

                uint32_t x = 20 + (rand() % 25);
                uint32_t y = 20 + (rand() % 25);

                features.push_back(new Feature("x", x));
                features.push_back(new Feature("y", y));
                sample.AddFeatures(features);

                // Run a test with the sample that contains outliers.
                double score = forest.Score(sample);
                double normalizedScore = forest.NormalizedScore(sample);
                avgOutlierScore += score;
                avgOutlierNormalizedScore += normalizedScore;

                if (outStream.is_open())
                {
                    outStream << "outlier," << x << "," << y << std::endl;
                }
            }
            avgOutlierScore /= numTestSamples;
            avgOutlierNormalizedScore /= numTestSamples;

            std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);

            std::cout << "Average of control test samples: " << avgControlScore << std::endl;
            std::cout << "Average of control test samples (normalized): " << avgControlNormalizedScore << std::endl;
            std::cout << "Average of outlier test samples: " << avgOutlierScore << std::endl;
            std::cout << "Average of outlier test samples (normalized): " << avgOutlierNormalizedScore << std::endl;
            std::cout << "Total time for Test 1: " << elapsedTime.count() << " seconds." << std::endl;

            if (dump)
            {
                forest.Dump();
            }
        }

        int main(int argc, const char * argv[])
        {
            const char* ARG_OUTFILE = "outfile";
            const char* ARG_DUMP = "dump";
            const size_t NUM_TRAINING_SAMPLES = 100;
            const size_t NUM_TEST_SAMPLES = 10;
            const uint32_t NUM_TREES_IN_FOREST = 10;
            const uint32_t SUBSAMPLING_SIZE = 10;

            std::ofstream outStream;
            bool dump = false;

            // Parse the command line arguments.
            for (int i = 1; i < argc; ++i)
            {
                if ((strncmp(argv[i], ARG_OUTFILE, strlen(ARG_OUTFILE)) == 0) && (i + 1 < argc))
                {
                    outStream.open(argv[i + 1]);
                }
                if (strncmp(argv[i], ARG_DUMP, strlen(ARG_DUMP)) == 0)
                {
                    dump = true;
                }
            }

            srand((unsigned int)time(NULL));

            std::cout << "Test 1:" << std::endl;
            std::cout << "-------" << std::endl;
            test(outStream, NUM_TRAINING_SAMPLES, NUM_TEST_SAMPLES, NUM_TREES_IN_FOREST, SUBSAMPLING_SIZE, dump);
            std::cout << std::endl;

            std::cout << "Test 2:" << std::endl;
            std::cout << "-------" << std::endl;
            test(outStream, NUM_TRAINING_SAMPLES * 10, NUM_TEST_SAMPLES * 10, NUM_TREES_IN_FOREST * 10, SUBSAMPLING_SIZE * 10, dump);
            std::cout << std::endl;

            if (outStream.is_open())
            {
                outStream.close();
            }

            return 0;
        }


**IsolationForest.h**
1. Cette classe représente une fonctionnalité. Chaque échantillon a une ou plusieurs caractéristiques. Chaque fonctionnalité a un nom et une valeur.

2. Cette classe représente un échantillon. Chaque échantillon a un nom et une liste de fonctionnalités.

3. Nœud d'arborescence, utilisé en interne.

4. Cette classe résume la génération de nombres aléatoires.
Héritez de cette classe si vous souhaitez fournir votre propre randomiseur.
Utilisez Forest::SetRandomizer pour remplacer le randomiseur par défaut par celui de votre choix.

        //#~/big-data/cerin24022022/cpp-LibIsolationForest/cpp/IsolationForest.h
        //	MIT License
        //
        //  Copyright © 2017 Michael J Simms. All rights reserved.
        //
        //	Permission is hereby granted, free of charge, to any person obtaining a copy
        //	of this software and associated documentation files (the "Software"), to deal
        //	in the Software without restriction, including without limitation the rights
        //	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        //	copies of the Software, and to permit persons to whom the Software is
        //	furnished to do so, subject to the following conditions:
        //
        //	The above copyright notice and this permission notice shall be included in all
        //	copies or substantial portions of the Software.
        //
        //	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        //	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        //	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        //	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        //	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        //	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        //	SOFTWARE.

        #pragma once

        #include <map>
        #include <random>
        #include <set>
        #include <stdint.h>
        #include <string>
        #include <time.h>
        #include <vector>

        namespace IsolationForest
        {
            /// This class represents a feature. Each sample has one or more features.
            /// Each feature has a name and value.
            class Feature
            {
            public:
                Feature(const std::string& name, uint64_t value) { m_name = name; m_value = value; };
                virtual ~Feature() {};

                virtual void Name(std::string& name) { m_name = name; };
                virtual std::string Name() const { return m_name; };

                virtual void Value(uint64_t value) { m_value = value; };
                virtual uint64_t Value() const { return m_value; };

            protected:
                std::string m_name;
                uint64_t m_value;

            private:
                Feature() {};
            };

            typedef Feature* FeaturePtr;
            typedef std::vector<FeaturePtr> FeaturePtrList;

            /// This class represents a sample.
            /// Each sample has a name and list of features.
            class Sample
            {
            public:
                Sample() {};
                Sample(const std::string& name) { m_name = name; };
                virtual ~Sample() {};

                virtual void AddFeatures(const FeaturePtrList& features) { m_features.insert(std::end(m_features), std::begin(features), std::end(features)); };
                virtual void AddFeature(const FeaturePtr feature) { m_features.push_back(feature); };
                virtual FeaturePtrList Features() const { return m_features; };

            private:
                std::string m_name;
                FeaturePtrList m_features;
            };

            typedef Sample* SamplePtr;
            typedef std::vector<SamplePtr> SamplePtrList;

            /// Tree node, used internally.
            class Node
            {
            public:
                Node();
                Node(const std::string& featureName, uint64_t splitValue);
                virtual ~Node();

                virtual std::string FeatureName() const { return m_featureName; };
                virtual uint64_t SplitValue() const { return m_splitValue; };

                Node* Left() const { return m_left; };
                Node* Right() const { return m_right; };

                void SetLeftSubTree(Node* subtree);
                void SetRightSubTree(Node* subtree);

                std::string Dump() const;

            private:
                std::string m_featureName;
                uint64_t m_splitValue;

                Node* m_left;
                Node* m_right;

                void DestroyLeftSubtree();
                void DestroyRightSubtree();
            };

            typedef Node* NodePtr;
            typedef std::vector<NodePtr> NodePtrList;

            /// This class abstracts the random number generation.
            /// Inherit from this class if you wish to provide your own randomizer.
            /// Use Forest::SetRandomizer to override the default randomizer with one of your choosing.
            class Randomizer
            {
            public:
                Randomizer() : m_gen(m_rand()) {} ;
                virtual ~Randomizer() { };

                virtual uint64_t Rand() { return m_dist(m_gen); };
                virtual uint64_t RandUInt64(uint64_t min, uint64_t max) { return min + (Rand() % (max - min + 1)); }

            private:
                std::random_device m_rand;
                std::mt19937_64 m_gen;
                std::uniform_int_distribution<uint64_t> m_dist;
            };

            typedef std::set<uint64_t> Uint64Set;
            typedef std::map<std::string, Uint64Set> FeatureNameToValuesMap;

            /// Isolation Forest implementation.
            class Forest
            {
            public:
                Forest();
                Forest(uint32_t numTrees, uint32_t subSamplingSize);
                virtual ~Forest();

                void SetRandomizer(Randomizer* newRandomizer);
                void AddSample(const Sample& sample);
                void Create();

                double Score(const Sample& sample);
                double NormalizedScore(const Sample& sample);

                std::string Dump() const;

            private:
                Randomizer* m_randomizer; // Performs random number generation
                FeatureNameToValuesMap m_featureValues; // Lists each feature and maps it to all unique values in the training set
                NodePtrList m_trees; // The decision trees that comprise the forest
                uint32_t m_numTreesToCreate; // The maximum number of trees to create
                uint32_t m_subSamplingSize; // The maximum depth of a tree

                NodePtr CreateTree(const FeatureNameToValuesMap& featureValues, size_t depth);
                double Score(const Sample& sample, const NodePtr tree);

                void Destroy();
                void DestroyRandomizer();
            };
        };


**IsolationForest.cpp**

Algorithme 
1. Constructeur.
2. Renvoie le nœud sous forme de chaîne JSON.
3. Constructeur.
4. Destructeur.
5. Ajoute chacune des fonctionnalités de l'échantillon à la liste des fonctionnalités connues avec l'ensemble correspondant de valeurs uniques.
6. Nous ne stockons pas l'échantillon directement, juste les fonctionnalités.
7. Créez ou mettez à jour le nombre de valeurs de caractéristiques.
8. Crée et renvoie un seul arbre. Comme il s'agit d'une fonction récursive, depth indique la profondeur actuelle de la récursivité.
9. Verification sanitaire.
10. Si nous avons dépassé la profondeur maximale souhaitée, alors arrêtez.
11. Sélection aléatoire d'une fonctionnalité.
12. Récupère la liste de valeurs à diviser.
13. Sélection aléatoire d'une valeur fractionnée.
14. Crée un nœud d'arbre pour contenir la valeur fractionnée. un pour le côté gauche de l'arbre et un pour le côté droit.
15. Crée le sous-arbre de gauche.
16. Crée la bonne sous-arborescence.
17. Crée une forêt contenant le nombre d'arbres spécifié au constructeur.
     Note l'échantillon par rapport à l'arbre spécifié.
18. Trouve la fonctionnalité suivante dans l'exemple.
19. Si l'arbre contient une entité qui n'est pas dans l'échantillon, alors prenez
 les deux côtés de l'arbre et faites la moyenne des scores ensemble.
20. Note l'échantillon par rapport à l'ensemble de la forêt d'arbres. Le résultat est la longueur moyenne du trajet.
21. Note l'échantillon par rapport à l'ensemble de la forêt d'arbres. Le résultat est normalisé de sorte que les valeurs proches de 1 indiquent des anomalies et des valeurs proches de zéro indiquent des valeurs normales.
22. Détruit toute la forêt d'arbres.
23. Libère l'objet randomiseur personnalisé (le cas échéant).
24. Renvoie la forêt sous la forme d'un objet JSON.
25.  Dernier élément

    //mamadou@port-lipn12:~/big-data/cerin24022022/cpp-LibIsolationForest/cpp$ cat IsolationForest.cpp 
        //	MIT License
    //
    //  Copyright © 2017 Michael J Simms. All rights reserved.
    //
    //	Permission is hereby granted, free of charge, to any person obtaining a copy
    //	of this software and associated documentation files (the "Software"), to deal
    //	in the Software without restriction, including without limitation the rights
    //	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    //	copies of the Software, and to permit persons to whom the Software is
    //	furnished to do so, subject to the following conditions:
    //
    //	The above copyright notice and this permission notice shall be included in all
    //	copies or substantial portions of the Software.
    //
    //	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    //	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    //	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    //	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    //	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    //	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    //	SOFTWARE.

    #include "IsolationForest.h"
    #include <math.h>

    namespace IsolationForest
    {
        /// Constructor.
        Node::Node() :
            m_splitValue(0),
            m_left(NULL),
            m_right(NULL)
        {
        }

        /// Constructor.
        Node::Node(const std::string& featureName, uint64_t splitValue) :
            m_featureName(featureName),
            m_splitValue(splitValue),
            m_left(NULL),
            m_right(NULL)
        {
        }

        /// Destructor.
        Node::~Node()
        {
            DestroyLeftSubtree();
            DestroyRightSubtree();
        }

        void Node::SetLeftSubTree(Node* subtree)
        {
            DestroyLeftSubtree();
            m_left = subtree;
        }

        void Node::SetRightSubTree(Node* subtree)
        {
            DestroyRightSubtree();
            m_right = subtree;
        }

        void Node::DestroyLeftSubtree()
        {
            if (m_left)
            {
                delete m_left;
                m_left = NULL;
            }
        }

        void Node::DestroyRightSubtree()
        {
            if (m_right)
            {
                delete m_right;
                m_right = NULL;
            }
        }

        /// Returns the node as a JSON string.
        std::string Node::Dump() const
        {
            std::string data = "{";

            data.append("'Feature Name': '");
            data.append(this->m_featureName);
            data.append("', 'Split Value': ");
            data.append(std::to_string(m_splitValue));
            data.append(", 'Left': ");
            if (this->m_left)
                data.append(this->m_left->Dump());
            else
                data.append("{}");
            data.append(", 'Right': ");
            if (this->m_right)
                data.append(this->m_right->Dump());
            else
                data.append("{}");
            data.append("}");
            return data;
        }

        /// Constructor.
        Forest::Forest() :
            m_randomizer(new Randomizer()),
            m_numTreesToCreate(10),
            m_subSamplingSize(0)
        {
        }

        /// Constructor.
        Forest::Forest(uint32_t numTrees, uint32_t subSamplingSize) :
            m_randomizer(new Randomizer()),
            m_numTreesToCreate(numTrees),
            m_subSamplingSize(subSamplingSize)
        {
        }

        /// Destructor.
        Forest::~Forest()
        {
            DestroyRandomizer();
            Destroy();
        }

        void Forest::SetRandomizer(Randomizer* newRandomizer)
        {
            DestroyRandomizer();
            m_randomizer = newRandomizer;
        }

        /// Adds each of the sample's features to the list of known features
        /// with the corresponding set of unique values.
        void Forest::AddSample(const Sample& sample)
        {
            // We don't store the sample directly, just the features.
            const FeaturePtrList& features = sample.Features();
            FeaturePtrList::const_iterator featureIter = features.begin();
            while (featureIter != features.end())
            {
                const FeaturePtr feature = (*featureIter);
                const std::string& featureName = feature->Name();
                uint64_t featureValue = feature->Value();

                // Either create or update the feature values count.
                if (m_featureValues.count(featureName) == 0)
                {
                    Uint64Set featureValueSet;
                    featureValueSet.insert(featureValue);
                    m_featureValues.insert(std::make_pair(featureName, featureValueSet));
                }
                else
                {
                    Uint64Set& featureValueSet = m_featureValues.at(featureName);
                    featureValueSet.insert(featureValue);
                }

                ++featureIter;
            }
        }

        /// Creates and returns a single tree. As this is a recursive function,
        /// depth indicates the current depth of the recursion.
        NodePtr Forest::CreateTree(const FeatureNameToValuesMap& featureValues, size_t depth)
        {
            // Sanity check.
            size_t featureValuesLen = featureValues.size();
            if (featureValuesLen <= 1)
            {
                return NULL;
            }

            // If we've exceeded the maximum desired depth, then stop.
            if ((m_subSamplingSize > 0) && (depth >= m_subSamplingSize))
            {
                return NULL;
            }

            // Randomly select a feature.
            size_t selectedFeatureIndex = (size_t)m_randomizer->RandUInt64(0, featureValuesLen - 1);
            FeatureNameToValuesMap::const_iterator featureIter = featureValues.begin();
            std::advance(featureIter, selectedFeatureIndex);
            const std::string& selectedFeatureName = (*featureIter).first;

            // Get the value list to split on.
            const Uint64Set& featureValueSet = (*featureIter).second;
            if (featureValueSet.size() == 0)
            {
                return NULL;
            }

            // Randomly select a split value.
            size_t splitValueIndex = 0;
            if (featureValueSet.size() > 1)
            {
                splitValueIndex = (size_t)m_randomizer->RandUInt64(0, featureValueSet.size() - 1);
            }
            Uint64Set::const_iterator splitValueIter = featureValueSet.begin();
            std::advance(splitValueIter, splitValueIndex);
            uint64_t splitValue = (*splitValueIter);

            // Create a tree node to hold the split value.
            NodePtr tree = new Node(selectedFeatureName, splitValue);
            if (tree)
            {
                // Create two versions of the feature value set that we just used,
                // one for the left side of the tree and one for the right.
                FeatureNameToValuesMap tempFeatureValues = featureValues;

                // Create the left subtree.
                Uint64Set leftFeatureValueSet = featureValueSet;
                splitValueIter = leftFeatureValueSet.begin();
                std::advance(splitValueIter, splitValueIndex);
                leftFeatureValueSet.erase(splitValueIter, leftFeatureValueSet.end());
                tempFeatureValues[selectedFeatureName] = leftFeatureValueSet;
                tree->SetLeftSubTree(CreateTree(tempFeatureValues, depth + 1));

                // Create the right subtree.
                if (splitValueIndex < featureValueSet.size() - 1)
                {
                    Uint64Set rightFeatureValueSet = featureValueSet;
                    splitValueIter = rightFeatureValueSet.begin();
                    std::advance(splitValueIter, splitValueIndex + 1);
                    rightFeatureValueSet.erase(rightFeatureValueSet.begin(), splitValueIter);
                    tempFeatureValues[selectedFeatureName] = rightFeatureValueSet;
                    tree->SetRightSubTree(CreateTree(tempFeatureValues, depth + 1));
                }
            }

            return tree;
        }

        /// Creates a forest containing the number of trees specified to the constructor.
        void Forest::Create()
        {
            m_trees.reserve(m_numTreesToCreate);

            for (size_t i = 0; i < m_numTreesToCreate; ++i)
            {
                NodePtr tree = CreateTree(m_featureValues, 0);
                if (tree)
                {
                    m_trees.push_back(tree);
                }
            }
        }

        /// Scores the sample against the specified tree.
        double Forest::Score(const Sample& sample, const NodePtr tree)
        {
            double depth = (double)0.0;

            const FeaturePtrList& features = sample.Features();

            NodePtr currentNode = tree;
            while (currentNode)
            {
                bool foundFeature = false;

                // Find the next feature in the sample.
                FeaturePtrList::const_iterator featureIter = features.begin();
                while (featureIter != features.end() && !foundFeature)
                {
                    const FeaturePtr currentFeature = (*featureIter);
                    if (currentFeature->Name().compare(currentNode->FeatureName()) == 0)
                    {
                        if (currentFeature->Value() < currentNode->SplitValue())
                        {
                            currentNode = currentNode->Left();
                        }
                        else
                        {
                            currentNode = currentNode->Right();
                        }
                        ++depth;
                        foundFeature = true;
                    }
                    ++featureIter;
                }

                // If the tree contained a feature not in the sample then take
                // both sides of the tree and average the scores together.
                if (!foundFeature)
                {
                    double leftDepth = depth + Score(sample, currentNode->Left());
                    double rightDepth = depth + Score(sample, currentNode->Right());
                    return (leftDepth + rightDepth) / (double)2.0;
                }
            }
            return depth;
        }

        /// Scores the sample against the entire forest of trees. Result is the average path length.
        double Forest::Score(const Sample& sample)
        {
            double avgPathLen = (double)0.0;

            if (m_trees.size() > 0)
            {
                NodePtrList::const_iterator treeIter = m_trees.begin();
                while (treeIter != m_trees.end())
                {
                    avgPathLen += (double)Score(sample, (*treeIter));
                    ++treeIter;
                }
                avgPathLen /= (double)m_trees.size();
            }
            return avgPathLen;
        }

        #define H(i) (log(i) + 0.5772156649)
        #define C(n) (2 * H(n - 1) - (2 * (n - 1) / n))

        /// Scores the sample against the entire forest of trees. Result is normalized so that values
        /// close to 1 indicate anomalies and values close to zero indicate normal values.
        double Forest::NormalizedScore(const Sample& sample)
        {
            double score = (double)0.0;
            size_t numTrees = m_trees.size();

            if (numTrees > 0)
            {
                double avgPathLen = (double)0.0;

                NodePtrList::const_iterator treeIter = m_trees.begin();
                while (treeIter != m_trees.end())
                {
                    avgPathLen += (double)Score(sample, (*treeIter));
                    ++treeIter;
                }
                avgPathLen /= (double)numTrees;

                if (numTrees > 1)
                {
                    double exponent = -1.0 * (avgPathLen / C(numTrees));
                    score = pow(2, exponent);
                }
            }
            return score;
        }

        /// Destroys the entire forest of trees.
        void Forest::Destroy()
        {
            std::vector<NodePtr>::iterator iter = m_trees.begin();
            while (iter != m_trees.end())
            {
                NodePtr tree = (*iter);
                if (tree)
                {
                    delete tree;
                }
                ++iter;
            }
            m_trees.clear();
        }

        /// Frees the custom randomizer object (if any).
        void Forest::DestroyRandomizer()
        {
            if (m_randomizer)
            {
                delete m_randomizer;
                m_randomizer = NULL;
            }
        }

        /// Returns the forest as a JSON object.
        std::string Forest::Dump() const
        {
            std::string data = "{";
            auto featureValuesIter = m_featureValues.begin();
            auto treeIter = m_trees.begin();

            data.append("'Sub Sampling Size': ");
            data.append(std::to_string(this->m_subSamplingSize));
            data.append(", 'Feature Values': [");
            while (featureValuesIter != m_featureValues.end())
            {
                data.append("'");
                data.append((*featureValuesIter).first);
                data.append("': [");

                auto valuesIter = (*featureValuesIter).second.begin();
                while (valuesIter != (*featureValuesIter).second.end())
                {
                    data.append(std::to_string(*valuesIter));
                    ++valuesIter;

                    // Last item?
                    if (valuesIter != (*featureValuesIter).second.end())
                        data.append(", ");
                }
                ++featureValuesIter;

                data.append("]");

                // Last item?
                if (featureValuesIter != m_featureValues.end())
                    data.append(", ");
            }
            data.append("], 'Trees': [");
            while (treeIter != m_trees.end())
            {
                data.append((*treeIter)->Dump());
                ++treeIter;

                // Last item?
                if (treeIter != m_trees.end())
                    data.append(", ");
            }
            data.append("]");
            data.append("}");
            return data;
        }
    }
    
    

        mamadou@port-lipn12:~/big-data/cerin24022022/cpp-LibIsolationForest/cpp
        ./isolationforestsimms 
        Test 1:
        -------
        Average of control test samples: 7.35
        Average of control test samples (normalized): 0.330412
        Average of outlier test samples: 6.14
        Average of outlier test samples (normalized): 0.393111
        Total time for Test 1: 0.00767787 seconds.

        Test 2:
        -------
        Average of control test samples: 8.8179
        Average of control test samples (normalized): 0.521212
        Average of outlier test samples: 5.3068
        Average of outlier test samples (normalized): 0.675219
        Total time for Test 1: 0.0556247 seconds.

