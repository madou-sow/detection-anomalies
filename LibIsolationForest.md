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

 Constructeur.
 
 Constructeur.
 
 Destructeur.
 
 Renvoie le nœud sous forme de chaîne JSON.
 
 Constructeur.
 
 Constructeur.
 
 Destructeur.
 
 Ajoute chacune des fonctionnalités de l'échantillon à la liste des fonctionnalités connues
 avec l'ensemble correspondant de valeurs uniques.
 
 Nous ne stockons pas l'échantillon directement, juste les fonctionnalités.
        
 Créez ou mettez à jour le nombre de valeurs de caractéristiques.

 Crée et renvoie un seul arbre. Comme il s'agit d'une fonction récursive,
 depth indique la profondeur actuelle de la récursivité.
 
 Verification sanitaire.

 Si nous avons dépassé la profondeur maximale souhaitée, alors arrêtez.

 Sélection aléatoire d'une fonctionnalité.

 Récupère la liste de valeurs à diviser.

 Sélection aléatoire d'une valeur fractionnée.

 Crée un nœud d'arbre pour contenir la valeur fractionnée.

 Crée deux versions de l'ensemble de valeurs de caractéristiques que nous venons d'utiliser,
 un pour le côté gauche de l'arbre et un pour le côté droit.

 Crée le sous-arbre de gauche.

 Crée la bonne sous-arborescence.

 Crée une forêt contenant le nombre d'arbres spécifié au constructeur.
     Note l'échantillon par rapport à l'arbre spécifié.
     
 Trouve la fonctionnalité suivante dans l'exemple.

 Si l'arbre contient une entité qui n'est pas dans l'échantillon, alors prenez
 les deux côtés de l'arbre et faites la moyenne des scores ensemble.

 Note l'échantillon par rapport à l'ensemble de la forêt d'arbres. Le résultat est la longueur moyenne du trajet.
 Note l'échantillon par rapport à l'ensemble de la forêt d'arbres. Le résultat est normalisé de sorte que les valeurs
     proches de 1 indiquent des anomalies et des valeurs proches de zéro indiquent des valeurs normales.
 Détruit toute la forêt d'arbres.
 Libère l'objet randomiseur personnalisé (le cas échéant).
 Renvoie la forêt sous la forme d'un objet JSON.
