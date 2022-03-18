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

