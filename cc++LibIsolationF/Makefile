#CFLAGS=-Wall -ggdb -I../../lib -I../../lib/cpp
#LDFLAGS=-L../../lib ../../lib/cpp/libmosquittopp.so.1 ../../lib/libmosquitto.so.1
CFLAGS=-Wall -O3 -std=c++11
#LDFLAGS=-L/usr/lib/arm-linux-gnueabihf/libmosquittopp.so.1 /usr/lib/arm-linux-gnueabihf/libmosquitto.so.1 -lmosquittopp -lmosquitto -lpthread
#LDFLAGS=-L/usr/local/lib/libmosquittopp.so.1 /usr/local/lib/libmosquittopp.so.1 -lmosquittopp -lmosquitto -lpthread

.PHONY: all clean 

#all : mqtt_date_amplitude_kmeansplusplus 
#all : mqtt_date_amplitude_kmeans 
#all : mqtt_date_amplitude-w2048-k3
all : isolationforestsimms

#
# date_amplitude_w2048_c64_k3.o 
#
#mqtt_date_amplitude-w2048-k3 : main.o date_amplitude_w2048_c64_k3.o kmeans.beautifier.o kmeansPlusPlus.beautifier.o sort.beautifier.o
#	${CXX} ${CFLAGS} $^ -o $@ ${LDFLAGS} 
isolationforestsimms : main.o IsolationForest.o 
	${CXX} ${CFLAGS} $^ -o $@ ${LDFLAGS} 

main.o : main.cpp
	${CXX} ${CFLAGS} -c $^ -o $@

#date_amplitude_kmeansplusplus.o : date_amplitude_kmeansplusplus.cpp
#	${CXX} ${CFLAGS} -c $^ -o $@
#date_amplitude_kmeans.o : date_amplitude_kmeans.cpp
#	${CXX} ${CFLAGS} -c $^ -o $@
#date_amplitude_w2048_c64_k3.o : date_amplitude_w2048_c64_k3.cpp
#	${CXX} ${CFLAGS} -c $^ -o $@
IsolationForest.o : IsolationForest.cpp
	${CXX} ${CFLAGS} -c $^ -o $@

#kmeans.beautifier.o : kmeans.beautifier.cpp
#	${CXX} ${CFLAGS} -c $^ -o $@

#kmeansPlusPlus.beautifier.o : kmeansPlusPlus.beautifier.cpp
#	${CXX} ${CFLAGS} -c $^ -o $@

#sort.beautifier.o : sort.beautifier.cpp
#	${CXX} ${CFLAGS} -c $^ -o $@

clean : 
	#-rm -f *.o mqtt_date_amplitude
	#-rm -f *.o isolationforestsimms
	-rm -f *.o 

