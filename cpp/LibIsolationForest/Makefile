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
	#-rm -f *.o isolationforestsimms
	-rm -f *.o 

