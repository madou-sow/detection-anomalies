CFLAGS=-Wall -O3 -std=c++11
LDFLAGS=-L/home/mamadou/big-data/cerin24022022/isotree/build/libisotree.so /usr/local/anaconda3/lib/libisotree.so /usr/local/anaconda3/lib/gcc/x86_64-conda_cos6-linux-gnu/7.3.0/libisotree.so /usr/local/lib/libisotree.so -lpthread -lisotree

.PHONY: all clean 

all : isotree_cpp_sow

isotree_cpp_sow : isotree_cpp_sow.o 
	${CXX} ${CFLAGS} $^ -o $@ ${LDFLAGS} 

isotree_cpp_sow.o : isotree_cpp_sow.cpp
	${CXX} ${CFLAGS} -c $^ -o $@

clean : 
	-rm -f *.o 

