# C compiler
CC = g++
ICC = gcc #icc
CC_FLAGS = -g -fopenmp -O2

all: gemv

gemv: gemv.cpp 
	$(CC) $(CC_FLAGS) gemv.cpp -o gemv 

clean:
	rm -rf gemv