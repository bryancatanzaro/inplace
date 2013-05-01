mkl_test: mkl_test.o
	g++ -o mkl_test mkl_test.o -L/home/bcatanzaro/p4/sw/sw/tools/linux/mkl/11.0.1/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -ldl -lpthread -lm -lgomp -L/usr/local/cuda/lib64 -lcuda -lcudart

mkl_test.o: mkl_test.cu
	nvcc -Xcompiler -O3 -o mkl_test.o -c -I/home/bcatanzaro/p4/sw/sw/tools/linux/mkl/11.0.1/include mkl_test.cu