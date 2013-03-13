CUDA_ARCH ?= sm_35

benchmark: benchmark.cu introspect.o save_array.o instantiation.o gcd.o sequential.o
	nvcc -Xptxas -v -arch=$(CUDA_ARCH) -o test benchmark.cu introspect.o save_array.o instantiation.o gcd.o sequential.o

gcd.o: gcd.cu
	nvcc -c -o gcd.o gcd.cu

introspect.o: introspect.h introspect.cu
	nvcc -c -o introspect.o introspect.cu

save_array.o: save_array.h save_array.cu
	nvcc -c -o save_array.o save_array.cu

instantiation.o: instantiation.cu c2r.h index.h
	nvcc -c -Xptxas -v -arch=$(CUDA_ARCH) -o instantiation.o instantiation.cu

sequential.o: sequential.cu c2r.h index.h
	nvcc -c -Xptxas -v -arch=$(CUDA_ARCH) -o sequential.o sequential.cu