CUDA_ARCH ?= sm_20

test: main.cu introspect.o save_array.o c2r.h temporary.h gcd.h transpose.h save_array.h index.h
	nvcc -arch=$(CUDA_ARCH) -o test main.cu introspect.o save_array.o

introspect.o: introspect.h introspect.cu
	nvcc -c -o introspect.o introspect.cu

save_array.o: save_array.h save_array.cu
	nvcc -c -o save_array.o save_array.cu
