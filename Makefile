test: main.cu introspect.o save_array.o r2c.h c2r.h temporary.h gcd.h transpose.h save_array.h
	nvcc -arch=sm_20 -o test main.cu introspect.o save_array.o

introspect.o: introspect.h introspect.cu
	nvcc -c -o introspect.o introspect.cu

save_array.o: save_array.h save_array.cu
	nvcc -c -o save_array.o save_array.cu