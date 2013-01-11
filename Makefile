test: main.cu introspect.o r2c.h c2r.h temporary.h gcd.h transpose.h
	nvcc -o test main.cu introspect.o

introspect.o: introspect.h introspect.cu
	nvcc -c -o introspect.o introspect.cu