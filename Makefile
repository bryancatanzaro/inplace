test: main.cu r2c.h c2r.h temporary.h gcd.h transpose.h
	nvcc -o test main.cu