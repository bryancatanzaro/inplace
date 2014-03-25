Inplace
=======

CUDA and OpenMP implementations of the C2R and R2C inplace
transposition algorithms.  These algorithms are described in our
[PPoPP paper](http://dl.acm.org/citation.cfm?id=2555253).

We have included a specialization for very tall, skinny matrices that
yields good performance for in-place conversions between Arrays of
Structures and Structures of Arrays.

The code includes OpenMP and CUDA implementations.
The OpenMP implementation is declared in `<inplace/openmp.h>`, while
the CUDA implementation is declared in `<inplace/transpose.h>`, and
carries the following signatures:

```c++
void transpose(bool row_major, float* data, int m, int n);
void transpose(bool row_major, double* data, int m, int n);
```
