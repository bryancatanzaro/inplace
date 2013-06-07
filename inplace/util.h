#pragma once

__host__ __device__ __forceinline__
unsigned int div_up(unsigned int a, unsigned int b) {
    return (a-1)/b + 1;
}

__host__ __device__ __forceinline__
unsigned int div_down(unsigned int a, unsigned int b) {
    return a / b;
}
