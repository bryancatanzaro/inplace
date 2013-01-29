#pragma once
namespace inplace {

__device__ __inline__ int ld_glb_cs(const int* a) {
    int r;
    asm("ld.global.cs.s32 %0, [%1];" : "=r"(r) : "l"(a));
    return r;
}

__device__ __inline__ void st_glb_cs(int* ptr, const int& val) {
    const int raw = reinterpret_cast<const int&>(val);
    asm("st.global.cs.s32 [%0], %1;" : : "l"(ptr), "r"(raw));
}

}
