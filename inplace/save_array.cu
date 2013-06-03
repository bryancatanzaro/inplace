#include "save_array.h"
namespace inplace {
void save_array(const char* name,
                int* gpu_d,
                int m,
                int n) {
    int* h_d = (int*)malloc(sizeof(int) * m * n);
    cudaMemcpy(h_d, gpu_d, m * n * sizeof(int), cudaMemcpyDeviceToHost);
    std::ofstream output;
    output.open(name);
    int* p = h_d;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            output << *p << " ";
            ++p;
        }
        output << std::endl;
    }
    output.close();
}
}
