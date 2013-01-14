#include "introspect.h"
#include <iostream>
namespace inplace {

namespace {
    introspect cached_properties;
}

int n_threads() {
    int maxThreadsPerBlock =
        cached_properties.properties.maxThreadsPerBlock;
    return (maxThreadsPerBlock > 0) ? maxThreadsPerBlock:
        256;
}


int n_ctas() {
    int n_sms = cached_properties.properties.multiProcessorCount;
    if (n_sms == 0) n_sms = 14;
    int n_threads_per_sm =
        cached_properties.properties.maxThreadsPerMultiProcessor;
    int blocks_per_sm = n_threads_per_sm / n_threads();
    return n_sms * blocks_per_sm;
}

size_t gpu_memory_size() {
    return cached_properties.properties.totalGlobalMem;
}


}
