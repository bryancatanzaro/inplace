#pragma once
namespace inplace {

struct timer {
    timer();
    ~timer();
    void start();
    float stop();
private:
    cudaEvent_t m_start, m_stop;
};


}
