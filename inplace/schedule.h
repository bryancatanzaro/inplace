#pragma once
#include "sm.h"

namespace inplace {
namespace detail {

struct nil{};

template<typename Head, typename Tail>
struct cons {
    typedef Head head;
    typedef Tail tail;
};

template<int w, int b>
struct reg {
    //Work per thread
    static const int wpt = w;
    //CUDA block size
    static const int blk = b;
    //Size limit for row or column
    static const int lim = w * b;    
};

template<typename T, typename SM, int blks>
struct smem {
    static const int smem_per_blk = SM::smem / blks;
    static const int vals_per_blk = smem_per_blk / sizeof(T);
    static const int lim = vals_per_blk;
    static const int blk = SM::blk;
};

struct memory{};

template<typename T, typename SM>
struct schedule{};

template<>
struct schedule<float, sm_35> {
    
    // typedef cons<smem<float, sm_35, 8>, cons<smem<float, sm_35, 6>, cons<smem<float, sm_35, 4>, cons<smem<float, sm_35, 2>, cons<smem<float, sm_35, 1>, cons<reg<5, 256>, cons<reg<5, 384>, cons<reg<5, 512>, cons<reg<12, 256>, cons<reg<12, 384>, cons<reg<12, 512>, cons<reg<20, 256>, cons<reg<20, 384>, cons<reg<20, 512>, cons<reg<30, 256>, cons<reg<30, 384>, cons<reg<30, 512>, cons<reg<40, 256>, cons<reg<40, 384>, cons<reg<40, 512>, cons<reg<54, 256>, cons<reg<54, 384>, cons<reg<54, 512>,  memory> > > > > > > > > > > > > > > > > > > > > > > type;
    typedef cons<smem<float, sm_35, 1>, cons<reg<31, 512>, cons<reg<60, 512>, memory> > > type;
};

template<>
struct schedule<double, sm_35> {
    // typedef cons<smem<double, sm_35, 8>, cons<smem<double, sm_35, 6>, cons<smem<double, sm_35, 4>, cons<smem<double, sm_35, 2>, cons<smem<double, sm_35, 1>, cons<reg<4, 256>, cons<reg<4, 384>, cons<reg<4, 512>, cons<reg<11, 256>, cons<reg<11, 384>, cons<reg<11, 512>, cons<reg<19, 256>, cons<reg<19, 384>, cons<reg<19, 512>, cons<reg<28, 256>, cons<reg<28, 384>, cons<reg<28, 512>, cons<reg<43, 256>, cons<reg<43, 384>, cons<reg<43, 512>, cons<reg<59, 256>, cons<reg<59, 384>, cons<reg<59, 512>, cons<reg<87, 256>, cons<reg<115, 256>, memory> > > > > > > > > > > > > > > > > > > > > > > > >type;
    // typedef cons<smem<double, sm_35, 8>, cons<reg<5, 256>, memory>
    // > type;
    typedef cons<smem<double, sm_35, 1>, cons<reg<16, 512>, cons<reg<18, 512>, cons<reg<59, 512>, memory > > > > type;
};


template<>
struct schedule<float, sm_20> {
    typedef cons<smem<float, sm_20, 8>, cons<reg<5, 256>, memory> > type;

};

template<>
struct schedule<double, sm_20> {
    typedef cons<smem<double, sm_20, 8>, cons<reg<5, 256>, memory> > type;
};

}
}
