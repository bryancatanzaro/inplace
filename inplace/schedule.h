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
    typedef cons<smem<float, sm_35, 8>, cons<reg<5, 256>, cons<reg<5, 384>, cons<reg<5, 512>, cons<reg<20, 256>, cons<reg<20, 384>, cons<reg<20, 512>, cons<reg<37, 256>, cons<reg<37, 384>, cons<reg<37, 512>, cons<reg<77, 256>, cons<reg<77, 384>, cons<reg<77, 512>, cons<reg<100, 256>, cons<reg<100, 384>, cons<reg<100, 512>, memory> > > > > > > > > > > > > > > > type;
    // typedef cons<smem<float, sm_35, 8>, cons<reg<5, 256>, memory> > type;
};

template<>
struct schedule<double, sm_35> {
    typedef cons<smem<double, sm_35, 8>, cons<smem<double, sm_35, 6>, cons<smem<double, sm_35, 4>, cons<smem<double, sm_35, 2>, cons<smem<double, sm_35, 1>, cons<reg<4, 256>, cons<reg<4, 384>, cons<reg<4, 512>, cons<reg<11, 256>, cons<reg<11, 384>, cons<reg<11, 512>, cons<reg<18, 256>, cons<reg<18, 384>, cons<reg<18, 512>, cons<reg<34, 256>, cons<reg<34, 384>, cons<reg<34, 512>, cons<reg<51, 256>, cons<reg<51, 384>, cons<reg<51, 512>, cons<reg<83, 256>, cons<reg<115, 256>, memory> > > > > > > > > > > > > > > > > > > > > > type;
    // typedef cons<smem<double, sm_35, 8>, cons<reg<5, 256>, memory> > type;
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
