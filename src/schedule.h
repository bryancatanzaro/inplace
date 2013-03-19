#pragma once
#include "sm.h"

namespace inplace {
namespace detail {

struct nil{};

template<int w, int b, typename Next=nil>
struct cons_tup {
    //Work per thread
    static const int wpt = w;
    //CUDA block size
    static const int blk = b;
    //Size limit for row or column
    static const int lim = w * b;
    typedef Next tail;
};

struct memory{};

template<typename T, typename SM>
struct schedule{};

template<>
struct schedule<float, sm_35> {
    typedef cons_tup<5, 256, cons_tup<5, 384, cons_tup<5, 512, cons_tup<20, 256, cons_tup<20, 384, cons_tup<20, 512, cons_tup<37, 256, cons_tup<37, 384, cons_tup<37, 512, cons_tup<77, 256, cons_tup<77, 384, cons_tup<77, 512, cons_tup<100, 256, cons_tup<100, 384, cons_tup<100, 512, memory> > > > > > > > > > > > > > > type;
};

template<>
struct schedule<double, sm_35> {
    typedef cons_tup<4, 256, cons_tup<4, 384, cons_tup<4, 512, cons_tup<11, 256, cons_tup<11, 384, cons_tup<11, 512, cons_tup<18, 256, cons_tup<18, 384, cons_tup<18, 512, cons_tup<34, 256, cons_tup<34, 384, cons_tup<34, 512, cons_tup<51, 256, cons_tup<51, 384, cons_tup<51, 512, cons_tup<83, 256, cons_tup<115, 256, memory> > > > > > > > > > > > > > > > > type;
};


template<>
struct schedule<float, sm_20> {
    typedef cons_tup<5, 256, memory> type;

};

template<>
struct schedule<double, sm_20> {
    typedef cons_tup<5, 256, memory> type;
};

}
}
