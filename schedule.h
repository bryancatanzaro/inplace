#pragma once

namespace inplace {

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

struct sm_10{};
struct sm_20{};
struct sm_30{};
struct sm_35{};

template<typename T, typename SM>
struct schedule{};


template<>
struct schedule<float, sm_35> {
    typedef cons_tup<35, 256, cons_tup<100, 256, cons_tup<100, 512> > > type;
};

template<>
struct schedule<double, sm_35> {
    typedef cons_tup<19, 256, cons_tup<50, 256, cons_tup<108, 256> > > type;
};

}
