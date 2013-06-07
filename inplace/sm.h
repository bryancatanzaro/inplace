#pragma once
namespace inplace {
namespace detail {

struct sm_10 {
    static const int smem = 16384;
    static const int blk = 128;
    static const int thrs = 768;
};
struct sm_12 {
    static const int smem = 16384;
    static const int blk = 128;
    static const int thrs = 1024;
};
    
struct sm_20{
    static const int blk = 256;
    static const int smem = 49152;
    static const int thrs = 1536;
};

struct sm_30{
    static const int blk = 256;
    static const int smem = 49152;
    static const int thrs = 2048;
};
struct sm_35{
    static const int blk = 256;
    static const int smem = 49152;
    static const int thrs = 2048;
};

}
}
