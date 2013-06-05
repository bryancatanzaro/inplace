#pragma once
namespace inplace {
namespace detail {

struct sm_10{
    static const int smem = 16384;
    static const int blk = 128;
};
struct sm_20{
    static const int blk = 256;
    static const int smem = 49152/(1536/blk);
};
struct sm_30{
    static const int blk = 256;
    static const int smem = 49152/(2048/blk);
};
struct sm_35{
    static const int blk = 256;
    static const int smem = 49152/(2048/blk);
};

}
}
