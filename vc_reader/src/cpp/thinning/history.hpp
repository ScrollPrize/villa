#pragma once
#include <vc_reader/thinning.hpp>
#ifdef __CUDACC__
#define VCR_HISTORY_HD __host__ __device__
#else
#define VCR_HISTORY_HD
#endif
namespace vcr::thinning::detail {
struct Penalties {float values[4][512][8];};
const Penalties& penalties();
struct PointHistory {
    Point last{};
    int count=0;
    unsigned code=0;
    VCR_HISTORY_HD void push_back(Point point) {
        if(count) {
            const int dx=point.x-last.x,dy=point.y-last.y;
            const unsigned index=unsigned((dy+1)*3+dx+1);
            const unsigned direction=index>4?index-1:index;
            code=((code<<3)|direction)&511;
        }
        count=count<4?count+1:4;last=point;
    }
};

}
#undef VCR_HISTORY_HD
