#pragma once
#include <vc_reader/thinning.hpp>
#if defined(__CUDACC__)
#define VCR_THIN_HD __host__ __device__
#else
#define VCR_THIN_HD
#endif
namespace vcr::thinning::detail {
// Clockwise neighbors N,NE,E,SE,S,SW,W,NW. Shared by both backends.
VCR_THIN_HD inline bool remove(unsigned code, int phase, Algorithm algorithm) {
    const bool n[8] = {bool(code&1),bool(code&2),bool(code&4),bool(code&8),
                       bool(code&16),bool(code&32),bool(code&64),bool(code&128)};
    if (algorithm == Algorithm::GuoHall) {
        const int transitions = (!n[0] && (n[1] || n[2])) + (!n[2] && (n[3] || n[4])) +
                                (!n[4] && (n[5] || n[6])) + (!n[6] && (n[7] || n[0]));
        const int a=(n[7]||n[0])+(n[1]||n[2])+(n[3]||n[4])+(n[5]||n[6]);
        const int b=(n[0]||n[1])+(n[2]||n[3])+(n[4]||n[5])+(n[6]||n[7]);
        const int neighbors=a<b?a:b;
        const bool preserve = phase == 0 ? ((n[4]||n[5]||!n[7])&&n[6]) : ((n[0]||n[1]||!n[3])&&n[2]);
        return transitions==1 && neighbors>=2 && neighbors<=3 && !preserve;
    }
    int neighbors=0,transitions=0;
    for(int k=0;k<8;++k) {neighbors+=n[k];transitions+=!n[k]&&n[(k+1)%8];}
    const bool preserve=phase==0 ? (n[0]&&n[2]&&n[4])||(n[2]&&n[4]&&n[6]) :
                                             (n[0]&&n[2]&&n[6])||(n[0]&&n[4]&&n[6]);
    return neighbors>=2 && neighbors<=6 && transitions==1 && !preserve;
}
VCR_THIN_HD inline unsigned neighbors(const std::uint8_t* image, std::size_t p, std::size_t rows, std::size_t cols) {
    const auto y=p/cols,x=p%cols;
    if(y>0 && y+1<rows && x>0 && x+1<cols)
        return unsigned(image[p-cols]!=0) | unsigned(image[p-cols+1]!=0)<<1 | unsigned(image[p+1]!=0)<<2 |
            unsigned(image[p+cols+1]!=0)<<3 | unsigned(image[p+cols]!=0)<<4 | unsigned(image[p+cols-1]!=0)<<5 |
            unsigned(image[p-1]!=0)<<6 | unsigned(image[p-cols-1]!=0)<<7;
    const int dx[8]={0,1,1,1,0,-1,-1,-1},dy[8]={-1,-1,0,1,1,1,0,-1};
    unsigned code=0;
    for(int k=0;k<8;++k){
        const auto yy=static_cast<long long>(y)+dy[k],xx=static_cast<long long>(x)+dx[k];
        if(yy>=0 && xx>=0 && yy<static_cast<long long>(rows) && xx<static_cast<long long>(cols))
            code |= unsigned(image[std::size_t(yy)*cols+std::size_t(xx)]!=0)<<k;
    }
    return code;
}
}
#undef VCR_THIN_HD
