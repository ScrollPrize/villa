#pragma once
#include "../core.hpp"
namespace vcr::transforms::surface_detail {
template<class Get> VCR_HD bool normal(Get get,std::size_t h,std::size_t w,std::size_t i,double* out) {
    const auto y=i/w,x=i%w,n=h*w;double row[3],col[3];
    for(std::size_t c=0;c<3;++c) {
        row[c]=get(c*n+(y+1<h?y+1:y)*w+x)-get(c*n+(y?y-1:y)*w+x);
        col[c]=get(c*n+y*w+(x+1<w?x+1:x))-get(c*n+y*w+(x?x-1:x));
    }
    out[0]=col[1]*row[2]-col[2]*row[1];out[1]=col[2]*row[0]-col[0]*row[2];out[2]=col[0]*row[1]-col[1]*row[0];
    double norm=0;for(int c=0;c<3;++c)norm+=out[c]*out[c];norm=std::sqrt(norm);
    for(int c=0;c<3;++c)out[c]=norm>1e-6?out[c]/norm:0;
    return norm>1e-6;
}
VCR_HD inline double distance(std::size_t i,std::size_t center,std::size_t w) {
    const double y=double(i/w)-double(center/w),x=double(i%w)-double(center%w);return y*y+x*x;
}
VCR_HD inline double contribution(double distance,double radius2,double sigma,double amplitude) {
    return distance<=radius2?amplitude*std::exp(-.5*distance/std::max(sigma*sigma,1e-6)):0;
}
}
