#pragma once
#include "volume_grid.hpp"
namespace vcr::augment::volume_detail {
struct Stats { double low,high,mean,stddev; };
template<class T> VCR_HD Stats stats(const T* v,std::size_t n) {
    Stats s{double(v[0]),double(v[0]),0,0};double m2=0;
    for(std::size_t i=0;i<n;++i) { double x=v[i];s.low=x<s.low?x:s.low;s.high=x>s.high?x:s.high;double d=x-s.mean;s.mean+=d/double(i+1);m2+=d*(x-s.mean); }
    s.stddev=sqrt(m2>0?m2/double(n):0);return s;
}
VCR_HD inline int bin(double x,double lo,double hi,int bins) {
    return hi>lo?int(clamp((x-lo)/(hi-lo)*bins,0,bins-1)):0;
}
template<class T> VCR_HD void histogram(const T* data,std::size_t n,double* h,int bins,double lo,double hi) {
    for(int b=0;b<bins;++b)h[b]=0;
    for(std::size_t i=0;i<n;++i)h[bin(data[i],lo,hi,bins)]+=1;
}
VCR_HD inline void cdf(double* h,int bins,double clip_limit=0) {
    double total=0;for(int b=0;b<bins;++b)total+=h[b];
    if(clip_limit>0) {
        double limit=fmax(1,clip_limit*total/bins),excess=0;
        for(int b=0;b<bins;++b)if(h[b]>limit){excess+=h[b]-limit;h[b]=limit;}
        for(int b=0;b<bins;++b)h[b]+=excess/bins;
    }
    double sum=0;for(int b=0;b<bins;++b){sum+=h[b];h[b]=total?sum/total:0;}
}

} // namespace vcr::augment::volume_detail

namespace vcr::augment::volume_detail {
struct HistogramGrid {
    std::size_t tiles[3],count=1;int bins;double clip_limit;
    HistogramGrid(const Grid3& g,const Shape& tile,int bins,double clip):bins(bins),clip_limit(clip) {
        require(bins>=2 && bins<=4096,"bins must be [2,4096]");require(tile.size()==3,"tile_shape needs three dimensions");
        for(int a=0;a<3;++a){require(tile[a]>0 && tile[a]<=65536,"tile sizes must be [1,65536]");tiles[a]=(g.n[a]+tile[a]-1)/tile[a];count*=tiles[a];}
    }
};
template<class T> VCR_HD inline void histogram_tile(const T* v,const Grid3& g,const Stats& s,const HistogramGrid& layout,std::size_t tile,double* table) {
    double* h=table+tile*layout.bins;for(int b=0;b<layout.bins;++b)h[b]=0;
    std::size_t pos[3]{tile/(layout.tiles[1]*layout.tiles[2]),tile/layout.tiles[2]%layout.tiles[1],tile%layout.tiles[2]},lo[3],hi[3];
    for(int a=0;a<3;++a){lo[a]=pos[a]*g.n[a]/layout.tiles[a];hi[a]=(pos[a]+1)*g.n[a]/layout.tiles[a];}
    for(std::size_t z=lo[0];z<hi[0];++z)for(std::size_t y=lo[1];y<hi[1];++y)for(std::size_t x=lo[2];x<hi[2];++x)
        h[bin(v[z*g.stride[0]+y*g.stride[1]+x],s.low,s.high,layout.bins)]+=1;
    cdf(h,layout.bins,layout.clip_limit);
}
VCR_HD inline double interpolate_histograms(const Grid3& g,const HistogramGrid& layout,std::size_t i,int b,const double* table) {
    std::size_t lo[3],hi[3];double f[3];
    for(int a=0;a<3;++a){double q=clamp((double(g.coord(i,a))+.5)*layout.tiles[a]/g.n[a]-.5,0,double(layout.tiles[a]-1));lo[a]=std::size_t(floor(q));hi[a]=lo[a]+1<layout.tiles[a]?lo[a]+1:lo[a];f[a]=q-lo[a];}
    double result=0;
    for(int corner=0;corner<8;++corner){std::size_t t=0;double w=1;for(int a=0;a<3;++a){bool bit=(corner>>a)&1;t=t*layout.tiles[a]+(bit?hi[a]:lo[a]);w*=bit?f[a]:1-f[a];}result+=w*table[t*layout.bins+b];}
    return result;
}
template<class T> inline std::vector<double> channel_histograms(const std::vector<T>& v,const Grid3& g,const HistogramGrid& layout) {
    auto summary=stats(v.data(),v.size());std::vector<double> table(layout.count*layout.bins);
    for(std::size_t tile=0;tile<layout.count;++tile)histogram_tile(v.data(),g,summary,layout,tile,table.data());return table;
}
} // namespace vcr::augment::volume_detail
