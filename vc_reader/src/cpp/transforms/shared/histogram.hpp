#pragma once
#include "volume_grid.hpp"
namespace vcr::transforms::volume_detail {
struct Stats { double low,high,mean,stddev; };
// stats<false>'s bounds in independent lanes. Strict comparisons keep the first of equal
// values, which only matters for signed zeros: a zero extreme takes the first zero's sign.
template<class T> VCR_TARGET_CLONES void lane_bounds(const T* v,std::size_t n,T& low,T& high) {
    constexpr std::size_t lanes=16;T lo[lanes],hi[lanes];
    for(std::size_t j=0;j<lanes;++j){lo[j]=v[0];hi[j]=v[0];}
    std::size_t i=0;
    for(;i+lanes<=n;i+=lanes)for(std::size_t j=0;j<lanes;++j){const T x=v[i+j];lo[j]=x<lo[j]?x:lo[j];hi[j]=x>hi[j]?x:hi[j];}
    for(;i<n;++i){lo[0]=v[i]<lo[0]?v[i]:lo[0];hi[0]=v[i]>hi[0]?v[i]:hi[0];}
    low=lo[0];high=hi[0];
    for(std::size_t j=1;j<lanes;++j){low=lo[j]<low?lo[j]:low;high=hi[j]>high?hi[j]:high;}
}
template<class T> Stats host_bounds(const T* v,std::size_t n) {
    T low,high;lane_bounds(v,n,low,high);
    if constexpr(std::is_floating_point_v<T>) if(low==0 || high==0) {
        const T* zero=std::find(v,v+n,T(0));
        if(low==0)low=*zero;
        if(high==0)high=*zero;
    }
    return Stats{double(low),double(high),0,0};
}
template<bool Moments=true,class T> VCR_HD Stats stats(const T* v,std::size_t n) {
#if !defined(__CUDACC__)
    if constexpr(!Moments) if(n) return host_bounds(v,n);
#endif
    Stats s{double(v[0]),double(v[0]),0,0};double m2=0;
    for(std::size_t i=0;i<n;++i) {
        double x=v[i];s.low=x<s.low?x:s.low;s.high=x>s.high?x:s.high;
        if constexpr(Moments){double d=x-s.mean;s.mean+=d/double(i+1);m2+=d*(x-s.mean);}
    }
    if constexpr(Moments)s.stddev=sqrt(m2>0?m2/double(n):0);
    return s;
}
VCR_HD inline int bin(double x,double lo,double hi,int bins) {
    return hi>lo?int(clamp((x-lo)/(hi-lo)*bins,0,bins-1)):0;
}
// bin() of every value. A reciprocal product is within a few ulps (relative ~1e-15) of bin()'s
// quotient fl(fl(fl(x-lo)/span)*bins), so both floor to the same bin unless q-t and q+t
// (t = 1e-9*(q+1)) truncate differently; those values (rare: near bin edges, including the
// maximum) are recomputed with bin() itself. Quotients are nonnegative, so the minimum always
// truncates to 0 on both sides. The result equals bin() for every input.
template<class T> VCR_TARGET_CLONES void bin_indices(const T* data,std::size_t n,double lo,double hi,int bins,std::uint16_t* out) {
    if(!(hi>lo)){std::fill_n(out,n,std::uint16_t(0));return;}
    const double rate=double(bins)/(hi-lo);const int top=bins-1;
    int uncertain=0;
    for(std::size_t i=0;i<n;++i){
        const double q=(double(data[i])-lo)*rate,t=1e-9*(q+1);
        const int f=int(q),near=int(int(q-t)!=int(q+t));
        out[i]=std::uint16_t(near?0xffff:(f<top?f:top));uncertain|=near;
    }
    if(uncertain)for(std::size_t i=0;i<n;++i)if(out[i]==0xffff)out[i]=std::uint16_t(bin(double(data[i]),lo,hi,bins));
}
// Integer counts of bin indices; four interleaved tables avoid store-to-load stalls on runs.
inline void count_bins(const std::uint16_t* index,std::size_t n,int bins,double* h) {
    std::vector<std::uint32_t> counts(4*std::size_t(bins),0u);
    std::size_t i=0;
    for(;i+4<=n;i+=4){++counts[index[i]];++counts[std::size_t(bins)+index[i+1]];++counts[2*std::size_t(bins)+index[i+2]];++counts[3*std::size_t(bins)+index[i+3]];}
    for(;i<n;++i)++counts[index[i]];
    for(int b=0;b<bins;++b)h[b]=double(std::uint64_t(counts[std::size_t(b)])+counts[std::size_t(bins+b)]+counts[2*std::size_t(bins)+std::size_t(b)]+counts[3*std::size_t(bins)+std::size_t(b)]);
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
// cdf() of `tiles` consecutive histograms. Each tile's sums keep cdf()'s order (they are serial
// dependency chains); interleaving eight tiles runs eight chains at once. A non-exceeding bin
// adds +0.0 to a nonnegative excess, which leaves it unchanged.
inline void cdf_many(double* h,std::size_t tiles,int bins,double clip_limit) {
    constexpr std::size_t L=8;std::size_t t=0;const std::size_t n=std::size_t(bins);
    for(;t+L<=tiles;t+=L) {
        double* base=h+t*n;double total[L]={},limit[L],excess[L]={},share[L],sum[L]={};
        for(std::size_t b=0;b<n;++b)for(std::size_t j=0;j<L;++j)total[j]+=base[j*n+b];
        if(clip_limit>0) {
            for(std::size_t j=0;j<L;++j)limit[j]=fmax(1,clip_limit*total[j]/bins);
            for(std::size_t b=0;b<n;++b)for(std::size_t j=0;j<L;++j){double& v=base[j*n+b];const bool over=v>limit[j];excess[j]+=over?v-limit[j]:0.;v=over?limit[j]:v;}
            for(std::size_t j=0;j<L;++j)share[j]=excess[j]/bins;
            for(std::size_t b=0;b<n;++b)for(std::size_t j=0;j<L;++j)base[j*n+b]+=share[j];
        }
        for(std::size_t b=0;b<n;++b)for(std::size_t j=0;j<L;++j){sum[j]+=base[j*n+b];base[j*n+b]=total[j]?sum[j]/total[j]:0;}
    }
    for(;t<tiles;++t)cdf(h+t*n,bins,clip_limit);
}

} // namespace vcr::transforms::volume_detail

namespace vcr::transforms::volume_detail {
struct HistogramGrid {
    std::size_t tiles[3],count=1;int bins;double clip_limit;
    template<int D> HistogramGrid(const GridND<D>& g,const Shape& requested,int bins,double clip):bins(bins),clip_limit(clip) {
        const auto tile=spatial_sizes(requested,D,16);
        require(bins>=2 && bins<=4096,"bins must be [2,4096]");
        for(int a=0;a<D;++a){require(tile[a]>0 && tile[a]<=65536,"tile sizes must be [1,65536]");tiles[a]=(g.n[a]+tile[a]-1)/tile[a];count*=tiles[a];}
    }
};
template<class T,int D> VCR_HD inline void histogram_tile(const T* v,const GridND<D>& g,const Stats& s,const HistogramGrid& layout,std::size_t tile,double* table) {
    double* h=table+tile*layout.bins;for(int b=0;b<layout.bins;++b)h[b]=0;
    std::size_t pos[3],lo[3],hi[3];
    if constexpr(D==3){pos[0]=tile/(layout.tiles[1]*layout.tiles[2]);pos[1]=tile/layout.tiles[2]%layout.tiles[1];pos[2]=tile%layout.tiles[2];}
    else {pos[0]=tile/layout.tiles[1];pos[1]=tile%layout.tiles[1];}
    for(int a=0;a<D;++a){lo[a]=pos[a]*g.n[a]/layout.tiles[a];hi[a]=(pos[a]+1)*g.n[a]/layout.tiles[a];}
    for(std::size_t z=D==3?lo[0]:0;z<(D==3?hi[0]:1);++z)for(std::size_t y=lo[D-2];y<hi[D-2];++y)for(std::size_t x=lo[D-1];x<hi[D-1];++x)
        h[bin(v[(D==3?z*g.stride[0]:0)+y*g.stride[D-2]+x],s.low,s.high,layout.bins)]+=1;
    cdf(h,layout.bins,layout.clip_limit);
}
template<int D> VCR_HD inline double interpolate_histograms(const GridND<D>& g,const HistogramGrid& layout,std::size_t i,int b,const double* table) {
    std::size_t lo[3],hi[3];double f[3];
    for(int a=0;a<D;++a){double q=clamp((double(g.coord(i,a))+.5)*layout.tiles[a]/g.n[a]-.5,0,double(layout.tiles[a]-1));lo[a]=std::size_t(floor(q));hi[a]=lo[a]+1<layout.tiles[a]?lo[a]+1:lo[a];f[a]=q-lo[a];}
    double result=0;
    for(int corner=0;corner<(1<<D);++corner){std::size_t t=0;double w=1;for(int a=0;a<D;++a){bool bit=(corner>>a)&1;t=t*layout.tiles[a]+(bit?hi[a]:lo[a]);w*=bit?f[a]:1-f[a];}result+=w*table[t*layout.bins+b];}
    return result;
}
} // namespace vcr::transforms::volume_detail
