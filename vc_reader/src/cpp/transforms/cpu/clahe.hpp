#pragma once
#include "../ops/clahe.hpp"
#include "../shared/voxel_transform.hpp"
namespace vcr::transforms {
inline ChannelPlan channel_parameters(const CLAHE& t,const Array& a,Random& r) {
    double clip=draw(t.clip_limit,r);require(clip>=0,"clip_limit must be nonnegative");
    volume_detail::with_grid(a.spatial(),[&](const auto& grid){volume_detail::HistogramGrid layout(grid,t.tile_shape,t.bins,clip);});ChannelPlan p;p.values={clip};return p;
}
struct CLAHEKernel {
    volume_detail::HistogramGrid layout;
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {
        using namespace volume_detail;double span=c.input.high-c.input.low;if(span==0)return v[i];
        return c.input.low+span*interpolate_histograms(c.grid,layout,i,bin(v[i],c.input.low,c.input.high,layout.bins),c.table);
    }
};
// CLAHEKernel row by row: interpolate_histograms' per-axis tile positions and weights depend
// only on that axis' coordinate, so they are tabulated once with the same expressions and
// combined in the same corner and multiplication order.
// From each voxel's bin (index), used for both its tile histogram and its mapping (histogram_tile's
// counts), into output; input holds the channel's bounds, with nonzero span.
template<class T> inline void clahe_map(const CLAHE& t,const Grid& g,const ChannelPlan& p,const volume_detail::Stats& input,
                                        const std::uint16_t* index,T* output) {
    using namespace volume_detail;
    with_grid(g.shape,[&](const auto& grid){
        constexpr int D=std::decay_t<decltype(grid)>::dim;
        HistogramGrid layout(grid,t.tile_shape,t.bins,p.values[0]);
        const double span=input.high-input.low;
        std::vector<double> table(layout.count*std::size_t(layout.bins),0.);
        std::vector<std::size_t> lo[3],hi[3];std::vector<double> f[3];
        for(int a=0;a<D;++a) {
            lo[a].resize(grid.n[a]);hi[a].resize(grid.n[a]);f[a].resize(grid.n[a]);
            for(std::size_t c=0;c<grid.n[a];++c) {
                double q=clamp((double(c)+.5)*layout.tiles[a]/grid.n[a]-.5,0,double(layout.tiles[a]-1));
                lo[a][c]=std::size_t(floor(q));hi[a][c]=lo[a][c]+1<layout.tiles[a]?lo[a][c]+1:lo[a][c];f[a][c]=q-lo[a][c];
            }
        }
        const std::size_t bins=std::size_t(layout.bins);
        const std::size_t nz=D==3?grid.n[0]:1,ny=grid.n[D-2],nx=grid.n[D-1];
        {
            // Tile of each coordinate along each axis: pos*n/tiles <= c < (pos+1)*n/tiles.
            std::vector<std::size_t> tile_of[3];
            for(int a=0;a<D;++a){tile_of[a].resize(grid.n[a]);for(std::size_t pos=0;pos<layout.tiles[a];++pos)
                for(std::size_t c=pos*grid.n[a]/layout.tiles[a];c<(pos+1)*grid.n[a]/layout.tiles[a];++c)tile_of[a][c]=pos;}
            std::vector<std::uint32_t> counts(table.size(),0u);
            for(std::size_t z=0;z<nz;++z)for(std::size_t y=0;y<ny;++y) {
                const std::size_t row_tile=D==3?(tile_of[0][z]*layout.tiles[1]+tile_of[1][y])*layout.tiles[2]:tile_of[0][y]*layout.tiles[1];
                const auto* row=index+(z*ny+y)*nx;
                for(std::size_t x=0;x<nx;++x)++counts[(row_tile+tile_of[D-1][x])*bins+row[x]];
            }
            for(std::size_t i=0;i<table.size();++i)table[i]=double(counts[i]);
            cdf_many(table.data(),layout.count,layout.bins,layout.clip_limit);
        }
        // Corner c takes axis a's upper tile when bit a is set; its weight is the product of the
        // axes' factors in axis order starting from 1 (so (wz*wy)*wx in 3D, wy*wx in 2D) and
        // corners are summed in order from 0, as interpolate_histograms does.
        const std::size_t tx=layout.tiles[D-1],ty=layout.tiles[D-2];
        std::vector<double> wx0(nx),wx1(nx);std::vector<std::size_t> cx0(nx),cx1(nx);
        for(std::size_t x=0;x<nx;++x){wx0[x]=1-f[D-1][x];wx1[x]=f[D-1][x];cx0[x]=lo[D-1][x]*bins;cx1[x]=hi[D-1][x]*bins;}
        for(std::size_t z=0;z<nz;++z)for(std::size_t y=0;y<ny;++y) {
            const std::size_t row=(z*ny+y)*nx;
            const std::uint16_t* __restrict index_row=index+row;
            T* __restrict out=output+row;
            if constexpr(D==2) {
                const double wy[2]={1-f[0][y],f[0][y]};
                const double* r0=table.data()+lo[0][y]*tx*bins;const double* r1=table.data()+hi[0][y]*tx*bins;
                for(std::size_t x=0;x<nx;++x) {
                    const std::size_t b=index_row[x];
                    double result=0;
                    result+=(wy[0]*wx0[x])*r0[cx0[x]+b];
                    result+=(wy[1]*wx0[x])*r1[cx0[x]+b];
                    result+=(wy[0]*wx1[x])*r0[cx1[x]+b];
                    result+=(wy[1]*wx1[x])*r1[cx1[x]+b];
                    out[x]=T(input.low+span*result);
                }
            } else {
                const double wz[2]={1-f[0][z],f[0][z]},wy[2]={1-f[1][y],f[1][y]};
                const std::size_t tz[2]={lo[0][z],hi[0][z]},tyy[2]={lo[1][y],hi[1][y]};
                const double* r[4];double wzy[4];
                for(int c=0;c<4;++c){const int bz=c&1,by=(c>>1)&1;r[c]=table.data()+(tz[bz]*ty+tyy[by])*tx*bins;wzy[c]=wz[bz]*wy[by];}
                for(std::size_t x=0;x<nx;++x) {
                    const std::size_t b=index_row[x];
                    double result=0;
                    for(int c=0;c<4;++c)result+=(wzy[c]*wx0[x])*r[c][cx0[x]+b];
                    for(int c=0;c<4;++c)result+=(wzy[c]*wx1[x])*r[c][cx1[x]+b];
                    out[x]=T(input.low+span*result);
                }
            }
        }
    });
    bool finite=true;for(std::size_t i=0;i<g.size;++i)finite&=std::isfinite(output[i]);
    require(finite,"transform produced nonfinite values");
}
template<class T> inline void channel_apply(const CLAHE& t,std::vector<T>& v,const Grid& g,const ChannelPlan& p) {
    using namespace volume_detail;
    const auto input=stats<false>(v.data(),v.size());
    if(input.high-input.low==0)return;
    std::vector<std::uint16_t> index(v.size());bin_indices(v.data(),v.size(),input.low,input.high,t.bins,index.data());
    std::vector<T> output(v.size());
    clahe_map(t,g,p,input,index.data(),output.data());
    v.swap(output);
}
// channel_apply on a uint8 channel's float32 values: bounds from value counts, bins per value.
inline void narrow_apply(const CLAHE& t,const ChannelPlan& p,const Grid& g,const std::uint8_t* in,float* out) {
    using namespace volume_detail;
    const NarrowValues values(in,g.size);
    const auto input=values.bounds();
    if(input.high-input.low==0){widen_values(in,out,g.size);return;}
    const auto bin_of=values.bins(t.bins);
    std::vector<std::uint16_t> index(g.size);lookup(in,index.data(),g.size,bin_of.data());
    clahe_map(t,g,p,input,index.data(),out);
}
inline void apply(const CLAHE& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::transforms
