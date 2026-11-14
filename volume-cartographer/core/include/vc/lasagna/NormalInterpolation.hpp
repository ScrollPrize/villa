#pragma once
#include "vc/lasagna/ChannelSampler.hpp"
#include <vc_reader/sampling.hpp>
#include <vc_reader/interpolation.hpp>

namespace vc::lasagna {
enum class NormalInterpolationStatus { Valid, Missing, ZeroDensity, Degenerate };
struct NormalInterpolationResult {
    cv::Vec3f normal{0.f,0.f,0.f};
    NormalInterpolationStatus status=NormalInterpolationStatus::Missing;
};

// Domain math stays shared with the corner-buffer sampler; vc_reader owns
// planning, chunk access, boundary gathers and invoking this policy.
template<class Store> struct NormalInterpolationPolicy {
    static constexpr bool custom=true,clamp_boundary=true;
    static constexpr std::size_t components=3;
    using real_type=float;
    std::array<std::int64_t,3> channels;
    bool mask,floatEncoded;
    Store output;
    std::int64_t offset(std::size_t c,std::size_t a) const {return a==0?channels[c]:0;}
    template<class Cell> void prepare_cell(Cell& cell) const {
        cell.k=3;cell.axes[0]=3;cell.axes[1]=2;cell.axes[2]=1;
    }
    template<class Cell,class Read> NormalInterpolationResult evaluate(const Cell& cell,Read&& read) const {
        std::array<uint8_t,8> density,nx,ny;
        for(std::size_t k=0;k<8;++k) {
            const auto byte=[&](auto value){return floatEncoded?encodedLasagnaByteValue(value):uint8_t(value);};
            density[k]=byte(read(k,0));nx[k]=byte(read(k,1));ny[k]=byte(read(k,2));
        }
        const auto weights=vcr::sampling::cornerWeights(std::array<float,3>{cell.frac[3],cell.frac[2],cell.frac[1]});
        if(!(vcr::sampling::weightedCorners(density,weights.data())>0.f))
            return {{0.f,0.f,0.f},NormalInterpolationStatus::ZeroDensity};
        const auto normal=interpolateLasagnaCompactAxisCorners(nx,ny,
            mask?validLasagnaCornerWeights(weights,density):weights);
        return {normal,normal.dot(normal)>1.e-12f?NormalInterpolationStatus::Valid:NormalInterpolationStatus::Degenerate};
    }
    void store(std::size_t p,const NormalInterpolationResult& value) const {output(p,value);}
    void outside(std::size_t p) const {output(p,NormalInterpolationResult{});}
};

// Float32 base XYZ -> source CZYX, using exactly the existing Vec division.
struct NormalInterpolationCoordinates {
    using value_type=float;
    static constexpr std::size_t kDimensions=4;
    const std::vector<cv::Vec3f>& points;
    float spacing;
    std::int64_t firstChannel;
    std::int64_t base(std::size_t a) const {return a==0?firstChannel:0;}
    const float* operator()(std::size_t p,float* out) const {
        const auto q=points[p]/spacing;
        out[0]=0.f;out[1]=q[2];out[2]=q[1];out[3]=q[0];return out;
    }
    void prefetch(std::size_t) const {}
};

template<class Store>
void sampleLasagnaInterpolatedNormals(const std::array<LasagnaChannelBinding,3>& channels,
    const std::vector<cv::Vec3f>& points,bool mask,Store output)
{
    if(!points.empty())vcr::diagnostics::recordCoordinates(points.front().val,points.size(),float(channels[0].spacing));
    for(const auto& c:channels)
        if(!c.packed||c.cuda||c.array!=channels[0].array||!sameLasagnaSamplingGrid(c,channels[0]))
            throw std::invalid_argument("fused normal policy requires CPU channels of one packed sampling grid");
    const auto first=std::min({channels[0].channelIndex,channels[1].channelIndex,channels[2].channelIndex});
    NormalInterpolationPolicy<Store> policy{{int64_t(channels[0].channelIndex-first),
        int64_t(channels[1].channelIndex-first),int64_t(channels[2].channelIndex-first)},mask,channels[0].float32,std::move(output)};
    const NormalInterpolationCoordinates coordinates{points,float(channels[0].spacing),int64_t(first)};
    channels[0].array->sample_interpolated_coordinates(coordinates,points.size(),
        channels[0].float32?vcr::ElementType{'f',4,false}:vcr::ElementType{'u',1,false},policy);
}

struct InterpolatedWindingSample {
    float encodedDensity=0.f;
    cv::Vec3f normal{0.f,0.f,0.f};
    bool valid=false;
};

// Runtime rank accommodates both individual ZYX arrays and packed CZYX
// channels, without changing float32 geometry or boundary replication.
struct ChannelInterpolationCoordinates {
    using value_type=float;
    const std::vector<cv::Vec3f>& points;
    const LasagnaChannelBinding& binding;
    float spacing=0.f;
    std::int64_t base(std::size_t a) const {return binding.packed&&a==0?int64_t(binding.channelIndex):0;}
    const float* operator()(std::size_t p,float* out) const {
        const auto q=points[p]/(spacing>0.f?spacing:float(binding.spacing));
        const auto first=binding.packed?1:0;
        if(first)out[0]=0.f;
        out[first]=q[2];out[first+1]=q[1];out[first+2]=q[0];return out;
    }
    void prefetch(std::size_t) const {}
};

template<class Store> struct DensityInterpolationPolicy {
    static constexpr bool custom=true,clamp_boundary=true;
    static constexpr std::size_t components=1;
    using real_type=float;
    bool packed,floatEncoded;
    Store output;
    std::int64_t offset(std::size_t,std::size_t) const {return 0;}
    template<class Cell> void prepare_cell(Cell& c) const {
        const auto end=packed?3:2;c.k=3;
        c.axes[0]=end;c.axes[1]=end-1;c.axes[2]=end-2;
    }
    template<class Cell,class Read> float evaluate(const Cell& c,Read&& read) const {
        std::array<uint8_t,8> corners;
        for(size_t k=0;k<8;++k)corners[k]=floatEncoded?encodedLasagnaByteValue(read(k,0)):uint8_t(read(k,0));
        const auto weights=vcr::sampling::cornerWeights(std::array<float,3>{c.frac[c.axes[0]],c.frac[c.axes[1]],c.frac[c.axes[2]]});
        return vcr::sampling::weightedCorners(corners,weights.data());
    }
    void store(size_t i,float value) const {output(i,value);}
    void outside(size_t i) const {output(i,std::numeric_limits<float>::quiet_NaN());}
};

template<class Store> struct AxisInterpolationPolicy {
    static constexpr bool custom=true,clamp_boundary=true;
    static constexpr size_t components=2;
    using real_type=float;
    std::array<bool,2> packed,floatEncoded;
    std::array<int64_t,2> channelOffsets;
    Store output;
    int64_t offset(size_t c,size_t a) const {return packed[0]&&a==0?channelOffsets[c]:0;}
    template<class Cell> void prepare_cell(Cell& c,size_t source=0) const {
        const auto end=packed[source]?3:2;c.k=3;
        c.axes[0]=end;c.axes[1]=end-1;c.axes[2]=end-2;
    }
    template<class Cell,class Read> cv::Vec3f evaluate(const Cell& c,Read&& read) const {
        std::array<uint8_t,8> nx,ny;
        for(size_t k=0;k<8;++k) {
            nx[k]=floatEncoded[0]?encodedLasagnaByteValue(read(k,0)):uint8_t(read(k,0));
            ny[k]=floatEncoded[1]?encodedLasagnaByteValue(read(k,1)):uint8_t(read(k,1));
        }
        const auto weights=vcr::sampling::cornerWeights(std::array<float,3>{c.frac[c.axes[0]],c.frac[c.axes[1]],c.frac[c.axes[2]]});
        return interpolateLasagnaCompactAxisCorners(nx,ny,weights);
    }
    void store(size_t i,const cv::Vec3f& v) const {output(i,v);}
    void outside(size_t i) const {output(i,cv::Vec3f{0.f,0.f,0.f});}
};

// Returns interpolated field values, never corner arrays. Density may have a
// different lattice from normals. Independent normal arrays retain only a
// bounded reader-internal tile of one encoded component for tensor coupling.
inline void sampleLasagnaInterpolatedWindingSamples(const std::array<LasagnaChannelBinding,3>& channels,
    const std::vector<cv::Vec3f>& points,std::vector<InterpolatedWindingSample>& samples)
{
    samples.assign(points.size(),{});
    const auto type=[](const auto& c){return c.float32?vcr::ElementType{'f',4,false}:vcr::ElementType{'u',1,false};};
    const auto& density=channels[0];const auto& nx=channels[1];const auto& ny=channels[2];
    if(density.cuda||nx.cuda||ny.cuda||!sameLasagnaSamplingGrid(nx,ny))
        throw std::invalid_argument("CPU winding samples require compatible normal grids");
    density.array->sample_interpolated_coordinates(ChannelInterpolationCoordinates{points,density},points.size(),type(density),
        DensityInterpolationPolicy{density.packed,density.float32,[&](size_t i,float v){
            samples[i].encodedDensity=v;samples[i].valid=std::isfinite(v);
        }});
    auto output=[&](size_t i,const cv::Vec3f& v){samples[i].normal=v;};
    AxisInterpolationPolicy policy{{nx.packed,ny.packed},{nx.float32,ny.float32},{0,0},output};
    // Corner-batch sampling uses one common normal lattice (the density
    // lattice too when all grids agree within the manifest tolerance).
    const auto normalSpacing=float(sameLasagnaSamplingGrid(density,nx)?density.spacing:nx.spacing);
    if(nx.array==ny.array) {
        auto first=nx;
        first.channelIndex=std::min(nx.channelIndex,ny.channelIndex);
        policy.channelOffsets={int64_t(nx.channelIndex-first.channelIndex),int64_t(ny.channelIndex-first.channelIndex)};
        nx.array->sample_interpolated_coordinates(ChannelInterpolationCoordinates{points,first,normalSpacing},points.size(),type(nx),policy);
    } else {
        vcr::sample_interpolated_pair_coordinates(*nx.array,ChannelInterpolationCoordinates{points,nx,normalSpacing},type(nx),
            *ny.array,ChannelInterpolationCoordinates{points,ny,normalSpacing},type(ny),points.size(),policy);
    }
}
} // namespace vc::lasagna
