#pragma once
#include "vc/lasagna/ChannelSampler.hpp"
#include <vc_reader/sampling.hpp>

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
} // namespace vc::lasagna
