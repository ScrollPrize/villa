#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/fiber_tracer/detail/FiberSamplingProfile.hpp"

namespace vc::fiber_tracer::detail {

// Native-lattice channels, either owned arrays or retained decoded cache chunks.
// Neither backend performs cache lookups, locks or I/O while interpolating.
// Presence-only construction requires a synchronous eligible seedGrid before
// companion sampling/packing; those methods throw otherwise. Do not call
// seedGrid concurrently with sampling: only completed tensors are read-shared.
// Scalar and batch sampling deliberately retain their different historical
// double/float coordinate and interpolation arithmetic.
class FiberChunkTensor final : public FiberPredictionSource, public vc::lasagna::NormalSampler {
public:
    FiberChunkTensor(const FiberPredictionField&,const vc::lasagna::LasagnaNormalSampler&,
        const cv::Vec3d& loTrace,const cv::Vec3d& hiTrace,bool presenceOnly=false,
        bool channelHints=true,std::size_t workerCacheBytes=0,bool retainCacheChunks=false);
    bool supportsConcurrentSampling() const noexcept override {return true;}
    FiberPredictionSample sample(const cv::Vec3d&,const cv::Vec3d&) const override;
    void sampleBatch(const std::vector<cv::Vec3d>&,const std::vector<cv::Vec3d>&,int,
        std::vector<FiberPredictionSample>&) const override;
    void sampleWidths(const std::vector<cv::Vec3f>&,std::vector<float>&) const override;
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override;
    // Scalar-equivalent frontier: float tracer positions widened to double for
    // all channels, one cube/address per position. Does not use float batch math.
    void sampleCandidateChannels(const std::vector<cv::Vec3f>& points,
        const std::vector<cv::Vec3f>& directions,bool withNormals,bool withWidths,
        std::vector<FiberPredictionSample>& predictions,
        std::vector<vc::lasagna::NormalSample>& normals,std::vector<float>& widths,
        FiberSamplingProfile* diagnosticProfile=nullptr) const;
    // Same presence/valid-direction decision as sampleBatch, omitting direction
    // work only for samples below the NMS threshold (cannot suppress a seed).
    float seedPresence(const cv::Vec3d&,double threshold) const;
    // Loads companions synchronously before returning if any presence-eligible
    // points exist. All direction validity checks precede caller-owned NMS.
    nlohmann::json seedGrid(const cv::Vec3d& loBase,double traceScale,double gridBase,
        int side,double threshold,std::vector<float>& output);
    // seedGrid/packGpu use the owned backend; direct retained windows only sample.
    void packGpu(std::vector<unsigned char>&,std::vector<float>&) const;
    const std::array<int,3>& origin() const {return origin_;} // XYZ storage indices
    const std::array<int,3>& shape() const {return shape_;}
    double spacing() const {return spacing_;}
    bool masksNormals() const {return maskNormals_;}
    size_t ownedVoxelBytes() const;
    size_t retainedVoxelBytes() const {return retained_?retained_->bytes():0;}
    size_t pointerTableBytes() const;
private:
    struct Cube {
        std::array<int,3> lo,hi; // Absolute lattice XYZ, retained backend.
        std::array<size_t,8> indices; // Owned backend only.
        vc::render::PinnedChunkGrid::CornerAddress address;
    };
    bool cube(const cv::Vec3d& lattice,bool scalar,Cube&,cv::Vec3d&) const;
    std::array<uint8_t,8> bytes(size_t,Cube&) const;
    // Existing seedGrid owns its tensor and already computes flat indices.
    std::array<uint8_t,8> bytes(size_t,const std::array<size_t,8>&) const;
    std::array<double,8> values(size_t,Cube&) const;
    template<class T> std::array<T,8> retainedCorners(size_t c,Cube& cube) const {
        const auto& grid=*retainedChannels_[c];
        if(!grid.accepts(cube.address))cube.address=grid.prepareCorners(cube.lo,cube.hi);
        return grid.corners<T>(cube.address);
    }
    FiberPredictionSampleOption batchOption(const cv::Vec3d&,const cv::Vec3d&,double threshold) const;
    float width(const cv::Vec3f&) const;
    FiberPredictionSampleOption scalarOption(Cube&,const cv::Vec3d&,const cv::Vec3d&) const;
    vc::lasagna::NormalSample normal(Cube&,const cv::Vec3d&) const;
    float width(Cube&,const cv::Vec3d&) const;
    template<bool Measured,class T> std::array<T,8> sampleCorners(size_t,Cube&,FiberSamplingTimer<Measured>&) const;
    template<bool Measured> FiberPredictionSampleOption scalarOption(Cube&,const cv::Vec3d&,
        const cv::Vec3d&,FiberSamplingTimer<Measured>&) const;
    template<bool Measured> vc::lasagna::NormalSample normal(Cube&,const cv::Vec3d&,FiberSamplingTimer<Measured>&) const;
    template<bool Measured> float width(Cube&,const cv::Vec3d&,FiberSamplingTimer<Measured>&) const;
    template<bool Measured> void sampleCandidateChannelsImpl(const std::vector<cv::Vec3f>&,
        const std::vector<cv::Vec3f>&,bool,bool,std::vector<FiberPredictionSample>&,
        std::vector<vc::lasagna::NormalSample>&,std::vector<float>&,FiberSamplingProfile*) const;
    void loadCompanions();
    void requireCompanions() const;
    std::array<vc::lasagna::LasagnaChannelBinding,7> pendingBindings_;
    bool companionsLoaded_=false;
    std::shared_ptr<vc::render::PinnedChunkRegion> retained_;
    std::array<std::unique_ptr<vc::render::PinnedChunkGrid>,7> retainedChannels_;
    std::array<Array3D<uint8_t>,6> channels_; // presence,nx,ny,grad_mag,sheet_nx,sheet_ny
    Array3D<uint8_t> byteWidth_;
    Array3D<float> floatWidth_;
    bool widthIsFloat_=false,maskNormals_=false;
    double spacing_=0,widthMultiplier_=1;
    std::array<int,3> origin_{},shape_{},volumeShape_{};
};
}
