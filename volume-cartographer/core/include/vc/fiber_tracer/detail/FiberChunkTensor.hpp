#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/core/types/Array3D.hpp"

namespace vc::fiber_tracer::detail {

// Owned native-lattice channels. No cache handles, locks or I/O after loading.
// Presence-only construction requires a synchronous eligible seedGrid before
// companion sampling/packing; those methods throw otherwise. Do not call
// seedGrid concurrently with sampling: only completed tensors are read-shared.
// Scalar and batch sampling deliberately retain their different historical
// double/float coordinate and interpolation arithmetic.
class FiberChunkTensor final : public FiberPredictionSource, public vc::lasagna::NormalSampler {
public:
    FiberChunkTensor(const FiberPredictionField&,const vc::lasagna::LasagnaNormalSampler&,
        const cv::Vec3d& loTrace,const cv::Vec3d& hiTrace,bool presenceOnly=false,
        bool channelHints=true,std::size_t workerCacheBytes=0);
    bool supportsConcurrentSampling() const noexcept override {return true;}
    FiberPredictionSample sample(const cv::Vec3d&,const cv::Vec3d&) const override;
    void sampleBatch(const std::vector<cv::Vec3d>&,const std::vector<cv::Vec3d>&,int,
        std::vector<FiberPredictionSample>&) const override;
    void sampleWidths(const std::vector<cv::Vec3f>&,std::vector<float>&) const override;
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override;
    // Same presence/valid-direction decision as sampleBatch, omitting direction
    // work only for samples below the NMS threshold (cannot suppress a seed).
    float seedPresence(const cv::Vec3d&,double threshold) const;
    // Loads companions synchronously before returning if any presence-eligible
    // points exist. All direction validity checks precede caller-owned NMS.
    nlohmann::json seedGrid(const cv::Vec3d& loBase,double traceScale,double gridBase,
        int side,double threshold,std::vector<float>& output);
    void packGpu(std::vector<unsigned char>&,std::vector<float>&) const;
    const std::array<int,3>& origin() const {return origin_;} // XYZ storage indices
    const std::array<int,3>& shape() const {return shape_;}
    double spacing() const {return spacing_;}
    bool masksNormals() const {return maskNormals_;}
private:
    bool cube(const cv::Vec3d& lattice,bool scalar,std::array<size_t,8>&,cv::Vec3d&) const;
    std::array<uint8_t,8> bytes(size_t,const std::array<size_t,8>&) const;
    FiberPredictionSampleOption batchOption(const cv::Vec3d&,const cv::Vec3d&,double threshold) const;
    float width(const cv::Vec3f&) const;
    void loadCompanions();
    void requireCompanions() const;
    std::array<vc::lasagna::LasagnaChannelBinding,7> pendingBindings_;
    bool companionsLoaded_=false;
    std::array<Array3D<uint8_t>,6> channels_; // presence,nx,ny,grad_mag,sheet_nx,sheet_ny
    Array3D<uint8_t> byteWidth_;
    Array3D<float> floatWidth_;
    bool widthIsFloat_=false,maskNormals_=false;
    double spacing_=0,widthMultiplier_=1;
    std::array<int,3> origin_{},shape_{},volumeShape_{};
};
}
