#pragma once

#include <array>
#include <chrono>
#include <cstdint>

namespace vc::fiber_tracer::detail {

enum class SamplingStage : unsigned {
    OutputControl, Coordinates, Address, Gather, DirectionTensor, DirectionAxis,
    Presence, PredictionOutput, NormalValidity, NormalTensor, NormalAxis,
    WidthInterpolation, WidthConversion, Count
};

// Diagnostic only. Counts/times accumulate across calls; one owner per thread.
// Exclusive nested intervals partition the timed root exactly. Clock overhead
// remains in these intervals: do not interpret them as uninstrumented costs.
struct FiberSamplingProfile {
    static constexpr std::array<const char*,unsigned(SamplingStage::Count)> names{
        "output_loop_control", "coordinates", "chunk_address", "corner_gather",
        "direction_tensor", "direction_axis_normalize", "presence_interpolation",
        "prediction_alignment_output", "normal_validity_interpolation", "normal_tensor",
        "normal_axis_normalize", "width_masked_interpolation", "width_conversion"};
    std::array<int64_t,names.size()> nanoseconds{};
    std::array<uint64_t,names.size()> entries{};
    int64_t elapsedNanoseconds=0;
    uint64_t calls=0;
};

template<bool Measured> class FiberSamplingTimer;

template<> class FiberSamplingTimer<false> {
public:
    explicit FiberSamplingTimer(FiberSamplingProfile* = nullptr) {}
    struct Scope {Scope(FiberSamplingTimer&,SamplingStage) {}};
};

template<> class FiberSamplingTimer<true> {
    using Clock=std::chrono::steady_clock;
    FiberSamplingProfile& profile_;
    int64_t start_,last_;
    SamplingStage stage_=SamplingStage::OutputControl;
    static int64_t now() {return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
    void account(int64_t now) {
        profile_.nanoseconds[unsigned(stage_)]+=now-last_;
        last_=now;
    }
public:
    explicit FiberSamplingTimer(FiberSamplingProfile* profile):
        profile_(*profile),start_(now()),last_(start_) {
        ++profile_.calls;++profile_.entries[unsigned(stage_)];
    }
    ~FiberSamplingTimer() {
        const auto end=now();account(end);
        profile_.elapsedNanoseconds+=end-start_;
    }
    class Scope {
        FiberSamplingTimer& timer_;
        SamplingStage previous_;
    public:
        Scope(FiberSamplingTimer& timer,SamplingStage stage):timer_(timer),previous_(timer.stage_) {
            timer_.account(now());timer_.stage_=stage;
            ++timer_.profile_.entries[unsigned(stage)];
        }
        ~Scope() {timer_.account(now());timer_.stage_=previous_;}
        Scope(const Scope&)=delete;
        Scope& operator=(const Scope&)=delete;
    };
};
}
