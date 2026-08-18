#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vc::fiber_tracer::detail
{

class FiberAnchorPeakGridLayout {
public:
    static constexpr int maximumExtent = 128;

    explicit FiberAnchorPeakGridLayout(int extent)
        : extent_{extent}
    {
        if (extent < 0 || extent > maximumExtent) {
            throw std::invalid_argument(
                "fiber anchor peak grid extent is outside [0, 128]");
        }
        const auto unsignedExtent = static_cast<size_t>(extent);
        if (unsignedExtent > (std::numeric_limits<size_t>::max() - 1) / 2)
            throw std::overflow_error("fiber anchor peak grid side overflows");
        side_ = 2 * unsignedExtent + 1;
        if (side_ > std::numeric_limits<size_t>::max() / side_)
            throw std::overflow_error("fiber anchor peak grid area overflows");
        size_ = side_ * side_;
    }

    [[nodiscard]] int extent() const noexcept { return extent_; }
    [[nodiscard]] size_t side() const noexcept { return side_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }

    [[nodiscard]] bool contains(int first, int second) const noexcept
    {
        return first >= -extent_ && first <= extent_ &&
            second >= -extent_ && second <= extent_;
    }

    [[nodiscard]] size_t indexUnchecked(int first, int second) const noexcept
    {
        const auto row = static_cast<size_t>(first + extent_);
        const auto column = static_cast<size_t>(second + extent_);
        return row * side_ + column;
    }

    [[nodiscard]] size_t index(int first, int second) const
    {
        if (!contains(first, second))
            throw std::out_of_range("fiber anchor peak grid index is outside its extent");
        return indexUnchecked(first, second);
    }

private:
    int extent_ = 0;
    size_t side_ = 0;
    size_t size_ = 0;
};

class FiberAnchorPeakResponseCache {
public:
    explicit FiberAnchorPeakResponseCache(FiberAnchorPeakGridLayout layout)
        : layout_{std::move(layout)}
        , values_(layout_.size())
        , computed_(layout_.size(), 0)
    {
    }

    [[nodiscard]] const FiberAnchorPeakGridLayout& layout() const noexcept
    {
        return layout_;
    }

    template <typename Compute>
    double getOrCompute(int first, int second, Compute&& compute)
    {
        const size_t slot = layout_.index(first, second);
        if (!computed_[slot]) {
            values_[slot] = std::forward<Compute>(compute)();
            computed_[slot] = 1;
        }
        return values_[slot];
    }

private:
    FiberAnchorPeakGridLayout layout_;
    std::vector<double> values_;
    std::vector<unsigned char> computed_;
};

} // namespace vc::fiber_tracer::detail
