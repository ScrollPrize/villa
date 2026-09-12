#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <stdexcept>
#include <vector>

// Captures copies of the final per-pixel rays, never changes renderer inputs.
// Dependency-free so tests exercise the same serializer as vc_render_tifxyz.
namespace vc::render {
class SamplingGridCapture {
public:
    using Pixel = std::array<float, 6>; // base XYZ, direction XYZ
    static constexpr std::size_t maxPixels = 512 * 512;
    static constexpr std::size_t maxOffsets = 65536;

    SamplingGridCapture(int height, int width, int level, int cropX, int cropY)
        : height_(height), width_(width), level_(level), cropX_(cropX), cropY_(cropY)
    {
        if (height < 2 || width < 2 || level < 0 || cropX < 0 || cropY < 0 ||
            std::size_t(height) > maxPixels / std::size_t(width))
            throw std::invalid_argument("sampling capture needs a 2D crop of at most 262144 pixels");
        pixels_.resize(std::size_t(height) * std::size_t(width));
        rows_.resize(std::size_t(height), false);
    }

    template<class PixelAt>
    void appendBand(int y0, int height, int width, const std::vector<float>& offsets,
                    PixelAt pixelAt)
    {
        if (y0 < 0 || height <= 0 || y0 > height_ || height > height_ - y0 || width != width_)
            throw std::invalid_argument("sampling capture band is outside the crop");
        if (offsets.empty() || offsets.size() > maxOffsets)
            throw std::invalid_argument("sampling capture needs a bounded nonempty offset list");
        for (float d : offsets)
            if (!std::isfinite(d)) throw std::invalid_argument("nonfinite sampling offset");
        if (!offsets_.empty() && offsets_ != offsets)
            throw std::invalid_argument("sampling offsets changed between bands");
        for (int r = 0; r < height; ++r)
            if (rows_[std::size_t(y0 + r)])
                throw std::invalid_argument("sampling capture contains duplicate rows");
        offsets_ = offsets;
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c)
                pixels_[std::size_t(y0 + r) * std::size_t(width_) + std::size_t(c)] = pixelAt(r, c);
            rows_[std::size_t(y0 + r)] = true;
        }
    }

    void writeJson(std::ostream& out) const
    {
        for (bool present : rows_)
            if (!present) throw std::runtime_error("incomplete sampling capture; not a full crop");
        // max_digits10 permits bit-exact finite float32 roundtrips. JSON null
        // preserves invalid pixels as invalid, never as an invented zero ray.
        out.imbue(std::locale::classic());
        out << std::defaultfloat << std::setprecision(std::numeric_limits<float>::max_digits10);
        out << "{\n\"schema\":\"vc-sampling-grid-v1\",\n"
               "\"stage\":\"final-base-and-dirs-before-readMultiSlice\",\n"
               "\"pixel_order\":\"row-major-baseXYZ-directionXYZ\",\n"
               "\"units\":\"level-index-voxel\",\n"
               "\"between_pixel_interpolation\":\"NOT_SPECIFIED_BY_RENDERER\",\n"
               "\"shape_hw\":[" << height_ << ',' << width_ << "],\n"
               "\"level\":" << level_ << ",\n\"crop_xy\":[" << cropX_ << ',' << cropY_ << "],\n"
               "\"offsets\":[";
        for (std::size_t i = 0; i < offsets_.size(); ++i) {
            if (i) out << ',';
            writeFinite(out, offsets_[i]);
        }
        out << "],\n\"pixels\":[\n";
        std::size_t invalid = 0;
        for (std::size_t i = 0; i < pixels_.size(); ++i) {
            if (i) out << ",\n";
            out << '[';
            bool valid = true;
            for (std::size_t j = 0; j < 6; ++j) {
                if (j) out << ',';
                const float x = pixels_[i][j];
                if (std::isfinite(x)) writeFinite(out, x);
                else { out << "null"; valid = false; }
            }
            if (!valid) ++invalid;
            out << ']';
        }
        out << "\n],\n\"invalid_pixels\":" << invalid << "\n}\n";
        if (!out) throw std::runtime_error("sampling capture write failed");
    }

private:
    static void writeFinite(std::ostream& out, float x)
    {
        // JSON decoders may parse "-0" as integer zero and lose its sign.
        if (x == 0.0f && std::signbit(x)) out << "-0.0";
        else out << x;
    }

    int height_, width_, level_, cropX_, cropY_;
    std::vector<Pixel> pixels_;
    std::vector<bool> rows_;
    std::vector<float> offsets_;
};
} // namespace vc::render
