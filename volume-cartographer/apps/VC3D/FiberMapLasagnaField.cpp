#include "FiberMapLasagnaField.hpp"
#include "FiberMapContentDigest.hpp"

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "utils/zarr.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vc3d::fiber_map::bent
{

namespace
{

using vc3d::fiber_map::ContentDigest;
using vc3d::fiber_map::hashString;
using vc3d::fiber_map::seededDigest;

// The layout's content digest, printed as hex.
std::string digestHex(const std::string& text)
{
    ContentDigest digest = seededDigest(0xF1E1D);
    hashString(digest, text);
    char buffer[40];
    std::snprintf(buffer, sizeof buffer, "%016llx%016llx", static_cast<unsigned long long>(digest.a),
                  static_cast<unsigned long long>(digest.b));
    return buffer;
}

std::string readText(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

// Every field of the array metadata as opened, v2 and v3 alike (the
// library's own serializer emits v3 only and drops v2 compressor settings,
// byte order and filters).
void describeCodecs(std::ostringstream& out, const std::vector<utils::ZarrCodecConfig>& codecs)
{
    out << "[";
    for (const auto& codec : codecs) {
        out << "{" << codec.name << ":";
        if (codec.configuration) {
            out << utils::json_serialize(*codec.configuration);
        }
        out << "}";
    }
    out << "]";
}

std::string describeMetadata(const utils::ZarrMetadata& meta)
{
    std::ostringstream out;
    out.precision(17);
    out << "version=" << (meta.version == utils::ZarrVersion::v2 ? "2" : "3");
    out << ";shape=";
    for (const auto s : meta.shape) {
        out << s << ",";
    }
    out << ";chunks=";
    for (const auto c : meta.chunks) {
        out << c << ",";
    }
    out << ";dtype=" << utils::dtype_string_v2(meta.dtype);
    out << ";fill=";
    if (meta.fill_value) {
        out << *meta.fill_value;
    }
    out << ";byte_order=" << meta.byte_order;
    out << ";compressor=" << meta.compressor_id << ":" << meta.compression_level;
    out << ";q=";
    if (meta.codec_q) {
        out << *meta.codec_q;
    }
    out << ";separator=" << meta.dimension_separator;
    out << ";filters=[";
    for (const auto& f : meta.filters) {
        out << "{" << static_cast<int>(f.id) << ":" << utils::dtype_string_v2(f.dtype) << ":"
            << utils::dtype_string_v2(f.astype) << ":" << f.offset << ":" << f.scale << ":" << f.digits << "}";
    }
    out << "];codecs=";
    describeCodecs(out, meta.codecs);
    out << ";chunk_key_encoding=" << meta.chunk_key_encoding;
    out << ";shard=";
    if (meta.shard_config) {
        out << "{sub=";
        for (const auto c : meta.shard_config->sub_chunks) {
            out << c << ",";
        }
        out << ";index=";
        describeCodecs(out, meta.shard_config->index_codecs);
        out << ";codecs=";
        describeCodecs(out, meta.shard_config->sub_codecs);
        out << ";index_location=" << meta.shard_config->index_location << "}";
    }
    out << ";node_type=" << meta.node_type;
    return out.str();
}

std::optional<cv::Vec3d> unitAxis(const cv::Vec3d& n)
{
    const double length = std::sqrt(n.dot(n));
    if (!(length > 1e-12) || !std::isfinite(length)) {
        return std::nullopt;
    }
    return n * (1.0 / length);
}

} // namespace

LasagnaSheetNormalField::LasagnaSheetNormalField(
    std::shared_ptr<const vc::lasagna::LasagnaNormalSampler> sampler, std::string identity, int threads)
    : sampler_(std::move(sampler)), identity_(std::move(identity)), threads_(threads)
{
}

LasagnaSheetNormalField::~LasagnaSheetNormalField() = default;

std::optional<cv::Vec3d> LasagnaSheetNormalField::axis(const cv::Vec3d& volumePoint) const
{
    const auto sample = sampler_->sampleNormal(volumePoint);
    if (!sample.valid) {
        return std::nullopt;
    }
    return unitAxis(sample.normal);
}

void LasagnaSheetNormalField::axes(const std::vector<cv::Vec3d>& points,
                                   std::vector<std::optional<cv::Vec3d>>& out) const
{
    out.assign(points.size(), std::nullopt);
    if (points.empty()) {
        return;
    }
    std::vector<vc::lasagna::NormalSampleWithDerivative> samples;
    (void)sampler_->sampleNormalBatch(points, false, threads_, samples);
    for (std::size_t i = 0; i < points.size() && i < samples.size(); ++i) {
        if (!samples[i].sample.valid) {
            continue;
        }
        out[i] = unitAxis(samples[i].sample.normal);
    }
}

std::string lasagnaFieldIdentity(const vc::lasagna::LasagnaDataset& dataset, double workingToBaseScale)
{
    const auto& manifest = dataset.manifest();
    std::string identity = "lasagna|" + manifest.manifestLocation + "|";
    identity += digestHex(readText(manifest.manifestPath));
    for (const auto& group : manifest.groups) {
        const utils::ZarrArray array = vc::lasagna::openLasagnaChannelArray(manifest, group);
        identity += "|" + group.name + ":" + digestHex(describeMetadata(array.metadata()));
    }
    char scale[48];
    std::snprintf(scale, sizeof scale, "|%.17g", workingToBaseScale);
    identity += scale;
    return identity;
}

double sheetFieldWorkingToBaseScale(const std::array<double, 3>& annotationExtentXyz,
                                    const std::optional<std::array<std::size_t, 3>>& baseShapeZYX)
{
    if (!baseShapeZYX) {
        return 1.0;
    }
    for (const double extent : annotationExtentXyz) {
        if (!(extent > 0.0) || !std::isfinite(extent)) {
            throw std::runtime_error(
                "the annotation frame's extent is unknown, so the sheet field's scale cannot be resolved");
        }
    }
    const std::array<std::size_t, 3> annotationZYX{
        static_cast<std::size_t>(std::llround(annotationExtentXyz[2])),
        static_cast<std::size_t>(std::llround(annotationExtentXyz[1])),
        static_cast<std::size_t>(std::llround(annotationExtentXyz[0]))};
    return vc::lasagna::dyadicCoordinateScaleBetweenShapes(annotationZYX, *baseShapeZYX, 5);
}

} // namespace vc3d::fiber_map::bent
