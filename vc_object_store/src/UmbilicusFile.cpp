#include "vc_object_store/objects/UmbilicusFile.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/VolumeUmbilicus.hpp"
#include "vc_object_store/Uuid.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace vc::objects {
namespace {
using Json = nlohmann::json;

enum class UmbilicusScaleSource {
    StampedDimensions,
    StampedVoxelSize,
    InferredFromGrid,
};
// The factor that carries the file's coordinates into a target grid's voxels.
struct UmbilicusScale {
    double factor = 1.0;
    UmbilicusScaleSource source = UmbilicusScaleSource::StampedDimensions;
    std::string description;
};
// What frame the file states: a complete positive dimension triplet, a voxel size.
struct UmbilicusFrameClaim {
    bool dimensions = false;
    bool voxelSize = false;
    [[nodiscard]] bool any() const { return dimensions || voxelSize; }
};


std::string trimmed(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    return value.substr(begin, value.find_last_not_of(" \t\r\n") - begin + 1);
}

// "z, y, x" per line; blank lines and '#' comments skipped.
std::vector<geometry::Point> readTextPoints(std::istream& stream)
{
    std::vector<geometry::Point> points;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(stream, line)) {
        ++lineNumber;
        const auto text = trimmed(line);
        if (text.empty() || text[0] == '#') continue;
        std::array<double, 3> values{};
        std::size_t count = 0;
        std::stringstream ss(text);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trimmed(token);
            if (token.empty()) continue;
            if (count >= values.size())
                throw std::runtime_error("Too many columns in umbilicus text line " + std::to_string(lineNumber));
            try {
                values[count] = std::stod(token);
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid numeric value in umbilicus text line " + std::to_string(lineNumber));
            }
            ++count;
        }
        if (count != values.size())
            throw std::runtime_error("Not enough columns in umbilicus text line " + std::to_string(lineNumber));
        points.push_back({values[2], values[1], values[0]});
    }
    if (points.empty()) throw std::runtime_error("Umbilicus text file contained no points");
    return points;
}

// A present-but-invalid field is recorded so a typo is told from a legacy file
// that never declared it: "<key>: <reason>, got <value-as-written>".
void readFrameMetadata(const Json& metadata, UmbilicusFile& file)
{
    const auto reject = [&file](const char* key, const char* reason, const Json& value) {
        file.metadataErrors.push_back(std::string(key) + ": " + reason + ", got " + value.dump());
    };
    if (metadata.contains("voxelsize_um")) {
        const auto& value = metadata.at("voxelsize_um");
        const double voxelSize = value.is_number() ? value.get<double>() : 0.0;
        if (value.is_number() && std::isfinite(voxelSize) && voxelSize > 0.0) file.voxelSizeUm = voxelSize;
        else reject("voxelsize_um", "expected a positive number", value);
    }
    if (metadata.contains("volume")) {
        const auto& value = metadata.at("volume");
        if (value.is_string() && !value.get<std::string>().empty()) file.volume = value.get<std::string>();
        else reject("volume", "expected a non-empty string", value);
    }
    bool dimensionRejected = false;
    const auto readDimension = [&](const char* key, std::optional<int>& out) {
        if (!metadata.contains(key)) return;
        const auto& value = metadata.at(key);
        if (value.is_number_integer() && value.get<int64_t>() > 0) {
            out = int(value.get<int64_t>());
        } else {
            dimensionRejected = true;
            reject(key, "expected a positive integer", value);
        }
    };
    readDimension("volume_width", file.volumeWidth);
    readDimension("volume_height", file.volumeHeight);
    readDimension("volume_slices", file.volumeSlices);
    // A partial triplet is a malformed statement, not an absent one: two of
    // three dimensions cannot describe a grid. Skipped when a dimension was
    // rejected; naming the bad value says more.
    const int declared = (file.volumeWidth ? 1 : 0) + (file.volumeHeight ? 1 : 0) + (file.volumeSlices ? 1 : 0);
    if (!dimensionRejected && declared > 0 && declared < 3) {
        std::string missing;
        for (const auto& [key, present] : {std::pair<const char*, bool>{"volume_width", file.volumeWidth.has_value()},
                                           std::pair<const char*, bool>{"volume_height", file.volumeHeight.has_value()},
                                           std::pair<const char*, bool>{"volume_slices", file.volumeSlices.has_value()}})
            if (!present) missing += (missing.empty() ? "" : ", ") + std::string(key);
        file.metadataErrors.push_back("volume dimensions: an incomplete triplet declares nothing; missing " + missing);
    }
}

UmbilicusFile readJsonFile(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Failed to open umbilicus json file: " + path.string());
    const auto document = Json::parse(stream);
    const Json* array = nullptr;
    if (document.is_array()) {
        array = &document;
    } else if (document.is_object() && (document.contains("points") || document.contains("control_points"))) {
        const char* key = document.contains("points") ? "points" : "control_points";
        array = &document.at(key);
        if (!array->is_array())
            throw std::runtime_error(std::string("'") + key + "' member in umbilicus json must be an array");
    } else {
        throw std::runtime_error("Umbilicus json root must be an array or contain a 'points' or 'control_points' array");
    }
    UmbilicusFile file;
    file.points.reserve(array->size());
    for (std::size_t i = 0; i < array->size(); ++i) {
        const auto& entry = (*array)[i];
        if (entry.is_array()) {
            if (entry.size() < 3)
                throw std::runtime_error("Umbilicus json entry at index " + std::to_string(i) + " expected three values");
            file.points.push_back({entry[2].get<double>(), entry[1].get<double>(), entry[0].get<double>()});
        } else if (entry.is_object()) {
            if (!entry.contains("z") || !entry.contains("y") || !entry.contains("x"))
                throw std::runtime_error("Umbilicus json entry at index " + std::to_string(i) + " missing z/y/x keys");
            file.points.push_back({entry.at("x").get<double>(), entry.at("y").get<double>(), entry.at("z").get<double>()});
        } else {
            throw std::runtime_error("Umbilicus json entry at index " + std::to_string(i) + " has unsupported type");
        }
    }
    if (file.points.empty()) throw std::runtime_error("Umbilicus json file contained no points");
    if (document.is_object() && document.contains("metadata")) {
        const auto& metadata = document.at("metadata");
        // Null is how a writer spells "no value"; any other non-object would
        // read as a legacy file that never declared a frame.
        if (metadata.is_object()) readFrameMetadata(metadata, file);
        else if (!metadata.is_null()) file.metadataErrors.push_back("metadata: expected an object, got " + metadata.dump());
    }
    return file;
}

bool haveStampedDimensions(const UmbilicusFile& file)
{
    return file.volumeWidth && file.volumeHeight && file.volumeSlices && *file.volumeWidth > 0 && *file.volumeHeight > 0 &&
           *file.volumeSlices > 0;
}

std::array<double, 3> stampedDimensions(const UmbilicusFile& file)
{
    return {double(*file.volumeWidth), double(*file.volumeHeight), double(*file.volumeSlices)};
}

std::string stampedGridText(const UmbilicusFile& file)
{
    return std::to_string(*file.volumeWidth) + "x" + std::to_string(*file.volumeHeight) + "x" +
           std::to_string(*file.volumeSlices);
}

std::optional<double> uniformRescaleFactor(const std::array<double, 3>& stampedXyz, const std::array<double, 3>& targetXyz)
{
    for (int axis = 0; axis < 3; ++axis)
        if (!std::isfinite(stampedXyz[axis]) || stampedXyz[axis] < 1.0 || !std::isfinite(targetXyz[axis]) || targetXyz[axis] < 1.0)
            return std::nullopt;
    // Downsampling by an integer factor n maps a count c to floor(c/n) or
    // ceil(c/n) and nothing else, so a candidate factor is tested exactly.
    // Both directions are tried: the stamped grid may be the coarser or the finer.
    const auto explains = [&](double factor) {
        for (int axis = 0; axis < 3; ++axis) {
            const double coarse = factor >= 1.0 ? targetXyz[axis] : stampedXyz[axis];
            const double fine = factor >= 1.0 ? stampedXyz[axis] : targetXyz[axis];
            const double exact = coarse / (factor >= 1.0 ? factor : 1.0 / factor);
            if (fine != std::floor(exact) && fine != std::ceil(exact)) return false;
        }
        return true;
    };
    // Six pyramid levels either way covers every store this format describes.
    constexpr int kMaxStep = 64;
    std::vector<double> candidates;
    for (int n = 1; n <= kMaxStep; ++n)
        if (explains(double(n))) candidates.push_back(double(n));
    for (int m = 2; m <= kMaxStep; ++m)
        if (explains(1.0 / double(m))) candidates.push_back(1.0 / double(m));
    // Several only happen on grids of a handful of voxels, where the rounding
    // windows overlap: picking one would be a guess.
    if (candidates.size() != 1) return std::nullopt;
    return candidates.front();
}

std::optional<UmbilicusScale> deriveUmbilicusScale(const UmbilicusFile& file, const std::array<double, 3>& targetGridXyz,
                                                   std::optional<double> targetVoxelSizeUm)
{
    const bool haveTarget = targetGridXyz[0] > 0.0 && targetGridXyz[1] > 0.0 && targetGridXyz[2] > 0.0;
    if (haveStampedDimensions(file) && haveTarget) {
        // No single factor explaining all three axes means this is no rescale
        // of the target grid, and neither weaker reading applies.
        const auto factor = uniformRescaleFactor(stampedDimensions(file), targetGridXyz);
        if (!factor) return std::nullopt;
        return UmbilicusScale{*factor, UmbilicusScaleSource::StampedDimensions, "stamped " + stampedGridText(file) + " grid"};
    }
    if (file.voxelSizeUm && targetVoxelSizeUm && *targetVoxelSizeUm > 0.0) {
        const double ratio = *file.voxelSizeUm / *targetVoxelSizeUm;
        if (std::isfinite(ratio) && ratio > 0.0)
            return UmbilicusScale{ratio, UmbilicusScaleSource::StampedVoxelSize,
                                  "stamped " + std::to_string(*file.voxelSizeUm) + " um voxels"};
    }
    if (!haveTarget || file.points.empty()) return std::nullopt;
    // Nothing stated: read the grid off the points. An umbilicus runs the length
    // of the scroll, so the right grid is the one it nearly fills and still fits
    // inside; candidates a factor of two apart make the match unique.
    constexpr double kMinZCoverage = 0.6;
    constexpr int kMaxDownsampleSteps = 5;
    std::array<double, 3> lo, hi;
    lo.fill(std::numeric_limits<double>::infinity());
    hi.fill(-std::numeric_limits<double>::infinity());
    for (const auto& point : file.points)
        for (int axis = 0; axis < 3; ++axis) {
            lo[axis] = std::min(lo[axis], point[axis]);
            hi[axis] = std::max(hi[axis], point[axis]);
        }
    std::optional<UmbilicusScale> match;
    for (int step = 0; step <= kMaxDownsampleSteps; ++step) {
        const double candidate = std::pow(2.0, step);
        bool fits = lo[0] >= 0.0 && lo[1] >= 0.0 && lo[2] >= 0.0;
        for (int axis = 0; axis < 3 && fits; ++axis) fits = hi[axis] <= targetGridXyz[axis] / candidate;
        if (!fits) continue;
        const double gridZ = targetGridXyz[2] / candidate;
        if ((gridZ > 0.0 ? (hi[2] - lo[2]) / gridZ : 0.0) < kMinZCoverage) continue;
        if (match) return std::nullopt;  // more than one grid fits: say nothing
        match = UmbilicusScale{candidate, UmbilicusScaleSource::InferredFromGrid, "inferred from the volume grid"};
    }
    return match;
}

UmbilicusFrameClaim umbilicusFrameClaim(const UmbilicusFile& file)
{
    // A partial triplet is not a claim: it cannot describe a grid.
    return {haveStampedDimensions(file), file.voxelSizeUm && *file.voxelSizeUm > 0.0};
}

std::optional<std::string> umbilicusStampContradiction(const UmbilicusFile& file,
                                                       const std::array<double, 3>& namedVolumeGridXyz,
                                                       std::optional<double> namedVolumeVoxelSizeUm)
{
    const auto claim = umbilicusFrameClaim(file);
    const bool haveGrid = namedVolumeGridXyz[0] > 0.0 && namedVolumeGridXyz[1] > 0.0 && namedVolumeGridXyz[2] > 0.0;
    // The named volume's voxel size carried onto the stamped grid. Only one the
    // dimension triplet implied is independent evidence a stamped voxel size
    // can contradict: a voxel-size difference on its own is the conversion
    // deriveUmbilicusScale applies.
    double impliedVoxelUm =
        namedVolumeVoxelSizeUm && std::isfinite(*namedVolumeVoxelSizeUm) && *namedVolumeVoxelSizeUm > 0.0
            ? *namedVolumeVoxelSizeUm : 0.0;
    bool impliedFromDimensions = false;
    if (claim.dimensions && haveGrid) {
        const auto storeFactor = uniformRescaleFactor(stampedDimensions(file), namedVolumeGridXyz);
        if (!storeFactor)
            return "the stamped " + stampedGridText(file) + " grid is not a uniform rescale of the named volume's " +
                   std::to_string((long long)namedVolumeGridXyz[0]) + "x" + std::to_string((long long)namedVolumeGridXyz[1]) +
                   "x" + std::to_string((long long)namedVolumeGridXyz[2]);
        impliedVoxelUm *= *storeFactor;
        impliedFromDimensions = true;
    }
    // 1% covers metadata that round-trips a micrometre figure imprecisely; a
    // real disagreement is far larger (the smallest rescale is x2).
    if (claim.voxelSize && impliedFromDimensions && impliedVoxelUm > 0.0 && std::isfinite(impliedVoxelUm) &&
        std::abs(impliedVoxelUm - *file.voxelSizeUm) > 0.01 * *file.voxelSizeUm)
        return "the stamped " + std::to_string(*file.voxelSizeUm) + " um voxel size disagrees with the " +
               std::to_string(impliedVoxelUm) + " um the named volume implies";
    return std::nullopt;
}


}

UmbilicusFile readUmbilicusFile(const std::filesystem::path& path)
{
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (extension == ".json") return readJsonFile(path);
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Failed to open umbilicus text file: " + path.string());
    UmbilicusFile file;
    file.points = readTextPoints(stream);
    return file;
}

UmbilicusBinding umbilicusBinding(const UmbilicusFile& file, const UmbilicusTargetGrid& target,
                                  const std::optional<UmbilicusTargetGrid>& named)
{
    UmbilicusBinding binding;
    if (!file.metadataErrors.empty()) {
        std::string errors;
        for (const auto& error : file.metadataErrors) errors += "\n" + error;
        binding.refusal = "it declares malformed frame metadata:" + errors;
        return binding;
    }
    const auto derived = deriveUmbilicusScale(file, target.xyz, target.voxelSizeUm);
    const auto contradiction = named && file.volume ? umbilicusStampContradiction(file, named->xyz, named->voxelSizeUm)
                                                    : std::nullopt;
    if (contradiction) {
        binding.refusal = *contradiction;
        return binding;
    }
    // A stated frame that derives no scale does not fit; it is never read as
    // though the file had stated nothing.
    if (!derived) {
        if (umbilicusFrameClaim(file).any()) binding.refusal = "it states a frame that does not fit the volume";
        else binding.confirm = true;
        return binding;
    }
    // A stated frame is taken as stated; an inferred one is confirmed.
    binding.storedUnitsPerVoxel = 1.0 / derived->factor;
    binding.frame = derived->description;
    binding.confirm = derived->source == UmbilicusScaleSource::InferredFromGrid;
    return binding;
}

std::string importVolumeUmbilicus(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& volume,
                                  const UmbilicusFile& file, double storedUnitsPerVoxel, const std::string& name,
                                  nlohmann::json provenance)
{
    ObjectRecord record;
    record.uuid = util::newUuid();
    record.kind = "umbilicus";
    record.name = name;
    record.meta = Json{{"coordinates", volumeCoordinates(volume, storedUnitsPerVoxel)}, {"provenance", std::move(provenance)}}.dump();
    RevisionSelection pins;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) { pins = volumeUmbilicusRevisions(db, txn, volume); });
    pins[record.uuid] = std::nullopt;
    session.act(attachment, pins, [&](auto& db, auto& txn) {
        setVolumeUmbilicus(db, txn, volume, record, geometry::Umbilicus{file.points});
    });
    return record.uuid;
}
}
