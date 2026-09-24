#include "vc/core/util/VoxelSizeMetadata.hpp"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <string_view>

namespace vc::metadata
{

namespace
{

// Follow an exact path of object keys. Returns nullopt as soon as a key is
// missing or a level is not an object, so a caller never has to pre-check the
// shape of a foreign document.
//
// The result is a copy, deliberately. utils::Json::operator[] hands out
// references into a bounded thread-local cache that evicts its oldest entries,
// so a reference held across a loop that itself indexes into the document is
// eventually left dangling. These documents are small.
std::optional<utils::Json> walk(const utils::Json& root,
                                std::initializer_list<std::string_view> keys)
{
    const utils::Json* current = &root;
    for (const std::string_view key : keys) {
        const std::string name(key);
        if (!current->is_object() || !current->contains(name))
            return std::nullopt;
        current = &(*current)[name];
    }
    return *current;
}

// Accept a number, or a string holding one. Several of these documents store
// numbers as strings -- `source.resolution` is `"2"`, not `2` -- and rejecting
// those would lose exactly the field that makes a derived store resolvable.
// Booleans are excluded: JSON true is not the number 1 here.
std::optional<double> asNumber(const utils::Json& value)
{
    if (value.is_boolean())
        return std::nullopt;
    if (value.is_number())
        return value.get_double();
    if (!value.is_string())
        return std::nullopt;

    const std::string text = value.get_string();
    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(text, &consumed);
        while (consumed < text.size() &&
               std::isspace(static_cast<unsigned char>(text[consumed])) != 0) {
            ++consumed;
        }
        if (consumed == text.size())
            return parsed;
    } catch (...) {
    }
    return std::nullopt;
}

std::optional<double> positiveNumber(const utils::Json& value)
{
    const auto number = asNumber(value);
    if (!number || !std::isfinite(*number) || *number <= 0.0)
        return std::nullopt;
    return number;
}

// A pyramid level must be an exact non-negative integer. The upper bound
// matches the 0..31 range the zarr opener enforces on dataset paths, and keeps
// the 2^level factor below from overflowing.
std::optional<int> asPyramidLevel(const utils::Json& value)
{
    const auto number = asNumber(value);
    if (!number || !std::isfinite(*number) || *number < 0.0 || *number > 31.0)
        return std::nullopt;
    const auto level = static_cast<int>(*number);
    if (static_cast<double>(level) != *number)
        return std::nullopt;
    return level;
}

// The ESRF/BM18 acquisition record, either under a "scan" wrapper or at the
// root of the document. `samplePixelSize` is in millimeters.
std::optional<double> detectorVoxelSize(const utils::Json& root, int sourceLevel)
{
    for (const bool scanWrapper : {true, false}) {
        const auto value =
            scanWrapper
                ? walk(root, {"scan", "tomo", "acquisition", "detector", "samplePixelSize"})
                : walk(root, {"tomo", "acquisition", "detector", "samplePixelSize"});
        if (!value)
            continue;
        const auto millimeters = positiveNumber(*value);
        if (!millimeters)
            continue;

        const double micrometers =
            *millimeters * 1000.0 * static_cast<double>(std::uint64_t{1} << sourceLevel);
        if (!std::isfinite(micrometers) || micrometers <= 0.0)
            return std::nullopt;
        return micrometers;
    }
    return std::nullopt;
}

// A voxel size the document states outright, at the top level only.
std::optional<double> explicitVoxelSize(const utils::Json& doc)
{
    if (!doc.is_object())
        return std::nullopt;
    for (const char* key : {"voxelsize", "voxel_size_um", "voxelSizeUm", "pixel_size_um",
                            "pixelSizeUm", "resolution_um"}) {
        if (!doc.contains(key))
            continue;
        if (const auto micrometers = positiveNumber(doc[key]))
            return micrometers;
    }
    return std::nullopt;
}

// `metadata.json` with a top-level `voxelsize` key *inside* its `scan` object.
//
// This is the legacy shape the renderer's own pre-patch reader handled, as its
// second candidate:
//
//     if (auto v = tryFile(volPath / "meta.json", nullptr))       return v;
//     if (auto v = tryFile(volPath / "metadata.json", "scan"))    return v;  <-- this
//     if (auto v = tryFile(volPath / "metadata.json", nullptr))   return v;
//
// `tryFile(path, "scan")` reads `root["scan"]` and then looks for `voxelsize` in
// it, so that candidate is exactly `scan.voxelsize` -- nothing deeper. It is
// restored here rather than in the renderer so that every caller of this resolver
// -- Volume construction included -- keeps agreeing on what a store says.
//
// Precedence, which is the point of restoring it rather than approximating it:
// the historical reader had no acquisition-record branch at all, so for a
// document carrying BOTH `scan.voxelsize` and
// `scan.tomo.acquisition.detector.samplePixelSize` it returned `scan.voxelsize`.
// That is why there is no `tomo` guard here, and why the caller consults this
// before `detectorVoxelSize()`: either would silently change which number such a
// document yields, and the review asked for the previously supported behaviour to
// be preserved, not reinterpreted. The resulting order is
//
//     top-level `voxelsize` > `scan.voxelsize` > acquisition record > source
//
// which reduces to the historical order for every document shape the old reader
// could read -- its only other candidate, `metadata.json`'s own root `voxelsize`,
// is the top-level case that already comes first -- and extends it with the
// `source` walk for derived stores that did not exist then.
//
// Shape is still pinned: the value must be reachable at exactly `scan.voxelsize`.
// A bare search for a nested `voxelsize` would also match the `voxelsize` of an
// unrelated sub-document, which is how a resolver reports a neighbour's
// measurement as this volume's.
//
// Unit: micrometers, decided from the historical code rather than assumed.
// `readVolumeVoxelSize()` returned this number with no conversion of any kind,
// and its caller treated every number it returned as micrometers -- the only
// input that received a unit conversion was an explicit `--voxel-size`, guarded
// by `voxelSizeFromCli`. Since the same reader produced both `meta.json`'s
// top-level `voxelsize` (which is micrometers: the published value 7.91 matches
// the volume's own `-7.910um-` name) and this field, and since a single code path
// cannot have meant two units, `scan.voxelsize` was micrometers too. Had it not
// been, the legacy volume would have produced a wrong TIFF resolution -- and that
// path is the one the repository's live-S3 test exercises.
std::optional<double> legacyScanVoxelSize(const utils::Json& doc)
{
    const auto scan = walk(doc, {"scan"});
    if (!scan || !scan->is_object())
        return std::nullopt;
    if (!scan->contains("voxelsize"))
        return std::nullopt;
    // Positive and finite only. The historical reader accepted any number here
    // and left the sign check to its caller, which rejected non-positive values;
    // rejecting them here is the shared resolver's documented contract and gives
    // the same answer for a store that publishes nonsense.
    return positiveNumber((*scan)["voxelsize"]);
}

} // namespace

std::optional<double> voxelSizeFromStoreMetadata(const utils::Json& doc)
{
    if (!doc.is_object())
        return std::nullopt;

    if (auto explicitSize = explicitVoxelSize(doc))
        return explicitSize;

    // The legacy `scan.voxelsize` comes BEFORE the acquisition record, because
    // that is where the pre-patch reader put it. The reader was:
    //
    //     tryFile(meta.json,     nullptr)   // meta.json root `voxelsize`
    //     tryFile(metadata.json, "scan")    // <- root["scan"]["voxelsize"]
    //     tryFile(metadata.json, nullptr)   // metadata.json root `voxelsize`
    //
    // and it had no acquisition-record branch at all. So for a document carrying
    // both `scan.voxelsize` and `scan.tomo...samplePixelSize`, the historical
    // answer is `scan.voxelsize`, and it has to keep winning here or the review's
    // "preserve previously supported local metadata" would be reinterpreted rather
    // than honoured.
    //
    // It also sits before the `source` walk: a derived store records its own
    // acquisition record under `scan` too, and a `voxelsize` found there describes
    // this document, not its source.
    if (auto legacy = legacyScanVoxelSize(doc))
        return legacy;

    // A source volume carries its own scan record. Reached only when neither an
    // explicit size nor a scan-wrapped one was stated.
    if (auto direct = detectorVoxelSize(doc, 0))
        return direct;

    // A derived store -- a surface or ink prediction -- records the volume it
    // ran on under `source`: that volume's scan record, and the pyramid level
    // of it that the run consumed.
    const auto source = walk(doc, {"source"});
    if (!source)
        return std::nullopt;
    const auto sourceMetadata = walk(*source, {"metadata"});
    if (!sourceMetadata)
        return std::nullopt;

    // The level is what converts the source volume's pixel size into this
    // store's, so it is not optional. Assuming 0 would report a voxel size too
    // small by whatever factor the run really used, and nothing downstream
    // could tell. Every derived store the catalog publishes states it.
    if (!source->is_object() || !source->contains("resolution"))
        return std::nullopt;
    const auto sourceLevel = asPyramidLevel((*source)["resolution"]);
    if (!sourceLevel)
        return std::nullopt;
    return detectorVoxelSize(*sourceMetadata, *sourceLevel);
}

std::optional<double> resolveLocalStoreVoxelSize(const std::filesystem::path& storeRoot)
{
    // Each candidate is tried in turn, and a candidate that yields nothing does
    // not stop the search. Two ways a present file can yield nothing:
    //
    //   * it parses but carries no schema this resolver recognises -- many
    //     published `meta.json` files state only dimensions, so stopping there
    //     would hide a perfectly good `metadata.json` next to them;
    //   * it is malformed or unreadable. A file we cannot parse tells us nothing
    //     about the store, and treating it as the final answer would make one
    //     corrupt file hide the resolution the store does publish.
    //
    // Order is the pre-patch renderer's: `meta.json` first because it is the
    // store's own canonical document, then `metadata.json`.
    for (const char* name : {"meta.json", "metadata.json"}) {
        const auto file = storeRoot / name;
        if (!std::filesystem::exists(file))
            continue;
        try {
            if (auto resolved = voxelSizeFromStoreMetadata(utils::Json::parse_file(file)))
                return resolved;
        } catch (const std::exception&) {
            // Fall through to the next candidate.
        }
    }
    return std::nullopt;
}

} // namespace vc::metadata
