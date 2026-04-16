#include "VolumeViewerCmaps.hpp"

namespace volume_viewer_cmaps
{

namespace {

std::vector<OverlayColormapEntry> buildEntries(const EntryScope scope)
{
    std::vector<OverlayColormapEntry> out;
    for (const auto& spec : vc::specs()) {
        if (scope == EntryScope::SharedOnly && spec.audience == ColormapAudience::OverlayOnly) {
            continue;
        }
        out.push_back({QString::fromStdString(spec.label), spec.id});
    }
    return out;
}

} // namespace

const std::vector<OverlayColormapEntry>& entries(const EntryScope scope)
{
    static const std::vector<OverlayColormapEntry> sharedEntries = buildEntries(EntryScope::SharedOnly);
    static const std::vector<OverlayColormapEntry> overlayEntries = buildEntries(EntryScope::OverlayCompatible);
    return scope == EntryScope::SharedOnly ? sharedEntries : overlayEntries;
}

} // namespace volume_viewer_cmaps
