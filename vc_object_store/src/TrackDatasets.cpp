#include "vc_object_store/objects/TrackDatasets.hpp"
#include "formats/TrackInput.hpp"

#include <map>
#include <stdexcept>

namespace vc::objects {

std::vector<TrackDatasetGroup> groupTrackDatasets(std::span<const TrackDatasetRow> rows)
{
    // Ordered maps: datasets by identity, a dataset's tracks by source ID.
    std::map<std::string_view, std::map<uint64_t, size_t>> datasets;
    for (size_t row = 0; row < rows.size(); ++row) {
        const auto& track = rows[row];
        const std::string uuid(track.uuid);
        if (track.dataset.empty()) throw std::runtime_error("track has no dataset_ref: " + uuid);
        uint64_t id = 0;
        try {
            id = detail::trackSourceId(std::string(track.externalKey));
        } catch (const std::runtime_error& error) {
            throw std::runtime_error(std::string(error.what()) + " (track " + uuid + ")");
        }
        const auto [first, inserted] = datasets[track.dataset].emplace(id, row);
        if (!inserted)
            throw std::runtime_error("duplicate registered track source ID " + std::string(track.externalKey)
                + " in dataset " + std::string(track.dataset) + ": tracks " + std::string(rows[first->second].uuid)
                + " and " + uuid);
    }
    std::vector<TrackDatasetGroup> groups;
    groups.reserve(datasets.size());
    for (const auto& [dataset, byId] : datasets) {
        auto& group = groups.emplace_back();
        group.dataset = std::string(dataset);
        group.rows.reserve(byId.size());
        group.sourceIds.reserve(byId.size());
        group.families.reserve(byId.size());
        for (const auto& [id, row] : byId) {
            const auto family = rows[row].family;
            if (!family.empty() && family != "horizontal" && family != "vertical")
                throw std::runtime_error("invalid stored track family '" + std::string(family) + "' of track "
                    + std::string(rows[row].uuid));
            group.rows.push_back(row);
            group.sourceIds.push_back(id);
            group.families.push_back(family == "horizontal" ? 0 : family == "vertical" ? 1 : -1);
        }
    }
    return groups;
}

}
