#include "helirec/projection_store.hpp"

#include <cstring>
#include <filesystem>
#include <map>

#include "helirec/common.hpp"
#include "vc_reader/array.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/zarr_write.hpp"

namespace helirec {
namespace fs = std::filesystem;
using vcr::zarr::Json;

struct ProjectionStoreLevel::Impl {
    vcr::zarr::OpenedArray opened;
};

namespace {

// the "helirec" attribute of a scan group, or null
Json scan_attribute(const std::string& path) {
    std::error_code ec;
    if (!fs::is_directory(path, ec) || !fs::exists(fs::path(path) / "zarr.json", ec)) return nullptr;
    try {
        const Json a = vcr::zarr::readAttributes(path);
        const auto h = a.find("helirec");
        if (h != a.end() && h->is_object() && h->contains("levels") && h->contains("frame_first")) return *h;
    } catch (const std::exception&) {
    }
    return nullptr;
}

}  // namespace

bool is_projection_store_scan(const std::string& path) { return !scan_attribute(path).is_null(); }

ProjectionStoreLevel open_projection_store(const std::string& scan_group, int reduce) {
    const Json h = scan_attribute(scan_group);
    if (h.is_null()) throw Error(scan_group + ": not a scan group of a projection store");
    ProjectionStoreLevel out;
    out.frames = h.at("frames").get<std::int64_t>();
    out.dtype = "uint16";
    try {
        const Json tile = vcr::zarr::readAttributes(fs::path(scan_group).parent_path()).value("helirec_tile", Json::object());
        out.stored_rows = tile.value("rows", 0), out.stored_cols = tile.value("cols", 0), out.source_chunk_rows = tile.value("chunk_rows", 0);
    } catch (const std::exception&) {
    }
    if (reduce < 0) return out;
    const Json* level = nullptr;
    for (const Json& l : h.at("levels"))
        if (l.value("reduce", -1) == reduce) level = &l;
    if (!level) throw Error(scan_group + ": the store has no level at 1/" + std::to_string(1 << reduce));
    if (!level->value("stored", false))
        throw Error(scan_group + ": the store declares 1/" + std::to_string(1 << reduce) + " of the detector and does not hold it (helirec-projections-zarr was given no copy at that scale)");
    out.impl = std::make_shared<ProjectionStoreLevel::Impl>();
    try {
        out.impl->opened = vcr::zarr::openArray((fs::path(scan_group) / level->at("path").get<std::string>()).string(), vcr::zarr::OpenOptions{});
    } catch (const std::exception& e) {
        throw Error(scan_group + ": " + e.what());
    }
    const auto& m = out.impl->opened.meta;
    if (m.shape.size() != 3 || m.chunks.size() != 3 || m.chunks[0] != 1 || m.chunks[2] != m.shape[2])
        throw Error(scan_group + ": a level's chunks must be whole rows of one frame");
    if (m.dtype.substr(1) != "u2" && m.dtype.substr(1) != "f4") throw Error(scan_group + ": values are neither uint16 nor float32");
    out.dtype = m.dtype.substr(1) == "u2" ? "uint16" : "float32";
    out.frames = m.shape[0];
    out.rows = static_cast<int>(m.shape[1]), out.cols = static_cast<int>(m.shape[2]);
    out.chunk_rows = static_cast<int>(m.chunks[1]);
    if (!out.stored_rows) out.stored_rows = out.rows << reduce, out.stored_cols = out.cols << reduce;
    return out;
}

void ProjectionStoreLevel::read_chunk(std::int64_t frame, int chunk, void* out) const {
    const int r0 = chunk * chunk_rows, nr = std::min(chunk_rows, rows - r0);
    if (!impl || frame < 0 || frame >= frames || chunk < 0 || nr < 1) throw Error("projection store: chunk outside the level");
    const std::size_t row_bytes = static_cast<std::size_t>(cols) * (dtype == "uint16" ? 2 : 4);
    auto* bytes = static_cast<std::uint8_t*>(out);
    if (nr < chunk_rows) std::memset(bytes + static_cast<std::size_t>(nr) * row_bytes, 0, static_cast<std::size_t>(chunk_rows - nr) * row_bytes);
    impl->opened.array->read_many({vcr::Index{frame, r0, 0}}, {vcr::Index{1, nr, cols}}, {bytes});
}

calibration::FrameList frames_in_projection_store(const calibration::FrameList& list, const std::string& tile_group) {
    struct Source {
        std::int64_t source_first, first, count;
    };
    struct Group {
        std::int32_t index;
        std::int64_t frame_first;
        std::map<std::string, std::vector<Source>> files;
    };
    calibration::FrameList out;
    out.dataset = list.dataset;
    std::map<std::string, Group> groups;  // by scan directory name
    std::vector<std::pair<const Group*, const std::vector<Source>*>> of_file(list.files.size(), {nullptr, nullptr});
    for (std::size_t i = 0; i < list.size(); ++i) {
        const std::size_t k = static_cast<std::size_t>(list.file_index[i]);
        if (!of_file.at(k).first) {
            const fs::path file = list.files[k];
            const std::string scan = file.parent_path().filename().string();
            auto it = groups.find(scan);
            if (it == groups.end()) {
                const std::string path = (fs::path(tile_group) / scan).string();
                const Json h = scan_attribute(path);
                if (h.is_null()) throw Error("projection store: " + tile_group + " has no scan " + scan + " (for " + file.string() + ")");
                if (!h.contains("files")) throw Error(path + ": the store does not record its source files; run helirec-projections-zarr on it again");
                Group g;
                g.index = static_cast<std::int32_t>(out.files.size());
                g.frame_first = h.at("frame_first").get<std::int64_t>();
                for (const Json& f : h.at("files"))
                    g.files[f.at("file").get<std::string>()].push_back({f.at("source_first").get<std::int64_t>(), f.at("first").get<std::int64_t>(), f.at("count").get<std::int64_t>()});
                out.files.push_back(path);
                it = groups.emplace(scan, std::move(g)).first;
            }
            const auto f = it->second.files.find(file.filename().string());
            if (f == it->second.files.end()) throw Error("projection store: " + tile_group + "/" + scan + " does not hold " + file.filename().string());
            of_file[k] = {&it->second, &f->second};
        }
        const std::int64_t frame = list.frame_index[i];
        const Source* found = nullptr;
        for (const Source& s : *of_file[k].second)
            if (frame >= s.source_first && frame < s.source_first + s.count) found = &s;
        if (!found) throw Error("projection store: frame " + std::to_string(frame) + " of " + list.files[k] + " is not in " + tile_group);
        out.file_index.push_back(of_file[k].first->index);
        out.frame_index.push_back(static_cast<std::int32_t>(found->first - of_file[k].first->frame_first + (frame - found->source_first)));
    }
    return out;
}

}  // namespace helirec
