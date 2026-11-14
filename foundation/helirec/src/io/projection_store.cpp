#include "helirec/projection_store.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_set>

#include "helirec/common.hpp"
#include "vc_reader/array.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/zarr_write.hpp"

namespace helirec {
namespace fs = std::filesystem;
using vcr::zarr::Json;

struct ProjectionStoreLevel::Impl {
    vcr::zarr::OpenedArray opened;
    // A shard is written whole (one file per 100 frames), and an absent one would read as zeros: the files found
    // are remembered, a missing one is an error.
    std::optional<vcr::zarr::Writer> layout;  // for the shards' paths only
    std::mutex mutex;
    std::unordered_set<std::int64_t> present;
    void require(std::int64_t frame) {
        const std::int64_t per = layout->meta().v3 && !layout->meta().v3->shards.empty() ? layout->meta().v3->shards[0] : 1;
        const std::int64_t shard = frame / per;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (present.count(shard)) return;
        }
        const fs::path file = layout->chunkPath({frame, 0, 0});
        if (!fs::exists(file)) throw Error("projection store: frame " + std::to_string(frame) + " of " + opened.location + " is not stored (no " + file.string() + ")");
        std::lock_guard<std::mutex> lock(mutex);
        present.insert(shard);
    }
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
        const std::string path = (fs::path(scan_group) / level->at("path").get<std::string>()).string();
        out.impl->opened = vcr::zarr::openArray(path, vcr::zarr::OpenOptions{});
        out.impl->layout.emplace(vcr::zarr::Writer::open(path));
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
    impl->require(frame);
    impl->opened.array->read_many({vcr::Index{frame, r0, 0}}, {vcr::Index{1, nr, cols}}, {bytes});
}

std::vector<ProjectionStoreRun> projection_store_runs(const std::string& scan_group) {
    const Json h = scan_attribute(scan_group);
    if (h.is_null()) throw Error(scan_group + ": not a scan group of a projection store");
    if (!h.contains("files")) throw Error(scan_group + ": the store does not record its source files; run helirec-projections-zarr on it again");
    const std::int64_t frame_first = h.at("frame_first").get<std::int64_t>();
    std::vector<ProjectionStoreRun> out;
    for (const Json& f : h.at("files"))
        out.push_back({f.at("file").get<std::string>(), f.at("source_first").get<std::int64_t>(), f.at("count").get<std::int64_t>(), f.at("first").get<std::int64_t>() - frame_first});
    return out;
}

calibration::FrameList frames_in_projection_store(const calibration::FrameList& list, const std::string& tile_group) {
    using Source = ProjectionStoreRun;
    struct Group {
        std::int32_t index;
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
                if (!is_projection_store_scan(path)) throw Error("projection store: " + tile_group + " has no scan " + scan + " (for " + file.string() + ")");
                Group g;
                g.index = static_cast<std::int32_t>(out.files.size());
                for (const ProjectionStoreRun& r : projection_store_runs(path)) g.files[r.file].push_back(r);
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
        out.frame_index.push_back(static_cast<std::int32_t>(found->index + (frame - found->source_first)));
    }
    return out;
}

std::vector<double> projection_store_sum(const std::string& scan_group, int reduce, const std::int64_t* frame, std::size_t n, const float* dark, const double* weight, int* rows_out,
                                         int* cols_out) {
    const ProjectionStoreLevel level = open_projection_store(scan_group, reduce);
    if (level.dtype != "float32") throw Error(scan_group + ": the store's darks and flats are float images");
    const int rows = level.rows, cols = level.cols;
    if (rows_out) *rows_out = rows;
    if (cols_out) *cols_out = cols;
    const std::size_t np = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const int chunks = (rows + level.chunk_rows - 1) / level.chunk_rows;
    std::vector<double> sum(np, 0.0);
#pragma omp parallel
    {
        std::vector<float> chunk(static_cast<std::size_t>(level.chunk_rows) * static_cast<std::size_t>(cols));
        std::vector<double> mine(np, 0.0);
#pragma omp for schedule(static) collapse(2)
        for (std::int64_t k = 0; k < static_cast<std::int64_t>(n); ++k)
            for (int c = 0; c < chunks; ++c) {
                level.read_chunk(frame[k], c, chunk.data());
                const int r0 = c * level.chunk_rows, nr = std::min(level.chunk_rows, rows - r0);
                const std::size_t at = static_cast<std::size_t>(r0) * static_cast<std::size_t>(cols), count = static_cast<std::size_t>(nr) * static_cast<std::size_t>(cols);
                double* to = mine.data() + at;
                if (dark && weight) {
                    const double w = weight[k];
                    for (std::size_t i = 0; i < count; ++i) to[i] += static_cast<double>(dark[at + i]) + (static_cast<double>(chunk[i]) - static_cast<double>(dark[at + i])) * w;
                } else {
                    for (std::size_t i = 0; i < count; ++i) to[i] += static_cast<double>(chunk[i]);
                }
            }
#pragma omp critical
        for (std::size_t i = 0; i < np; ++i) sum[i] += mine[i];
    }
    return sum;
}

// ---------------------------------------------------------------- writing

namespace {

constexpr int kLevels = 4, kChunkRows = 128, kShardFrames = 100;

std::string typestr(const std::string& dtype) {
    if (dtype == "uint16") return "<u2";
    if (dtype == "float32") return "<f4";
    throw Error("projection store: values are uint16 or float32, not " + dtype);
}

vcr::zarr::CreateOptions level_options(const vcr::Index& shape, const std::string& type, int clevel) {
    vcr::zarr::CreateOptions o;
    o.version = 3;
    o.shape = shape;
    o.dtype = type;
    o.separator = "/";
    o.compressor = "blosc";
    o.cname = "lz4";
    o.shuffle = 2;
    o.level = clevel;
    const std::int64_t rows = std::min<std::int64_t>(kChunkRows, shape[1]);
    o.chunks = {1, rows, shape[2]};
    o.shardShape = {kShardFrames, (shape[1] + rows - 1) / rows * rows, shape[2]};
    return o;
}

}  // namespace

void declare_projection_store_scan(const std::string& root, const std::string& spec_text, int reduce, const std::string& source,
                                   const std::string& dtype, int clevel) {
    try {
        const Json spec = Json::parse(spec_text);
        const std::string tile = spec.at("tile").get<std::string>(), scan = spec.at("scan").get<std::string>(), role = spec.at("role").get<std::string>();
        const std::int64_t rows = spec.at("rows").get<std::int64_t>(), cols = spec.at("cols").get<std::int64_t>();
        const std::int64_t frame_first = spec.at("frame_first").get<std::int64_t>(), frames = spec.at("frames").get<std::int64_t>();
        if (reduce < 0 || reduce >= kLevels) throw Error("reduce must be 0, 1, 2 or 3");
        if (frames < 1 || rows < 1 || cols < 1 || rows % (1 << (kLevels - 1)) || cols % (1 << (kLevels - 1))) throw Error("the detector's size must be a multiple of 8");
        const std::string type = typestr(dtype);
        const fs::path group = fs::path(root) / tile / scan;
        const Json before = scan_attribute(group.string());
        if (!before.is_null() && (before.at("frame_first").get<std::int64_t>() != frame_first || before.at("frames").get<std::int64_t>() != frames))
            throw Error("the store holds frames [" + std::to_string(before.at("frame_first").get<std::int64_t>()) + ", +" + std::to_string(before.at("frames").get<std::int64_t>()) +
                        ") of this scan, not [" + std::to_string(frame_first) + ", +" + std::to_string(frames) + ")");
        vcr::zarr::createGroup(root, Json{{"helirec_projections", {{"format_version", 1}}}}, 3);
        vcr::zarr::createGroup(fs::path(root) / tile, Json{{"helirec_tile", spec.value("tile_attributes", Json::object())}}, 3);
        Json levels = Json::array();
        for (int l = 0; l < kLevels; ++l) {
            Json entry{{"path", std::to_string(l)}, {"reduce", l}, {"stored", false}};
            if (!before.is_null())
                for (const Json& e : before.at("levels"))
                    if (e.value("reduce", -1) == l && e.value("stored", false)) entry = e;
            const bool was = entry.at("stored").get<bool>();
            if (l == reduce) entry["stored"] = true, entry["source"] = source;
            // a level without files is declared as its files would be: the detector's uint16, or a reduced dark's float32
            const std::string want = l == reduce ? type : (role == "projection" || l == 0 ? "<u2" : "<f4");
            const fs::path path = group / std::to_string(l);
            vcr::zarr::CreateOptions o = level_options({frames, rows >> l, cols >> l}, want, clevel);
            if (!fs::exists(path / "zarr.json")) {
                vcr::zarr::Writer::create(path, o);
            } else {
                const auto meta = vcr::zarr::Writer::open(path).meta();
                if (meta.shape != o.shape || meta.chunks != o.chunks) throw Error(path.string() + " exists with another shape or chunking");
                if (l == reduce && meta.dtype.substr(1) != type.substr(1)) {
                    if (was) throw Error(path.string() + " is stored with another dtype");
                    o.overwrite = true;  // declared only: nothing is lost
                    vcr::zarr::Writer::create(path, o);
                }
            }
            levels.push_back(std::move(entry));
        }
        Json files = spec.value("files", Json::array());
        if (files.empty() && !before.is_null() && before.contains("files")) files = before.at("files");
        vcr::zarr::MultiscalesOptions m;
        m.name = tile + "/" + scan, m.levels = kLevels - 1, m.factors = {1, 2, 2}, m.axes = {"frame", "row", "col"};
        m.reduction = vcr::zarr::Reduction::Nearest;  // no translation: level values sit at stored pixels (j, k) 2^l
        Json multiscales = vcr::zarr::omeMultiscales(m);
        multiscales["metadata"] = {{"downsampling_method", "JPEG 2000 9/7 wavelet low band"}};
        vcr::zarr::createGroup(group,
                               Json{{"multiscales", Json::array({multiscales})},
                                    {"helirec", {{"role", role}, {"tile", tile}, {"scan", scan}, {"frame_first", frame_first}, {"frames", frames}, {"levels", levels}, {"files", files}}}},
                               3);
        // one value per frame
        const std::size_t n = static_cast<std::size_t>(frames);
        const auto write = [&](const std::string& name, const std::string& t, const void* data, const Json& fill) {
            vcr::zarr::CreateOptions o;
            o.version = 3, o.shape = {frames}, o.chunks = {frames}, o.dtype = t, o.separator = "/";
            o.compressor = "blosc", o.cname = "lz4", o.shuffle = 1, o.fillValue = fill, o.overwrite = true;
            auto w = vcr::zarr::Writer::create(group / name, o);
            w.writeChunk({0}, static_cast<const std::uint8_t*>(data), n * 8);
        };
        std::vector<std::int64_t> original(n, -1);
        for (const Json& f : files) {
            const std::int64_t first = f.at("first").get<std::int64_t>(), count = f.at("count").get<std::int64_t>();
            if (first < frame_first || count < 0 || first + count > frame_first + frames) throw Error("a file's frames lie outside the scan's");
            for (std::int64_t i = 0; i < count; ++i) original[static_cast<std::size_t>(first - frame_first + i)] = first + i;
        }
        write("original_frame_index", "<i8", original.data(), -1);
        const Json frame_values = spec.value("frame_values", Json::object());
        for (const auto& [key, list] : frame_values.items()) {
            if (!list.is_array() || list.size() != n) throw Error("frame_values." + key + " must hold one value per frame");
            std::vector<double> v(n, std::numeric_limits<double>::quiet_NaN());
            for (std::size_t i = 0; i < n; ++i)
                if (list[i].is_number()) v[i] = list[i].get<double>();
            write(key, "<f8", v.data(), "NaN");
        }
    } catch (const Error&) {
        throw;
    } catch (const std::exception& e) {
        throw Error(std::string("projection store: ") + e.what());
    }
}

struct ProjectionStoreWriter::Impl {
    vcr::zarr::Writer writer;
    std::int64_t frame_first = 0;
    std::string dtype, group;
    explicit Impl(vcr::zarr::Writer w) : writer(std::move(w)) {}
    vcr::Index start(std::int64_t first, std::int64_t count) const {
        const auto& shape = writer.meta().shape;
        if (count < 1 || first < frame_first || first - frame_first + count > shape[0]) throw Error(group + ": frames [" + std::to_string(first) + ", +" + std::to_string(count) + ") are outside the scan's");
        return {first - frame_first, 0, 0};
    }
};

ProjectionStoreWriter::ProjectionStoreWriter(const std::string& scan_group, int reduce) {
    const Json h = scan_attribute(scan_group);
    if (h.is_null()) throw Error(scan_group + ": not a scan group of a projection store (declare it first)");
    bool stored = false;
    for (const Json& l : h.at("levels")) stored |= l.value("reduce", -1) == reduce && l.value("stored", false);
    if (!stored) throw Error(scan_group + ": level " + std::to_string(reduce) + " is not declared as stored");
    try {
        impl_ = std::make_unique<Impl>(vcr::zarr::Writer::open(fs::path(scan_group) / std::to_string(reduce)));
    } catch (const std::exception& e) {
        throw Error(scan_group + ": " + e.what());
    }
    impl_->frame_first = h.at("frame_first").get<std::int64_t>();
    impl_->group = scan_group;
    impl_->dtype = impl_->writer.meta().dtype.substr(1) == "u2" ? "uint16" : "float32";
}
ProjectionStoreWriter::~ProjectionStoreWriter() = default;
ProjectionStoreWriter::ProjectionStoreWriter(ProjectionStoreWriter&&) noexcept = default;
int ProjectionStoreWriter::rows() const { return static_cast<int>(impl_->writer.meta().shape[1]); }
int ProjectionStoreWriter::cols() const { return static_cast<int>(impl_->writer.meta().shape[2]); }
const std::string& ProjectionStoreWriter::dtype() const { return impl_->dtype; }

bool ProjectionStoreWriter::holds(std::int64_t first, std::int64_t count) const {
    const vcr::Index s = impl_->start(first, count);
    const auto& m = impl_->writer.meta();
    try {
        return impl_->writer.chunkExists({s[0], 0, 0}) && impl_->writer.chunkExists({s[0] + count - 1, (m.shape[1] - 1) / m.chunks[1], 0});
    } catch (const std::exception& e) {
        throw Error(impl_->group + ": " + e.what());
    }
}

void ProjectionStoreWriter::write(std::int64_t first, std::int64_t count, const void* data, bool verify) {
    const vcr::Index s = impl_->start(first, count);
    const auto& m = impl_->writer.meta();
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    try {
        impl_->writer.writeRegion(s, {count, m.shape[1], m.shape[2]}, bytes);
        if (!verify) return;
        const std::size_t frame = static_cast<std::size_t>(m.shape[1] * m.shape[2]) * m.itemsize();
        std::vector<std::uint8_t> back(frame);
        for (std::int64_t i = 0; i < count; ++i) {
            impl_->writer.reader().read_many({vcr::Index{s[0] + i, 0, 0}}, {vcr::Index{1, m.shape[1], m.shape[2]}}, {back.data()});
            if (std::memcmp(back.data(), bytes + static_cast<std::size_t>(i) * frame, frame) != 0)
                throw Error("frame " + std::to_string(first + i) + " reads back with other values");
        }
    } catch (const Error& e) {
        throw Error(impl_->group + ": " + e.what());
    } catch (const std::exception& e) {
        throw Error(impl_->group + ": " + e.what());
    }
}

}  // namespace helirec
