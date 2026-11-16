#include "helirec/projection_subset.hpp"
#include "helirec/common.hpp"
#include <hdf5.h>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace helirec {
ProjectionSubset plan_projection_subset(const std::string& master, double center_mm, int slices, double context_turns) {
    if (!std::isfinite(center_mm) || slices < 1 || !std::isfinite(context_turns) || context_turns < 0)
        throw Error("projection subset: invalid slab or context");
    const auto rec = calibration::read_record(master, false);
    const calibration::Stretch* helix = nullptr;
    for (const auto& s : rec.stretches)
        if (s.role == calibration::Role::Projection && !s.z_center.empty()) {
            if (helix) throw Error("projection subset: expected exactly one helix");
            helix = &s;
        }
    if (!helix || helix->n_frames < 2 || helix->z_center.size() != static_cast<std::size_t>(helix->n_frames))
        throw Error("projection subset: missing per-frame helix heights");
    ProjectionSubset out;
    out.master = master; out.label = rec.label; out.detector = rec.detector;
    out.rows = helix->rows; out.cols = helix->cols;
    out.pixel_um = calibration::pixel_um(rec);
    out.center_mm = center_mm; out.slices = slices;
    const double turn = rec.technique.number("scan/tomo_n");
    if (!std::isfinite(out.pixel_um) || out.pixel_um <= 0 || !std::isfinite(turn) || turn <= 0)
        throw Error("projection subset: invalid pixel size or projections per turn");
    out.thickness_mm = slices * out.pixel_um / 1000;
    out.pitch_mm = std::abs(helix->z_center.back() - helix->z_center.front()) * turn / (helix->n_frames - 1);
    out.context_mm = std::max(out.rows * out.pixel_um / 1000, context_turns * out.pitch_mm);
    const double lo = center_mm - out.thickness_mm / 2 - out.context_mm;
    const double hi = center_mm + out.thickness_mm / 2 + out.context_mm;
    const auto [zlo, zhi] = std::minmax_element(helix->z_center.begin(), helix->z_center.end());
    if (lo < *zlo || hi > *zhi) throw Error("projection subset: requested slab and context extend beyond the helix");
    std::int64_t first = helix->n_frames, end = 0;
    for (std::int64_t i = 0; i < helix->n_frames; ++i) {
        const double z = helix->z_center[i];
        if (!std::isfinite(z)) throw Error("projection subset: non-finite helix height");
        if (z >= lo && z <= hi) { first = std::min(first, i); end = i + 1; }
    }
    if (end <= first) throw Error("projection subset: no frames cover the slab");
    for (const auto& s : rec.stretches) {
        if (&s != helix && s.role != calibration::Role::Dark && s.role != calibration::Role::Flat) continue;
        if (s.rows != out.rows || s.cols != out.cols) throw Error("projection subset: detector shape changes within acquisition");
        ProjectionSubsetSeries dst;
        dst.entry = s.entry; dst.role = calibration::role_name(s.role); dst.original_count = s.n_frames;
        const auto a = &s == helix ? first : 0, b = &s == helix ? end : s.n_frames;
        for (const auto& piece : s.pieces) {
            auto p = piece;
            if (p.first < 0 || p.count <= 0 || p.first > s.n_frames || p.count > s.n_frames - p.first || p.source_first < 0)
                throw Error("projection subset: invalid source-file mapping range");
            if (p.first >= b || p.first + p.count <= a) continue;
            // Source paths are relative to the original master, not this metadata copy.
            const auto relative = std::filesystem::path(p.file).lexically_relative(rec.directory);
            if (relative.empty() || *relative.begin() == ".." || relative.is_absolute())
                throw Error("projection subset: source outside the master directory: " + p.file);
            p.file = relative.generic_string();
            if (!dst.pieces.empty() && p.first != dst.pieces.back().first + dst.pieces.back().count)
                throw Error("projection subset: gaps or overlapping source-file mappings");
            dst.pieces.push_back(std::move(p));
        }
        if (dst.pieces.empty()) throw Error("projection subset: missing source mappings");
        dst.first = dst.pieces.front().first;
        dst.count = dst.pieces.back().first + dst.pieces.back().count - dst.first;
        if (dst.first > a || dst.first + dst.count < b) throw Error("projection subset: incomplete frame coverage");
        const auto slice = [&](const std::vector<double>& v) {
            if (v.empty()) return std::vector<double>{};
            if (v.size() != static_cast<std::size_t>(s.n_frames)) throw Error("projection subset: per-frame metadata length mismatch");
            return std::vector<double>(v.begin() + dst.first, v.begin() + dst.first + dst.count);
        };
        dst.angle_deg = slice(calibration::stated_angle(s));
        dst.height_mm = slice(calibration::stated_height(rec, s));
        dst.epoch = slice(s.epoch_trig.empty() ? s.epoch : s.epoch_trig);
        if (&s == helix && dst.angle_deg.size() != static_cast<std::size_t>(dst.count))
            throw Error("projection subset: missing per-frame projection angles");
        out.series.push_back(std::move(dst));
    }
    return out;
}

struct UncompressedFrameWriter::Impl {
    hid_t file = -1, dataset = -1;
    hsize_t frames = 0, rows = 0, cols = 0;
    bool floating = false;
    ~Impl() { if (dataset >= 0) H5Dclose(dataset); if (file >= 0) H5Fclose(file); }
    void write(std::int64_t first, std::int64_t count, const void* data, bool is_float) {
        if (dataset < 0 || is_float != floating || first < 0 || count < 0 ||
            static_cast<std::uint64_t>(first) > frames || static_cast<std::uint64_t>(count) > frames - first)
            throw Error("uncompressed frame writer: invalid write range, dtype, or closed file");
        if (!count) return;
        hsize_t offset[3] = {static_cast<hsize_t>(first), 0, 0}, shape[3] = {static_cast<hsize_t>(count), rows, cols};
        const hid_t fs = H5Dget_space(dataset), ms = H5Screate_simple(3, shape, nullptr);
        bool ok = fs >= 0 && ms >= 0 && H5Sselect_hyperslab(fs, H5S_SELECT_SET, offset, nullptr, shape, nullptr) >= 0;
        if (ok) ok = H5Dwrite(dataset, floating ? H5T_NATIVE_FLOAT : H5T_NATIVE_UINT16, ms, fs, H5P_DEFAULT, data) >= 0;
        if (ms >= 0) H5Sclose(ms);
        if (fs >= 0) H5Sclose(fs);
        if (!ok) throw Error("uncompressed frame writer: HDF5 write failed");
    }
};
UncompressedFrameWriter::UncompressedFrameWriter(const std::string& path, const std::string& dataset,
        std::int64_t frames, int rows, int cols, const std::string& dtype) : impl_(std::make_unique<Impl>()) {
    if (frames < 1 || rows < 1 || cols < 1 || (dtype != "uint16" && dtype != "float32"))
        throw Error("uncompressed frame writer: invalid shape or dtype");
    auto& p = *impl_; p.frames = frames; p.rows = rows; p.cols = cols; p.floating = dtype == "float32";
    p.file = H5Fcreate(path.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    if (p.file < 0) throw Error("uncompressed frame writer: cannot create " + path);
    hsize_t shape[3] = {p.frames, p.rows, p.cols}, chunk[3] = {1, p.rows, p.cols};
    hid_t sp = H5Screate_simple(3, shape, nullptr), dcpl = H5Pcreate(H5P_DATASET_CREATE), lcpl = H5Pcreate(H5P_LINK_CREATE);
    bool ok = sp >= 0 && dcpl >= 0 && lcpl >= 0 && H5Pset_chunk(dcpl, 3, chunk) >= 0 && H5Pset_create_intermediate_group(lcpl, 1) >= 0;
    if (ok) p.dataset = H5Dcreate2(p.file, dataset.c_str(), p.floating ? H5T_IEEE_F32LE : H5T_STD_U16LE, sp, lcpl, dcpl, H5P_DEFAULT);
    if (sp >= 0) H5Sclose(sp);
    if (dcpl >= 0) H5Pclose(dcpl);
    if (lcpl >= 0) H5Pclose(lcpl);
    if (!ok || p.dataset < 0) throw Error("uncompressed frame writer: cannot create dataset");
}
UncompressedFrameWriter::~UncompressedFrameWriter() = default;
int UncompressedFrameWriter::rows() const { return static_cast<int>(impl_->rows); }
int UncompressedFrameWriter::cols() const { return static_cast<int>(impl_->cols); }
void UncompressedFrameWriter::write(std::int64_t first, std::int64_t count, const std::uint16_t* data) { impl_->write(first, count, data, false); }
void UncompressedFrameWriter::write(std::int64_t first, std::int64_t count, const float* data) { impl_->write(first, count, data, true); }
void UncompressedFrameWriter::close() {
    auto& p = *impl_;
    bool ok = true;
    if (p.dataset >= 0) { ok = H5Dclose(p.dataset) >= 0; p.dataset = -1; }
    if (p.file >= 0) { ok = H5Fclose(p.file) >= 0 && ok; p.file = -1; }
    if (!ok) throw Error("uncompressed frame writer: closing HDF5 output failed");
}
} // namespace helirec
