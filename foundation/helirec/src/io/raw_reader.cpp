#include "helirec/raw_reader.hpp"

#include <fcntl.h>
#include <hdf5.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <map>
#include <mutex>

#include "io/j2k_chunk.hpp"

namespace helirec {

struct RawReader::FileInfo {
    std::vector<std::int64_t> frame_offset;  // byte offset of each frame in the file (plain values)
    // Compressed frames (io/j2k_chunk.hpp): chunks of chunk_rows rows, each one JPEG 2000 codestream.
    int chunk_rows = 0;                      // 0: plain values
    int chunks_per_frame = 0;
    std::vector<std::int64_t> chunk_offset;  // (frame, chunk)
    std::vector<std::uint32_t> chunk_bytes;
    std::size_t frames() const { return chunk_rows ? chunk_offset.size() / static_cast<std::size_t>(chunks_per_frame) : frame_offset.size(); }
};

struct RawReader::Sync {
    std::mutex hdf5;  // libhdf5 (serial build) is not thread safe
};

namespace {

struct H5Handle {
    hid_t id;
    herr_t (*close)(hid_t);
    H5Handle(hid_t i, herr_t (*c)(hid_t)) : id(i), close(c) {}
    ~H5Handle() {
        if (id >= 0) close(id);
    }
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
};

}  // namespace

bool jpeg2000_frames_supported() { return j2k::available(); }
std::vector<std::uint8_t> jpeg2000_chunk(const std::uint16_t* values, int rows, int cols) { return j2k::encode_chunk(values, cols, rows); }

RawReader::RawReader(std::vector<std::string> files, std::string dataset, int rows, int cols)
    : files_(std::move(files)), dataset_(std::move(dataset)), rows_(rows), cols_(cols), sync_(std::make_unique<Sync>()) {
    infos_.resize(files_.size());
    static std::once_flag quiet;
    std::call_once(quiet, [] { H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); });
}

RawReader::~RawReader() = default;

const RawReader::FileInfo& RawReader::info(int file) {
    std::lock_guard<std::mutex> lock(sync_->hdf5);
    if (file < 0 || static_cast<std::size_t>(file) >= files_.size()) throw Error("RawReader: bad file index");
    if (infos_[file]) return *infos_[file];
    const std::string& path = files_[file];
    H5Handle f(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (f.id < 0) throw Error("RawReader: cannot open " + path);
    H5Handle d(H5Dopen2(f.id, dataset_.c_str(), H5P_DEFAULT), H5Dclose);
    if (d.id < 0) throw Error("RawReader: no dataset " + dataset_ + " in " + path);
    H5Handle type(H5Dget_type(d.id), H5Tclose);
    if (H5Tequal(type.id, H5T_STD_U16LE) <= 0) throw Error("RawReader: " + path + ": dataset is not uint16 LE");
    H5Handle space(H5Dget_space(d.id), H5Sclose);
    hsize_t dims[3] = {0, 0, 0};
    if (H5Sget_simple_extent_ndims(space.id) != 3) throw Error("RawReader: " + path + ": dataset is not 3D");
    H5Sget_simple_extent_dims(space.id, dims, nullptr);
    if (static_cast<int>(dims[1]) != rows_ || static_cast<int>(dims[2]) != cols_)
        throw Error("RawReader: " + path + ": unexpected frame shape");
    H5Handle plist(H5Dget_create_plist(d.id), H5Pclose);
    auto fi = std::make_unique<FileInfo>();
    fi->frame_offset.resize(dims[0], -1);
    const std::int64_t frame_bytes = static_cast<std::int64_t>(rows_) * cols_ * 2;
    const H5D_layout_t layout = H5Pget_layout(plist.id);
    if (layout == H5D_CONTIGUOUS) {
        const haddr_t base = H5Dget_offset(d.id);
        if (base == HADDR_UNDEF) throw Error("RawReader: " + path + ": contiguous dataset without storage");
        for (hsize_t k = 0; k < dims[0]; ++k) fi->frame_offset[k] = static_cast<std::int64_t>(base) + k * frame_bytes;
    } else if (layout == H5D_CHUNKED && H5Pget_nfilters(plist.id) != 0) {
        // one JPEG 2000 codestream per chunk, through the blosc2 filter; nothing else compressed is read
        unsigned flags = 0;
        std::size_t n_values = 0;
        hsize_t cdims[3];
        H5Pget_chunk(plist.id, 3, cdims);
        if (H5Pget_nfilters(plist.id) != 1 || H5Pget_filter2(plist.id, 0, &flags, &n_values, nullptr, 0, nullptr, nullptr) != static_cast<H5Z_filter_t>(j2k::kBlosc2FilterId))
            throw Error("RawReader: " + path + ": compressed data is not supported, but for JPEG 2000 chunks of the blosc2 filter (32026)");
        if (cdims[0] != 1 || cdims[1] < 1 || static_cast<int>(cdims[2]) != cols_)
            throw Error("RawReader: " + path + ": compressed chunks must be whole rows of one frame");
        if (!j2k::available())
            throw Error("RawReader: " + path + ": the frames are stored as JPEG 2000; build helirec with OpenJPEG (libopenjp2 >= 2.5: apt install libopenjp2-7-dev, brew install openjpeg)");
        fi->chunk_rows = static_cast<int>(cdims[1]);
        fi->chunks_per_frame = (rows_ + fi->chunk_rows - 1) / fi->chunk_rows;
        fi->chunk_offset.resize(dims[0] * fi->chunks_per_frame);
        fi->chunk_bytes.resize(dims[0] * fi->chunks_per_frame);
        for (hsize_t k = 0; k < dims[0]; ++k)
            for (int c = 0; c < fi->chunks_per_frame; ++c) {
                hsize_t off[3] = {k, static_cast<hsize_t>(c) * fi->chunk_rows, 0};
                unsigned mask = 0;
                haddr_t addr = HADDR_UNDEF;
                hsize_t size = 0;
                if (H5Dget_chunk_info_by_coord(d.id, off, &mask, &addr, &size) < 0 || addr == HADDR_UNDEF)
                    throw Error("RawReader: " + path + ": frame " + std::to_string(k) + " not written");
                if (mask != 0) throw Error("RawReader: " + path + ": a chunk of frame " + std::to_string(k) + " was stored without its filter");
                fi->chunk_offset[k * fi->chunks_per_frame + c] = static_cast<std::int64_t>(addr);
                fi->chunk_bytes[k * fi->chunks_per_frame + c] = static_cast<std::uint32_t>(size);
            }
    } else if (layout == H5D_CHUNKED) {
        hsize_t cdims[3];
        H5Pget_chunk(plist.id, 3, cdims);
        if (cdims[0] != 1 || static_cast<int>(cdims[1]) != rows_ || static_cast<int>(cdims[2]) != cols_)
            throw Error("RawReader: " + path + ": chunks must be one whole frame");
        for (hsize_t k = 0; k < dims[0]; ++k) {
            hsize_t off[3] = {k, 0, 0};
            unsigned mask = 0;
            haddr_t addr = HADDR_UNDEF;
            hsize_t size = 0;
            if (H5Dget_chunk_info_by_coord(d.id, off, &mask, &addr, &size) < 0 || addr == HADDR_UNDEF)
                throw Error("RawReader: " + path + ": frame " + std::to_string(k) + " not written");
            if (static_cast<std::int64_t>(size) != frame_bytes) throw Error("RawReader: " + path + ": bad chunk size");
            fi->frame_offset[k] = static_cast<std::int64_t>(addr);
        }
    } else {
        throw Error("RawReader: " + path + ": unsupported layout (virtual datasets must be resolved to source files)");
    }
    infos_[file] = std::move(fi);
    return *infos_[file];
}

namespace {

void pread_all(int fd, void* buf, std::size_t n, std::int64_t off, const std::string& path) {
    char* p = static_cast<char*>(buf);
    while (n > 0) {
        const ssize_t r = ::pread(fd, p, n, off);
        if (r < 0) {
            if (errno == EINTR) continue;
            throw Error("RawReader: read error in " + path + ": " + std::strerror(errno));
        }
        if (r == 0) throw Error("RawReader: unexpected end of file in " + path);
        p += r;
        n -= static_cast<std::size_t>(r);
        off += r;
    }
}

// rows [k chunk_rows, (k + 1) chunk_rows) of frame `frame`, decoded -> tile (chunk_rows, cols)
void decode_chunk(int fd, const std::string& path, std::int64_t offset, std::size_t bytes, int chunk_rows, int cols, std::vector<std::uint8_t>& raw,
                  std::uint16_t* tile) {
    raw.resize(bytes);
    pread_all(fd, raw.data(), bytes, offset, path);
    j2k::decode(j2k::codestream_of_chunk(raw.data(), bytes, static_cast<std::size_t>(chunk_rows) * cols * 2), cols, chunk_rows, tile);
}

// the reading threads of the compressed paths keep one file open each
struct OpenFile {
    int fd = -1, file = -1;
    ~OpenFile() {
        if (fd >= 0) ::close(fd);
    }
    int get(int f, const std::string& path) {
        if (f == file) return fd;
        if (fd >= 0) ::close(fd);
        file = f;
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            file = -1;
            throw Error("RawReader: cannot open " + path);
        }
        return fd;
    }
};

}  // namespace

RawReader::Storage RawReader::storage(int file) {
    const FileInfo& fi = info(file);
    Storage st;
    st.frames = static_cast<std::int64_t>(fi.frames());
    st.bytes_per_frame = static_cast<double>(rows_) * cols_ * 2;
    if (!fi.chunk_rows || fi.chunk_offset.empty()) return st;
    st.compressed = true;
    st.chunk_rows = fi.chunk_rows;
    double total = 0;
    for (std::uint32_t b : fi.chunk_bytes) total += b;
    st.bytes_per_frame = total / static_cast<double>(st.frames);
    const std::string& path = files_[file];
    std::vector<std::uint8_t> raw(fi.chunk_bytes[0]);
    OpenFile of;
    pread_all(of.get(file, path), raw.data(), raw.size(), fi.chunk_offset[0], path);
    const j2k::Info ji = j2k::info(j2k::codestream_of_chunk(raw.data(), raw.size(), static_cast<std::size_t>(fi.chunk_rows) * cols_ * 2));
    st.reversible = ji.reversible;
    st.bit_depth = ji.bit_depth;
    st.codec = ji.high_throughput ? "JPEG 2000 (HTJ2K)" : "JPEG 2000";
    return st;
}

void RawReader::read_rows(const int32_t* file_index, const int32_t* frame_index, const int32_t* row0, std::size_t n,
                          int nrows, uint16_t* out, int n_threads, std::size_t out_stride) {
    if (!out_stride) out_stride = static_cast<std::size_t>(nrows) * cols_;
    // group requests by file so each file is opened once
    std::map<int, std::vector<std::size_t>> by_file;
    for (std::size_t i = 0; i < n; ++i) by_file[file_index[i]].push_back(i);
    std::vector<std::pair<int, const std::vector<std::size_t>*>> groups;
    for (auto& [f, v] : by_file) {
        info(f);  // inspect serially (HDF5)
        groups.push_back({f, &v});
    }
    const std::size_t row_bytes = static_cast<std::size_t>(cols_) * 2;
    const int nt = n_threads > 0 ? n_threads : std::min(32, omp_get_max_threads());
    std::string error;
    // compressed files: one task per chunk a request touches, decoded in parallel
    struct Task {
        std::size_t i;
        int chunk;
    };
    std::vector<Task> tasks;
    for (auto& [f, v] : by_file) {
        const FileInfo& fi = *infos_[f];
        if (!fi.chunk_rows) continue;
        for (std::size_t i : v) {
            const int k = frame_index[i];
            if (k < 0 || static_cast<std::size_t>(k) >= fi.frames()) throw Error("RawReader: frame index out of range in " + files_[f]);
            const int a = std::max(0, row0[i]), b = std::min(rows_, row0[i] + nrows);
            if (a > row0[i] || b < row0[i] + nrows) std::memset(out + i * out_stride, 0, static_cast<std::size_t>(nrows) * row_bytes);
            for (int c = a / fi.chunk_rows; b > a && c <= (b - 1) / fi.chunk_rows; ++c) tasks.push_back({i, c});
        }
    }
    if (!tasks.empty()) {
#pragma omp parallel num_threads(nt)
        {
            OpenFile of;
            std::vector<std::uint8_t> raw;
            std::vector<uint16_t> tile;
#pragma omp for schedule(dynamic, 1)
            for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(tasks.size()); ++t) {
                try {
                    const std::size_t i = tasks[t].i;
                    const int f = file_index[i], c = tasks[t].chunk;
                    const FileInfo& fi = *infos_[f];
                    const std::size_t at = static_cast<std::size_t>(frame_index[i]) * fi.chunks_per_frame + c;
                    tile.resize(static_cast<std::size_t>(fi.chunk_rows) * cols_);
                    decode_chunk(of.get(f, files_[f]), files_[f], fi.chunk_offset[at], fi.chunk_bytes[at], fi.chunk_rows, cols_, raw, tile.data());
                    const int a = std::max({0, row0[i], c * fi.chunk_rows}), b = std::min({rows_, row0[i] + nrows, (c + 1) * fi.chunk_rows});
                    if (b > a)
                        std::memcpy(out + i * out_stride + static_cast<std::size_t>(a - row0[i]) * cols_,
                                    tile.data() + static_cast<std::size_t>(a - c * fi.chunk_rows) * cols_, static_cast<std::size_t>(b - a) * row_bytes);
                } catch (const std::exception& e) {
#pragma omp critical(helirec_reader_error)
                    if (error.empty()) error = e.what();
                }
            }
        }
        if (!error.empty()) throw Error(error);
    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
    for (std::ptrdiff_t g = 0; g < static_cast<std::ptrdiff_t>(groups.size()); ++g) {
        const int f = groups[g].first;
        const std::string& path = files_[f];
        if (infos_[f]->chunk_rows) continue;  // done above
        try {
            const int fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) throw Error("RawReader: cannot open " + path);
            const FileInfo& fi = *infos_[f];
            try {
                for (std::size_t i : *groups[g].second) {
                    const int k = frame_index[i];
                    if (k < 0 || static_cast<std::size_t>(k) >= fi.frame_offset.size())
                        throw Error("RawReader: frame index out of range in " + path);
                    uint16_t* dst = out + i * out_stride;
                    const int r0 = row0[i];
                    const int a = std::max(0, r0), b = std::min(rows_, r0 + nrows);
                    if (a > r0 || b < r0 + nrows) std::memset(dst, 0, static_cast<std::size_t>(nrows) * row_bytes);
                    if (b > a)
                        pread_all(fd, dst + static_cast<std::size_t>(a - r0) * cols_, static_cast<std::size_t>(b - a) * row_bytes,
                                  fi.frame_offset[k] + static_cast<std::int64_t>(a) * row_bytes, path);
                }
            } catch (...) {
                ::close(fd);
                throw;
            }
            ::close(fd);
        } catch (const std::exception& e) {
#pragma omp critical(helirec_reader_error)
            if (error.empty()) error = e.what();
        }
    }
    if (!error.empty()) throw Error(error);
}

void RawReader::visit_frames(const int32_t* file_index, const int32_t* frame_index, std::size_t n, int nrows,
                             const std::function<void(std::size_t, int, int, const uint16_t*)>& visit, int n_threads) {
    if (nrows < 1) throw Error("RawReader: visit_frames needs nrows >= 1");
    std::map<int, std::vector<std::size_t>> by_file;
    for (std::size_t i = 0; i < n; ++i) by_file[file_index[i]].push_back(i);
    std::vector<std::pair<int, const std::vector<std::size_t>*>> groups;
    for (auto& [f, v] : by_file) {
        info(f);  // inspect serially (HDF5)
        groups.push_back({f, &v});
    }
    const std::size_t row_bytes = static_cast<std::size_t>(cols_) * 2;
    const int nt = n_threads > 0 ? n_threads : std::min(32, omp_get_max_threads());
    std::string error;
    // compressed files: one task per frame, the whole frame decoded and handed over in pieces
    std::vector<std::size_t> packed;
    for (auto& [f, v] : by_file)
        if (infos_[f]->chunk_rows) {
            for (std::size_t i : v)
                if (frame_index[i] < 0 || static_cast<std::size_t>(frame_index[i]) >= infos_[f]->frames()) throw Error("RawReader: frame index out of range in " + files_[f]);
            packed.insert(packed.end(), v.begin(), v.end());
        }
    if (!packed.empty()) {
#pragma omp parallel num_threads(nt)
        {
            OpenFile of;
            std::vector<std::uint8_t> raw;
            std::vector<uint16_t> frame;
#pragma omp for schedule(dynamic, 1)
            for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(packed.size()); ++t) {
                try {
                    const std::size_t i = packed[t];
                    const int f = file_index[i];
                    const FileInfo& fi = *infos_[f];
                    frame.resize(static_cast<std::size_t>(fi.chunks_per_frame) * fi.chunk_rows * cols_);
                    for (int c = 0; c < fi.chunks_per_frame; ++c) {
                        const std::size_t at = static_cast<std::size_t>(frame_index[i]) * fi.chunks_per_frame + c;
                        decode_chunk(of.get(f, files_[f]), files_[f], fi.chunk_offset[at], fi.chunk_bytes[at], fi.chunk_rows, cols_, raw,
                                     frame.data() + static_cast<std::size_t>(c) * fi.chunk_rows * cols_);
                    }
                    for (int r0 = 0; r0 < rows_; r0 += nrows) visit(i, r0, std::min(nrows, rows_ - r0), frame.data() + static_cast<std::size_t>(r0) * cols_);
                } catch (const std::exception& e) {
#pragma omp critical(helirec_reader_error)
                    if (error.empty()) error = e.what();
                }
            }
        }
        if (!error.empty()) throw Error(error);
    }
#pragma omp parallel num_threads(nt)
    {
        std::vector<uint16_t> buf(static_cast<std::size_t>(nrows) * cols_);
#pragma omp for schedule(dynamic, 1)
        for (std::ptrdiff_t g = 0; g < static_cast<std::ptrdiff_t>(groups.size()); ++g) {
            const int f = groups[g].first;
            const std::string& path = files_[f];
            if (infos_[f]->chunk_rows) continue;  // done above
            try {
                const int fd = ::open(path.c_str(), O_RDONLY);
                if (fd < 0) throw Error("RawReader: cannot open " + path);
                const FileInfo& fi = *infos_[f];
                try {
                    for (std::size_t i : *groups[g].second) {
                        const int k = frame_index[i];
                        if (k < 0 || static_cast<std::size_t>(k) >= fi.frame_offset.size())
                            throw Error("RawReader: frame index out of range in " + path);
                        for (int r0 = 0; r0 < rows_; r0 += nrows) {
                            const int nr = std::min(nrows, rows_ - r0);
                            pread_all(fd, buf.data(), static_cast<std::size_t>(nr) * row_bytes,
                                      fi.frame_offset[k] + static_cast<std::int64_t>(r0) * row_bytes, path);
                            visit(i, r0, nr, buf.data());
                        }
                    }
                } catch (...) {
                    ::close(fd);
                    throw;
                }
                ::close(fd);
            } catch (const std::exception& e) {
#pragma omp critical(helirec_reader_error)
                if (error.empty()) error = e.what();
            }
        }
    }
    if (!error.empty()) throw Error(error);
}

void RawReader::read_frames(const int32_t* file_index, const int32_t* frame_index, std::size_t n, uint16_t* out,
                            int n_threads) {
    std::vector<int32_t> row0(n, 0);
    read_rows(file_index, frame_index, row0.data(), n, rows_, out, n_threads);
}

}  // namespace helirec
