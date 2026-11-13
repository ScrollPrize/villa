#include "helirec/raw_reader.hpp"

#include <fcntl.h>
#include <hdf5.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "io/j2k_chunk.hpp"

namespace helirec {

struct RawReader::FileInfo {
    std::vector<std::int64_t> frame_offset;  // byte offset of each frame in the file (plain values)
    // Compressed frames (io/j2k_chunk.hpp): chunks of chunk_rows rows, each one JPEG 2000 codestream.
    int chunk_rows = 0;                      // 0: plain values. Rows of a chunk as read (at the reader's resolution)
    int stored_chunk_rows = 0;
    int chunks_per_frame = 0;
    std::vector<std::int64_t> chunk_offset;  // (frame, chunk)
    std::vector<std::uint32_t> chunk_bytes;
    std::size_t frames() const { return chunk_rows ? chunk_offset.size() / static_cast<std::size_t>(chunks_per_frame) : frame_offset.size(); }
};

struct RawReader::Sync {
    std::atomic<std::uint64_t> bytes_read{0};
    std::mutex hdf5;  // libhdf5 (serial build) is not thread safe
};

// Decoded chunks of compressed frames, least recently used first out
struct RawReader::Cache {
    struct Buffer {
        std::vector<std::uint16_t> u16;
        std::vector<float> f32;
        Buffer(std::size_t n, int bytes) { if (bytes == 2) u16.resize(n); else f32.resize(n); }
        void* data() { return u16.empty() ? static_cast<void*>(f32.data()) : u16.data(); }
        const void* data() const { return u16.empty() ? static_cast<const void*>(f32.data()) : u16.data(); }
        std::size_t bytes() const { return u16.size() * 2 + f32.size() * 4; }
    };
    using Tile = std::shared_ptr<const Buffer>;
    std::mutex mutex;
    std::size_t limit = 0, bytes = 0;
    std::uint64_t hits = 0, decoded = 0;
    std::list<std::uint64_t> order;  // most recent first
    std::unordered_map<std::uint64_t, std::pair<Tile, std::list<std::uint64_t>::iterator>> held;
    static std::uint64_t key(int file, int frame, int chunk) {
        return (static_cast<std::uint64_t>(file) << 40) | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(frame)) << 8) | static_cast<std::uint64_t>(chunk);
    }
    Tile get(std::uint64_t k) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = held.find(k);
        if (it == held.end()) return nullptr;
        order.splice(order.begin(), order, it->second.second);
        ++hits;
        return it->second.first;
    }
    void put(std::uint64_t k, Tile tile) {
        std::lock_guard<std::mutex> lock(mutex);
        ++decoded;
        const std::size_t size = tile->bytes();
        if (!limit || size > limit || held.count(k)) return;
        order.push_front(k);
        held.emplace(k, std::make_pair(std::move(tile), order.begin()));
        bytes += size;
        while (bytes > limit) {
            const auto last = held.find(order.back());
            bytes -= last->second.first->bytes();
            held.erase(last);
            order.pop_back();
        }
    }
};

// Chunks announced and not yet in the cache, and the workers that decode them
struct RawReader::Ahead {
    struct Job {
        int file, frame, chunk;
    };
    std::mutex mutex;
    std::condition_variable work, done;
    std::deque<Job> queue;
    std::unordered_set<std::uint64_t> pending;  // queued or being decoded
    std::vector<std::thread> workers;
    bool stop = false;
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

bool htj2k_float_frames_supported() { return j2k::openjph_available(); }
bool jpeg2000_frames_supported() { return j2k::available(); }
std::vector<std::uint8_t> jpeg2000_chunk(const std::uint16_t* values, int rows, int cols) { return j2k::encode_chunk(values, cols, rows); }

RawReader::RawReader(std::vector<std::string> files, std::string dataset, int rows, int cols, int reduce, std::string dtype, bool selective, std::string decoder)
    : dtype_(std::move(dtype)), sample_bytes_(dtype_ == "float32" ? 4 : 2), selective_(selective), use_openjph_(decoder == "openjph" || (decoder == "auto" && sample_bytes_ == 4)), files_(std::move(files)), dataset_(std::move(dataset)), rows_(rows), cols_(cols), stored_rows_(rows), stored_cols_(cols), reduce_(reduce),
      sync_(std::make_unique<Sync>()), cache_(std::make_unique<Cache>()) {
    if (decoder != "auto" && decoder != "openjph") throw Error("RawReader: decoder must be auto or openjph");
    if (dtype_ != "uint16" && dtype_ != "float32") throw Error("RawReader: dtype must be uint16 or float32");
    if (rows < 1 || cols < 1) throw Error("RawReader: rows and cols must be positive");
    if (reduce < 0 || reduce > 3) throw Error("RawReader: reduce must be 0 (as stored), 1 (a half), 2 (a quarter) or 3 (an eighth)");
    if (sample_bytes_ == 2 && (rows % (1 << reduce) || cols % (1 << reduce)))
        throw Error("RawReader: frames of " + std::to_string(rows) + " x " + std::to_string(cols) + " cannot be read at 1/" + std::to_string(1 << reduce) + " (not multiples of it)");
    rows_ = 1 + ((rows - 1) >> reduce), cols_ = 1 + ((cols - 1) >> reduce);
    infos_.resize(files_.size());
    static std::once_flag quiet;
    std::call_once(quiet, [] { H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); });
}

RawReader::~RawReader() { set_prefetch_threads(0); }

void RawReader::set_cache_bytes(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(cache_->mutex);
    cache_->limit = bytes;
    while (cache_->bytes > cache_->limit && !cache_->order.empty()) {
        const auto last = cache_->held.find(cache_->order.back());
        cache_->bytes -= last->second.first->bytes();
        cache_->held.erase(last);
        cache_->order.pop_back();
    }
}
void RawReader::set_fetched(int file, const std::vector<std::int64_t>& offsets, const std::vector<std::int64_t>& lengths) {
    if (file < 0 || static_cast<std::size_t>(file) >= files_.size()) throw Error("RawReader: bad file index");
    if (offsets.size() != lengths.size()) throw Error("RawReader: set_fetched needs one length per offset");
    std::vector<std::pair<std::int64_t, std::int64_t>> r;
    r.reserve(offsets.size());
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        if (offsets[i] < 0 || lengths[i] < 0) throw Error("RawReader: negative fetched range");
        if (lengths[i]) r.emplace_back(offsets[i], offsets[i] + lengths[i]);
    }
    std::sort(r.begin(), r.end());
    std::vector<std::pair<std::int64_t, std::int64_t>> merged;
    for (const auto& x : r)
        if (!merged.empty() && x.first <= merged.back().second) merged.back().second = std::max(merged.back().second, x.second);
        else merged.push_back(x);
    if (fetched_.size() != files_.size()) fetched_.resize(files_.size());
    fetched_[file] = std::move(merged);
}

void RawReader::require_fetched(int file, std::int64_t offset, std::size_t n) const {
    if (fetched_.empty() || fetched_[file].empty()) return;
    const auto& r = fetched_[file];
    const std::int64_t end = offset + static_cast<std::int64_t>(n);
    auto it = std::upper_bound(r.begin(), r.end(), std::make_pair(offset, std::numeric_limits<std::int64_t>::max()));
    if (it == r.begin() || std::prev(it)->second < end)
        throw Error("RawReader: bytes [" + std::to_string(offset) + ", " + std::to_string(end) + ") of " + files_[file] +
                    " were not fetched");
}

std::uint64_t RawReader::bytes_read() const { return sync_->bytes_read.load(std::memory_order_relaxed); }
void RawReader::cache_counts(std::uint64_t& hits, std::uint64_t& decoded) const {
    std::lock_guard<std::mutex> lock(cache_->mutex);
    hits = cache_->hits, decoded = cache_->decoded;
}

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
    if (H5Tequal(type.id, sample_bytes_ == 2 ? H5T_STD_U16LE : H5T_IEEE_F32LE) <= 0)
        throw Error("RawReader: " + path + ": dataset is not " + dtype_ + " LE");
    H5Handle space(H5Dget_space(d.id), H5Sclose);
    hsize_t dims[3] = {0, 0, 0};
    if (H5Sget_simple_extent_ndims(space.id) != 3) throw Error("RawReader: " + path + ": dataset is not 3D");
    H5Sget_simple_extent_dims(space.id, dims, nullptr);
    if (static_cast<int>(dims[1]) != stored_rows_ || static_cast<int>(dims[2]) != stored_cols_)
        throw Error("RawReader: " + path + ": unexpected frame shape");
    H5Handle plist(H5Dget_create_plist(d.id), H5Pclose);
    auto fi = std::make_unique<FileInfo>();
    fi->frame_offset.resize(dims[0], -1);
    const std::int64_t frame_bytes = static_cast<std::int64_t>(rows_) * cols_ * sample_bytes_;
    const H5D_layout_t layout = H5Pget_layout(plist.id);
    if (reduce_ && !(layout == H5D_CHUNKED && H5Pget_nfilters(plist.id) != 0))
        throw Error("RawReader: " + path + ": a reduced resolution is read from frames stored as JPEG 2000 only; these are plain values");
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
        if (cdims[0] != 1 || cdims[1] < 1 || static_cast<int>(cdims[2]) != stored_cols_)
            throw Error("RawReader: " + path + ": compressed chunks must be whole rows of one frame");
        if (cdims[1] % (1u << reduce_) && (sample_bytes_ == 2 || cdims[1] != dims[1]))
            throw Error("RawReader: " + path + ": chunks of " + std::to_string(cdims[1]) + " rows cannot be read at 1/" + std::to_string(1 << reduce_));
        if (use_openjph_ && !j2k::openjph_available())
            throw Error("RawReader: HTJ2K with OpenJPH needs a build with HELIREC_OPENJPH=ON");
        if (!use_openjph_ && !j2k::available())
            throw Error("RawReader: " + path + ": the frames are stored as JPEG 2000; build helirec with OpenJPEG (libopenjp2 >= 2.5: apt install libopenjp2-7-dev, brew install openjpeg)");
        fi->stored_chunk_rows = static_cast<int>(cdims[1]);
        fi->chunk_rows = 1 + ((fi->stored_chunk_rows - 1) >> reduce_);
        fi->chunks_per_frame = (stored_rows_ + fi->stored_chunk_rows - 1) / fi->stored_chunk_rows;
        fi->chunk_offset.resize(dims[0] * fi->chunks_per_frame);
        fi->chunk_bytes.resize(dims[0] * fi->chunks_per_frame);
#if H5_VERSION_GE(1, 14, 0)
        // every chunk in one pass over the dataset's index (asking for them one by one is a search each: 1900 a
        // file of 100 frames, and a slab of a reconstruction opens a hundred files)
        struct Walk {
            FileInfo* fi;
            hsize_t frames;
            std::int64_t unfiltered = -1;
        } walk{fi.get(), dims[0]};
        std::fill(fi->chunk_offset.begin(), fi->chunk_offset.end(), std::int64_t{-1});
        const H5D_chunk_iter_op_t each = [](const hsize_t* off, unsigned mask, haddr_t addr, hsize_t size, void* data) -> int {
            Walk& w = *static_cast<Walk*>(data);
            const hsize_t c = off[1] / static_cast<hsize_t>(w.fi->stored_chunk_rows);
            if (off[0] >= w.frames || c >= static_cast<hsize_t>(w.fi->chunks_per_frame) || addr == HADDR_UNDEF) return 0;
            if (mask != 0 && w.unfiltered < 0) w.unfiltered = static_cast<std::int64_t>(off[0]);
            w.fi->chunk_offset[off[0] * w.fi->chunks_per_frame + c] = static_cast<std::int64_t>(addr);
            w.fi->chunk_bytes[off[0] * w.fi->chunks_per_frame + c] = static_cast<std::uint32_t>(size);
            return 0;
        };
        if (H5Dchunk_iter(d.id, H5P_DEFAULT, each, &walk) < 0) throw Error("RawReader: " + path + ": the chunks cannot be listed");
        if (walk.unfiltered >= 0) throw Error("RawReader: " + path + ": a chunk of frame " + std::to_string(walk.unfiltered) + " was stored without its filter");
        for (std::size_t at = 0; at < fi->chunk_offset.size(); ++at)
            if (fi->chunk_offset[at] < 0) throw Error("RawReader: " + path + ": frame " + std::to_string(at / fi->chunks_per_frame) + " not written");
#else
        for (hsize_t k = 0; k < dims[0]; ++k)
            for (int c = 0; c < fi->chunks_per_frame; ++c) {
                hsize_t off[3] = {k, static_cast<hsize_t>(c) * fi->stored_chunk_rows, 0};
                unsigned mask = 0;
                haddr_t addr = HADDR_UNDEF;
                hsize_t size = 0;
                if (H5Dget_chunk_info_by_coord(d.id, off, &mask, &addr, &size) < 0 || addr == HADDR_UNDEF)
                    throw Error("RawReader: " + path + ": frame " + std::to_string(k) + " not written");
                if (mask != 0) throw Error("RawReader: " + path + ": a chunk of frame " + std::to_string(k) + " was stored without its filter");
                fi->chunk_offset[k * fi->chunks_per_frame + c] = static_cast<std::int64_t>(addr);
                fi->chunk_bytes[k * fi->chunks_per_frame + c] = static_cast<std::uint32_t>(size);
            }
#endif
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

void RawReader::decode_chunk(int fd, int file, const FileInfo& fi, std::size_t at,
                             std::vector<std::uint8_t>& raw, void* tile) {
    const auto bytes = fi.chunk_bytes[at];
    const auto nbytes = static_cast<std::size_t>(fi.stored_chunk_rows) * stored_cols_ * sample_bytes_;
    const j2k::ReadAt read = [&](void* dst, std::size_t n, std::size_t offset) {
        if (offset > bytes || n > bytes - offset) throw Error("RawReader: read outside compressed chunk");
        require_fetched(file, fi.chunk_offset[at] + static_cast<std::int64_t>(offset), n);
        pread_all(fd, dst, n, fi.chunk_offset[at] + offset, files_[file]);
        sync_->bytes_read.fetch_add(n, std::memory_order_relaxed);
    };
    if (use_openjph_) {
        j2k::decode_htj2k(read, bytes, nbytes, sample_bytes_, stored_cols_, fi.stored_chunk_rows, reduce_, tile, selective_);
    } else {
        raw.resize(bytes);
        read(raw.data(), bytes, 0);
        j2k::decode(j2k::codestream_of_chunk(raw.data(), bytes, nbytes), cols_, fi.chunk_rows, static_cast<std::uint16_t*>(tile), reduce_);
    }
}

void RawReader::set_prefetch_threads(int threads) {
    if (ahead_) {
        {
            std::lock_guard<std::mutex> lock(ahead_->mutex);
            ahead_->stop = true;
        }
        ahead_->work.notify_all();
        for (std::thread& t : ahead_->workers) t.join();
        ahead_->done.notify_all();
        ahead_.reset();
    }
    if (threads <= 0) return;
    ahead_ = std::make_unique<Ahead>();
    Ahead* const A = ahead_.get();
    for (int w = 0; w < threads; ++w)
        A->workers.emplace_back([this, A] {
            OpenFile of;
            std::vector<std::uint8_t> raw;
            for (;;) {
                Ahead::Job job;
                {
                    std::unique_lock<std::mutex> lock(A->mutex);
                    A->work.wait(lock, [A] { return A->stop || !A->queue.empty(); });
                    if (A->stop) return;
                    job = A->queue.front();
                    A->queue.pop_front();
                }
                const std::uint64_t key = Cache::key(job.file, job.frame, job.chunk);
                try {
                    const FileInfo& fi = *infos_[job.file];
                    const std::size_t at = static_cast<std::size_t>(job.frame) * fi.chunks_per_frame + job.chunk;
                    auto fresh = std::make_shared<Cache::Buffer>(static_cast<std::size_t>(fi.chunk_rows) * cols_, sample_bytes_);
                    decode_chunk(of.get(job.file, files_[job.file]), job.file, fi, at, raw, fresh->data());
                    cache_->put(key, fresh);
                } catch (const std::exception&) {
                    // the read that needs the chunk decodes it again and reports what is wrong with it
                }
                {
                    std::lock_guard<std::mutex> lock(A->mutex);
                    A->pending.erase(key);
                }
                A->done.notify_all();
            }
        });
}

void RawReader::prefetch_rows(const int32_t* file_index, const int32_t* frame_index, const int32_t* row0, std::size_t n, int nrows) {
    if (!ahead_) return;
    {
        std::lock_guard<std::mutex> lock(cache_->mutex);
        if (!cache_->limit) return;
    }
    try {
        std::vector<Ahead::Job> jobs;
        for (std::size_t i = 0; i < n; ++i) {
            const int f = file_index[i];
            const FileInfo& fi = info(f);
            if (!fi.chunk_rows) continue;  // plain frames are read, not decoded
            const int k = frame_index[i];
            if (k < 0 || static_cast<std::size_t>(k) >= fi.frames()) continue;
            const int a = std::max(0, row0[i]), b = std::min(rows_, row0[i] + nrows);
            for (int c = a / fi.chunk_rows; b > a && c <= (b - 1) / fi.chunk_rows; ++c) jobs.push_back({f, k, c});
        }
        std::size_t added = 0;
        for (const Ahead::Job& j : jobs) {
            const std::uint64_t key = Cache::key(j.file, j.frame, j.chunk);
            {
                std::lock_guard<std::mutex> lock(cache_->mutex);
                if (cache_->held.count(key)) continue;
            }
            std::lock_guard<std::mutex> lock(ahead_->mutex);
            if (ahead_->pending.insert(key).second) {
                ahead_->queue.push_back(j);
                ++added;
            }
        }
        if (added) ahead_->work.notify_all();
    } catch (const std::exception&) {
        // a file that cannot be inspected: the read reports it
    }
}

RawReader::Storage RawReader::storage(int file) {
    const FileInfo& fi = info(file);
    Storage st;
    st.frames = static_cast<std::int64_t>(fi.frames());
    st.bytes_per_frame = static_cast<double>(stored_rows_) * stored_cols_ * sample_bytes_;
    st.bit_depth = sample_bytes_ * 8;
    if (!fi.chunk_rows || fi.chunk_offset.empty()) return st;
    st.compressed = true;
    st.chunk_rows = fi.stored_chunk_rows;
    double total = 0;
    for (std::uint32_t b : fi.chunk_bytes) total += b;
    st.bytes_per_frame = total / static_cast<double>(st.frames);
    const std::string& path = files_[file];
    OpenFile of;
    const int fd = of.get(file, path);
    const j2k::ReadAt read = [&](void* dst, std::size_t n, std::size_t offset) {
        require_fetched(file, fi.chunk_offset[0] + static_cast<std::int64_t>(offset), n);
        pread_all(fd, dst, n, fi.chunk_offset[0] + offset, path);
        sync_->bytes_read.fetch_add(n, std::memory_order_relaxed);
    };
    const auto h = j2k::chunk_header(read, fi.chunk_bytes[0], static_cast<std::size_t>(fi.stored_chunk_rows) * stored_cols_ * sample_bytes_, sample_bytes_);
    if (h.constant) { st.codec = "B2JPF32 (constant)"; return st; }
    const auto ji = j2k::info(read, h);
    st.reversible = ji.reversible;
    st.bit_depth = ji.bit_depth;
    st.codec = ji.high_throughput ? "JPEG 2000 (HTJ2K)" : "JPEG 2000";
    return st;
}

void RawReader::read_rows(const int32_t* files, const int32_t* frames, const int32_t* row0, std::size_t n,
                          int nrows, uint16_t* out, int threads, std::size_t stride) {
    if (sample_bytes_ != 2) throw Error("RawReader: output type does not match reader dtype");
    read_rows_impl(files, frames, row0, n, nrows, out, threads, stride);
}

void RawReader::read_rows(const int32_t* files, const int32_t* frames, const int32_t* row0, std::size_t n,
                          int nrows, float* out, int threads, std::size_t stride) {
    if (sample_bytes_ != 4) throw Error("RawReader: output type does not match reader dtype");
    read_rows_impl(files, frames, row0, n, nrows, out, threads, stride);
}

void RawReader::read_rows_impl(const int32_t* file_index, const int32_t* frame_index, const int32_t* row0, std::size_t n,
                          int nrows, void* output, int n_threads, std::size_t out_stride) {
    if (nrows < 0) throw Error("RawReader: nrows must be nonnegative");
    if (nrows == 0 || n == 0) return;
    if (out_stride && out_stride < static_cast<std::size_t>(nrows) * cols_) throw Error("RawReader: output stride is too small");
    auto* out = static_cast<std::uint8_t*>(output);
    if (!out_stride) out_stride = static_cast<std::size_t>(nrows) * cols_;
    out_stride *= sample_bytes_;
    // group requests by file so each file is opened once
    std::map<int, std::vector<std::size_t>> by_file;
    for (std::size_t i = 0; i < n; ++i) by_file[file_index[i]].push_back(i);
    std::vector<std::pair<int, const std::vector<std::size_t>*>> groups;
    for (auto& [f, v] : by_file) {
        info(f);  // inspect serially (HDF5)
        groups.push_back({f, &v});
    }
    const std::size_t row_bytes = static_cast<std::size_t>(cols_) * sample_bytes_;
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
#pragma omp for schedule(dynamic, 1)
            for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(tasks.size()); ++t) {
                try {
                    const std::size_t i = tasks[t].i;
                    const int f = file_index[i], c = tasks[t].chunk;
                    const FileInfo& fi = *infos_[f];
                    const std::size_t at = static_cast<std::size_t>(frame_index[i]) * fi.chunks_per_frame + c;
                    const std::uint64_t key = Cache::key(f, frame_index[i], c);
                    Cache::Tile tile = cache_->get(key);
                    if (!tile && ahead_) {  // being decoded ahead: wait for it rather than decode it twice
                        std::unique_lock<std::mutex> lock(ahead_->mutex);
                        if (ahead_->pending.count(key)) {
                            ahead_->done.wait(lock, [&] { return !ahead_->pending.count(key); });
                            lock.unlock();
                            tile = cache_->get(key);
                        }
                    }
                    if (!tile) {
                        auto fresh = std::make_shared<Cache::Buffer>(static_cast<std::size_t>(fi.chunk_rows) * cols_, sample_bytes_);
                        decode_chunk(of.get(f, files_[f]), f, fi, at, raw, fresh->data());
                        tile = fresh;
                        cache_->put(key, tile);
                    }
                    const int a = std::max({0, row0[i], c * fi.chunk_rows}), b = std::min({rows_, row0[i] + nrows, (c + 1) * fi.chunk_rows});
                    if (b > a)
                        std::memcpy(out + i * out_stride + static_cast<std::size_t>(a - row0[i]) * row_bytes,
                                    static_cast<const std::uint8_t*>(tile->data()) + static_cast<std::size_t>(a - c * fi.chunk_rows) * row_bytes, static_cast<std::size_t>(b - a) * row_bytes);
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
                    std::uint8_t* dst = out + i * out_stride;
                    const int r0 = row0[i];
                    const int a = std::max(0, r0), b = std::min(rows_, r0 + nrows);
                    if (a > r0 || b < r0 + nrows) std::memset(dst, 0, static_cast<std::size_t>(nrows) * row_bytes);
                    if (b > a) {
                        pread_all(fd, dst + static_cast<std::size_t>(a - r0) * row_bytes, static_cast<std::size_t>(b - a) * row_bytes,
                                  fi.frame_offset[k] + static_cast<std::int64_t>(a) * row_bytes, path);
                        sync_->bytes_read.fetch_add(static_cast<std::size_t>(b - a) * row_bytes, std::memory_order_relaxed);
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
    if (!error.empty()) throw Error(error);
}

void RawReader::visit_frames(const int32_t* file_index, const int32_t* frame_index, std::size_t n, int nrows,
                             const std::function<void(std::size_t, int, int, const uint16_t*)>& visit, int n_threads) {
    if (sample_bytes_ != 2) throw Error("RawReader: uint16 visit_frames cannot read float32 data; use read_rows");
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
                        decode_chunk(of.get(f, files_[f]), f, fi, at, raw,
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
                            sync_->bytes_read.fetch_add(static_cast<std::size_t>(nr) * row_bytes, std::memory_order_relaxed);
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

void RawReader::read_frames(const int32_t* files, const int32_t* frames, std::size_t n, float* out, int threads) {
    std::vector<int32_t> row0(n, 0);
    read_rows(files, frames, row0.data(), n, rows_, out, threads);
}

}  // namespace helirec
