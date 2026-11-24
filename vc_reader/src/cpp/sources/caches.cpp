#include "vc_reader/source_reader.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include "../lru.hpp"
#include <system_error>
#include <thread>
#include <unordered_map>

#include "../disk_cache.hpp"
#include "../output_pool.hpp"
#include "../reader_types.hpp"
#include "../sigv4.hpp"

namespace fs = std::filesystem;

namespace vcr::sources {

// ---- blocks ----
std::string canonicalDtype(std::string_view dtype)
{
    static const std::map<std::string_view, const char*> names{
        {"bool", "|b1"}, {"uint8", "|u1"}, {"int8", "|i1"}, {"uint16", "<u2"}, {"int16", "<i2"}, {"uint32", "<u4"}, {"int32", "<i4"},
        {"uint64", "<u8"}, {"int64", "<i8"}, {"float32", "<f4"}, {"float64", "<f8"}};
    if (const auto it = names.find(dtype); it != names.end()) return it->second;
    if (dtype.size() == 3 && std::string_view("<>|=").find(dtype[0]) != std::string_view::npos)
        return parseDtype(dtype).str();
    throw std::invalid_argument("unsupported dtype '" + std::string(dtype) + "'");
}

std::string dtypeName(std::string_view typestr)
{
    const auto canonical = canonicalDtype(typestr);
    if (canonical[0] == '>') return canonical;
    if (canonical[1] == 'b') return "bool";
    const char* kind = canonical[1] == 'u' ? "uint" : canonical[1] == 'i' ? "int" : "float";
    return kind + std::to_string((canonical[2] - '0') * 8);
}

Block Block::allocate(std::vector<std::int64_t> shape, std::string dtype)
{
    Block block;
    block.dtype = std::move(dtype);
    std::size_t bytes = block.dtype[2] - '0';
    for (const auto n : shape) bytes *= std::size_t(n);
    block.shape = std::move(shape);
    block.nbytes = bytes;
    auto* memory = static_cast<std::uint8_t*>(OutputPool::instance().acquire(bytes ? bytes : 1));
    block.data = std::shared_ptr<std::uint8_t[]>(memory, [](std::uint8_t* p) { OutputPool::instance().release(p); });
    return block;
}
std::size_t Block::itemsize() const { return std::size_t(dtype[2] - '0'); }

// ---- decoded cache ----
struct DecodedCache::State {
    long pid = platform::process_id();
    std::recursive_mutex mutex;
    explicit State(std::size_t limit) : entries(limit) {}
    Lru<Block> entries;
    std::size_t hits = 0, misses = 0, peakWorking = 0, working = 0;
};
struct DecodedCache::Lease::Impl {
    std::shared_ptr<State> state;
    std::unique_lock<std::recursive_mutex> lock;
    std::size_t bytes = 0;
};
DecodedCache::Lease::Lease(Lease&& other) noexcept = default;
DecodedCache::Lease& DecodedCache::Lease::operator=(Lease&& other) noexcept
{
    release();
    impl_ = std::move(other.impl_);
    return *this;
}
DecodedCache::Lease::~Lease() { release(); }
void DecodedCache::Lease::release()
{
    if (impl_) {
        impl_->state->working -= impl_->bytes;
        impl_.reset();
    }
}

DecodedCache::DecodedCache(std::size_t maxBytes, std::size_t maxDecodeBytes) : maxBytes_(maxBytes), maxDecodeBytes_(maxDecodeBytes), state_(std::make_shared<State>(maxBytes)) {}

std::shared_ptr<DecodedCache::State> DecodedCache::state()
{
    std::lock_guard guard(stateMutex_);
    if (state_->pid != platform::process_id()) {
        // The inherited lock may have been held across fork: leave it alone.
        new std::shared_ptr<State>(state_);
        state_ = std::make_shared<State>(maxBytes_);
    }
    return state_;
}

DecodedCache::Stats DecodedCache::stats()
{
    const auto s = state();
    std::lock_guard lock(s->mutex);
    return {s->entries.bytes(), s->entries.size(), s->hits, s->misses, s->peakWorking};
}

void DecodedCache::clear()
{
    const auto s = state();
    std::lock_guard lock(s->mutex);
    s->entries.clear();
}

std::pair<Block, DecodedCache::Lease> DecodedCache::use(const std::string& key, std::size_t decodeBytes, std::size_t scratchBytes, const Load& load)
{
    const auto s = state();
    Lease lease;
    lease.impl_ = std::make_unique<Lease::Impl>();
    lease.impl_->state = s;
    lease.impl_->lock = std::unique_lock(s->mutex);
    const auto* found = s->entries.peek(key);
    const bool cached = found != nullptr;
    const std::size_t needed = scratchBytes + (cached ? 0 : decodeBytes);
    if (needed + s->working > maxDecodeBytes_)
        throw DecodeBudgetError("decode working set needs " + std::to_string(needed) + " bytes; max_decode_bytes=" + std::to_string(maxDecodeBytes_) +
                                ". Increase the limit or use a chunked source.");
    s->peakWorking = std::max(s->peakWorking, needed + s->working);
    s->working += needed;
    lease.impl_->bytes = needed;
    Block block;
    if (cached) {
        ++s->hits;
        block = *s->entries.get(key);
    } else {
        ++s->misses;
        block = load();
        if (block.nbytes > decodeBytes) throw ImageFormatError("decoder returned more bytes than its declared working set");
        if (block.nbytes > 0) s->entries.put(key, block, block.nbytes);
    }
    return {std::move(block), std::move(lease)};
}

DecodedCache::Lease DecodedCache::working(std::size_t nbytes)
{
    const auto s = state();
    Lease lease;
    lease.impl_ = std::make_unique<Lease::Impl>();
    lease.impl_->state = s;
    lease.impl_->lock = std::unique_lock(s->mutex);
    if (nbytes + s->working > maxDecodeBytes_)
        throw DecodeBudgetError("read scratch needs " + std::to_string(nbytes) + " bytes; max_decode_bytes=" + std::to_string(maxDecodeBytes_) +
                                ". Increase the limit or read a smaller region.");
    s->peakWorking = std::max(s->peakWorking, nbytes + s->working);
    s->working += nbytes;
    lease.impl_->bytes = nbytes;
    return lease;
}

// ---- encoded files ----
OptionsFor defaultOptions()
{
    return [](const std::string& location) {
        zarr::OpenOptions options;
        options.access = zarr::resolveAccess(location);
        return options;
    };
}

EncodedFiles::Opened::~Opened()
{
    if (fd >= 0) ::close(fd);
}

EncodedFiles::EncodedFiles(OptionsFor options) : options_(std::move(options)), pid_(platform::process_id()) {}
EncodedFiles::~EncodedFiles()
{
    try {
        close();
    } catch (...) {
    }
}

void EncodedFiles::process()
{
    if (pid_ != platform::process_id()) {
        // A child must not unlink the parent's temporary cache.
        temporary_.clear();
        identities_.clear();
        resolved_.clear();
        bytesRead_ = cacheHits_ = 0;
        pid_ = platform::process_id();
    }
}

std::string EncodedFiles::temporaryDirectory() const { return temporary_; }

void EncodedFiles::close()
{
    std::lock_guard guard(mutex_);
    process();
    if (!temporary_.empty()) {
        std::error_code ignored;
        fs::remove_all(temporary_, ignored);
        temporary_.clear();
    }
}

std::string EncodedFiles::identity(const std::string& location, int fd)
{
    platform::FileStat st;
    if (platform::fstat(fd, st) != 0) throw std::system_error(errno, std::generic_category(), "fstat " + location);
    const auto signature = std::to_string(st.device) + ":" + std::to_string(st.inode) + ":" + std::to_string(st.size) + ":" +
                           std::to_string(st.modified.seconds) + ":" + std::to_string(st.changed.seconds);
    {
        std::lock_guard guard(mutex_);
        const auto known = identities_.find(location);
        if (known != identities_.end() && known->second.first == signature) return known->second.second;
    }
    // Bounded scratch; identities match online and offline, where no validator can be consulted.
    // Hashed outside the lock, so one file's digest does not hold up another's download.
    Sha256 hash;
    std::vector<char> block(1 << 16);
    std::int64_t at = 0;
    for (;;) {
        const auto n = platform::pread(fd, block.data(), block.size(), at);
        if (n < 0) throw std::system_error(errno, std::generic_category(), "read " + location);
        if (n == 0) break;
        hash.update(block.data(), std::size_t(n));
        at += n;
    }
    static constexpr char hex[] = "0123456789abcdef";
    std::string token = location + "|";
    for (const auto byte : hash.finish()) {
        token += hex[byte >> 4];
        token += hex[byte & 15];
    }
    std::lock_guard guard(mutex_);
    identities_[location] = {signature, token};
    return token;
}

zarr::OpenOptions EncodedFiles::resolve(const std::string& location, std::vector<fs::path>& roots)
{
    std::lock_guard guard(mutex_);
    process();
    auto known = resolved_.find(location);
    if (known == resolved_.end()) known = resolved_.emplace(location, options_(location)).first;
    const auto& options = known->second;
    if (std::find(options.cacheDirs.begin(), options.cacheDirs.end(), "vc3d") != options.cacheDirs.end())
        throw std::invalid_argument("image files need explicit cache directories, not the vc3d chunk-cache alias");
    for (const auto& dir : options.cacheDirs) {
        std::string path = dir;
        if (path.starts_with("~/"))
            if (const char* home = std::getenv("HOME")) path = std::string(home) + path.substr(1);
        roots.push_back(fs::weakly_canonical(path));
    }
    return options;
}

namespace {
// One pooled client per credentials and connection count, shared by every reader in
// the process, so successive range reads reuse connections.
std::shared_ptr<HttpClient> rangeClient(const Auth& auth, int connections)
{
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<HttpClient>> clients;
    const auto key = auth.kind + '\0' + auth.user + '\0' + auth.secret + '\0' + auth.token + '\0' + auth.region + '\0' +
                     std::to_string(connections);
    std::lock_guard lock(mutex);
    auto& client = clients[key];
    if (!client) {
        HttpOptions options;
        options.auth = auth;
        options.connections = connections;
        client = std::make_shared<HttpClient>(options);
    }
    return client;
}

// The total of a Content-Range answer ("bytes 0-51/7340032"); nullopt for "*".
std::optional<std::uint64_t> rangeTotal(const std::string& contentRange)
{
    const auto slash = contentRange.rfind('/');
    if (slash == std::string::npos || slash + 1 >= contentRange.size()) return std::nullopt;
    std::uint64_t total = 0;
    const auto* first = contentRange.data() + slash + 1;
    const auto* last = contentRange.data() + contentRange.size();
    const auto [end, error] = std::from_chars(first, last, total);
    if (error != std::errc() || end != last) return std::nullopt;
    return total;
}
}  // namespace

EncodedFiles::Ranges EncodedFiles::readRanges(const std::string& location,
                                              const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges,
                                              const std::string& version)
{
    Ranges out;
    out.parts.resize(ranges.size());
    const auto inPlace = [&](int fd, const std::string& where) {
        File file(fd);
        platform::FileStat st;
        if (platform::fstat(fd, st) != 0) throw std::system_error(errno, std::generic_category(), "fstat " + where);
        const auto size = static_cast<std::uint64_t>(st.size);
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            const auto [offset, length] = ranges[i];
            if (offset > size || length > size - offset)
                throw std::out_of_range(where + ": range past the end of the file");
            auto& part = out.parts[i];
            part.resize(length);
            std::uint64_t done = 0;
            while (done < length) {
                const auto n = platform::pread(fd, part.data() + done, length - done, std::int64_t(offset + done));
                if (n < 0) throw std::system_error(errno, std::generic_category(), "read " + where);
                if (n == 0) throw std::runtime_error(where + ": file shrank while it was read");
                done += std::uint64_t(n);
            }
        }
        out.size = size;
    };
    if (!zarr::isRemote(location)) {
        std::string path = location;
        if (path.starts_with("~/") || path == "~")
            if (const char* home = std::getenv("HOME")) path = std::string(home) + path.substr(1);
        const int fd = platform::open_file(path, O_RDONLY);
        if (fd < 0) {
            if (errno == ENOENT || errno == ENOTDIR) throw zarr::NotFound(path + ": no such file");
            throw fs::filesystem_error("open", path, std::error_code(errno, std::generic_category()));
        }
        inPlace(fd, path);
        return out;
    }
    std::vector<fs::path> roots;
    const auto options = resolve(location, roots);
    {
        std::lock_guard guard(mutex_);
        if (!temporary_.empty()) roots.emplace_back(temporary_);
    }
    for (const auto& root : roots) {
        const auto path = zarr::decodedCachePath(root, location);
        const int fd = platform::open_file(path, O_RDONLY);
        if (fd < 0) {
            if (errno == ENOENT || errno == ENOTDIR) continue;
            throw fs::filesystem_error("open", path, std::error_code(errno, std::generic_category()));
        }
        inPlace(fd, path.string());
        std::lock_guard guard(mutex_);
        ++cacheHits_;
        return out;
    }
    if (options.offline) throw CacheMiss(location + ": encoded file is not cached (offline)");

    const auto url = options.access.url(location);
    auto client = rangeClient(options.access.auth, std::max(1, options.connections));
    std::mutex answers;
    std::exception_ptr failure;
    bool missing = false;
    std::optional<std::string> seen = version.empty() ? std::nullopt : std::optional<std::string>(version);
    std::size_t received = 0;
    std::atomic<std::size_t> next{0};
    const auto work = [&] {
        for (std::size_t i; (i = next.fetch_add(1)) < ranges.size();) {
            {
                std::lock_guard lock(answers);
                if (failure || missing) return;
            }
            const auto [offset, length] = ranges[i];
            if (length == 0) continue;
            try {
                std::vector<std::uint8_t> body;
                const auto response = client->get_range(url, body,
                                                        std::to_string(offset) + "-" + std::to_string(offset + length - 1),
                                                        length);
                std::lock_guard lock(answers);
                if (response.status == 404) {
                    missing = true;
                    return;
                }
                if (seen && !response.etag.empty() && response.etag != *seen)
                    throw Changed(location + ": the object changed between range reads (ETag " + *seen + " then " +
                                  response.etag + ")");
                if (!seen && !response.etag.empty()) seen = response.etag;
                if (!out.size) out.size = rangeTotal(response.content_range);
                out.parts[i].assign(reinterpret_cast<const char*>(body.data()), body.size());
                received += body.size();
            } catch (...) {
                std::lock_guard lock(answers);
                if (!failure) failure = std::current_exception();
                return;
            }
        }
    };
    const auto workers = std::min<std::size_t>(ranges.size(), std::size_t(std::max(1, options.connections)));
    std::vector<std::thread> threads;
    for (std::size_t w = 1; w < workers; ++w) {
        try {
            threads.emplace_back(work);
        } catch (const std::system_error&) {
            break;  // out of threads: the ones started (and this one) read the rest
        }
    }
    work();
    for (auto& thread : threads) thread.join();
    {
        std::lock_guard guard(mutex_);
        bytesRead_ += received;
    }
    if (failure) std::rethrow_exception(failure);
    if (missing) throw zarr::NotFound(location + ": remote file does not exist");
    out.version = seen.value_or(std::string());
    return out;
}

bool EncodedFiles::chargeFile(const std::string& location, const std::string& path, std::int64_t bytes)
{
    if (!zarr::isRemote(location)) return true;
    std::vector<fs::path> roots;
    const auto options = resolve(location, roots);
    if (roots.empty() || !options.cacheWrites) return true;
    auto budget = zarr::diskBudget(roots.front().string(), options.cacheMaxBytes);
    if (options.cacheMinFreeBytes) budget->set_min_free_bytes(*options.cacheMinFreeBytes);
    return budget->account_file(fs::weakly_canonical(path).string(), bytes);
}

EncodedFiles::Opened EncodedFiles::open(const std::string& location)
{
    if (!zarr::isRemote(location)) {
        std::string path = location;
        if (path.starts_with("~/") || path == "~")
            if (const char* home = std::getenv("HOME")) path = std::string(home) + path.substr(1);
        const int fd = platform::open_file(path, O_RDONLY);
        if (fd < 0) {
            if (errno == ENOENT || errno == ENOTDIR) throw zarr::NotFound(path + ": no such file");
            throw fs::filesystem_error("open", path, std::error_code(errno, std::generic_category()));
        }
        Opened opened(fd, {});
        platform::FileStat st;
        if (platform::fstat(fd, st) != 0) throw fs::filesystem_error("fstat", path, std::error_code(errno, std::generic_category()));
        if (st.directory) throw fs::filesystem_error("open", path, std::make_error_code(std::errc::is_a_directory));
        opened.token = fs::absolute(path).string() + "|" + std::to_string(st.device) + "|" + std::to_string(st.inode) + "|" + std::to_string(st.size) + "|" +
                       std::to_string(st.modified.seconds) + "." + std::to_string(st.modified.nanoseconds) + "|" +
                       std::to_string(st.changed.seconds) + "." + std::to_string(st.changed.nanoseconds);
        return opened;
    }
    // The lock covers this object's state only: downloads of different files overlap, and
    // concurrent misses for one file wait on the budget's per-file lock instead.
    std::vector<fs::path> roots, lookIn;
    const auto options = resolve(location, roots);
    lookIn = roots;
    {
        std::lock_guard guard(mutex_);
        if (!temporary_.empty()) lookIn.emplace_back(temporary_);
    }
    for (const auto& root : lookIn) {
        const auto path = zarr::decodedCachePath(root, location);
        const int fd = platform::open_file(path, O_RDONLY);
        if (fd < 0) {
            if (errno == ENOENT || errno == ENOTDIR) continue;
            throw fs::filesystem_error("open", path, std::error_code(errno, std::generic_category()));
        }
        Opened opened(fd, {});
        opened.token = identity(location, fd);
        std::lock_guard guard(mutex_);
        ++cacheHits_;
        return opened;
    }
    if (options.offline) throw CacheMiss(location + ": encoded file is not cached (offline)");
    fs::path root;
    auto limit = options.cacheMaxBytes;
    if (!roots.empty() && options.cacheWrites) {
        root = roots.front();
    } else {
        // Scratch space the user did not choose: a large file must not fill it unasked.
        limit = limit.value_or(std::int64_t{1} << 30);
        std::lock_guard guard(mutex_);
        if (temporary_.empty()) {
            temporary_ = platform::make_temp_directory((fs::temp_directory_path() / "vc-reader-images-XXXXXX").string());
        }
        root = temporary_;
    }
    // No limit reuses the directory's configured one, unlimited for a new directory; it
    // never resets another reader's explicit budget.
    auto budget = zarr::diskBudget(root.string(), limit);
    if (options.cacheMinFreeBytes) budget->set_min_free_bytes(*options.cacheMinFreeBytes);
    auto [fd, received] = budget->open_file(zarr::decodedCachePath(root, location).string(), options.access.url(location), options.access.auth);
    {
        std::lock_guard guard(mutex_);
        bytesRead_ += received;
        if (!received && fd >= 0) ++cacheHits_;
    }
    if (fd < 0) throw zarr::NotFound(location + ": remote image does not exist");
    Opened opened(fd, {});
    opened.token = identity(location, fd);
    return opened;
}
} // namespace vcr::sources
