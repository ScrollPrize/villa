#include "vc_reader/source_reader.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <list>
#include <system_error>
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
    if (dtype.size() == 3 && (dtype[0] == '<' || dtype[0] == '>' || dtype[0] == '|' || dtype[0] == '=')) {
        std::string out(dtype);
        if (out[0] == '=' || out[0] == '|') out[0] = out[2] == '1' ? '|' : '<';
        if (out[2] == '1') out[0] = '|';
        if ((out[1] == 'u' || out[1] == 'i' || out[1] == 'f' || out[1] == 'b') && std::string_view("1248").find(out[2]) != std::string_view::npos) return out;
    }
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
    long pid = ::getpid();
    std::recursive_mutex mutex;
    std::list<std::pair<std::string, Block>> order;  // oldest first
    std::unordered_map<std::string, std::list<std::pair<std::string, Block>>::iterator> entries;
    std::size_t nbytes = 0, hits = 0, misses = 0, peakWorking = 0, working = 0;
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

DecodedCache::DecodedCache(std::size_t maxBytes, std::size_t maxDecodeBytes) : maxBytes_(maxBytes), maxDecodeBytes_(maxDecodeBytes), state_(std::make_shared<State>()) {}

std::shared_ptr<DecodedCache::State> DecodedCache::state()
{
    std::lock_guard guard(stateMutex_);
    if (state_->pid != ::getpid()) {
        // The inherited lock may have been held across fork: leave it alone.
        new std::shared_ptr<State>(state_);
        state_ = std::make_shared<State>();
    }
    return state_;
}

DecodedCache::Stats DecodedCache::stats()
{
    const auto s = state();
    std::lock_guard lock(s->mutex);
    return {s->nbytes, s->entries.size(), s->hits, s->misses, s->peakWorking};
}

void DecodedCache::clear()
{
    const auto s = state();
    std::lock_guard lock(s->mutex);
    s->entries.clear();
    s->order.clear();
    s->nbytes = 0;
}

std::pair<Block, DecodedCache::Lease> DecodedCache::use(const std::string& key, std::size_t decodeBytes, std::size_t scratchBytes, const Load& load)
{
    const auto s = state();
    Lease lease;
    lease.impl_ = std::make_unique<Lease::Impl>();
    lease.impl_->state = s;
    lease.impl_->lock = std::unique_lock(s->mutex);
    const auto found = s->entries.find(key);
    const bool cached = found != s->entries.end();
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
        s->order.splice(s->order.end(), s->order, found->second);
        block = found->second->second;
    } else {
        ++s->misses;
        block = load();
        if (block.nbytes > decodeBytes) throw ImageFormatError("decoder returned more bytes than its declared working set");
        if (block.nbytes > 0 && block.nbytes <= maxBytes_) {
            while (!s->order.empty() && s->nbytes + block.nbytes > maxBytes_) {
                s->nbytes -= s->order.front().second.nbytes;
                s->entries.erase(s->order.front().first);
                s->order.pop_front();
            }
            s->order.emplace_back(key, block);
            s->entries[key] = std::prev(s->order.end());
            s->nbytes += block.nbytes;
        }
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

EncodedFiles::EncodedFiles(OptionsFor options) : options_(std::move(options)), pid_(::getpid()) {}
EncodedFiles::~EncodedFiles()
{
    try {
        close();
    } catch (...) {
    }
}

void EncodedFiles::process()
{
    if (pid_ != ::getpid()) {
        // A child must not unlink the parent's temporary cache.
        temporary_.clear();
        identities_.clear();
        resolved_.clear();
        bytesRead_ = cacheHits_ = 0;
        pid_ = ::getpid();
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
    struct stat st{};
    if (::fstat(fd, &st) != 0) throw std::system_error(errno, std::generic_category(), "fstat " + location);
    const auto signature = std::to_string(st.st_dev) + ":" + std::to_string(st.st_ino) + ":" + std::to_string(st.st_size) + ":" +
                           std::to_string(st.st_mtime) + ":" + std::to_string(st.st_ctime);
    {
        std::lock_guard guard(mutex_);
        const auto known = identities_.find(location);
        if (known != identities_.end() && known->second.first == signature) return known->second.second;
    }
    // Bounded scratch; identities match online and offline, where no validator can be consulted.
    // Hashed outside the lock, so one file's digest does not hold up another's download.
    Sha256 hash;
    std::vector<char> block(1 << 16);
    off_t at = 0;
    for (;;) {
        const auto n = ::pread(fd, block.data(), block.size(), at);
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

EncodedFiles::Opened EncodedFiles::open(const std::string& location)
{
    if (!zarr::isRemote(location)) {
        std::string path = location;
        if (path.starts_with("~/") || path == "~")
            if (const char* home = std::getenv("HOME")) path = std::string(home) + path.substr(1);
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            if (errno == ENOENT || errno == ENOTDIR) throw zarr::NotFound(path + ": no such file");
            throw fs::filesystem_error("open", path, std::error_code(errno, std::generic_category()));
        }
        Opened opened(fd, {});
        struct stat st{};
        if (::fstat(fd, &st) != 0) throw fs::filesystem_error("fstat", path, std::error_code(errno, std::generic_category()));
        if (S_ISDIR(st.st_mode)) throw fs::filesystem_error("open", path, std::make_error_code(std::errc::is_a_directory));
        opened.token = fs::absolute(path).string() + "|" + std::to_string(st.st_dev) + "|" + std::to_string(st.st_ino) + "|" + std::to_string(st.st_size) + "|" +
                       std::to_string(st.st_mtim.tv_sec) + "." + std::to_string(st.st_mtim.tv_nsec) + "|" +
                       std::to_string(st.st_ctim.tv_sec) + "." + std::to_string(st.st_ctim.tv_nsec);
        return opened;
    }
    // The lock covers this object's state only: downloads of different files overlap, and
    // concurrent misses for one file wait on the budget's per-file lock instead.
    zarr::OpenOptions options;
    std::vector<fs::path> roots, lookIn;
    {
        std::lock_guard guard(mutex_);
        process();
        auto known = resolved_.find(location);
        if (known == resolved_.end()) known = resolved_.emplace(location, options_(location)).first;
        options = known->second;
        if (std::find(options.cacheDirs.begin(), options.cacheDirs.end(), "vc3d") != options.cacheDirs.end())
            throw std::invalid_argument("image files need explicit cache directories, not the vc3d chunk-cache alias");
        for (const auto& dir : options.cacheDirs) {
            std::string path = dir;
            if (path.starts_with("~/"))
                if (const char* home = std::getenv("HOME")) path = std::string(home) + path.substr(1);
            roots.push_back(fs::weakly_canonical(path));
        }
        lookIn = roots;
        if (!temporary_.empty()) lookIn.emplace_back(temporary_);
    }
    for (const auto& root : lookIn) {
        const auto path = zarr::decodedCachePath(root, location);
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
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
            auto pattern = (fs::temp_directory_path() / "vc-reader-images-XXXXXX").string();
            if (!::mkdtemp(pattern.data())) throw std::system_error(errno, std::generic_category(), "mkdtemp");
            temporary_ = pattern;
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
