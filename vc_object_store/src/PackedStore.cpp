#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/Uuid.hpp"
#include "internal/Publication.hpp"
#include "internal/Sqlite.hpp"
#include <vc_reader/native/platform_io.hpp>
#include <sqlite3.h>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cerrno>
#include <deque>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace vc::objects {
namespace {
using Json = nlohmann::json;
constexpr std::string_view readPrefix = "vcobj-packed:";
constexpr std::string_view buildPrefix = "vcobj-building:";
constexpr std::size_t headerBytes = 80;
constexpr char magic[] = "VCOBJPK2";
constexpr auto maxOffset = std::uint64_t(std::numeric_limits<sqlite3_int64>::max());
thread_local std::string lastError;
void rememberError(const char* value) noexcept { try { lastError = value; } catch (...) {} }

void check(int result, const char* what)
{
    if (result != SQLITE_OK) throw std::runtime_error(std::string(what) + ": SQLite I/O error " + std::to_string(result));
}
std::uint64_t checkedEnd(std::uint64_t offset, std::uint64_t bytes)
{
    if (offset > maxOffset || bytes > maxOffset - offset) throw std::runtime_error("packed file offset overflow");
    return offset + bytes;
}

// Use SQLite's platform VFS instead of a second OS-specific file I/O layer.
class PhysicalFile {
    std::string path_;
    std::vector<std::max_align_t> storage_;
    sqlite3_file* file_ = nullptr;
    std::mutex mutex_;
public:
    PhysicalFile(const std::filesystem::path& path, bool writable, bool create = false) : path_(path.string())
    {
        auto* vfs = sqlite3_vfs_find(nullptr);
        if (!vfs) throw std::runtime_error("no default SQLite VFS");
        storage_.resize((vfs->szOsFile + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t));
        file_ = reinterpret_cast<sqlite3_file*>(storage_.data());
        const int flags = SQLITE_OPEN_MAIN_DB | (writable ? SQLITE_OPEN_READWRITE : SQLITE_OPEN_READONLY)
            | (create ? SQLITE_OPEN_CREATE | SQLITE_OPEN_EXCLUSIVE : 0);
        int actual = 0;
        const auto result = vfs->xOpen(vfs, path_.c_str(), file_, flags, &actual);
        if (result != SQLITE_OK) {
            if (file_->pMethods) file_->pMethods->xClose(file_);
            file_ = nullptr;
            check(result, "open packed backing file");
        }
    }
    ~PhysicalFile() { if (file_) file_->pMethods->xClose(file_); }
    void read(void* data, std::size_t count, std::uint64_t offset)
    {
        checkedEnd(offset, count);
        std::lock_guard lock(mutex_);
        auto* p = static_cast<char*>(data);
        while (count) {
            // SQLite's native VFS is a page-I/O interface. Some unix VFS
            // versions mask counts above 0x1ffff internally; use the largest
            // supported SQLite page size rather than passing MiB extents.
            const auto n = int(std::min<std::size_t>(count, 65536));
            check(file_->pMethods->xRead(file_, p, n, sqlite3_int64(offset)), "read packed backing file");
            p += n; offset += n; count -= n;
        }
    }
    void write(const void* data, std::size_t count, std::uint64_t offset)
    {
        checkedEnd(offset, count);
        std::lock_guard lock(mutex_);
        const auto* p = static_cast<const char*>(data);
        while (count) {
            const auto n = int(std::min<std::size_t>(count, 65536));
            const auto result = file_->pMethods->xWrite(file_, p, n, sqlite3_int64(offset));
            if (result != SQLITE_OK) {
                const int callErrno = errno;
                int osError = 0; file_->pMethods->xFileControl(file_, SQLITE_FCNTL_LAST_ERRNO, &osError);
                throw std::runtime_error("write packed backing file: SQLite I/O error " + std::to_string(result)
                    + ", OS error " + std::to_string(osError) + ", errno " + std::to_string(callErrno) + ", offset " + std::to_string(offset) + ", bytes " + std::to_string(n));
            }
            p += n; offset += n; count -= n;
        }
    }
    std::uint64_t size()
    {
        std::lock_guard lock(mutex_);
        sqlite3_int64 size = 0;
        check(file_->pMethods->xFileSize(file_, &size), "size packed backing file");
        if (size < 0) throw std::runtime_error("negative packed file size");
        return std::uint64_t(size);
    }
    void sync() { std::lock_guard lock(mutex_); check(file_->pMethods->xSync(file_, SQLITE_SYNC_FULL), "sync packed backing file"); }
};

std::string hexEncode(std::string_view in)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(in.size() * 2);
    for (const auto c : in) { result += digits[static_cast<unsigned char>(c) >> 4]; result += digits[c & 15]; }
    return result;
}
std::string hexDecode(std::string_view in)
{
    if (in.size() % 2) throw std::runtime_error("invalid packed locator path");
    auto digit = [](char c) -> unsigned {
        if (c >= '0' && c <= '9') return unsigned(c - '0');
        if (c >= 'a' && c <= 'f') return unsigned(c - 'a' + 10);
        throw std::runtime_error("invalid packed locator path");
    };
    std::string result;
    result.reserve(in.size() / 2);
    for (std::size_t i = 0; i < in.size(); i += 2) result += char(digit(in[i]) * 16 + digit(in[i + 1]));
    if (result.find('\0') != std::string::npos) throw std::runtime_error("null byte in packed locator");
    return result;
}
struct Locator { bool building; std::string source, member; };
Locator parseLocator(const std::string& name)
{
    const bool building = name.starts_with(buildPrefix);
    const auto prefix = building ? buildPrefix : readPrefix;
    if (!name.starts_with(prefix)) throw std::runtime_error("not a packed database locator");
    const auto slash = name.find('/', prefix.size());
    if (slash == std::string::npos) throw std::runtime_error("invalid packed database locator");
    return {building, building ? name.substr(prefix.size(), slash - prefix.size())
                              : hexDecode(std::string_view(name).substr(prefix.size(), slash - prefix.size())),
            name.substr(slash + 1)};
}
void put64(unsigned char* p, std::uint64_t n) { for (int i = 0; i < 8; ++i) p[i] = static_cast<unsigned char>(n >> (8 * i)); }
std::uint64_t get64(const unsigned char* p) { std::uint64_t n = 0; for (int i = 0; i < 8; ++i) n |= std::uint64_t(p[i]) << (8 * i); return n; }

struct Member {
    std::uint64_t size = 0;
    // Extent offsets for builders/noncompact files; a single range for compact files.
    std::vector<std::uint64_t> offsets;
    bool contiguous = false;
};
struct Published {
    std::shared_ptr<PhysicalFile> file;
    std::size_t extent = 0;
    std::string root;
    std::map<std::string, Member> members;
};
Published readDirectory(const std::filesystem::path& path)
{
    Published out;
    out.file = std::make_shared<PhysicalFile>(path, false);
    std::array<unsigned char, headerBytes> header{};
    out.file->read(header.data(), header.size(), 0);
    if (std::memcmp(header.data(), magic, 8) || get64(header.data() + 8) != packedStoreFormatVersion)
        throw std::runtime_error("not a supported finalized packed vcobj container");
    const auto offset = get64(header.data() + 16), length = get64(header.data() + 24);
    if (offset < headerBytes || length == 0 || length > 256u * 1024u * 1024u || checkedEnd(offset, length) != out.file->size())
        throw std::runtime_error("invalid packed container directory bounds");
    std::string text(std::size_t(length), '\0');
    out.file->read(text.data(), text.size(), offset);
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256(reinterpret_cast<const unsigned char*>(text.data()), text.size(), digest.data());
    if (std::memcmp(header.data() + 32, digest.data(), digest.size())) throw std::runtime_error("packed directory checksum mismatch");
    const auto json = Json::parse(text);
    out.root = util::canonicalUuid(json.at("root").get<std::string>());
    out.extent = json.at("extent_bytes").get<std::size_t>();
    if (out.extent < 1024u * 1024u || out.extent > 1024u * 1024u * 1024u || (out.extent & (out.extent - 1)))
        throw std::runtime_error("invalid packed extent size");
    std::vector<std::pair<std::uint64_t, std::uint64_t>> occupied;
    for (const auto& value : json.at("databases")) {
        const auto uuid = util::canonicalUuid(value.at("uuid").get<std::string>());
        Member member;
        member.size = value.at("size").get<std::uint64_t>();
        if (member.size > maxOffset) throw std::runtime_error("oversize packed database");
        member.contiguous = value.at("contiguous").get<bool>();
        member.offsets = value.at("offsets").get<std::vector<std::uint64_t>>();
        const auto expected = member.contiguous ? 1u : member.size / out.extent + (member.size % out.extent != 0);
        if (member.offsets.size() != expected) throw std::runtime_error("invalid packed extent count");
        for (const auto pos : member.offsets) {
            const auto end = checkedEnd(pos, member.contiguous ? member.size : out.extent);
            if (pos < headerBytes || end > offset) throw std::runtime_error("packed database outside payload");
            if (end != pos) occupied.emplace_back(pos, end);
        }
        if (!out.members.emplace(uuid, std::move(member)).second) throw std::runtime_error("duplicate packed database UUID");
    }
    std::sort(occupied.begin(), occupied.end());
    for (std::size_t i = 1; i < occupied.size(); ++i)
        if (occupied[i].first < occupied[i - 1].second) throw std::runtime_error("overlapping packed extents");
    if (!out.members.contains(out.root)) throw std::runtime_error("packed root database missing");
    return out;
}
void readMember(Published& source, const Member& member, void* data, std::size_t bytes, std::uint64_t offset)
{
    if (checkedEnd(offset, bytes) > member.size) throw std::runtime_error("read outside packed member");
    auto* p = static_cast<char*>(data);
    while (bytes) {
        const auto block = offset / source.extent, within = offset % source.extent;
        const auto n = member.contiguous ? bytes : std::min(bytes, source.extent - std::size_t(within));
        source.file->read(p, n, member.contiguous ? member.offsets.front() + offset : member.offsets.at(block) + within);
        p += n; offset += n; bytes -= n;
    }
}

struct Block {
    std::uint64_t physical = 0, version = 0, flushed = 0, touched = 0;
    bool persisted = false, inFlight = false, queued = false, retired = false;
    std::vector<char> data;
    // Allocation remains MiB-sized, but SQLite mutates individual pages. Never
    // rewrite the entire extent for each small page update after initial fill.
    std::vector<bool> dirtyPages;
    std::list<std::shared_ptr<Block>>::iterator lru;
};
struct MutableMember {
    std::uint64_t size = 0;
    std::vector<std::shared_ptr<Block>> blocks;
    std::map<const void*, int> locks;
    unsigned handles = 0;
    bool database = false;
    bool deletePending = false;
    bool retiring = false;
    // SQLite permits one writer per logical database. Independent members must
    // not wait for one another's native file I/O through a global cache lock.
    std::mutex operations;
    std::shared_ptr<PhysicalFile> directFile;
};
struct BuildState {
    PackedStoreOptions options;
    std::filesystem::path destination, staging, compactStaging;
    std::shared_ptr<PhysicalFile> file;
    std::map<std::string, std::shared_ptr<MutableMember>> members;
    std::list<std::shared_ptr<Block>> lru;
    std::deque<std::weak_ptr<Block>> dirty;
    std::vector<std::uint64_t> freeExtents;
    std::mutex mutex;
    std::condition_variable wake;
    std::thread worker;
    std::exception_ptr failure;
    std::uint64_t next = headerBytes, clock = 0;
    std::size_t resident = 0;
    bool stopping = false, finalized = false, closing = false;

    ~BuildState()
    {
        { std::lock_guard lock(mutex); stopping = true; wake.notify_all(); }
        if (worker.joinable()) worker.join();
        for(auto& [_,member]:members)member->directFile.reset();
        file.reset();
        std::error_code ec;
        std::filesystem::remove(staging, ec);
        if (!compactStaging.empty()) std::filesystem::remove(compactStaging, ec);
    }
    void healthy() { if (failure) std::rethrow_exception(failure); if (closing || finalized) throw std::runtime_error("packed builder is closed"); }
    static constexpr std::size_t dirtyPageBytes = 4096;
    void markDirty(const std::shared_ptr<Block>& block, std::size_t offset, std::size_t bytes)
    {
        ++block->version;
        if(block->dirtyPages.empty())block->dirtyPages.resize(options.extentBytes/dirtyPageBytes);
        for(auto page=offset/dirtyPageBytes;page<(offset+bytes+dirtyPageBytes-1)/dirtyPageBytes;++page)block->dirtyPages[page]=true;
        if (!block->queued) { dirty.emplace_back(block); block->queued = true; }
    }
    struct DirtySpan {std::size_t offset;std::vector<char> bytes;};
    std::vector<DirtySpan> snapshot(Block& block)
    {
        std::vector<DirtySpan> spans;
        // First persistence initializes the complete extent, including zero
        // tails and recycled addresses; later writes touch dirty subranges only.
        if(!block.persisted)spans.push_back({0,block.data});
        else for(std::size_t p=0;p<block.dirtyPages.size();){
            if(!block.dirtyPages[p]){++p;continue;}
            const auto first=p++;while(p<block.dirtyPages.size()&&block.dirtyPages[p])++p;
            const auto begin=first*dirtyPageBytes,end=p*dirtyPageBytes;
            spans.push_back({begin,{block.data.begin()+std::ptrdiff_t(begin),block.data.begin()+std::ptrdiff_t(end)}});
        }
        std::fill(block.dirtyPages.begin(),block.dirtyPages.end(),false);
        return spans;
    }
    void persist(const Block& block,const std::vector<DirtySpan>& spans)
    {for(const auto& span:spans)file->write(span.bytes.data(),span.bytes.size(),block.physical+span.offset);}
    void discard(const std::shared_ptr<Block>& block)
    {
        if (!block->data.empty()) {
            lru.erase(block->lru);
            std::vector<char>().swap(block->data);
            resident -= options.extentBytes;
        }
    }
    void retire(const std::shared_ptr<Block>& block, std::unique_lock<std::mutex>& lock)
    {
        // Do not recycle an address while an old asynchronous write can still
        // arrive there. A retired queue entry is ignored even if another owner
        // temporarily retains the Block object.
        wake.wait(lock, [&] { return !block->inFlight; });
        block->retired = true;
        discard(block);
        freeExtents.push_back(block->physical);
    }
    void flush(Block& block)
    {
        if (block.data.empty() || block.flushed == block.version) return;
        // The cache mutex is already held for synchronous eviction/sync. Write
        // directly from resident bytes, avoiding another extent-sized snapshot.
        try{
            if(!block.persisted)file->write(block.data.data(),block.data.size(),block.physical);
            else for(std::size_t p=0;p<block.dirtyPages.size();){
                if(!block.dirtyPages[p]){++p;continue;}
                const auto first=p++;while(p<block.dirtyPages.size()&&block.dirtyPages[p])++p;
                file->write(block.data.data()+first*dirtyPageBytes,(p-first)*dirtyPageBytes,block.physical+first*dirtyPageBytes);
            }
        }catch(...){failure=std::current_exception();throw;}
        std::fill(block.dirtyPages.begin(),block.dirtyPages.end(),false);
        block.persisted = true;
        block.flushed = block.version;
    }
    void load(const std::shared_ptr<Block>& block, std::unique_lock<std::mutex>& lock, bool overwrite = false)
    {
        if (!block->data.empty()) { block->touched = ++clock; lru.splice(lru.begin(), lru, block->lru); return; }
        while (resident + options.extentBytes > options.residentBytes) {
            std::shared_ptr<Block> oldest;
            for (auto it = lru.rbegin(); it != lru.rend(); ++it)
                if (!(*it)->inFlight) { oldest = *it; break; }
            if (!oldest) {
                wake.wait(lock); healthy();
                // Another reader can have loaded this exact block while the
                // condition wait released the cache mutex.
                if (!block->data.empty()) { block->touched = ++clock; lru.splice(lru.begin(), lru, block->lru); return; }
                continue;
            }
            flush(*oldest);
            discard(oldest);
        }
        block->data.assign(options.extentBytes, 0);
        resident += options.extentBytes;
        lru.push_front(block); block->lru = lru.begin();
        if (block->persisted && !overwrite) file->read(block->data.data(), block->data.size(), block->physical);
        block->touched = ++clock;
    }
    void ensure(MutableMember& member, std::uint64_t end)
    {
        const auto count = end / options.extentBytes + (end % options.extentBytes != 0);
        while (member.blocks.size() < count) {
            auto block = std::make_shared<Block>();
            if (freeExtents.empty()) { block->physical = next; next = checkedEnd(next, options.extentBytes); }
            else { block->physical = freeExtents.back(); freeExtents.pop_back(); }
            member.blocks.push_back(std::move(block));
        }
    }
    template<class F> void directIo(MutableMember& member,const std::shared_ptr<Block>& block,
                                    std::unique_lock<std::mutex>& lock,F&& operation)
    {
        if(!member.directFile)member.directFile=std::make_shared<PhysicalFile>(staging,true);
        // The member operation mutex excludes concurrent mutation/truncation of
        // this logical file. The extent flag also prevents physical retirement.
        // A separate native handle avoids relying on concurrent sqlite3_file
        // method calls, which a platform VFS need not support.
        block->inFlight=true;
        lock.unlock();
        try{operation(*member.directFile);}
        catch(...){lock.lock();failure=std::current_exception();block->inFlight=false;wake.notify_all();throw;}
        lock.lock();block->inFlight=false;wake.notify_all();healthy();
    }
    void write(MutableMember& member, const void* data, std::size_t bytes, std::uint64_t offset)
    {
        std::lock_guard operation(member.operations);
        std::unique_lock lock(mutex); healthy();
        const auto end = checkedEnd(offset, bytes);
        ensure(member, end);
        const auto* p = static_cast<const char*>(data);
        while (bytes) {
            auto block = member.blocks.at(offset / options.extentBytes);
            const auto within = std::size_t(offset % options.extentBytes), n = std::min(bytes, options.extentBytes - within);
            if(block->data.empty()&&block->persisted){
                // SQLite already caches pages. Reloading a complete MiB extent
                // for one evicted index-page write creates extreme amplification.
                // Eviction excludes in-flight blocks, so direct partial I/O
                // cannot race an older asynchronous snapshot of this extent.
                directIo(member,block,lock,[&](PhysicalFile& target){target.write(p,n,block->physical+within);});
                block->flushed=++block->version;
                p+=n;offset+=n;bytes-=n;continue;
            }
            load(block, lock, within == 0 && n == options.extentBytes);
            std::memcpy(block->data.data() + within, p, n);
            markDirty(block,within,n);
            p += n; offset += n; bytes -= n;
        }
        member.size = std::max(member.size, end);
        wake.notify_all();
    }
    void read(MutableMember& member, void* data, std::size_t bytes, std::uint64_t offset)
    {
        std::lock_guard operation(member.operations);
        std::unique_lock lock(mutex); healthy();
        auto* p = static_cast<char*>(data);
        while (bytes) {
            auto block = member.blocks.at(offset / options.extentBytes);
            const auto within = std::size_t(offset % options.extentBytes), n = std::min(bytes, options.extentBytes - within);
            if(block->data.empty()&&block->persisted)directIo(member,block,lock,[&](PhysicalFile& source){source.read(p,n,block->physical+within);});
            else {load(block, lock);std::memcpy(p, block->data.data() + within, n);}
            p += n; offset += n; bytes -= n;
        }
    }
    void truncate(MutableMember& member, std::uint64_t size)
    {
        std::lock_guard operation(member.operations);
        std::unique_lock lock(mutex); healthy();
        checkedEnd(size, 0);
        ensure(member, size);
        // Zero the retained tail. Re-extending a file must not expose old data.
        if (size < member.size && size % options.extentBytes) {
            auto block = member.blocks.at(size / options.extentBytes);
            load(block, lock);
            std::fill(block->data.begin() + std::ptrdiff_t(size % options.extentBytes), block->data.end(), 0);
            markDirty(block,std::size_t(size%options.extentBytes),options.extentBytes-std::size_t(size%options.extentBytes));
        }
        const auto count = size / options.extentBytes + (size % options.extentBytes != 0);
        for (std::size_t i = std::size_t(count); i < member.blocks.size(); ++i)
            retire(member.blocks[i], lock);
        member.blocks.resize(std::size_t(count));
        member.size = size;
        wake.notify_all();
    }
    void sync(MutableMember& member)
    {
        std::lock_guard operation(member.operations);
        std::unique_lock lock(mutex); healthy();
        if (options.disposableBulk) return;
        wake.wait(lock, [&] { return failure || std::none_of(member.blocks.begin(), member.blocks.end(), [](const auto& b) { return b->inFlight; }); });
        healthy();
        for (const auto& block : member.blocks) flush(*block);
        file->sync();
    }
    void runLoop()
    {
        std::unique_lock lock(mutex);
        for (;;) {
            std::shared_ptr<Block> selected;
            while (!dirty.empty() && !selected) {
                auto b = dirty.front().lock(); dirty.pop_front();
                if (!b) continue;
                b->queued = false;
                if (!b->retired && !b->data.empty() && b->flushed != b->version) selected = std::move(b);
            }
            if (!selected) { if (stopping) return; wake.wait(lock); continue; }
            selected->inFlight = true;
            const auto version = selected->version;
            const auto spans = snapshot(*selected);
            lock.unlock();
            try { persist(*selected,spans); }
            catch (...) { lock.lock(); failure = std::current_exception(); selected->inFlight = false; wake.notify_all(); return; }
            lock.lock();
            selected->persisted = true;
            selected->flushed = version;
            selected->inFlight = false;
            wake.notify_all();
        }
    }
    void run() noexcept
    {
        try { runLoop(); }
        catch (...) {
            std::lock_guard lock(mutex);
            failure = std::current_exception();
            for (const auto& block : lru) block->inFlight = false;
            wake.notify_all();
        }
    }
};

std::mutex registryMutex;
std::map<std::string, std::weak_ptr<BuildState>> builders;
std::shared_ptr<BuildState> building(const std::string& token)
{
    std::lock_guard lock(registryMutex);
    const auto it = builders.find(token);
    auto value = it == builders.end() ? nullptr : it->second.lock();
    if (!value) throw std::runtime_error("packed builder no longer exists");
    return value;
}

struct PackedFile {
    sqlite3_file base{};
    std::shared_ptr<BuildState> build;
    std::shared_ptr<MutableMember> mutableMember;
    std::unique_ptr<Published> published;
    Member member;
    bool readOnly = true, deleteOnClose = false;
    std::string name;
};
PackedFile& cast(sqlite3_file* file) { return *reinterpret_cast<PackedFile*>(file); }
template<class F> int io(F&& f, int code = SQLITE_IOERR) noexcept
{
    try { f(); return SQLITE_OK; }
    catch (const std::bad_alloc&) { rememberError("allocation failed in packed VFS"); return SQLITE_NOMEM; }
    catch (const std::exception& error) { rememberError(error.what()); return code; }
    catch (...) { rememberError("unknown packed VFS error"); return code; }
}
int closeFile(sqlite3_file* raw)
{
    auto& f = cast(raw);
    const auto result = io([&] { if (f.build) {
        std::lock_guard operation(f.mutableMember->operations);
        std::unique_lock lock(f.build->mutex);
        f.mutableMember->locks.erase(&f);
        f.mutableMember->deletePending|=f.deleteOnClose;
        if (f.mutableMember->deletePending&&f.mutableMember->handles==1) {
            f.mutableMember->retiring=true;
            for (const auto& b : f.mutableMember->blocks) f.build->retire(b, lock);
            f.build->members.erase(f.name);
        }
        // Finalization must not observe a closed handle before its pending I/O
        // and physical extent retirement have actually completed.
        --f.mutableMember->handles;
        f.build->wake.notify_all();
    } }, SQLITE_IOERR_CLOSE);
    f.~PackedFile();
    return result;
}
int readFile(sqlite3_file* raw, void* data, int n, sqlite3_int64 offset)
{
    if (n < 0 || offset < 0) return SQLITE_IOERR_READ;
    auto& f = cast(raw);
    bool shortRead = false;
    const auto status = io([&] {
        std::uint64_t size;
        if (f.build) { std::lock_guard lock(f.build->mutex); size = f.mutableMember->size; }
        else size = f.member.size;
        const auto count = std::size_t(std::min<std::uint64_t>(std::uint64_t(n), std::uint64_t(offset) < size ? size - std::uint64_t(offset) : 0));
        if (count != std::size_t(n)) { std::memset(static_cast<char*>(data) + count, 0, std::size_t(n) - count); shortRead = true; }
        if (count) {
            if (f.build) f.build->read(*f.mutableMember, data, count, std::uint64_t(offset));
            else readMember(*f.published, f.member, data, count, std::uint64_t(offset));
        }
    }, SQLITE_IOERR_READ);
    return status == SQLITE_OK && shortRead ? SQLITE_IOERR_SHORT_READ : status;
}
int writeFile(sqlite3_file* raw, const void* data, int n, sqlite3_int64 offset)
{
    auto& f = cast(raw);
    if (f.readOnly) return SQLITE_READONLY;
    if (n < 0 || offset < 0) return SQLITE_IOERR_WRITE;
    return io([&] { f.build->write(*f.mutableMember, data, std::size_t(n), std::uint64_t(offset)); }, SQLITE_IOERR_WRITE);
}
int truncateFile(sqlite3_file* raw, sqlite3_int64 size)
{
    auto& f = cast(raw);
    if (f.readOnly) return SQLITE_READONLY;
    if (size < 0) return SQLITE_IOERR_TRUNCATE;
    return io([&] { f.build->truncate(*f.mutableMember, std::uint64_t(size)); }, SQLITE_IOERR_TRUNCATE);
}
int syncFile(sqlite3_file* raw, int)
{
    auto& f = cast(raw);
    return f.build ? io([&] { f.build->sync(*f.mutableMember); }, SQLITE_IOERR_FSYNC) : SQLITE_OK;
}
int sizeFile(sqlite3_file* raw, sqlite3_int64* size)
{
    auto& f = cast(raw);
    return io([&]{
        if (f.build) { std::lock_guard lock(f.build->mutex); *size = sqlite3_int64(f.mutableMember->size); }
        else *size = sqlite3_int64(f.member.size);
    },SQLITE_IOERR_FSTAT);
}
int lockFile(sqlite3_file* raw, int level)
{
    auto& f = cast(raw);
    if (!f.build) return SQLITE_OK;
    bool busy=false;
    const auto status=io([&]{
        std::lock_guard lock(f.build->mutex);
        const auto current = f.mutableMember->locks.find(&f);
        const auto old = current == f.mutableMember->locks.end() ? SQLITE_LOCK_NONE : current->second;
        for (const auto& [owner, held] : f.mutableMember->locks) {
            if (owner == &f) continue;
            if ((level >= SQLITE_LOCK_EXCLUSIVE && held > SQLITE_LOCK_NONE)
                || (level >= SQLITE_LOCK_RESERVED && held >= SQLITE_LOCK_RESERVED)
                || (level == SQLITE_LOCK_SHARED && old == SQLITE_LOCK_NONE && held >= SQLITE_LOCK_PENDING)) {busy=true;return;}
        }
        f.mutableMember->locks[&f] = std::max(old, level);
    }, SQLITE_IOERR_LOCK);
    return status==SQLITE_OK&&busy?SQLITE_BUSY:status;
}
int unlockFile(sqlite3_file* raw, int level)
{
    auto& f = cast(raw);
    return io([&]{if (f.build) { std::lock_guard lock(f.build->mutex); if (level == SQLITE_LOCK_NONE) f.mutableMember->locks.erase(&f); else f.mutableMember->locks[&f] = level; }},SQLITE_IOERR_UNLOCK);
}
int reservedFile(sqlite3_file* raw, int* result)
{
    auto& f = cast(raw); *result = 0;
    return io([&]{if (f.build) { std::lock_guard lock(f.build->mutex); for (const auto& [_, level] : f.mutableMember->locks) if (level >= SQLITE_LOCK_RESERVED) *result = 1; }},SQLITE_IOERR_CHECKRESERVEDLOCK);
}
int controlFile(sqlite3_file* raw, int op, void* value)
{
    auto& f = cast(raw);
    if (op == SQLITE_FCNTL_VFSNAME) { *static_cast<char**>(value) = sqlite3_mprintf("vcobj-packed-v1"); return *static_cast<char**>(value) ? SQLITE_OK : SQLITE_NOMEM; }
    if (op == SQLITE_FCNTL_HAS_MOVED) { *static_cast<int*>(value) = 0; return SQLITE_OK; }
    if (op == SQLITE_FCNTL_LOCKSTATE) {
        auto* result = static_cast<int*>(value); *result = SQLITE_LOCK_NONE;
        return io([&]{if (f.build) { std::lock_guard lock(f.build->mutex); const auto it = f.mutableMember->locks.find(&f); if (it != f.mutableMember->locks.end()) *result = it->second; }},SQLITE_IOERR);
    }
    return SQLITE_NOTFOUND;
}
int sectorFile(sqlite3_file*) { return 4096; }
int characteristicsFile(sqlite3_file* raw) { return cast(raw).published ? SQLITE_IOCAP_IMMUTABLE : 0; }
const sqlite3_io_methods methods = {1, closeFile, readFile, writeFile, truncateFile, syncFile, sizeFile,
    lockFile, unlockFile, reservedFile, controlFile, sectorFile, characteristicsFile,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

int openFile(sqlite3_vfs*, const char* name, sqlite3_file* raw, int flags, int* actual)
{
    auto* f = new (raw) PackedFile;
    const auto result = io([&] {
        if (!name || (flags & SQLITE_OPEN_WAL)) throw std::runtime_error("packed VFS requires named rollback-journal databases; WAL is unsupported");
        const auto locator = parseLocator(name);
        f->name = locator.member;
        f->readOnly = (flags & SQLITE_OPEN_READONLY) != 0;
        f->deleteOnClose = (flags & SQLITE_OPEN_DELETEONCLOSE) != 0;
        if (locator.building) {
            auto state = building(locator.source);
            std::lock_guard lock(state->mutex); state->healthy();
            auto it = state->members.find(locator.member);
            if (it == state->members.end()) {
                if (!(flags & SQLITE_OPEN_CREATE) || (flags & SQLITE_OPEN_MAIN_DB)) throw std::runtime_error("unknown packed member");
                it = state->members.emplace(locator.member, std::make_shared<MutableMember>()).first;
            }
            if(it->second->retiring)throw std::runtime_error("packed member is being retired");
            f->mutableMember = it->second;
            ++f->mutableMember->handles;
            f->build = std::move(state);
        } else {
            if (!f->readOnly) throw std::runtime_error("published packed databases are immutable");
            f->published = std::make_unique<Published>(readDirectory(locator.source));
            const auto member = locator.member.empty() ? f->published->root : util::canonicalUuid(locator.member);
            f->member = f->published->members.at(member);
        }
        f->base.pMethods = &methods;
        if (actual) *actual = flags;
    }, SQLITE_CANTOPEN);
    if (result != SQLITE_OK) { f->~PackedFile(); raw->pMethods = nullptr; }
    return result;
}
int deleteFile(sqlite3_vfs*, const char* name, int syncDir)
{
    return io([&] {
        const auto loc = parseLocator(name);
        if (!loc.building) throw std::runtime_error("published packed container is immutable");
        auto state = building(loc.source);
        std::unique_lock lock(state->mutex); state->healthy();
        const auto it = state->members.find(loc.member);
        if (it == state->members.end()) return;
        if (it->second->database || it->second->handles || it->second->retiring)
            throw std::runtime_error("cannot delete open/retiring member or primary database");
        it->second->retiring=true;
        for (const auto& b : it->second->blocks) state->retire(b, lock);
        state->members.erase(it);
        if (syncDir && !state->options.disposableBulk) state->file->sync();
    }, SQLITE_IOERR_DELETE);
}
int accessFile(sqlite3_vfs*, const char* name, int flag, int* result)
{
    *result = 0;
    return io([&] {
        const auto loc = parseLocator(name);
        if (loc.building) {
            auto state = building(loc.source); std::lock_guard lock(state->mutex);
            const auto it = state->members.find(loc.member);
            *result = it != state->members.end() && (flag != SQLITE_ACCESS_EXISTS || it->second->size != 0);
        } else {
            if (flag == SQLITE_ACCESS_READWRITE || (!loc.member.empty() && !util::isUuid(loc.member))) return;
            auto source = readDirectory(loc.source);
            *result = loc.member.empty() || source.members.contains(util::canonicalUuid(loc.member));
        }
    }, SQLITE_IOERR_ACCESS);
}
int fullPath(sqlite3_vfs*, const char* name, int count, char* result)
{
    if (!name || std::strlen(name) >= std::size_t(count)) return SQLITE_CANTOPEN;
    std::memcpy(result, name, std::strlen(name) + 1); return SQLITE_OK;
}
int randomVfs(sqlite3_vfs*, int n, char* out) { auto* v = sqlite3_vfs_find(nullptr); return v->xRandomness(v, n, out); }
int sleepVfs(sqlite3_vfs*, int microseconds) { auto* v = sqlite3_vfs_find(nullptr); return v->xSleep(v, microseconds); }
int timeVfs(sqlite3_vfs*, double* out) { auto* v = sqlite3_vfs_find(nullptr); return v->xCurrentTime(v, out); }
void* dlOpen(sqlite3_vfs*, const char* name) { auto* v = sqlite3_vfs_find(nullptr); return v->xDlOpen ? v->xDlOpen(v, name) : nullptr; }
void dlError(sqlite3_vfs*, int count, char* message) { auto* v = sqlite3_vfs_find(nullptr); if (v->xDlError) v->xDlError(v, count, message); }
void (*dlSym(sqlite3_vfs*, void* handle, const char* name))(void) { auto* v = sqlite3_vfs_find(nullptr); return v->xDlSym ? v->xDlSym(v, handle, name) : nullptr; }
void dlClose(sqlite3_vfs*, void* handle) { auto* v = sqlite3_vfs_find(nullptr); if (v->xDlClose) v->xDlClose(v, handle); }
int getLastError(sqlite3_vfs*, int count, char* message) { sqlite3_snprintf(count, message, "%s", lastError.c_str()); return 0; }

void writeDirectory(PhysicalFile& file, const Json& directory, std::uint64_t offset)
{
    const auto text = directory.dump();
    file.write(text.data(), text.size(), offset);
    file.sync();
    std::array<unsigned char, headerBytes> header{};
    std::memcpy(header.data(), magic, 8);
    put64(header.data() + 8, packedStoreFormatVersion);
    put64(header.data() + 16, offset);
    put64(header.data() + 24, text.size());
    SHA256(reinterpret_cast<const unsigned char*>(text.data()), text.size(), header.data() + 32);
    file.write(header.data(), header.size(), 0);
    file.sync();
}
} // namespace

struct PackedStoreBuilder::Impl { std::shared_ptr<BuildState> state; std::string token; };

const char* packedStoreVfs()
{
    static sqlite3_vfs vfs{};
    static std::once_flag once;
    std::call_once(once, [] {
        sqlite3_initialize();
        vfs.iVersion = 1; vfs.szOsFile = sizeof(PackedFile); vfs.mxPathname = 32768;
        vfs.zName = "vcobj-packed-v1";
        vfs.xOpen = openFile; vfs.xDelete = deleteFile; vfs.xAccess = accessFile; vfs.xFullPathname = fullPath;
        vfs.xDlOpen = dlOpen; vfs.xDlError = dlError; vfs.xDlSym = dlSym; vfs.xDlClose = dlClose; vfs.xGetLastError = getLastError;
        vfs.xRandomness = randomVfs; vfs.xSleep = sleepVfs; vfs.xCurrentTime = timeVfs;
        check(sqlite3_vfs_register(&vfs, 0), "register packed VFS");
    });
    return vfs.zName;
}
std::string packedStoreLastError() { return lastError; }
bool isPackedDatabasePath(const std::filesystem::path& path)
{
    const auto value = path.string(); return value.starts_with(readPrefix) || value.starts_with(buildPrefix);
}
bool isBuildingPackedDatabasePath(const std::filesystem::path& path) { return path.string().starts_with(buildPrefix); }
bool isPackedStore(const std::filesystem::path& path)
{
    if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) < 8) return false;
    PhysicalFile file(path, false);
    char value[8]; file.read(value, sizeof(value), 0);
    return std::memcmp(value, magic, sizeof(value)) == 0;
}
std::filesystem::path packedDatabasePath(const std::filesystem::path& container, const std::string& databaseUuid)
{
    auto directory = readDirectory(container);
    const auto uuid = databaseUuid.empty() ? directory.root : util::canonicalUuid(databaseUuid);
    if (!directory.members.contains(uuid)) throw std::runtime_error("database UUID missing from packed container");
    return std::string(readPrefix) + hexEncode(std::filesystem::absolute(container).lexically_normal().string()) + "/" + uuid;
}
std::filesystem::path packedContainerPath(const std::filesystem::path& locator)
{
    const auto value = parseLocator(locator.string());
    if (value.building) return building(value.source)->destination;
    return value.source;
}
std::string packedDatabaseUuid(const std::filesystem::path& locatorOrContainer)
{
    if (!isPackedDatabasePath(locatorOrContainer)) return readDirectory(locatorOrContainer).root;
    return util::canonicalUuid(parseLocator(locatorOrContainer.string()).member);
}
std::string packedMemberReference(const std::string& databaseUuid) { return "@packed:" + util::canonicalUuid(databaseUuid); }
std::filesystem::path resolvePackedDatabasePath(const std::filesystem::path& context, const std::string& memberReference)
{
    constexpr std::string_view prefix = "@packed:";
    if (!memberReference.starts_with(prefix)) throw std::invalid_argument("not a packed member reference");
    const auto uuid = util::canonicalUuid(memberReference.substr(prefix.size()));
    if (isBuildingPackedDatabasePath(context)) {
        const auto source = parseLocator(context.string());
        return std::string(buildPrefix) + source.source + "/" + uuid;
    }
    return packedDatabasePath(isPackedDatabasePath(context) ? packedContainerPath(context) : context, uuid);
}
std::vector<PackedDatabaseInfo> packedDatabases(const std::filesystem::path& container)
{
    std::vector<PackedDatabaseInfo> out;
    for (const auto& [uuid, member] : readDirectory(container).members) out.push_back({uuid, member.size});
    return out;
}
PackedStoreBuilder::PackedStoreBuilder(const std::filesystem::path& destination, const PackedStoreOptions& options)
    : impl_(std::make_shared<Impl>())
{
    if (options.extentBytes < 1024u * 1024u || options.extentBytes > 1024u * 1024u * 1024u
        || (options.extentBytes & (options.extentBytes - 1)) || options.residentBytes < options.extentBytes)
        throw std::invalid_argument("packed builder requires power-of-two MiB extents and at least one resident extent");
    packedStoreVfs();
    auto state = std::make_shared<BuildState>();
    state->options = options;
    state->destination = std::filesystem::absolute(destination).lexically_normal();
    if (std::filesystem::exists(state->destination)) throw std::runtime_error("packed destination already exists");
    impl_->token = util::newUuid();
    state->staging = state->destination.string() + ".building-" + impl_->token;
    state->file = std::make_shared<PhysicalFile>(state->staging, true, true);
    const std::array<unsigned char, headerBytes> empty{};
    state->file->write(empty.data(), empty.size(), 0);
    impl_->state = state;
    { std::lock_guard lock(registryMutex); builders.emplace(impl_->token, state); }
    state->worker = std::thread([pointer = state.get()] { pointer->run(); });
}
PackedStoreBuilder::~PackedStoreBuilder()
{
    if (impl_) {
        { std::lock_guard lock(registryMutex); builders.erase(impl_->token); }
        auto& state = *impl_->state;
        { std::lock_guard lock(state.mutex); state.closing = true; state.stopping = true; state.wake.notify_all(); }
        if (state.worker.joinable()) state.worker.join();
    }
}
std::filesystem::path PackedStoreBuilder::addDatabase(const std::string& databaseUuid)
{
    const auto uuid = util::canonicalUuid(databaseUuid);
    auto& state = *impl_->state;
    std::lock_guard lock(state.mutex); state.healthy();
    auto member = std::make_shared<MutableMember>(); member->database = true;
    if (!state.members.emplace(uuid, member).second) throw std::runtime_error("duplicate packed database UUID");
    return std::string(buildPrefix) + impl_->token + "/" + uuid;
}
std::filesystem::path PackedStoreBuilder::databasePath(const std::string& databaseUuid) const
{
    const auto uuid = util::canonicalUuid(databaseUuid);
    auto& state = *impl_->state;
    std::lock_guard lock(state.mutex); state.healthy();
    const auto it = state.members.find(uuid);
    if (it == state.members.end() || !it->second->database) throw std::runtime_error("packed member does not exist");
    return std::string(buildPrefix) + impl_->token + "/" + uuid;
}
void PackedStoreBuilder::importDatabase(const std::string& databaseUuid, const std::filesystem::path& source)
{
    const auto destination = addDatabase(databaseUuid);
    sqlite3 *input = nullptr, *output = nullptr;
    sqlite3_backup* backup = nullptr;
    try {
        const auto path = isPackedDatabasePath(source) ? source : isPackedStore(source) ? packedDatabasePath(source) : source;
        check(sqlite3_open_v2(path.string().c_str(), &input, SQLITE_OPEN_READONLY, isPackedDatabasePath(path) ? packedStoreVfs() : nullptr), "open packed import source");
        {
            detail::Statement identityTable(input, "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='database_identity'");
            if (identityTable.row()) {
                detail::Statement identity(input, "SELECT uuid FROM database_identity WHERE singleton=1");
                if (!identity.row() || detail::formatUuid(identity.blob16(0)) != util::canonicalUuid(databaseUuid))
                    throw std::invalid_argument("packed import database UUID does not match source identity");
            }
        }
        check(sqlite3_open_v2(destination.string().c_str(), &output, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, packedStoreVfs()), "open packed import destination");
        backup = sqlite3_backup_init(output, "main", input, "main");
        if (!backup) throw std::runtime_error(sqlite3_errmsg(output));
        const auto result = sqlite3_backup_step(backup, -1);
        if (result != SQLITE_DONE) check(result, "copy packed database backup");
        check(sqlite3_backup_finish(backup), "finish packed database backup"); backup = nullptr;
        check(sqlite3_close(input), "close packed import source"); input = nullptr;
        check(sqlite3_close(output), "close packed import destination"); output = nullptr;
        // Backup has materialized every committed page, but it also copies the
        // source header's WAL read/write-version bytes. Before any reader opens
        // this private checkpoint, mark those bytes as rollback-journal format.
        // Asking PRAGMA journal_mode to do this would first try to open a WAL
        // through a deliberately WAL-free VFS. No logical/schema data changes.
        const std::array<unsigned char, 2> rollbackVersion{1, 1};
        auto state = impl_->state;
        std::shared_ptr<MutableMember> member;
        { std::lock_guard lock(state->mutex); member = state->members.at(util::canonicalUuid(databaseUuid)); }
        if (member->size >= 100) state->write(*member, rollbackVersion.data(), rollbackVersion.size(), 18);
    } catch (...) {
        if (backup) sqlite3_backup_finish(backup);
        if (input) sqlite3_close(input);
        if (output) sqlite3_close(output);
        { std::lock_guard lock(impl_->state->mutex); impl_->state->failure = std::current_exception(); impl_->state->stopping = true; impl_->state->wake.notify_all(); }
        throw;
    }
}
void PackedStoreBuilder::finalize(const std::string& rootDatabaseUuid, bool compact)
{
    const auto root = util::canonicalUuid(rootDatabaseUuid);
    auto& state = *impl_->state;
    {
        std::unique_lock lock(state.mutex); state.healthy();
        if (!state.members.contains(root) || !state.members.at(root)->database) throw std::runtime_error("packed root database missing");
        for (const auto& [_, member] : state.members) if (member->handles) throw std::runtime_error("close all packed database handles before finalize");
        state.closing = true; state.stopping = true; state.wake.notify_all();
    }
    if (state.worker.joinable()) state.worker.join();
    if (state.failure) std::rethrow_exception(state.failure);
    Json directory{{"root", root}, {"extent_bytes", state.options.extentBytes}, {"databases", Json::array()}};
    std::shared_ptr<PhysicalFile> target = state.file;
    auto position = compact ? std::uint64_t(headerBytes) : state.next;
    if (compact) {
        state.compactStaging = state.destination.string() + ".compact-" + impl_->token;
        target = std::make_shared<PhysicalFile>(state.compactStaging, true, true);
        const std::array<unsigned char, headerBytes> empty{}; target->write(empty.data(), empty.size(), 0);
    }
    std::vector<char> buffer(state.options.extentBytes);
    for (const auto& [uuid, member] : state.members) {
        if (!member->database) continue;
        std::vector<std::uint64_t> offsets;
        if (compact) offsets.push_back(position);
        std::uint64_t copied = 0;
        for (const auto& block : member->blocks) {
            state.flush(*block);
            // Sparse extended ranges have no prior writes; persist zero extents.
            if (!block->persisted) {
                std::fill(buffer.begin(), buffer.end(), 0);
                state.file->write(buffer.data(), buffer.size(), block->physical);
                block->persisted = true;
            }
            if (compact) {
                const auto n = std::size_t(std::min<std::uint64_t>(buffer.size(), member->size - copied));
                state.file->read(buffer.data(), n, block->physical);
                target->write(buffer.data(), n, position);
                position = checkedEnd(position, n); copied += n;
            } else offsets.push_back(block->physical);
        }
        directory["databases"].push_back({{"uuid", uuid}, {"size", member->size}, {"contiguous", compact}, {"offsets", offsets}});
    }
    writeDirectory(*target, directory, position);
    const auto publish = compact ? state.compactStaging : state.staging;
    // Verify the complete directory before exposing it. No partially built file
    // is ever accepted by the reader (magic is only written at finalization).
    readDirectory(publish);
    target.reset();
    for(auto& [_,member]:state.members)member->directFile.reset();
    state.file.reset();
    // Link publication is atomic and fails if another publisher won the name;
    // an exists()+rename pair would silently clobber that competing result.
    detail::publishNewFile(publish, state.destination);
    state.finalized = true;
    std::error_code ec;
    if (compact) std::filesystem::remove(state.staging, ec);
}
void compactPackedStore(const std::filesystem::path& source, const std::filesystem::path& destination, const PackedStoreOptions& options)
{
    const auto directory = readDirectory(source);
    PackedStoreBuilder builder(destination, options);
    for (const auto& [uuid, _] : directory.members) builder.importDatabase(uuid, packedDatabasePath(source, uuid));
    builder.finalize(directory.root, true);
}
} // namespace vc::objects
