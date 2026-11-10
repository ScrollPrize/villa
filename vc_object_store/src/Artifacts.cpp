// Content-addressed artifact bytes beside the database, and the manifests
// that name them in object metadata (objects/Artifacts.hpp).
#include "vc_object_store/objects/Artifacts.hpp"

#include "vc_object_store/objects/TransactionJson.hpp"
#include "internal/Artifacts.hpp"
#include "internal/Database.hpp"
#include "internal/ReferenceResolver.hpp"
#include "internal/RecordResolver.hpp"
#include "vc_object_store/objects/PackedStore.hpp"

#include <vc_reader/native/atomic_file.hpp>
#include <vc_reader/native/platform_io.hpp>

#include <fcntl.h>

#include <algorithm>
#include <set>

namespace vc::objects {
void detail::writeArtifactFile(const ArtifactFileRef& file, const std::filesystem::path& destination, uint64_t range,
                               const std::function<std::vector<uint8_t>(uint64_t, uint64_t)>& read)
{
    vcr::AtomicFile out(destination.string());
    PayloadHasher hasher;
    uint64_t written = 0;
    while (written < file.length) {
        const auto bytes = read(written, std::min(range, file.length - written));
        if (bytes.empty()) break;
        hasher.update(bytes.data(), bytes.size());
        out.write(bytes.data(), bytes.size());
        written += bytes.size();
    }
    if (written != file.length || hasher.finish() != file.sha256)
        throw std::runtime_error("artifact '" + file.name + "' does not match its manifest");
    out.publish(false);
}

using Json = nlohmann::json;
namespace platform = vcr::platform;

namespace {
constexpr std::size_t kBlock = 1 << 20;

void checkName(const std::string& name)
{
    if (name.empty() || name.size() > 1024 || name.front() == '/' || name.find('\\') != std::string::npos ||
        name.find('\0') != std::string::npos)
        throw std::invalid_argument("artifact: invalid file name '" + name + "'");
    std::size_t start = 0;
    for (;;) {
        const auto end = name.find('/', start);
        const auto part = name.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (part.empty() || part == "." || part == "..")
            throw std::invalid_argument("artifact: invalid file name '" + name + "'");
        if (end == std::string::npos) break;
        start = end + 1;
    }
}

void checkFile(const ArtifactFile& file)
{
    checkName(file.name);
    if (!isArtifactHash(file.sha256)) throw std::invalid_argument("artifact: invalid SHA-256 for '" + file.name + "'");
}

// Strict parse: the canonical form artifactManifestJson writes.
std::vector<ArtifactFile> parseManifest(const Json& value)
{
    if (!value.is_array()) throw std::invalid_argument("artifact: manifest is not an array");
    std::vector<ArtifactFile> files;
    for (const auto& entry : value) {
        if (!entry.is_object() || entry.size() != 3 || !entry.contains("name") || !entry.contains("sha256") ||
            !entry.contains("length") || !entry.at("length").is_number_unsigned())
            throw std::invalid_argument("artifact: manifest entries are {name, sha256, length}");
        ArtifactFile file{entry.at("name").get<std::string>(), entry.at("sha256").get<std::string>(),
                          entry.at("length").get<uint64_t>()};
        checkFile(file);
        if (!files.empty() && !(files.back().name < file.name))
            throw std::invalid_argument("artifact: manifest names must be unique and sorted");
        files.push_back(std::move(file));
    }
    return files;
}

struct Descriptor {
    int fd = -1;
    explicit Descriptor(const std::filesystem::path& path, int flags) : fd(platform::open_file(path, flags))
    {
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "open " + path.string());
    }
    ~Descriptor() { if (fd >= 0) ::close(fd); }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
};

// Reads [offset, offset + size) of `fd` into `out`, calling `chunk` per block.
template<class F> void readBlocks(int fd, uint64_t offset, uint64_t size, F&& chunk)
{
    std::vector<uint8_t> buffer(std::min<uint64_t>(std::max<uint64_t>(size, 1), kBlock));
    for (uint64_t done = 0; done < size;) {
        const auto want = std::min<uint64_t>(buffer.size(), size - done);
        const auto got = platform::pread(fd, buffer.data(), std::size_t(want), int64_t(offset + done));
        if (got <= 0) throw std::runtime_error("artifact: short read");
        chunk(buffer.data(), std::size_t(got));
        done += uint64_t(got);
    }
}
} // namespace

bool detail::artifactFileMatches(const ArtifactFileRef& file, const std::filesystem::path& path)
{
    try {
        Descriptor input(path, O_RDONLY);
        platform::FileStat status;
        if (platform::fstat(input.fd, status) != 0 || !status.regular || uint64_t(status.size) != file.length) return false;
        PayloadHasher hasher;
        readBlocks(input.fd, 0, file.length, [&](const uint8_t* data, size_t size) { hasher.update(data, size); });
        return hasher.finish() == file.sha256;
    } catch (const std::exception&) { return false; }
}

bool isArtifactHash(const std::string& text)
{
    return text.size() == 64 && std::all_of(text.begin(), text.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

std::filesystem::path artifactBlobDirectory(const std::filesystem::path& storePath)
{
    const auto path=isPackedDatabasePath(storePath)?packedContainerPath(storePath):storePath;
    return std::filesystem::path(path.string() + ".artifacts") / "sha256";
}

std::filesystem::path artifactBlobPath(const std::filesystem::path& storePath, const std::string& sha256)
{
    if (!isArtifactHash(sha256)) throw std::invalid_argument("artifact: invalid SHA-256 " + sha256);
    return artifactBlobDirectory(storePath) / sha256.substr(0, 2) / sha256;
}

detail::ArtifactExport::ArtifactExport(const std::vector<std::filesystem::path>& sources,
                                      const std::filesystem::path& destination,
                                      const std::vector<std::filesystem::path>& snapshots)
{
    std::set<std::filesystem::path> roots;
    for(const auto& source:sources)roots.insert(artifactBlobDirectory(source));
    try {
        for(const auto& root:roots){
            if(!std::filesystem::exists(root))continue;
            for(const auto& entry:std::filesystem::recursive_directory_iterator(root)){
                if(!entry.is_regular_file())continue;
                const auto hash=entry.path().filename().string();
                if(!isArtifactHash(hash)||entry.path().parent_path().filename()!=hash.substr(0,2))
                    throw std::runtime_error("artifact: unexpected payload file in source sidecar");
                if(owned_.empty()){
                    auto target=artifactBlobDirectory(destination).parent_path();
                    if(!std::filesystem::create_directory(target))
                        throw std::runtime_error("artifact destination already exists: "+target.string());
                    owned_=std::move(target);
                }
                const auto target=artifactBlobPath(destination,hash);
                ArtifactFileRef file{hash,hash,entry.file_size()};
                if(std::filesystem::exists(target)){
                    if(!artifactFileMatches(file,entry.path()))throw std::runtime_error("artifact: corrupt duplicate payload");
                    continue;
                }
                std::filesystem::create_directories(target.parent_path());
                Descriptor input(entry.path(),O_RDONLY);
                writeArtifactFile(file,target,kBlock,[&](uint64_t offset,uint64_t count){
                    std::vector<uint8_t> bytes;bytes.reserve(count);
                    readBlocks(input.fd,offset,count,[&](const uint8_t* data,size_t size){bytes.insert(bytes.end(),data,data+size);});
                    return bytes;
                });
            }
        }
        // Copying present files cannot detect a missing referenced blob. Check
        // the copied state, including manifests retained only in history. The
        // bytes above are already hash-verified; here only presence/length remain.
        const auto check=[&](const ObjectRecord& record){
            for(const auto& file:artifactManifest(record)){
                std::error_code error;
                const auto size=std::filesystem::file_size(artifactBlobPath(destination,file.sha256),error);
                if(error||size!=file.length)throw std::runtime_error("artifact export: missing or wrong-length retained payload "+file.sha256);
            }
        };
        for(const auto& snapshot:snapshots){
            std::unique_ptr<sqlite3,decltype(&sqlite3_close_v2)> db(
                detail::openConnection(snapshot,SQLITE_OPEN_READONLY,{},true),sqlite3_close_v2);
            const auto hasTable=[&](const char* name){detail::Statement table(db.get(),"SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?",name);return table.row();};
            if(!hasTable("store_manifest")||!hasTable("objects"))continue; // Other SQLite members have no vcobj manifests.
            detail::Statement current(db.get(),"SELECT meta FROM objects WHERE json_type(meta,'$.artifacts') IS NOT NULL");
            while(current.row()){ObjectRecord record;record.meta=current.text(0);check(record);}
            if(!hasTable("object_revisions"))continue;
            // Only revisions introducing/changing an artifact manifest can add
            // a required hash. Reuse the history resolver, not a second delta codec.
            detail::Statement history(db.get(),"SELECT uuid FROM object_revisions WHERE EXISTS ("
                "SELECT 1 FROM json_tree(metadata_delta) WHERE key='artifacts' OR "
                "(key='path' AND (value='/meta' OR value LIKE '/meta/artifacts%')))");
            while(history.row()){
                const auto revision=detail::resolveObjectRevision(db.get(),history.uuid(0));
                if(!revision)throw std::runtime_error("artifact export: unavailable retained revision");
                check(revision->revision.record);
            }
        }
    }catch(...){
        if(!owned_.empty()){std::error_code error;std::filesystem::remove_all(owned_,error);}
        throw;
    }
}
detail::ArtifactExport::~ArtifactExport()
{
    if(!committed_&&!owned_.empty()){std::error_code error;std::filesystem::remove_all(owned_,error);}
}

std::string artifactManifestJson(std::vector<ArtifactFile> files)
{
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
    auto result = Json::array();
    for (std::size_t i = 0; i < files.size(); ++i) {
        checkFile(files[i]);
        if (i && files[i].name == files[i - 1].name)
            throw std::invalid_argument("artifact: duplicate file name '" + files[i].name + "'");
        result.push_back({{"length", files[i].length}, {"name", files[i].name}, {"sha256", files[i].sha256}});
    }
    return result.dump();
}

std::vector<ArtifactFile> artifactManifest(const ObjectRecord& record)
{
    const auto meta = Json::parse(record.meta);
    if (!meta.is_object() || !meta.contains(kArtifactsMetaKey)) return {};
    return parseManifest(meta.at(kArtifactsMetaKey));
}

namespace {
struct ResolvedArtifact {std::filesystem::path path;uint64_t length;};
std::optional<ResolvedArtifact> resolveArtifact(sqlite3* db,const std::string& hash)
{
    if(!isArtifactHash(hash))throw std::invalid_argument("artifact: invalid SHA-256");
    std::optional<ResolvedArtifact> result;
    detail::resolveReference(db,[&](sqlite3* source){
        const auto path=artifactBlobPath(sqlite3_db_filename(source,"main"),hash);
        std::error_code error;const auto size=std::filesystem::file_size(path,error);
        if(error)return false;
        result=ResolvedArtifact{path,size};return true;
    });
    return result;
}
}
std::optional<uint64_t> detail::resolvedArtifactLength(sqlite3* db,const std::string& hash)
{
    const auto artifact=resolveArtifact(db,hash);
    return artifact?std::optional(artifact->length):std::nullopt;
}
void detail::checkArtifactReferences(sqlite3* db, const std::string& objectUuid,
                                     const Json& manifest)
{
    for (const auto& file : parseManifest(manifest)) {
        const auto size = resolvedArtifactLength(db,file.sha256);
        if (!size || *size != file.length)
            throw std::invalid_argument("artifact: " + objectUuid + " names '" + file.name + "' (" + file.sha256 +
                                        ") which this store does not hold; put its bytes first");
    }
}

std::string ObjectStore::putArtifact(const std::filesystem::path& file, bool move)
{
    if (!impl_->writer) throw std::runtime_error("object store: putArtifact on a read-only handle");
    const auto incoming = artifactBlobDirectory(impl_->path) / "incoming" / util::newUuid();
    std::filesystem::create_directories(incoming.parent_path());
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { if (!path.empty()) { std::error_code ignored; std::filesystem::remove(path, ignored); } }
    } cleanup;
    PayloadHasher hasher;
    uint64_t length = 0;
    std::error_code moveError;
    if (move) std::filesystem::rename(file, incoming, moveError);
    if (move && !moveError) {
        // Same filesystem: hash the moved file in place and sync it.
        cleanup.path = incoming;
        Descriptor in(incoming, O_RDONLY);
        length = std::filesystem::file_size(incoming);
        readBlocks(in.fd, 0, length, [&](const uint8_t* data, std::size_t size) { hasher.update(data, size); });
        if (platform::fsync(in.fd)) throw std::system_error(errno, std::generic_category(), "fsync " + incoming.string());
    } else {
        Descriptor in(file, O_RDONLY);
        length = std::filesystem::file_size(file);
        vcr::AtomicFile out(incoming.string());
        readBlocks(in.fd, 0, length, [&](const uint8_t* data, std::size_t size) {
            hasher.update(data, size);
            out.write(data, size);
        });
        out.publish(true);
        cleanup.path = incoming;
        if (move) {
            std::error_code ignored;
            std::filesystem::remove(file, ignored);
        }
    }
    const auto hash = hasher.finish();
    const auto target = artifactBlobPath(impl_->path, hash);
    std::error_code ec;
    if (std::filesystem::file_size(target, ec) == length && !ec) return hash;  // already held
    std::filesystem::create_directories(target.parent_path());
    std::filesystem::permissions(incoming, std::filesystem::perms::owner_read | std::filesystem::perms::group_read |
                                               std::filesystem::perms::others_read);
    platform::replace_file(incoming, target, true);
    cleanup.path.clear();
    platform::sync_directory(target.parent_path(), target.string());
    return hash;
}

std::optional<uint64_t> ObjectStore::artifactLength(const std::string& sha256) const
{
    auto txn=read();
    return detail::resolvedArtifactLength(ObjectStoreImplAccess::conn(txn),sha256);
}

std::vector<uint8_t> ObjectStore::readArtifact(const std::string& sha256, uint64_t offset,
                                               const std::optional<uint64_t>& count) const
{
    auto txn=read();
    return detail::readResolvedArtifact(ObjectStoreImplAccess::conn(txn),sha256,offset,count);
}
std::vector<uint8_t> detail::readResolvedArtifact(sqlite3* db,const std::string& sha256,uint64_t offset,
                                               const std::optional<uint64_t>& count)
{
    const auto artifact=resolveArtifact(db,sha256);
    if (!artifact) throw std::invalid_argument("artifact: unavailable hash " + sha256);
    if (offset > artifact->length) throw std::out_of_range("artifact: invalid byte offset");
    const auto size = count.value_or(artifact->length - offset);
    if (size > artifact->length - offset) throw std::out_of_range("artifact: invalid byte count");
    std::vector<uint8_t> result;
    result.reserve(std::size_t(size));
    Descriptor in(artifact->path, O_RDONLY);
    readBlocks(in.fd, offset, size, [&](const uint8_t* data, std::size_t n) { result.insert(result.end(), data, data + n); });
    return result;
}

} // namespace vc::objects
