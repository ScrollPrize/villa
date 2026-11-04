// Content-addressed artifact bytes beside the database, and the manifests
// that name them in object metadata (objects/Artifacts.hpp).
#include "vc_object_store/objects/Artifacts.hpp"

#include "vc_object_store/objects/TransactionJson.hpp"
#include "internal/Artifacts.hpp"
#include "internal/Database.hpp"

#include <vc_reader/native/atomic_file.hpp>
#include <vc_reader/native/platform_io.hpp>

#include <fcntl.h>

#include <algorithm>
#include <set>

namespace vc::objects {
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

bool isArtifactHash(const std::string& text)
{
    return text.size() == 64 && std::all_of(text.begin(), text.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

std::filesystem::path artifactBlobDirectory(const std::filesystem::path& storePath)
{
    return std::filesystem::path(storePath.string() + ".artifacts") / "sha256";
}

std::filesystem::path artifactBlobPath(const std::filesystem::path& storePath, const std::string& sha256)
{
    if (!isArtifactHash(sha256)) throw std::invalid_argument("artifact: invalid SHA-256 " + sha256);
    return artifactBlobDirectory(storePath) / sha256.substr(0, 2) / sha256;
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

void detail::checkArtifactReferences(const std::filesystem::path& storePath, const std::string& objectUuid,
                                     const Json& manifest)
{
    for (const auto& file : parseManifest(manifest)) {
        std::error_code ec;
        const auto size = std::filesystem::file_size(artifactBlobPath(storePath, file.sha256), ec);
        if (ec || size != file.length)
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
    std::error_code ec;
    const auto size = std::filesystem::file_size(artifactBlobPath(impl_->path, sha256), ec);
    if (ec) return std::nullopt;
    return size;
}

std::vector<uint8_t> ObjectStore::readArtifact(const std::string& sha256, uint64_t offset,
                                               const std::optional<uint64_t>& count) const
{
    const auto path = artifactBlobPath(impl_->path, sha256);
    const auto length = artifactLength(sha256);
    if (!length) throw std::invalid_argument("artifact: this store does not hold " + sha256);
    if (offset > *length) throw std::out_of_range("artifact: invalid byte offset");
    const auto size = count.value_or(*length - offset);
    if (size > *length - offset) throw std::out_of_range("artifact: invalid byte count");
    std::vector<uint8_t> result;
    result.reserve(std::size_t(size));
    Descriptor in(path, O_RDONLY);
    readBlocks(in.fd, offset, size, [&](const uint8_t* data, std::size_t n) { result.insert(result.end(), data, data + n); });
    return result;
}

} // namespace vc::objects
