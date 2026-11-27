#include "DraftCheckpoints.hpp"
#include "internal/SurfaceDrafts.hpp"
#include "internal/ObjectVersionJson.hpp"

#include <vc_reader/native/atomic_file.hpp>

#include <nlohmann/json.hpp>

#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <string_view>

namespace vc::objects::detail {
namespace {
using Json = nlohmann::json;
constexpr std::string_view kCanonicalDraft = "VCDRAFT1", kDisplayDraft = "VCDRAFT2";
constexpr std::size_t kHeaderBytes = kCanonicalDraft.size() + 2 * sizeof(int64_t);
const char* kManifest = "checkpoint.json";

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("draft checkpoint: cannot read " + path.string());
    return {std::istreambuf_iterator<char>(in), {}};
}

void writeDurably(const std::filesystem::path& path, const void* data, std::size_t size)
{
    vcr::AtomicFile file(path.string());
    file.write(static_cast<const uint8_t*>(data), size);
    file.publish(true);
}

Json versionJson(const std::optional<ObjectVersion>& v)
{
    if (!v) return nullptr;
    return objectVersionJson(*v);
}

std::optional<ObjectVersion> versionFrom(const Json& j)
{
    if (j.is_null()) return std::nullopt;
    return objectVersionFrom(j);
}

template <class T> Json nullable(const std::optional<T>& value) { return value ? Json(*value) : Json(nullptr); }

// Canonical drafts keep the original file format. Display-frame drafts add one
// presence byte per vertex in a separate plane after the validity array.
std::vector<uint8_t> encodeDraft(const SurfaceDraft& draft)
{
    const auto grid = draftGrid(draft);
    const auto count = grid.points.size();
    const auto magic = grid.coordinatePresence.empty() ? kCanonicalDraft : kDisplayDraft;
    std::vector<uint8_t> bytes(kHeaderBytes + count * (sizeof(vc::geometry::SurfacePoint) + 1) +
                               grid.coordinatePresence.size());
    auto* out = bytes.data();
    std::memcpy(out, magic.data(), magic.size()); out += magic.size();
    const int64_t shape[2] = {draft.rows, draft.columns};
    std::memcpy(out, shape, sizeof shape); out += sizeof shape;
    if (count) {
        std::memcpy(out, grid.points.data(), count * sizeof(vc::geometry::SurfacePoint)); out += count * sizeof(vc::geometry::SurfacePoint);
        std::memcpy(out, grid.validity.data(), count); out += count;
        if (!grid.coordinatePresence.empty()) std::memcpy(out, grid.coordinatePresence.data(), count);
    }
    return bytes;
}

vc::geometry::Surface decodeDraft(const std::string& bytes, int64_t rows, int64_t columns)
{
    const auto magic = std::string_view(bytes).substr(0, kCanonicalDraft.size());
    const bool explicitPresence = magic == kDisplayDraft;
    if ((!explicitPresence && magic != kCanonicalDraft) || bytes.size() < kHeaderBytes)
        throw std::runtime_error("draft checkpoint: malformed draft header");
    const auto bytesPerVertex = sizeof(vc::geometry::SurfacePoint) + 1 + explicitPresence;
    const auto payloadBytes = bytes.size() - kHeaderBytes;
    if (rows < 0 || columns < 0 || (columns && rows > INT64_MAX / columns) || payloadBytes % bytesPerVertex ||
        payloadBytes / bytesPerVertex != std::size_t(rows * columns))
        throw std::runtime_error("draft checkpoint: malformed draft file");
    int64_t shape[2];
    std::memcpy(shape, bytes.data() + magic.size(), sizeof shape);
    if (shape[0] != rows || shape[1] != columns) throw std::runtime_error("draft checkpoint: draft shape mismatch");
    const auto count = payloadBytes / bytesPerVertex;
    vc::geometry::Surface surface{rows, columns, {1, 1}, {}};
    surface.points.resize(count);
    if (count) {
        const auto* in = bytes.data() + kHeaderBytes;
        std::memcpy(surface.points.data(), in, count * sizeof(vc::geometry::SurfacePoint)); in += count * sizeof(vc::geometry::SurfacePoint);
        surface.validity.assign(in, in + count); in += count;
        if (explicitPresence) surface.coordinatePresence.assign(in, in + count);
    }
    return surface;
}

bool startsWith(const std::string& text, const std::string& prefix) { return text.rfind(prefix, 0) == 0; }
} // namespace

DraftDirectory::DraftDirectory(const std::filesystem::path& directory) : directory_(directory)
{
    std::filesystem::create_directories(directory_);
    try {
        guard_ = std::make_unique<OwnershipGuard>(directory_ / kManifest, OwnershipGuard::Mode::Service);
    } catch (const std::exception&) {
        throw std::runtime_error("draft checkpoint: " + directory_.string() + " is in use by another session");
    }
    if (std::filesystem::exists(directory_ / kManifest))
        sequence_ = Json::parse(readFile(directory_ / kManifest)).at("sequence").get<uint64_t>();
}

DraftDirectory::Loaded DraftDirectory::load()
{
    Loaded loaded;
    if (!std::filesystem::exists(directory_ / kManifest)) return loaded;
    const auto manifest = Json::parse(readFile(directory_ / kManifest));
    if (manifest.at("format") != 1) throw std::runtime_error("draft checkpoint: unsupported format");
    loaded.sequence = manifest.at("sequence");
    for (const auto& [uuid, d] : manifest.at("drafts").items()) {
        RecoveredDraft draft;
        draft.objectUuid = uuid;
        draft.coordinates = d.at("coordinates");
        files_[uuid] = d.at("file").get<std::string>();
        draft.surface = decodeDraft(readFile(directory_ / d.at("file").get<std::string>()), d.at("rows"), d.at("columns"));
        if (d.contains("channels")) {
            for (const auto& channel : jsonArrayField(d, "channels")) {
                const auto bytes = readFile(directory_ / channel.at("file").get<std::string>());
                draft.surface.channels.push_back({channel.at("name"), channel.at("dtype"), channel.at("shape"),
                                                  std::vector<uint8_t>(bytes.begin(), bytes.end())});
            }
        }
        draft.surface.validateDraft();
        if (!d.at("store").is_null()) draft.meta.storeUuid = d.at("store").get<std::string>();
        draft.meta.base = versionFrom(d.at("base"));
        if (!d.at("submitted_revision").is_null()) draft.meta.submittedRevision = d.at("submitted_revision").get<uint64_t>();
        if (!d.at("pending_request").is_null()) draft.meta.pendingRequest = d.at("pending_request").get<std::string>();
        draft.meta.conflict = d.at("conflict");
        draft.submittedAsStored = d.at("submitted_as_stored");
        loaded.drafts.push_back(std::move(draft));
    }
    for (const auto& p : jsonArrayField(manifest, "pending"))
        loaded.pending.push_back({p.at("request"), p.at("store"), p.at("drafts").get<std::map<std::string, uint64_t>>()});
    return loaded;
}

uint64_t DraftDirectory::write(const std::vector<Capture>& drafts, const std::vector<PendingRecord>& pending)
{
    const auto sequence = sequence_ + 1;
    std::map<std::string, std::string> files;
    std::map<std::string, uint64_t> written;
    std::set<std::string> channelFiles;
    Json entries = Json::object();
    for (const auto& capture : drafts) {
        const auto& draft = *capture.draft;
        auto file = files_.contains(capture.objectUuid) && written_.contains(capture.objectUuid) &&
                    written_.at(capture.objectUuid) == draft.revision
            ? files_.at(capture.objectUuid) : std::string();
        const bool writeFiles = file.empty();
        if (writeFiles) {
            file = "draft-" + capture.objectUuid + "-" + std::to_string(sequence) + ".bin";
            const auto bytes = encodeDraft(draft);
            writeDurably(directory_ / file, bytes.data(), bytes.size());
        }
        files[capture.objectUuid] = file;
        written[capture.objectUuid] = draft.revision;
        entries[capture.objectUuid] = {
            {"file", file}, {"rows", draft.rows}, {"columns", draft.columns}, {"coordinates", draft.coordinates},
            {"store", nullable(capture.meta.storeUuid)}, {"base", versionJson(capture.meta.base)},
            {"submitted_revision", nullable(capture.meta.submittedRevision)},
            {"pending_request", nullable(capture.meta.pendingRequest)}, {"conflict", capture.meta.conflict},
            {"submitted_as_stored", capture.meta.submittedRevision == draft.revision}};
        auto& channels = entries[capture.objectUuid]["channels"] = Json::array();
        if (draft.channels) {
            for (size_t i = 0; i < draft.channels->size(); ++i) {
                const auto& channel = (*draft.channels)[i];
                const auto channelFile = file + ".channel-" + std::to_string(i);
                if (writeFiles) writeDurably(directory_ / channelFile, channel.bytes.data(), channel.bytes.size());
                channelFiles.insert(channelFile);
                channels.push_back({{"name", channel.name}, {"dtype", channel.dtype}, {"shape", channel.shape},
                                    {"file", channelFile}});
            }
        }
    }
    Json requests = Json::array();
    for (const auto& p : pending) requests.push_back({{"request", p.requestUuid}, {"store", p.storeUuid}, {"drafts", p.drafts}});
    const auto manifest = Json{{"format", 1}, {"sequence", sequence}, {"drafts", entries}, {"pending", requests}}.dump(1);
    writeDurably(directory_ / kManifest, manifest.data(), manifest.size());
    sequence_ = sequence;
    files_ = std::move(files);
    written_ = std::move(written);
    // What the new checkpoint no longer names.
    std::set<std::string> keep = std::move(channelFiles);
    for (const auto& [object, file] : files_) keep.insert(file);
    std::set<std::string> requestsKept;
    for (const auto& p : pending) requestsKept.insert("request-" + p.requestUuid);
    for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
        const auto name = entry.path().filename().string();
        const bool staleDraft = startsWith(name, "draft-") && !keep.contains(name);
        const bool staleRequest = startsWith(name, "request-") &&
            std::none_of(requestsKept.begin(), requestsKept.end(), [&](const auto& prefix) { return startsWith(name, prefix); });
        std::error_code ignored;
        if (staleDraft || staleRequest) std::filesystem::remove(entry.path(), ignored);
    }
    return sequence;
}

void DraftDirectory::storeRequest(const std::string& requestUuid, const std::string& canonicalJson, const TransactionPayloads& payloads)
{
    for (const auto& [hash, bytes] : payloads)
        writeDurably(directory_ / ("request-" + requestUuid + "-" + hash + ".bin"), bytes.data(), bytes.size());
    writeDurably(directory_ / ("request-" + requestUuid + ".json"), canonicalJson.data(), canonicalJson.size());
}

TransactionRequest DraftDirectory::loadRequest(const std::string& requestUuid) const
{
    const auto prefix = "request-" + requestUuid + "-";
    TransactionPayloads payloads;
    for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
        const auto name = entry.path().filename().string();
        if (!startsWith(name, prefix) || entry.path().extension() != ".bin") continue;
        const auto bytes = readFile(entry.path());
        payloads.emplace(name.substr(prefix.size(), name.size() - prefix.size() - 4), std::vector<uint8_t>(bytes.begin(), bytes.end()));
    }
    return transactionRequestFromJson(readFile(directory_ / ("request-" + requestUuid + ".json")), payloads);
}

} // namespace vc::objects::detail
