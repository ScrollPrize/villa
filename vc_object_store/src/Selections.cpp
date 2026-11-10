#include "vc_object_store/objects/Selections.hpp"

#include "internal/Sqlite.hpp"
#include "internal/History.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace vc::objects {
using Json = nlohmann::json;
using detail::formatUuid;
using detail::parseUuid;

namespace {
constexpr char kMagic[8] = {'V', 'C', 'S', 'E', 'L', '\0', '\2', '\n'};
constexpr std::size_t kEntryBytes = 24;

void putU32(std::vector<uint8_t>& out, uint32_t value)
{
    for (int i = 0; i < 4; ++i) out.push_back(uint8_t(value >> (8 * i)));
}

uint32_t getU32(const uint8_t* p)
{
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

Json versionJson(const ObjectVersion& v)
{
    return Json{{"store", v.storeUuid}, {"generation", v.generation}, {"object", v.objectUuid}, {"version", v.version}};
}

std::string uuidField(const Json& j, const char* key)
{
    return formatUuid(parseUuid(j.at(key).get<std::string>()));
}
} // namespace

std::vector<uint8_t> encodeSelectionRecord(const SelectionRecord& record)
{
    using Origin=std::pair<std::string,std::string>;
    std::vector<Origin> origins;
    std::map<Origin, uint32_t> originIndex;
    for (const auto& object : record.objects) {
        const Origin origin{object.storeUuid,object.generation};
        if (originIndex.emplace(origin, uint32_t(origins.size())).second)
            origins.push_back(origin);
    }
    auto transforms = Json::array();
    for (const auto& transform : record.transforms) transforms.push_back(versionJson(transform));
    const auto header = Json{{"store", record.storeUuid}, {"generation", record.generation},
                             {"commit_sequence", record.commitSequence}, {"query", Json::parse(record.queryJson)},
                             {"transforms", std::move(transforms)}, {"origins", origins},
                             {"count", record.objects.size()}}.dump();
    if (header.size() > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("selection record: header too large");
    std::vector<uint8_t> out(kMagic, kMagic + sizeof kMagic);
    out.reserve(sizeof kMagic + 4 + header.size() + record.objects.size() * kEntryBytes);
    putU32(out, uint32_t(header.size()));
    out.insert(out.end(), header.begin(), header.end());
    for (const auto& object : record.objects) {
        if (object.version <= 0 || object.version > int64_t(std::numeric_limits<uint32_t>::max()))
            throw std::invalid_argument("selection record: version out of range for " + object.objectUuid);
        const auto uuid = parseUuid(object.objectUuid);
        out.insert(out.end(), uuid.begin(), uuid.end());
        putU32(out, originIndex.at({object.storeUuid,object.generation}));
        putU32(out, uint32_t(object.version));
    }
    return out;
}

SelectionRecord decodeSelectionRecord(std::span<const uint8_t> bytes)
{
    auto bad = [](const std::string& why) { return std::invalid_argument("selection record: " + why); };
    if (bytes.size() < sizeof kMagic + 4 || std::memcmp(bytes.data(), kMagic, 6) != 0 ||
        (bytes[6]!=1&&bytes[6]!=2) || bytes[7]!=kMagic[7])
        throw bad("not a selection record (or an unknown format version)");
    const std::size_t headerBytes = getU32(bytes.data() + sizeof kMagic);
    const std::size_t entriesAt = sizeof kMagic + 4 + headerBytes;
    if (headerBytes > bytes.size() - sizeof kMagic - 4) throw bad("truncated header");
    SelectionRecord record;
    std::vector<std::pair<std::string,std::string>> origins;
    uint64_t count = 0;
    try {
        const auto header = Json::parse(bytes.begin() + sizeof kMagic + 4, bytes.begin() + entriesAt);
        record.storeUuid = uuidField(header, "store");
        record.generation = uuidField(header, "generation");
        record.commitSequence = header.at("commit_sequence").get<int64_t>();
        record.queryJson = header.at("query").dump();
        for (const auto& t : header.at("transforms")) {
            const auto version = t.at("version").get<int64_t>();
            if (version <= 0) throw bad("transform version must be positive");
            record.transforms.push_back({bytes[6]==1?record.storeUuid:uuidField(t,"store"), uuidField(t, "generation"), uuidField(t, "object"), version});
        }
        if(bytes[6]==1) {
            for(const auto& g:header.at("generations"))origins.emplace_back(record.storeUuid,formatUuid(parseUuid(g.get<std::string>())));
        } else for(const auto& origin:header.at("origins")) {
            if(!origin.is_array()||origin.size()!=2)throw bad("invalid origin");
            origins.emplace_back(formatUuid(parseUuid(origin.at(0).get<std::string>())),formatUuid(parseUuid(origin.at(1).get<std::string>())));
        }
        count = header.at("count").get<uint64_t>();
    } catch (const Json::exception& e) {
        throw bad(std::string("malformed header: ") + e.what());
    }
    if ((bytes.size() - entriesAt) % kEntryBytes != 0 || (bytes.size() - entriesAt) / kEntryBytes != count)
        throw bad("entry count does not match its length");
    record.objects.reserve(count);
    for (std::size_t at = entriesAt; at < bytes.size(); at += kEntryBytes) {
        std::array<unsigned char, 16> uuid{};
        std::memcpy(uuid.data(), bytes.data() + at, 16);
        const auto generation = getU32(bytes.data() + at + 16);
        const auto version = getU32(bytes.data() + at + 20);
        if (generation >= origins.size()) throw bad("origin index out of range");
        if (version == 0) throw bad("version must be positive");
        record.objects.push_back({origins[generation].first, origins[generation].second, formatUuid(uuid), int64_t(version)});
    }
    return record;
}

bool selectionUsesTransforms(const ObjectSelection& selection)
{
    return selection.volume.has_value() || selection.coordinates.has_value();
}

SelectionRecord recordSelection(const ObjectStore& store, const Txn& txn, const ObjectSelection& selection)
{
    detail::requirePublishedGraphRead(txn);
    if (selection.transforms)
        throw std::invalid_argument("selection record: selections use the store's recorded transforms, not explicit ones");
    if (selection.limit) throw std::invalid_argument("selection record: records a whole selection; remove the limit");
    const auto status = store.status(txn);
    SelectionRecord record{status.storeUuid, status.generation, status.commitSequence,
                           objectQueryRequestToJson(selection, std::nullopt), {}, {}};
    auto versions = [&](const std::vector<ObjectRecord>& records, std::vector<ObjectVersion>& out) {
        for (const auto& r : records) {
            if (!r.version)
                throw std::invalid_argument("selection record: requires a read transaction (published versions)");
            out.push_back(r.reference());
        }
    };
    versions(store.selectObjects(txn, selection, std::vector<std::string>{}), record.objects);
    if (selectionUsesTransforms(selection)) {
        // Client catalog registrations the store has not recorded would apply
        // to this query but cannot be named by the record.
        std::set<std::string> recorded;
        for (const auto& transform : store.objectsByKind(txn, "volume_transform")) {
            const auto meta = Json::parse(transform.meta);
            recorded.insert(meta.contains("catalog_registration")
                ? meta.at("catalog_registration").at("id").get<std::string>() : transform.uuid);
        }
        const auto available = store.volumeTransforms(txn);  // the loop reads its registrations by reference
        for (const auto& registration : available.graph.registrations())
            if (!recorded.contains(registration.id))
                throw std::invalid_argument("selection record: catalog registration " + registration.id +
                                            " is not recorded in the store; record it first (recordCatalogTransforms)");
        std::vector<ObjectRecord> transforms;
        for (const auto* kind : {"volume_transform", "volume_transform_choice"}) {
            ObjectSelection byKind;
            byKind.kind = kind;
            auto part = store.selectObjects(txn, byKind, std::vector<std::string>{});
            transforms.insert(transforms.end(), part.begin(), part.end());
        }
        std::sort(transforms.begin(), transforms.end(),
                  [](const ObjectRecord& a, const ObjectRecord& b) { return a.uuid < b.uuid; });
        versions(transforms, record.transforms);
    }
    return record;
}

} // namespace vc::objects
