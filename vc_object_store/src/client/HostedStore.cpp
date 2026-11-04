#include "vc_object_store/client/HostedStore.hpp"

#include "vc_object_store/objects/TransactionJson.hpp"

#include <nlohmann/json.hpp>
#include <vc_reader/http.hpp>

#include <stdexcept>

namespace vc::objects {
using Json = nlohmann::json;

namespace {
std::string serverError(const vcr::http::Response& response)
{
    try {
        const auto j = Json::parse(response.body);
        if (j.contains("error")) return j.at("error").get<std::string>();
    } catch (const Json::exception&) {
    }
    return response.body.substr(0, 300);
}
} // namespace

struct HostedStore::Impl {
    explicit Impl(const HostedStoreOptions& o)
        : http(vcr::http::Options{o.connections, o.connectTimeoutMs, o.stallTimeoutS, o.retries, o.requestTimeoutMs}) {}
    vcr::http::Session http;
};

HostedStore::HostedStore(std::string endpoint, HostedStoreOptions options)
    : endpoint_(std::move(endpoint)), impl_(std::make_unique<Impl>(options))
{
    while (!endpoint_.empty() && endpoint_.back() == '/') endpoint_.pop_back();
    if (endpoint_.empty()) throw std::invalid_argument("hosted store: endpoint required");
}

HostedStore::~HostedStore() = default;

namespace {
// The answer of `response`, or the matching exception: 409 conflicts are
// TransactionConflict, other 4xx are invalid requests.
const vcr::http::Response& expectOk(const vcr::http::Response& response, const std::string& what)
{
    if (response.status == 200) return response;
    if (response.status == 409) throw transactionConflictFromJson(response.body);
    const auto message = "hosted store: " + what + ": HTTP " + std::to_string(response.status) + ": " + serverError(response);
    if (response.status / 100 == 4) throw std::invalid_argument(message);
    throw std::runtime_error(message);
}
} // namespace

StoreStatus HostedStore::status()
{
    return storeStatusFromJson(expectOk(impl_->http.request("GET", endpoint_ + "/v1/status"), "status").body);
}

TransactionReceipt HostedStore::submit(const TransactionRequest& request)
{
    TransactionPayloads payloads;
    const auto body = transactionRequestToJson(request, &payloads);
    auto upload = [&](const std::string& hash) {
        const auto& bytes = payloads.at(hash);
        const auto response = impl_->http.request("PUT", endpoint_ + "/v1/payloads/" + hash,
            std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()),
            {{"Content-Type", "application/octet-stream"}});
        expectOk(response, "stage payload " + hash);
    };
    for (const auto& [hash, bytes] : payloads) upload(hash);
    // Staging can expire between upload and submit (or before a retry): the
    // server names what it lacks and the identical request is sent again.
    for (int attempt = 0;; ++attempt) {
        const auto response = impl_->http.request("POST", endpoint_ + "/v1/transactions", body,
            {{"Content-Type", "application/json"}});
        if (response.status != 422 || attempt == 3) {
            return transactionReceiptFromJson(expectOk(response, "submit " + request.requestUuid).body);
        }
        for (const auto& hash : Json::parse(response.body).at("missing")) {
            const auto name = hash.get<std::string>();
            if (!payloads.contains(name)) throw std::runtime_error("hosted store: server asked for an unknown payload " + name);
            upload(name);
        }
    }
}

std::optional<TransactionReceipt> HostedStore::receipt(const std::string& requestUuid)
{
    const auto response = impl_->http.request("GET", endpoint_ + "/v1/receipts/" + requestUuid);
    if (response.status == 404) return std::nullopt;
    return transactionReceiptFromJson(expectOk(response, "receipt").body);
}

ChangePage HostedStore::changesSince(const FeedCursor& cursor, int64_t maxChanges)
{
    const auto url = endpoint_ + "/v1/changes?" + feedCursorQuery(cursor) + "&max=" + std::to_string(maxChanges);
    return changePageFromJson(expectOk(impl_->http.request("GET", url), "changes").body);
}

std::optional<ObjectRevision> HostedStore::currentObjectRevision(const std::string& objectUuid, StoreStatus* asOf)
{
    const auto response = impl_->http.request("GET", endpoint_ + "/v1/objects/" + objectUuid);
    if (response.status == 404) return std::nullopt;
    const auto j = Json::parse(expectOk(response, "object").body);
    if (asOf) *asOf = storeStatusFromJson(j.at("status").dump());
    return objectRevisionFromJson(j.at("revision").dump());
}

std::optional<int64_t> HostedStore::currentVersion(const std::string& objectUuid)
{
    const auto revision = currentObjectRevision(objectUuid);
    if (!revision || revision->deleted) return std::nullopt;
    return revision->version;
}

std::optional<ObjectRevision> HostedStore::objectAtVersion(const ObjectVersion& reference)
{
    const auto url = endpoint_ + "/v1/objects/" + reference.objectUuid + "/versions/" + std::to_string(reference.version) +
                     "?store=" + reference.storeUuid + "&generation=" + reference.generation;
    const auto response = impl_->http.request("GET", url);
    if (response.status == 404) return std::nullopt;
    return objectRevisionFromJson(expectOk(response, "object version").body);
}

std::optional<GeometryRevision> HostedStore::geometryRevision(const std::string& revisionUuid)
{
    const auto response = impl_->http.request("GET", endpoint_ + "/v1/geometry/" + revisionUuid);
    if (response.status == 404) return std::nullopt;
    return geometryRevisionFromJson(expectOk(response, "geometry").body);
}

std::vector<uint8_t> HostedStore::readGeometryArray(const std::string& revisionUuid, const std::string& arrayName,
                                                    int64_t byteOffset, const std::optional<int64_t>& byteCount)
{
    auto url = endpoint_ + "/v1/geometry/" + revisionUuid + "/arrays/" + arrayName + "?offset=" + std::to_string(byteOffset);
    if (byteCount) url += "&count=" + std::to_string(*byteCount);
    const auto response = impl_->http.request("GET", url);
    const auto& body = expectOk(response, "geometry array").body;
    return {body.begin(), body.end()};
}

namespace {
// Pages a listing until a short page, restarting whenever a page comes from
// another snapshot than the first. `page(cursorFrom)` reads one page after
// the cursor state captured from the previous page's last record.
template<class Record, class Fetch, class Advance>
std::vector<Record> pageAll(int64_t pageSize, int maxRestarts, StoreStatus* asOf, Fetch fetch, Advance advance)
{
    if (pageSize <= 0) throw std::invalid_argument("hosted store: page size must be positive");
    for (int attempt = 0; attempt <= maxRestarts; ++attempt) {
        std::vector<Record> all;
        StoreStatus first, current;
        bool restart = false;
        for (bool start = true;; start = false) {
            auto page = fetch(start, all, start ? &first : &current);
            if (!start && (current.storeUuid != first.storeUuid || current.generation != first.generation ||
                           current.commitSequence != first.commitSequence)) {
                restart = true;
                break;
            }
            const bool last = int64_t(page.size()) < pageSize;
            advance(page);
            all.insert(all.end(), std::make_move_iterator(page.begin()), std::make_move_iterator(page.end()));
            if (last) break;
        }
        if (restart) continue;
        if (asOf) *asOf = first;
        return all;
    }
    throw StoreBusy("hosted store: the store kept changing while paging a listing; retry later");
}
} // namespace

std::vector<ObjectRecord> HostedStore::selectObjects(const ObjectSelection& selection,
    const std::optional<std::vector<std::string>>& fields, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", endpoint_ + "/v1/query/objects",
        objectQueryRequestToJson(selection, fields), {{"Content-Type", "application/json"}});
    return objectQueryResultFromJson(expectOk(response, "object query").body, asOf);
}

std::vector<EdgeRecord> HostedStore::selectEdges(const EdgeSelection& selection,
    const std::optional<std::vector<std::string>>& fields, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", endpoint_ + "/v1/query/edges",
        edgeQueryRequestToJson(selection, fields), {{"Content-Type", "application/json"}});
    return edgeQueryResultFromJson(expectOk(response, "edge query").body, asOf);
}

std::vector<ObjectRecord> HostedStore::selectAllObjects(ObjectSelection selection,
    const std::optional<std::vector<std::string>>& fields, int64_t pageSize, int maxRestarts, StoreStatus* asOf)
{
    if (selection.limit) throw std::invalid_argument("hosted store: selectAllObjects pages itself; remove the limit");
    // UUID order pages by cursor. Length order and volume listings with
    // unbound objects page by offset, which is exact within one snapshot.
    const bool byCursor = selection.orderBy == "uuid" && !selection.includeUnbound;
    if (!byCursor && (selection.afterUuid || selection.afterLength))
        throw std::invalid_argument("hosted store: selectAllObjects starts length/unbound listings at an offset, not a cursor");
    const auto start = selection;
    selection.limit = pageSize;
    return pageAll<ObjectRecord>(pageSize, maxRestarts, asOf,
        [&](bool first, const std::vector<ObjectRecord>& all, StoreStatus* status) {
            if (first) {
                selection.afterUuid = start.afterUuid;
                selection.offset = start.offset;
            } else if (!byCursor) {
                selection.offset = start.offset + int64_t(all.size());
            }
            return selectObjects(selection, fields, status);
        },
        [&](const std::vector<ObjectRecord>& page) {
            if (byCursor && !page.empty()) {
                selection.afterUuid = page.back().uuid;
                selection.offset = 0;
            }
        });
}

std::vector<EdgeRecord> HostedStore::selectAllEdges(EdgeSelection selection,
    const std::optional<std::vector<std::string>>& fields, int64_t pageSize, int maxRestarts, StoreStatus* asOf)
{
    if (selection.limit) throw std::invalid_argument("hosted store: selectAllEdges pages itself; remove the limit");
    const auto start = selection.afterUuid;
    selection.limit = pageSize;
    return pageAll<EdgeRecord>(pageSize, maxRestarts, asOf,
        [&](bool first, const std::vector<EdgeRecord>&, StoreStatus* status) {
            if (first) selection.afterUuid = start;
            return selectEdges(selection, fields, status);
        },
        [&](const std::vector<EdgeRecord>& page) {
            if (!page.empty()) selection.afterUuid = page.back().uuid;
        });
}

std::vector<ObjectRevision> HostedStore::objectHistory(const std::string& objectUuid, StoreStatus* asOf)
{
    const auto response = impl_->http.request("GET", endpoint_ + "/v1/objects/" + objectUuid + "/history");
    if (response.status == 404) {
        if (asOf) *asOf = status();
        return {};
    }
    return objectHistoryFromJson(expectOk(response, "object history").body, asOf);
}

std::vector<SelectedRelationship> HostedStore::relationshipsAt(const std::vector<ObjectVersion>& versions, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", endpoint_ + "/v1/query/relationships",
        relationshipQueryRequestToJson(versions), {{"Content-Type", "application/json"}});
    return relationshipQueryResultFromJson(expectOk(response, "relationship query").body, asOf);
}

RegionResult HostedStore::geometryRegion(const RegionSlice& slice, const RegionOptions& options, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", endpoint_ + "/v1/query/region",
        regionQueryRequestToJson(slice, options), {{"Content-Type", "application/json"}});
    return regionQueryResultFromJson(expectOk(response, "region query").body, asOf);
}

std::vector<SurfaceProximityHit> HostedStore::nearestSurfaces(const SurfaceProximityQuery& query, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", endpoint_ + "/v1/query/nearest_surfaces",
        proximityQueryRequestToJson(query), {{"Content-Type", "application/json"}});
    return proximityQueryResultFromJson(expectOk(response, "proximity query").body, asOf);
}

} // namespace vc::objects
