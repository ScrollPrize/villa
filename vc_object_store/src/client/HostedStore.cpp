#include "vc_object_store/client/HostedStore.hpp"

#include "vc_object_store/objects/TransactionJson.hpp"
#include "internal/Artifacts.hpp"

#include <nlohmann/json.hpp>
#include <vc_reader/native/atomic_file.hpp>
#include <vc_reader/native/platform_io.hpp>

#include <fcntl.h>
#include <vc_reader/http.hpp>
#include <variant>
#include <vc_reader/sftp.hpp>
#include <vc_reader/native/sigv4.hpp>

#include <algorithm>
#include <stdexcept>
#include <set>
#include <map>
#include <functional>
#include <iterator>

namespace vc::objects {
using Json = nlohmann::json;

namespace {
std::string arrayPathSegment(const std::string& name)
{
    // Dot-only names are valid arrays but URL clients normalize literal dot segments.
    if (name == ".") return "%2E";
    if (name == "..") return "%2E%2E";
    return vcr::aws_uri_encode(name, false);
}

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
    explicit Impl(const HostedStoreOptions& o, bool ssh)
        : http(vcr::http::Options{o.connections, o.connectTimeoutMs, o.stallTimeoutS, o.retries, o.requestTimeoutMs, true, ssh}),
          linkHttp(vcr::http::Options{1, o.connectTimeoutMs, o.linkTimeoutS, o.retries, 0, true, ssh}) {}
    std::unique_ptr<vcr::sftp::Forward> forward;
    vcr::http::Session http;
    vcr::http::Session linkHttp; // link_objects requests: the detection's duration, not a stall
};

HostedStore::HostedStore(std::string endpoint, HostedStoreOptions options)
    : endpoint_(std::move(endpoint)), rangeBytes_(std::max<int64_t>(options.rangeBytes, 1)),
      impl_(std::make_unique<Impl>(options, vcr::sftp::isSftpUrl(endpoint_)))
{
    while (!endpoint_.empty() && endpoint_.back() == '/') endpoint_.pop_back();
    if (endpoint_.empty()) throw std::invalid_argument("hosted store: endpoint required");
    if (vcr::sftp::isSftpUrl(endpoint_))
        impl_->forward = std::make_unique<vcr::sftp::Forward>(endpoint_);
    requestEndpoint_ = impl_->forward ? impl_->forward->endpoint() : endpoint_;
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
    return storeStatusFromJson(expectOk(impl_->http.request("GET", requestEndpoint_ + "/v1/status"), "status").body);
}

std::string HostedStore::serviceReport()
{
    return expectOk(impl_->http.request("GET", requestEndpoint_ + "/v1/service"), "service").body;
}

TransactionReceipt HostedStore::submit(const TransactionRequest& request)
{
    TransactionPayloads payloads;
    const auto body = transactionRequestToJson(request, &payloads);
    auto upload = [&](const std::string& hash) {
        const auto& bytes = payloads.at(hash);
        const auto response = impl_->http.request("PUT", requestEndpoint_ + "/v1/payloads/" + hash,
            std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()),
            {{"Content-Type", "application/octet-stream"}});
        expectOk(response, "stage payload " + hash);
    };
    for (const auto& [hash, bytes] : payloads) upload(hash);
    bool prepared = false;
    for (const auto& op : request.operations) prepared = prepared || std::holds_alternative<LinkObjectsOp>(op);
    auto& http = prepared ? impl_->linkHttp : impl_->http;
    // Staging can expire between upload and submit (or before a retry): the
    // server names what it lacks and the identical request is sent again.
    for (int attempt = 0;; ++attempt) {
        const auto response = http.request("POST", requestEndpoint_ + "/v1/transactions", body,
            {{"Content-Type", "application/json"}});
        if (response.status != 422 || attempt == 3) {
            return transactionReceiptFromJson(expectOk(response, "submit " + request.requestUuid).body);
        }
        const auto answer = Json::parse(response.body);  // the loop reads into it
        if (!answer.contains("missing"))
            throw PreparationFailed("hosted store: submit " + request.requestUuid + ": " + serverError(response));
        for (const auto& hash : answer.at("missing")) {
            const auto name = hash.get<std::string>();
            if (!payloads.contains(name)) throw std::runtime_error("hosted store: server asked for an unknown payload " + name);
            upload(name);
        }
    }
}

std::optional<TransactionReceipt> HostedStore::receipt(const std::string& requestUuid)
{
    const auto response = impl_->http.request("GET", requestEndpoint_ + "/v1/receipts/" + requestUuid);
    if (response.status == 404) return std::nullopt;
    return transactionReceiptFromJson(expectOk(response, "receipt").body);
}

ChangePage HostedStore::changesSince(const FeedCursor& cursor, int64_t maxChanges)
{
    const auto url = requestEndpoint_ + "/v1/changes?" + feedCursorQuery(cursor) + "&max=" + std::to_string(maxChanges);
    return changePageFromJson(expectOk(impl_->http.request("GET", url), "changes").body);
}

std::optional<ObjectRevision> HostedStore::currentObjectRevision(const std::string& objectUuid, StoreStatus* asOf)
{
    const auto response = impl_->http.request("GET", requestEndpoint_ + "/v1/objects/" + objectUuid);
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
    const auto url = requestEndpoint_ + "/v1/objects/" + reference.objectUuid + "/versions/" + std::to_string(reference.version) +
                     "?store=" + reference.storeUuid + "&generation=" + reference.generation;
    const auto response = impl_->http.request("GET", url);
    if (response.status == 404) return std::nullopt;
    return objectRevisionFromJson(expectOk(response, "object version").body);
}

std::optional<GeometryRevision> HostedStore::geometryRevision(const std::string& revisionUuid)
{
    const auto response = impl_->http.request("GET", requestEndpoint_ + "/v1/geometry/" + revisionUuid);
    if (response.status == 404) return std::nullopt;
    return geometryRevisionFromJson(expectOk(response, "geometry").body);
}

std::vector<uint8_t> HostedStore::readGeometryArray(const std::string& revisionUuid, const std::string& arrayName,
                                                    int64_t byteOffset, const std::optional<int64_t>& byteCount)
{
    auto get = [&](int64_t offset, const std::optional<int64_t>& count) {
        auto url = requestEndpoint_ + "/v1/geometry/" + revisionUuid + "/arrays/" + arrayPathSegment(arrayName) + "?offset=" + std::to_string(offset);
        if (count) url += "&count=" + std::to_string(*count);
        return impl_->http.request("GET", url);
    };
    auto ranged = [&](int64_t total, int64_t range) {
        std::vector<uint8_t> result;
        result.reserve(static_cast<std::size_t>(total));
        for (int64_t done = 0; done < total;) {
            const auto want = std::min(range, total - done);
            const auto response = get(byteOffset + done, want);
            if (response.status == 413) {
                int64_t serverMax = 0;
                try { serverMax = Json::parse(response.body).value("max_response_bytes", int64_t(0)); } catch (const Json::exception&) {}
                if (serverMax > 0 && serverMax < want) {
                    range = serverMax;
                    continue; // retry the same offset within the server's limit
                }
            }
            const auto& body = expectOk(response, "geometry array").body;
            result.insert(result.end(), body.begin(), body.end());
            done += want;
        }
        return result;
    };
    if (byteCount && *byteCount > rangeBytes_) return ranged(*byteCount, rangeBytes_);
    const auto response = get(byteOffset, byteCount);
    if (response.status == 413) {
        // Larger than the server answers at once: read ranges it accepts.
        int64_t serverMax = 0;
        try { serverMax = Json::parse(response.body).value("max_response_bytes", int64_t(0)); } catch (const Json::exception&) {}
        if (serverMax > 0) {
            if (byteCount) return ranged(*byteCount, std::min(rangeBytes_, serverMax));
            const auto revision = geometryRevision(revisionUuid);
            if (!revision) throw std::invalid_argument("hosted store: no geometry revision " + revisionUuid);
            for (const auto& info : revision->arrays)
                if (info.name == arrayName) return ranged(info.byteLength - byteOffset, std::min(rangeBytes_, serverMax));
        }
    }
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

// Catch-up listing: pages may come from different snapshots. Commits that
// land while paging are read from the change feed, and the records they
// touched are selected again (restricted to their UUIDs, one snapshot per
// chunk) until the feed is quiet. The result is the exact listing as of the
// feed position it ends at, whatever the store size; it restarts only when a
// change can move membership for records it did not touch.
enum class ChangeEffect { Ignore, Recheck, Restart };

template<class Record>
struct CatchUp {
    HostedStore& store;
    int64_t pageSize;
    int maxRestarts, maxRounds = 32;
    // (cursor uuid or nullopt for the first page, status out) -> one page
    std::function<std::vector<Record>(const std::optional<std::string>&, StoreStatus*)> page;
    // (uuids to re-select, status out) -> their current matches
    std::function<std::vector<Record>(const std::vector<std::string>&, StoreStatus*)> requery;
    // How one feed change affects this listing.
    std::function<ChangeEffect(const ChangeRecord&)> effect;
    // Extra check of the rechecked UUIDs (e.g. transform records): true restarts.
    std::function<bool(const std::vector<std::string>&)> restartFor;

    std::vector<Record> run(StoreStatus* asOf)
    {
        if (pageSize <= 0) throw std::invalid_argument("hosted store: page size must be positive");
        for (int attempt = 0; attempt <= maxRestarts; ++attempt)
            if (auto result = once(asOf)) return std::move(*result);
        throw StoreBusy("hosted store: the store kept changing while paging a listing; retry later");
    }

private:
    bool sameLineage(const StoreStatus& a, const StoreStatus& b)
    {
        return a.storeUuid == b.storeUuid && a.generation == b.generation;
    }

    std::optional<std::vector<Record>> once(StoreStatus* asOf)
    {
        std::map<std::string, Record> found;
        StoreStatus first, status;
        int64_t high = 0;
        std::optional<std::string> cursor;
        for (bool start = true;; start = false) {
            auto records = page(cursor, start ? &first : &status);
            if (start) status = first;
            if (!sameLineage(first, status)) return std::nullopt;
            // With no retained feed, every page must come from the same committed state.
            if (!first.historyEnabled && status.commitSequence != first.commitSequence) return std::nullopt;
            high = std::max(high, status.commitSequence);
            const bool last = int64_t(records.size()) < pageSize;
            if (!records.empty()) cursor = records.back().uuid;
            for (auto& record : records) found.insert_or_assign(record.uuid, std::move(record));
            if (last) break;
        }
        FeedCursor feed{first.storeUuid, first.generation, first.commitSequence, 0};
        if (!first.historyEnabled) {
            const auto current = store.status();
            if (!sameLineage(first, current) || first.commitSequence != current.commitSequence) return std::nullopt;
            if (asOf) *asOf = current;
            std::vector<Record> result;
            result.reserve(found.size());
            for (auto& [uuid, record] : found) result.push_back(std::move(record));
            return result;
        }
        for (int round = 0; round < maxRounds; ++round) {
            std::set<std::string> changed;
            StoreStatus end = first;
            for (;;) {
                const auto changes = store.changesSince(feed);
                if (!changes.cursorValid) return std::nullopt;  // retention or lineage moved: start over
                for (const auto& commit : changes.commits)
                    for (const auto& change : commit.changes) {
                        const auto e = effect(change);
                        if (e == ChangeEffect::Restart) return std::nullopt;
                        if (e == ChangeEffect::Recheck) changed.insert(change.uuid);
                    }
                feed = changes.next;
                if (!changes.more) break;
            }
            end.commitSequence = feed.sequence;
            if (changed.empty() && feed.sequence >= high) {
                if (asOf) *asOf = end;
                std::vector<Record> result;
                result.reserve(found.size());
                for (auto& [uuid, record] : found) result.push_back(std::move(record));
                return result;
            }
            const std::vector<std::string> uuids(changed.begin(), changed.end());
            if (restartFor && restartFor(uuids)) return std::nullopt;
            for (std::size_t i = 0; i < uuids.size(); i += std::size_t(pageSize)) {
                const std::vector<std::string> chunk(uuids.begin() + i,
                    uuids.begin() + std::min(uuids.size(), i + std::size_t(pageSize)));
                StoreStatus at;
                auto matches = requery(chunk, &at);
                if (!sameLineage(first, at)) return std::nullopt;
                high = std::max(high, at.commitSequence);
                for (const auto& uuid : chunk) found.erase(uuid);
                for (auto& record : matches) found.insert_or_assign(record.uuid, std::move(record));
            }
        }
        return std::nullopt;
    }
};

// Whether any of these objects is (or, deleted, was) a recorded transform or
// route choice, retired ones included: those move volume matches.
bool touchesTransforms(HostedStore& store, const std::vector<std::string>& uuids, const std::set<std::string>& deleted,
                       int64_t pageSize)
{
    auto transformKind = [](const std::string& kind) { return kind.starts_with("volume_transform"); };
    for (std::size_t i = 0; i < uuids.size(); i += std::size_t(pageSize)) {
        ObjectSelection kinds;
        kinds.uuids.assign(uuids.begin() + i, uuids.begin() + std::min(uuids.size(), i + std::size_t(pageSize)));
        kinds.limit = int64_t(kinds.uuids.size());
        kinds.includeRetired = true;
        for (const auto& record : store.selectObjects(kinds, std::vector<std::string>{"kind"}))
            if (transformKind(record.kind)) return true;
    }
    for (const auto& uuid : uuids)
        if (deleted.contains(uuid))
            for (const auto& revision : store.objectHistory(uuid))
                if (transformKind(revision.record.kind)) return true;
    return false;
}

std::vector<std::string> restrictTo(const std::vector<std::string>& chunk, const std::vector<std::string>& allowed)
{
    if (allowed.empty()) return chunk;
    std::set<std::string> keep(allowed.begin(), allowed.end());
    std::vector<std::string> result;
    for (const auto& uuid : chunk)
        if (keep.contains(uuid)) result.push_back(uuid);
    return result;
}
} // namespace

std::vector<ObjectRecord> HostedStore::selectObjects(const ObjectSelection& selection,
    const std::optional<std::vector<std::string>>& fields, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", requestEndpoint_ + "/v1/query/objects",
        objectQueryRequestToJson(selection, fields), {{"Content-Type", "application/json"}});
    return objectQueryResultFromJson(expectOk(response, "object query").body, asOf);
}

std::vector<EdgeRecord> HostedStore::selectEdges(const EdgeSelection& selection,
    const std::optional<std::vector<std::string>>& fields, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", requestEndpoint_ + "/v1/query/edges",
        edgeQueryRequestToJson(selection, fields), {{"Content-Type", "application/json"}});
    return edgeQueryResultFromJson(expectOk(response, "edge query").body, asOf);
}

std::vector<ObjectRecord> HostedStore::selectAllObjects(ObjectSelection selection,
    const std::optional<std::vector<std::string>>& fields, int64_t pageSize, int maxRestarts, StoreStatus* asOf)
{
    if (selection.limit) throw std::invalid_argument("hosted store: selectAllObjects pages itself; remove the limit");
    // UUID order pages by cursor and catches up from the feed. Length order
    // and volume listings with unbound objects, or an explicit initial offset,
    // page by offset within one snapshot: they restart when a commit lands.
    const bool byCursor = selection.orderBy == "uuid" && !selection.includeUnbound && selection.offset == 0;
    if (!byCursor && (selection.afterUuid || selection.afterLength))
        throw std::invalid_argument("hosted store: selectAllObjects cannot combine a cursor with offset paging");
    const auto start = selection;
    if (byCursor) {
        CatchUp<ObjectRecord> listing{*this, pageSize, maxRestarts};
        listing.page = [&](const std::optional<std::string>& after, StoreStatus* status) {
            auto page = start;
            page.limit = pageSize;
            page.afterUuid = after ? after : start.afterUuid;
            return selectObjects(page, fields, status);
        };
        listing.requery = [&](const std::vector<std::string>& uuids, StoreStatus* status) {
            auto again = start;
            again.uuids = restrictTo(uuids, start.uuids);
            if (again.uuids.empty()) {
                *status = this->status();
                return std::vector<ObjectRecord>{};
            }
            again.limit = int64_t(again.uuids.size());
            if (start.afterUuid) {  // the listing starts after this UUID
                std::erase_if(again.uuids, [&](const std::string& u) { return u <= *start.afterUuid; });
                if (again.uuids.empty()) { *status = this->status(); return std::vector<ObjectRecord>{}; }
                again.afterUuid.reset();
            }
            return selectObjects(again, fields, status);
        };
        // Membership through relationships (connectedTo) changes with them.
        const bool graph = start.connectedTo.has_value();
        std::set<std::string> deleted;
        listing.effect = [&](const ChangeRecord& change) {
            if (change.recordKind == "object") {
                if (change.deleted) deleted.insert(change.uuid);
                return ChangeEffect::Recheck;
            }
            return graph ? ChangeEffect::Restart : ChangeEffect::Ignore;
        };
        // A recorded transform or route choice can move every volume match.
        if (selectionUsesTransforms(start))
            listing.restartFor = [&](const std::vector<std::string>& uuids) {
                return touchesTransforms(*this, uuids, deleted, pageSize);
            };
        return listing.run(asOf);
    }
    selection.limit = pageSize;
    return pageAll<ObjectRecord>(pageSize, maxRestarts, asOf,
        [&](bool first, const std::vector<ObjectRecord>& all, StoreStatus* status) {
            if (first) selection.offset = start.offset;
            else selection.offset = start.offset + int64_t(all.size());
            return selectObjects(selection, fields, status);
        },
        [&](const std::vector<ObjectRecord>&) {});
}

std::vector<EdgeRecord> HostedStore::selectAllEdges(EdgeSelection selection,
    const std::optional<std::vector<std::string>>& fields, int64_t pageSize, int maxRestarts, StoreStatus* asOf)
{
    if (selection.limit) throw std::invalid_argument("hosted store: selectAllEdges pages itself; remove the limit");
    const auto start = selection;
    CatchUp<EdgeRecord> listing{*this, pageSize, maxRestarts};
    listing.page = [&](const std::optional<std::string>& after, StoreStatus* status) {
        auto page = start;
        page.limit = pageSize;
        page.afterUuid = after ? after : start.afterUuid;
        return selectEdges(page, fields, status);
    };
    listing.requery = [&](const std::vector<std::string>& uuids, StoreStatus* status) {
        auto again = start;
        auto chosen = restrictTo(uuids, start.uuids ? *start.uuids : std::vector<std::string>{});
        if (start.uuids && start.uuids->empty()) chosen.clear();
        if (start.afterUuid) std::erase_if(chosen, [&](const std::string& u) { return u <= *start.afterUuid; });
        if (chosen.empty()) {
            *status = this->status();
            return std::vector<EdgeRecord>{};
        }
        again.uuids = chosen;
        again.limit = int64_t(chosen.size());
        again.afterUuid.reset();
        return selectEdges(again, fields, status);
    };
    // Applicability depends on the endpoints' current state.
    const bool endpoints = start.applicableOnly;
    listing.effect = [&](const ChangeRecord& change) {
        if (change.recordKind == "relationship") return ChangeEffect::Recheck;
        return endpoints ? ChangeEffect::Restart : ChangeEffect::Ignore;
    };
    return listing.run(asOf);
}

SelectionRecord HostedStore::recordSelection(const ObjectSelection& selection, int64_t pageSize, int maxRestarts)
{
    if (selection.transforms)
        throw std::invalid_argument("hosted store: selections use the store's recorded transforms, not explicit ones");
    const auto query = objectQueryRequestToJson(selection, std::nullopt);
    auto versions = [](const std::vector<ObjectRecord>& records, std::vector<ObjectVersion>& out) {
        for (const auto& r : records) {
            if (!r.version) throw std::runtime_error("hosted store: selected record without a version");
            out.push_back(r.reference());
        }
    };
    for (int attempt = 0; attempt <= maxRestarts; ++attempt) {
        StoreStatus asOf;
        const auto objects = selectAllObjects(selection, std::vector<std::string>{}, pageSize, maxRestarts, &asOf);
        SelectionRecord record{asOf.storeUuid, asOf.generation, asOf.commitSequence, query, {}, {}};
        versions(objects, record.objects);
        if (!selectionUsesTransforms(selection)) return record;
        // The transforms in force at asOf: read them, then make sure the feed
        // since asOf touched none of them (the listing itself restarts on
        // transform changes while it pages).
        std::vector<ObjectRecord> transforms;
        int64_t readAt = asOf.commitSequence;
        bool sameLineage = true;
        for (const auto* kind : {"volume_transform", "volume_transform_choice"}) {
            ObjectSelection byKind;
            byKind.kind = kind;
            StoreStatus at;
            auto part = selectAllObjects(byKind, std::vector<std::string>{}, pageSize, maxRestarts, &at);
            sameLineage = sameLineage && at.storeUuid == asOf.storeUuid && at.generation == asOf.generation;
            readAt = std::max(readAt, at.commitSequence);
            transforms.insert(transforms.end(), std::make_move_iterator(part.begin()), std::make_move_iterator(part.end()));
        }
        if (!sameLineage) continue;
        FeedCursor cursor{asOf.storeUuid, asOf.generation, asOf.commitSequence, 0};
        std::vector<std::string> changed;
        std::set<std::string> deleted;
        bool valid = true;
        while (cursor.sequence < readAt || cursor.offset > 0) {
            const auto page = changesSince(cursor);
            if (!page.cursorValid) { valid = false; break; }
            for (const auto& commit : page.commits)
                for (const auto& change : commit.changes)
                    if (change.recordKind == "object") {
                        changed.push_back(change.uuid);
                        if (change.deleted) deleted.insert(change.uuid);
                    }
            cursor = page.next;
            if (!page.more) break;
        }
        if (!valid || touchesTransforms(*this, changed, deleted, pageSize)) continue;
        std::sort(transforms.begin(), transforms.end(),
                  [](const ObjectRecord& a, const ObjectRecord& b) { return a.uuid < b.uuid; });
        versions(transforms, record.transforms);
        return record;
    }
    throw StoreBusy("hosted store: transforms kept changing while recording a selection; retry later");
}

std::vector<ObjectRevision> HostedStore::objectHistory(const std::string& objectUuid, StoreStatus* asOf)
{
    const auto response = impl_->http.request("GET", requestEndpoint_ + "/v1/objects/" + objectUuid + "/history");
    if (response.status == 404) {
        if (asOf) *asOf = status();
        return {};
    }
    return objectHistoryFromJson(expectOk(response, "object history").body, asOf);
}

std::vector<std::optional<ObjectRevision>> HostedStore::objectsAtVersions(const std::vector<ObjectVersion>& references,
                                                                       int64_t batch)
{
    if (batch <= 0) throw std::invalid_argument("hosted store: batch size must be positive");
    std::vector<std::optional<ObjectRevision>> result;
    result.reserve(references.size());
    for (std::size_t i = 0; i < references.size(); i += std::size_t(batch)) {
        const std::vector<ObjectVersion> part(references.begin() + i,
            references.begin() + std::min(references.size(), i + std::size_t(batch)));
        const auto response = impl_->http.request("POST", requestEndpoint_ + "/v1/query/versions",
            objectVersionsRequestToJson(part), {{"Content-Type", "application/json"}});
        auto revisions = objectsAtVersionsResultFromJson(expectOk(response, "versions query").body, nullptr);
        if (revisions.size() != part.size()) throw std::runtime_error("hosted store: versions answer has the wrong length");
        std::move(revisions.begin(), revisions.end(), std::back_inserter(result));
    }
    return result;
}

nlohmann::json HostedStore::geometryPointAnnotations(const std::string& revisionUuid, const std::string& arrayName)
{
    const auto response = impl_->http.request("GET", requestEndpoint_ + "/v1/geometry/" + revisionUuid + "/annotations/" + arrayPathSegment(arrayName));
    if (response.status == 404) throw std::invalid_argument("geometry revision not found: " + revisionUuid);
    return Json::parse(expectOk(response, "point annotations").body);
}

std::vector<std::string> HostedStore::traverse(const GraphQuery& query, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", requestEndpoint_ + "/v1/query/graph",
        graphQueryRequestToJson(query), {{"Content-Type", "application/json"}});
    return uuidListResultFromJson(expectOk(response, "graph query").body, asOf);
}

int64_t HostedStore::objectCount(const std::string& kind, bool includeRetired, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", requestEndpoint_ + "/v1/query/count",
        kindQueryRequestToJson({kind, includeRetired, std::nullopt}), {{"Content-Type", "application/json"}});
    return objectCountResultFromJson(expectOk(response, "object count").body, asOf);
}

int64_t HostedStore::unboundObjectCount(const std::string& kind, const std::string& volume, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", requestEndpoint_ + "/v1/query/count",
        kindQueryRequestToJson({kind, false, volume}), {{"Content-Type", "application/json"}});
    return objectCountResultFromJson(expectOk(response, "unbound object count").body, asOf);
}

std::vector<std::string> HostedStore::objectTags(const std::optional<std::string>& kind, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", requestEndpoint_ + "/v1/query/tags",
        kindQueryRequestToJson({kind, false, std::nullopt}), {{"Content-Type", "application/json"}});
    return objectTagsResultFromJson(expectOk(response, "object tags").body, asOf);
}

std::vector<SelectedRelationship> HostedStore::relationshipsAt(const std::vector<ObjectVersion>& versions, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", requestEndpoint_ + "/v1/query/relationships",
        objectVersionsRequestToJson(versions), {{"Content-Type", "application/json"}});
    return relationshipQueryResultFromJson(expectOk(response, "relationship query").body, asOf);
}

RegionResult HostedStore::geometryRegion(const RegionSlice& slice, const RegionOptions& options, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", requestEndpoint_ + "/v1/query/region",
        regionQueryRequestToJson(slice, options), {{"Content-Type", "application/json"}});
    return regionQueryResultFromJson(expectOk(response, "region query").body, asOf);
}

std::vector<SurfaceProximityHit> HostedStore::nearestSurfaces(const SurfaceProximityQuery& query, StoreStatus* asOf)
{
    const auto response = impl_->http.request("POST", requestEndpoint_ + "/v1/query/nearest_surfaces",
        proximityQueryRequestToJson(query), {{"Content-Type", "application/json"}});
    return proximityQueryResultFromJson(expectOk(response, "proximity query").body, asOf);
}

std::optional<uint64_t> HostedStore::artifactLength(const std::string& sha256)
{
    if (!isArtifactHash(sha256)) throw std::invalid_argument("hosted store: invalid artifact SHA-256 " + sha256);
    const auto response = impl_->http.request("GET", requestEndpoint_ + "/v1/artifacts/" + sha256 + "/info");
    if (response.status == 404) return std::nullopt;
    return Json::parse(expectOk(response, "artifact info").body).at("length").get<uint64_t>();
}

std::string HostedStore::putArtifact(const std::filesystem::path& file)
{
    // The upload reads a mapping of the file, not a copy in memory.
    struct Mapping {
        int fd = -1;
        const uint8_t* data = nullptr;
        std::size_t size = 0;
        ~Mapping()
        {
            if (data) vcr::platform::unmap(data, size);
            if (fd >= 0) ::close(fd);
        }
    } mapping;
    mapping.fd = vcr::platform::open_file(file, O_RDONLY);
    if (mapping.fd < 0) throw std::system_error(errno, std::generic_category(), "open " + file.string());
    mapping.size = std::filesystem::file_size(file);
    if (mapping.size && !(mapping.data = vcr::platform::map_read(mapping.fd, mapping.size)))
        throw std::runtime_error("hosted store: cannot map " + file.string());
    const std::string_view bytes(reinterpret_cast<const char*>(mapping.data), mapping.size);
    PayloadHasher hasher;
    hasher.update(bytes.data(), bytes.size());
    const auto hash = hasher.finish();
    if (artifactLength(hash) == uint64_t(bytes.size())) return hash;
    expectOk(impl_->http.request("PUT", requestEndpoint_ + "/v1/artifacts/" + hash, bytes,
                                 {{"Content-Type", "application/octet-stream"}}), "put artifact " + hash);
    return hash;
}

std::vector<uint8_t> HostedStore::readArtifact(const std::string& sha256, uint64_t offset,
                                               const std::optional<uint64_t>& count)
{
    if (!isArtifactHash(sha256)) throw std::invalid_argument("hosted store: invalid artifact SHA-256 " + sha256);
    auto get = [&](uint64_t from, const std::optional<uint64_t>& n) {
        auto url = requestEndpoint_ + "/v1/artifacts/" + sha256 + "?offset=" + std::to_string(from);
        if (n) url += "&count=" + std::to_string(*n);
        return impl_->http.request("GET", url);
    };
    auto total = count;
    if (!total) {
        const auto length = artifactLength(sha256);
        if (!length) throw std::invalid_argument("hosted store: the server does not hold artifact " + sha256);
        if (offset > *length) throw std::out_of_range("hosted store: invalid artifact offset");
        total = *length - offset;
    }
    if (!*total) {  // still an answer about this artifact and offset
        expectOk(get(offset, 0), "artifact");
        return {};
    }
    auto range = uint64_t(rangeBytes_);
    std::vector<uint8_t> result;
    result.reserve(std::size_t(*total));
    for (uint64_t done = 0; done < *total;) {
        const auto want = std::min(range, *total - done);
        const auto response = get(offset + done, want);
        if (response.status == 413) {
            // The server answers smaller ranges than ours: use its limit.
            const auto serverMax = Json::parse(response.body).value("max_response_bytes", uint64_t(0));
            if (!serverMax || serverMax >= range) expectOk(response, "artifact");
            range = serverMax;
            continue;
        }
        const auto& body = expectOk(response, "artifact").body;
        result.insert(result.end(), body.begin(), body.end());
        done += want;
    }
    return result;
}

void HostedStore::fetchArtifact(const ArtifactFile& file, const std::filesystem::path& destination)
{
    detail::writeArtifactFile({file.name, file.sha256, file.length}, destination, uint64_t(rangeBytes_),
                              [&](uint64_t offset, uint64_t count) { return readArtifact(file.sha256, offset, count); });
}

} // namespace vc::objects
