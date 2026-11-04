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

} // namespace vc::objects
