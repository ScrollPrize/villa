#pragma once
#include "internal/Database.hpp"
#include "geometry/GeometryInternal.hpp"

namespace vc::objects::detail {

// Scoped to a single operation and its transaction snapshot. Never reuse verified
// bytes across transactions: a later operation must still detect disk corruption.
class GeometryReader {
public:
    GeometryReader(const Txn& txn, const std::string& revision, bool cacheChunks = true);
    GeometryReader(sqlite3* db, const std::string& revision, bool cacheChunks = true);
    const nlohmann::json& manifest() const { return manifest_; }
    std::vector<uint8_t> read(const std::string& name, int64_t offset, std::optional<int64_t> count = std::nullopt);

private:
    sqlite3* db_;
    std::array<unsigned char, 16> revision_;
    bool cacheChunks_;
    nlohmann::json manifest_;
    std::map<std::pair<std::string, int64_t>, std::vector<uint8_t>> chunks_;
};
}
