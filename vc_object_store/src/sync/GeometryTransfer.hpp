#pragma once
#include <sqlite3.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
namespace vc::objects::detail {
// Geometry history read from a source store into memory. Reading needs no
// destination lock, so a slow source (a hub over SSH) is never read while the
// destination's writers wait; installing touches only the destination.
class GeometryStage {
public:
    // Stage retained revisions after afterRow, verifying chunk hashes. Payloads
    // of revisions that `known` already holds are not read.
    void read(sqlite3* source, sqlite3* known, const std::string& objectUuid, int64_t afterRow = 0);
    bool contains(const std::string& objectUuid) const { return objects_.contains(objectUuid); }
    // Copy the staged revisions, comparing identities with the destination's.
    void install(sqlite3* destination, const std::string& objectUuid) const;
private:
    // Column values as read, keeping their SQLite storage classes.
    using Row = std::vector<std::unique_ptr<sqlite3_value, void (*)(sqlite3_value*)>>;
    struct Chunk { std::string array, hash; int64_t index = 0; std::vector<uint8_t> data; };
    struct Revision {
        std::string uuid, manifest;
        Row row, delta;
        bool payload = false;
        std::vector<std::vector<uint8_t>> metadata;
        std::vector<Chunk> chunks;
    };
    std::map<std::string, std::vector<Revision>> objects_;
};
// Copy retained immutable history, comparing identities and verifying chunk hashes.
void transferGeometry(sqlite3* source, sqlite3* destination, const std::string& objectUuid, int64_t afterRow = 0);
}
