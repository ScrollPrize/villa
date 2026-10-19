#pragma once

// Read-only relationship queries on an ObjectStore from VC3D. Opened only
// when --object-db or objects/db_path is configured. Worker queries retain
// the store handle through controller close/reopen.

#include <QString>

#include <optional>
#include <memory>
#include <string>
#include <vector>

#include "vc/core/objects/ObjectStore.hpp"

class GraphDbController {
public:
    // Opens the object database at path read-only. Returns false and
    // fills *errorMessage on failure (leaving any previously open DB
    // closed). An empty path just closes and returns true.
    bool open(const QString& path, QString* errorMessage = nullptr);
    void close();
    bool isOpen() const { return bool(db_); }
    QString path() const { return path_; }
    // Acquire on the GUI thread; workers retain the open DB through close/reopen.
    std::shared_ptr<vc::objects::ObjectStore> readHandle() const { requireDb(); return db_; }

    // Query passthroughs. Each runs in its own short read transaction.
    // Throws std::runtime_error when no object DB is open or on a
    // vc_objects error (unknown UUID, malformed UUID, ...).
    std::optional<vc::objects::ObjectRecord> objectByUuid(const std::string& uuid) const;
    std::vector<vc::objects::ObjectRecord> objectsByKind(const std::string& kind) const;
    std::vector<std::string> neighbours(const std::string& uuid,
                                        const std::optional<std::string>& edgeKind = std::nullopt,
                                        const std::optional<std::string>& neighbourKind = std::nullopt) const;
    std::vector<vc::objects::EdgeRecord> edgesOf(const std::string& uuid) const;
    std::vector<std::string> componentOf(const std::string& uuid,
                                         const std::vector<std::string>& edgeKindFilter = {}) const;

private:
    vc::objects::ObjectStore& requireDb() const;
    std::shared_ptr<vc::objects::ObjectStore> db_;
    QString path_;
};
