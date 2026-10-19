#include "GraphDbController.hpp"

#include <exception>
#include <filesystem>
#include <stdexcept>

bool GraphDbController::open(const QString& path, QString* errorMessage)
{
    close();
    if (path.trimmed().isEmpty())
        return true;
    try {
        vc::graph::OpenOptions options;
        options.readOnly = true;
        options.readerPoolSize = 4;
        db_ = std::make_shared<vc::graph::GraphDb>(vc::graph::GraphDb::open(std::filesystem::path(path.toStdString()), options));
        path_ = path;
        return true;
    } catch (const std::exception& e) {
        if (errorMessage)
            *errorMessage = QStringLiteral("Cannot open graph database %1: %2").arg(path, e.what());
        return false;
    }
}

void GraphDbController::close()
{
    db_.reset();
    path_.clear();
}

vc::graph::GraphDb& GraphDbController::requireDb() const
{
    if (!db_)
        throw std::runtime_error("no graph database is open");
    return *db_;
}

std::optional<vc::graph::ObjectRecord> GraphDbController::objectByUuid(const std::string& uuid) const
{
    auto& db = requireDb();
    auto txn = db.read();
    return db.objectByUuid(txn, uuid);
}

std::vector<vc::graph::ObjectRecord> GraphDbController::objectsByKind(const std::string& kind) const
{
    auto& db = requireDb();
    auto txn = db.read();
    return db.objectsByKind(txn, kind);
}

std::vector<std::string> GraphDbController::neighbours(const std::string& uuid,
                                                       const std::optional<std::string>& edgeKind,
                                                       const std::optional<std::string>& neighbourKind) const
{
    auto& db = requireDb();
    auto txn = db.read();
    return db.neighbours(txn, uuid, edgeKind, neighbourKind);
}

std::vector<vc::graph::EdgeRecord> GraphDbController::edgesOf(const std::string& uuid) const
{
    auto& db = requireDb();
    auto txn = db.read();
    return db.edgesOf(txn, uuid);
}

std::vector<std::string> GraphDbController::componentOf(const std::string& uuid,
                                                        const std::vector<std::string>& edgeKindFilter) const
{
    auto& db = requireDb();
    auto txn = db.read();
    return db.componentOf(txn, uuid, edgeKindFilter);
}
