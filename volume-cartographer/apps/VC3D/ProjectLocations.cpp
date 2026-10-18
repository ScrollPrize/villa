#include "ProjectLocations.hpp"

#include <string>

namespace fs = std::filesystem;

namespace vc3d::project_locations {

bool looksRemote(const QString& location)
{
    const QString trimmed = location.trimmed();
    return trimmed.contains(QStringLiteral("://")) &&
           !trimmed.startsWith(QStringLiteral("file://"), Qt::CaseInsensitive);
}

QString storedLocalLocation(const QString& location, const fs::path& projectDir)
{
    fs::path path = vc::project::resolveLocalPath(location.trimmed().toStdString(), projectDir)
                        .lexically_normal();
    while (path.has_relative_path() && path.filename().empty()) {
        path = path.parent_path();
    }
    return QString::fromStdString(path.string());
}

QString findEntry(const std::vector<vc::project::Entry>& entries,
                  const QString& location,
                  const fs::path& projectDir)
{
    const std::string wanted = location.toStdString();
    for (const auto& entry : entries) {
        if (vc::project::sameLocation(entry.location, wanted, projectDir)) {
            return QString::fromStdString(entry.location);
        }
    }
    return {};
}

}  // namespace vc3d::project_locations
