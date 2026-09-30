#include "FiberCollectionGenerator.hpp"

#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLocale>
#include <algorithm>

namespace vc3d::fibergen
{

QStringList arguments(const Request& request)
{
    auto numbers = [](const std::array<int, 3>& values) {
        return QStringList{QString::number(values[0]), QString::number(values[1]), QString::number(values[2])};
    };
    QStringList args{"-m", kModule, "--volume", request.volume, "--level", QString::number(request.level)};
    args << "--origin" << numbers(request.origin) << "--size" << numbers(request.size);
    args << "--output" << request.output << "--coordinate-space" << request.coordinateSpace;
    args << "--native-scale" << QString::number(request.nativeScale);
    auto decimal = [](double value) { return QString::number(value, 'g', QLocale::FloatingPointShortest); };
    if (request.voxelSizeUm > 0)
        args << "--voxel-size" << decimal(request.voxelSizeUm);
    if (!request.sourcePath.isEmpty())
        args << "--source-path" << request.sourcePath;
    args << "--model" << request.model << "--threshold" << decimal(request.thresholdPercent);
    if (!request.mirror)
        args << "--no-mirror";
    args << "--progress" << "json";
    return args;
}

std::optional<Event> parseEvent(const QByteArray& line)
{
    const auto document = QJsonDocument::fromJson(line.trimmed());
    if (!document.isObject())
        return std::nullopt;
    const auto object = document.object();
    const auto name = object.value("event").toString();
    Event event;
    if (name == "progress")
        event.kind = Event::Kind::Progress;
    else if (name == "warning")
        event.kind = Event::Kind::Warning;
    else if (name == "done")
        event.kind = Event::Kind::Done;
    else if (name == "error")
        event.kind = Event::Kind::Error;
    else
        return std::nullopt;
    event.message = object.value("message").toString();
    event.fraction = std::clamp(object.value("fraction").toDouble(), 0.0, 1.0);
    event.output = object.value("output").toString();
    event.fibers = object.value("fibers").toInteger();
    if (event.kind == Event::Kind::Done && event.output.isEmpty())
        return std::nullopt;
    return event;
}

std::array<int, 3> zoneOrigin(const std::array<int, 3>& center, const std::array<int, 3>& size, const std::array<int, 3>& shape)
{
    std::array<int, 3> origin{};
    for (size_t i = 0; i < 3; ++i) {
        const int extent = std::clamp(size[i], 1, std::max(shape[i], 1));
        origin[i] = std::clamp(center[i] - extent / 2, 0, std::max(shape[i] - extent, 0));
    }
    return origin;
}

QString vesuviusSourceDirectory(const QString& applicationDirectory)
{
    // The same development layouts as the Neural Trace service.
    const QDir app(applicationDirectory);
    for (const auto* relative : {"../../vesuvius/src", "../../../vesuvius/src"}) {
        const QDir source(app.filePath(relative));
        if (QFileInfo::exists(source.filePath("vesuvius/afv_spline_generator/__main__.py")))
            return QDir::cleanPath(source.absolutePath());
    }
    return {};
}

QProcessEnvironment environment(QProcessEnvironment environment, const QString& vesuviusSource)
{
    if (vesuviusSource.isEmpty())
        return environment;
    const auto existing = environment.value("PYTHONPATH");
    environment.insert("PYTHONPATH", existing.isEmpty() ? vesuviusSource : vesuviusSource + QDir::listSeparator() + existing);
    return environment;
}

}  // namespace vc3d::fibergen
