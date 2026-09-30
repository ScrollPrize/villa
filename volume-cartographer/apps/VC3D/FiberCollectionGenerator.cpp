#include "FiberCollectionGenerator.hpp"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
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
    args << "--block-size" << QString::number(request.blockSize);
    if (!request.previewDirectory.isEmpty())
        args << "--preview-dir" << request.previewDirectory;
    args << "--progress" << "json";
    return args;
}

std::vector<Block> zoneBlocks(const std::array<int, 3>& origin, const std::array<int, 3>& size, int blockSize)
{
    std::vector<Block> blocks;
    if (blockSize < 1 || std::any_of(size.begin(), size.end(), [](int s) { return s < 1; }))
        return blocks;
    for (int z = 0; z < size[2]; z += blockSize)
        for (int y = 0; y < size[1]; y += blockSize)
            for (int x = 0; x < size[0]; x += blockSize)
                blocks.push_back({{origin[0] + x, origin[1] + y, origin[2] + z},
                                  {std::min(blockSize, size[0] - x), std::min(blockSize, size[1] - y), std::min(blockSize, size[2] - z)}});
    return blocks;
}

namespace
{
std::optional<std::array<int, 3>> triple(const QJsonValue& value)
{
    const auto array = value.toArray();
    if (array.size() != 3)
        return std::nullopt;
    std::array<int, 3> result{};
    for (int i = 0; i < 3; ++i) {
        if (!array[i].isDouble())
            return std::nullopt;
        result[i] = array[i].toInt();
    }
    return result;
}
}  // namespace

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
    else if (name == "plan")
        event.kind = Event::Kind::Plan;
    else if (name == "block")
        event.kind = Event::Kind::Block;
    else if (name == "preview")
        event.kind = Event::Kind::Preview;
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
    if (event.kind == Event::Kind::Plan) {
        for (const auto& value : object.value("blocks").toArray()) {
            const auto origin = triple(value.toObject().value("origin"));
            const auto size = triple(value.toObject().value("size"));
            if (!origin || !size)
                return std::nullopt;
            event.blocks.push_back({*origin, *size});
        }
    }
    if (event.kind == Event::Kind::Block) {
        event.index = object.value("index").toInt(-1);
        event.state = object.value("state").toString();
        if (event.index < 0 || event.state.isEmpty())
            return std::nullopt;
    }
    if (event.kind == Event::Kind::Preview) {
        event.path = object.value("path").toString();
        if (event.path.isEmpty())
            return std::nullopt;
    }
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
