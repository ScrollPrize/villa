#include "SpiralPreviewWindings.hpp"

#include <QJsonObject>
#include <QObject>

#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace vc3d::spiral {

std::optional<std::vector<PreviewComponent>> parseWindingBounds(
    const QJsonArray& bounds, const QJsonArray& windingIds, QString* error)
{
    if (bounds.isEmpty() || bounds.size() != windingIds.size()) {
        *error = QObject::tr("Invalid Spiral preview winding mapping");
        return std::nullopt;
    }
    std::vector<PreviewComponent> components;
    std::set<int> declared;
    components.reserve(bounds.size());
    for (int index = 0; index < bounds.size(); ++index) {
        const QJsonObject entry = bounds[index].toObject();
        const PreviewComponent component{
            entry.value(QStringLiteral("row_begin")).toInt(-1),
            entry.value(QStringLiteral("row_end")).toInt(-1),
            entry.value(QStringLiteral("column_begin")).toInt(-1),
            entry.value(QStringLiteral("column_end")).toInt(-1),
            entry.value(QStringLiteral("winding")).toInt(-1),
        };
        if (component.rowBegin < 0 || component.rowEnd <= component.rowBegin
            || component.columnBegin < 0
            || component.columnEnd <= component.columnBegin
            || component.winding != windingIds[index].toInt()
            || component.winding < 0
            || !declared.insert(component.winding).second) {
            *error = QObject::tr("Invalid Spiral winding bounds");
            return std::nullopt;
        }
        components.push_back(component);
    }
    return components;
}

std::optional<cv::Mat_<int32_t>> windingIdsFromImage(const cv::Mat& image, QString* error)
{
    if (image.empty() || image.channels() != 1) {
        *error = QObject::tr("Cannot read Spiral winding-ID map");
        return std::nullopt;
    }
    cv::Mat values;
    image.convertTo(values, CV_64F);
    cv::Mat_<int32_t> windings(image.rows, image.cols);
    for (int row = 0; row < values.rows; ++row) {
        const double* line = values.ptr<double>(row);
        for (int column = 0; column < values.cols; ++column) {
            const double value = line[column];
            const double rounded = std::nearbyint(value);
            if (!std::isfinite(value) || std::abs(value - rounded) > 1.0e-6
                || rounded < -1.0
                || rounded > std::numeric_limits<int32_t>::max()) {
                *error = QObject::tr("Spiral winding-ID map contains a non-integer value");
                return std::nullopt;
            }
            windings(row, column) = static_cast<int>(rounded);
        }
    }
    return windings;
}

bool windingMapMatchesBounds(const cv::Mat_<int32_t>& windingIds,
                             const std::vector<PreviewComponent>& components,
                             QString* error)
{
    std::set<int> declared;
    for (const PreviewComponent& component : components) declared.insert(component.winding);
    std::map<int, PreviewComponent> actual;
    for (int row = 0; row < windingIds.rows; ++row) {
        for (int column = 0; column < windingIds.cols; ++column) {
            const int winding = windingIds(row, column);
            if (winding == -1) continue;
            if (!declared.contains(winding)) {
                *error = QObject::tr("Spiral winding-ID map contains an undeclared winding");
                return false;
            }
            auto [found, inserted] = actual.try_emplace(
                winding, PreviewComponent{row, row + 1, column, column + 1, winding});
            if (!inserted) {
                found->second.rowBegin = std::min(found->second.rowBegin, row);
                found->second.rowEnd = std::max(found->second.rowEnd, row + 1);
                found->second.columnBegin = std::min(found->second.columnBegin, column);
                found->second.columnEnd = std::max(found->second.columnEnd, column + 1);
            }
        }
    }
    if (actual.size() != components.size()) {
        *error = QObject::tr("Spiral winding-ID map omits a declared winding");
        return false;
    }
    for (const PreviewComponent& expected : components) {
        const auto found = actual.find(expected.winding);
        if (found == actual.end()
            || found->second.rowBegin != expected.rowBegin
            || found->second.rowEnd != expected.rowEnd
            || found->second.columnBegin != expected.columnBegin
            || found->second.columnEnd != expected.columnEnd) {
            *error = QObject::tr("Spiral winding bounds do not match the winding-ID map");
            return false;
        }
    }
    return true;
}

}  // namespace vc3d::spiral
