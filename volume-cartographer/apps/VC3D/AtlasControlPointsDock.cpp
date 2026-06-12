#include "AtlasControlPointsDock.hpp"

#include <QCheckBox>
#include <QFile>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QSignalBlocker>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace
{
constexpr int kPointIndexRole = Qt::UserRole + 1;

float jsonFloat(const QJsonObject& obj, const QString& key)
{
    const QJsonValue value = obj.value(key);
    return value.isDouble() ? static_cast<float>(value.toDouble()) : NAN;
}

int jsonInt(const QJsonObject& obj, const QString& key, int fallback = -1)
{
    const QJsonValue value = obj.value(key);
    return value.isDouble() ? value.toInt() : fallback;
}

QString jsonString(const QJsonObject& obj, const QString& key, const QString& fallback = {})
{
    const QJsonValue value = obj.value(key);
    return value.isString() ? value.toString() : fallback;
}

cv::Vec3f jsonVec3(const QJsonObject& obj, const QString& key)
{
    const QJsonArray arr = obj.value(key).toArray();
    if (arr.size() < 3) {
        return {NAN, NAN, NAN};
    }
    return {
        arr.at(0).isDouble() ? static_cast<float>(arr.at(0).toDouble()) : NAN,
        arr.at(1).isDouble() ? static_cast<float>(arr.at(1).toDouble()) : NAN,
        arr.at(2).isDouble() ? static_cast<float>(arr.at(2).toDouble()) : NAN,
    };
}

QString fmtFloat(float value, int precision = 4)
{
    return std::isfinite(value) ? QString::number(value, 'f', precision) : QStringLiteral("-");
}

QString fmtVec3(const cv::Vec3f& point)
{
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
        return QStringLiteral("-");
    }
    return QStringLiteral("%1, %2, %3")
        .arg(QString::number(point[0], 'f', 1))
        .arg(QString::number(point[1], 'f', 1))
        .arg(QString::number(point[2], 'f', 1));
}

QString fmtSnapStatus(const AtlasControlPointResult& point)
{
    return point.hasSnapStatus ? point.snapStatus : QStringLiteral("-");
}

AtlasControlPointResult parsePoint(const QJsonObject& obj)
{
    AtlasControlPointResult point;
    point.fiberId = jsonString(obj, QStringLiteral("fiber_id"),
                               jsonString(obj, QStringLiteral("object_id"), QStringLiteral("unknown")));
    point.objectId = jsonString(obj, QStringLiteral("object_id"), point.fiberId);
    point.sourceIndex = jsonInt(obj, QStringLiteral("source_index"));
    point.controlIndex = jsonInt(obj, QStringLiteral("control_index"));
    point.layerIndex = jsonInt(obj, QStringLiteral("layer_index"));
    point.valid = obj.value(QStringLiteral("valid")).toBool(false);
    point.distance = jsonFloat(obj, QStringLiteral("distance"));
    point.signedDelta = jsonFloat(obj, QStringLiteral("signed_delta"));
    point.hasSnapStatus = obj.contains(QStringLiteral("snap_status"));
    point.snapStatus = jsonString(obj, QStringLiteral("snap_status"), QStringLiteral("-"));
    point.snapValid = obj.value(QStringLiteral("snap_valid")).toBool(false);
    point.snapSignedDelta = jsonFloat(obj, QStringLiteral("snap_signed_delta"));
    point.snapTargetXyz = jsonVec3(obj, QStringLiteral("snap_target_xyz"));
    point.targetXyz = jsonVec3(obj, QStringLiteral("target_xyz"));
    point.meshXyz = jsonVec3(obj, QStringLiteral("mesh_xyz"));
    point.modelH = jsonFloat(obj, QStringLiteral("model_h"));
    point.modelW = jsonFloat(obj, QStringLiteral("model_w"));
    return point;
}
}

AtlasControlPointsDock::AtlasControlPointsDock(QWidget* parent)
    : QDockWidget(tr("Lasagna Atlas Control Diff"), parent)
{
    setObjectName(QStringLiteral("las-ctl-diff"));

    auto* content = new QWidget(this);
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(6, 6, 6, 6);
    layout->setSpacing(6);

    _overlayCheck = new QCheckBox(tr("Overlay control points"), content);
    _overlayCheck->setObjectName(QStringLiteral("atlasControlOverlayCheck"));
    _overlayCheck->setChecked(false);
    _overlayCheck->setEnabled(false);
    layout->addWidget(_overlayCheck);

    _statusLabel = new QLabel(tr("No atlas control results for the active segment."), content);
    _statusLabel->setObjectName(QStringLiteral("atlasControlStatusLabel"));
    layout->addWidget(_statusLabel);

    _tree = new QTreeWidget(content);
    _tree->setObjectName(QStringLiteral("atlasControlResultsTree"));
    _tree->setColumnCount(7);
    _tree->setHeaderLabels({
        tr("Control/Source"),
        tr("Valid"),
        tr("Snap"),
        tr("Snap Delta"),
        tr("Snap XYZ"),
        tr("Distance"),
        tr("Signed Delta"),
    });
    _tree->header()->setStretchLastSection(false);
    _tree->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
    layout->addWidget(_tree, 1);

    setWidget(content);

    connect(_overlayCheck, &QCheckBox::toggled, this, &AtlasControlPointsDock::overlayToggled);
    connect(_tree, &QTreeWidget::itemSelectionChanged, this, [this]() {
        const auto selected = _tree->selectedItems();
        if (selected.isEmpty()) {
            return;
        }
        if (const auto* point = pointForItem(selected.first())) {
            emit controlPointSelected(*point);
        }
    });
    connect(_tree, &QTreeWidget::itemDoubleClicked, this, [this](QTreeWidgetItem* item, int) {
        if (const auto* point = pointForItem(item)) {
            emit controlPointActivated(*point);
        }
    });
}

bool AtlasControlPointsDock::overlayChecked() const
{
    return _overlayCheck && _overlayCheck->isChecked();
}

void AtlasControlPointsDock::loadResults(const std::filesystem::path& jsonPath)
{
    if (jsonPath.empty() || !std::filesystem::exists(jsonPath)) {
        clearResults();
        return;
    }
    try {
        _results = parseResults(jsonPath);
        rebuildTree();
        emit resultsChanged(_results);
    } catch (const std::exception& e) {
        _results.clear();
        rebuildTree();
        setEmptyState(tr("Could not read atlas control results: %1").arg(QString::fromUtf8(e.what())));
        emit resultsChanged(_results);
    }
}

void AtlasControlPointsDock::clearResults()
{
    _results.clear();
    {
        const QSignalBlocker blocker(_overlayCheck);
        _overlayCheck->setChecked(false);
    }
    emit overlayToggled(false);
    rebuildTree();
    emit resultsChanged(_results);
}

AtlasControlPointResults AtlasControlPointsDock::parseResults(const std::filesystem::path& jsonPath) const
{
    QFile file(QString::fromStdString(jsonPath.string()));
    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("open failed");
    }
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    if (!doc.isObject()) {
        throw std::runtime_error("top-level value is not an object");
    }
    const QJsonObject root = doc.object();

    AtlasControlPointResults results;
    const QJsonArray records = root.value(QStringLiteral("records")).toArray();
    if (!records.isEmpty()) {
        results.reserve(records.size());
        for (const QJsonValue& value : records) {
            if (value.isObject()) {
                results.push_back(parsePoint(value.toObject()));
            }
        }
    } else {
        const QJsonArray fibers = root.value(QStringLiteral("fibers")).toArray();
        for (const QJsonValue& fiberValue : fibers) {
            if (!fiberValue.isObject()) {
                continue;
            }
            const QJsonObject fiber = fiberValue.toObject();
            const QString fiberId = jsonString(fiber, QStringLiteral("fiber_id"),
                                               jsonString(fiber, QStringLiteral("object_id"), QStringLiteral("unknown")));
            const QJsonArray points = fiber.value(QStringLiteral("control_points")).toArray();
            for (const QJsonValue& pointValue : points) {
                if (!pointValue.isObject()) {
                    continue;
                }
                AtlasControlPointResult point = parsePoint(pointValue.toObject());
                if (point.fiberId.isEmpty()) {
                    point.fiberId = fiberId;
                }
                results.push_back(point);
            }
        }
    }

    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        if (a.fiberId != b.fiberId) {
            return a.fiberId < b.fiberId;
        }
        if (a.layerIndex != b.layerIndex) {
            return a.layerIndex < b.layerIndex;
        }
        if (a.sourceIndex != b.sourceIndex) {
            return a.sourceIndex < b.sourceIndex;
        }
        return a.controlIndex < b.controlIndex;
    });
    return results;
}

void AtlasControlPointsDock::rebuildTree()
{
    _tree->clear();
    _overlayCheck->setEnabled(!_results.empty());
    if (_results.empty()) {
        setEmptyState(tr("No atlas control results for the active segment."));
        return;
    }

    _statusLabel->setText(tr("%1 control points").arg(_results.size()));
    std::map<QString, QTreeWidgetItem*> groups;
    for (int i = 0; i < static_cast<int>(_results.size()); ++i) {
        const AtlasControlPointResult& point = _results[static_cast<size_t>(i)];
        QTreeWidgetItem* group = nullptr;
        auto it = groups.find(point.fiberId);
        if (it == groups.end()) {
            group = new QTreeWidgetItem(_tree);
            group->setText(0, point.fiberId);
            group->setFirstColumnSpanned(true);
            groups.emplace(point.fiberId, group);
        } else {
            group = it->second;
        }

        auto* row = new QTreeWidgetItem(group);
        row->setData(0, kPointIndexRole, i);
        row->setText(0, QStringLiteral("%1 / %2").arg(point.controlIndex).arg(point.sourceIndex));
        row->setText(1, point.valid ? tr("yes") : tr("no"));
        row->setText(2, fmtSnapStatus(point));
        row->setText(3, fmtFloat(point.snapSignedDelta));
        row->setText(4, fmtVec3(point.snapTargetXyz));
        row->setText(5, fmtFloat(point.distance));
        row->setText(6, fmtFloat(point.signedDelta));
    }
    _tree->expandAll();
}

void AtlasControlPointsDock::setEmptyState(const QString& text)
{
    _statusLabel->setText(text);
    _overlayCheck->setEnabled(false);
}

const AtlasControlPointResult* AtlasControlPointsDock::pointForItem(QTreeWidgetItem* item) const
{
    if (!item) {
        return nullptr;
    }
    bool ok = false;
    const int idx = item->data(0, kPointIndexRole).toInt(&ok);
    if (!ok || idx < 0 || idx >= static_cast<int>(_results.size())) {
        return nullptr;
    }
    return &_results[static_cast<size_t>(idx)];
}
