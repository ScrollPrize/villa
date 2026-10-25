#pragma once

#include "AtlasControlPointsTypes.hpp"

#include <QDockWidget>
#include <QJsonObject>

class QCheckBox;
class QLabel;
class QTreeWidget;
class QTreeWidgetItem;

class AtlasControlPointsDock : public QDockWidget
{
    Q_OBJECT

public:
    explicit AtlasControlPointsDock(QWidget* parent = nullptr);

    // Lasagna's atlas control-point results for the surface (its records).
    void loadResults(const QJsonObject& results);
    void clearResults();
    [[nodiscard]] const AtlasControlPointResults& results() const { return _results; }
    [[nodiscard]] bool overlayChecked() const;

signals:
    void overlayToggled(bool enabled);
    void resultsChanged(const AtlasControlPointResults& results);
    void controlPointActivated(const AtlasControlPointResult& point);
    void controlPointSelected(const AtlasControlPointResult& point);

private:
    void rebuildTree();
    void setEmptyState(const QString& text);
    AtlasControlPointResults parseResults(const QJsonObject& root) const;
    const AtlasControlPointResult* pointForItem(QTreeWidgetItem* item) const;

    QCheckBox* _overlayCheck = nullptr;
    QLabel* _statusLabel = nullptr;
    QTreeWidget* _tree = nullptr;
    AtlasControlPointResults _results;
};
