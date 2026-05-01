#pragma once

#include <QWidget>

#include <opencv2/core.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "vc/core/util/SurfacePatchIndex.hpp"

class QLabel;
class QPushButton;
class QCheckBox;
class QDoubleSpinBox;
class PatchGraphOverlayController;
class SegmentationModule;

class PatchGraphWidget : public QWidget
{
    Q_OBJECT

public:
    struct Endpoint {
        int node{-1};
        cv::Vec3f world{0.0f, 0.0f, 0.0f};
    };

    PatchGraphWidget(SegmentationModule* segmentationModule,
                     PatchGraphOverlayController* overlay,
                     QWidget* parent = nullptr);

signals:
    void captureActiveChanged(bool active);

public slots:
    void onViewerMouseMove(const cv::Vec3f& worldPos,
                           Qt::MouseButtons buttons,
                           Qt::KeyboardModifiers modifiers);
    void onViewerMouseRelease(const cv::Vec3f& worldPos,
                              Qt::MouseButton button,
                              Qt::KeyboardModifiers modifiers);
    void setEditingEnabled(bool enabled);

private:
    struct GraphNode {
        cv::Vec3f world{0.0f, 0.0f, 0.0f};
        int patch{-1};
        int row{-1};
        int col{-1};
    };

    struct Edge {
        int to{-1};
        double weight{0.0};
    };

    void chooseFolder();
    void rebuildGraph();
    void addPatchNodes(int patchIndex);
    void addPatchGridEdges(int patchIndex);
    void addCrossPatchEdges(double linkDistance);
    void addEdge(int from, int to, double weight);
    std::optional<int> nodeNearSurfaceHit(const SurfacePatchIndex::LookupResult& hit,
                                          const cv::Vec3f& referencePoint) const;
    std::optional<int> nearestNode(const cv::Vec3f& worldPos, double maxDistance) const;
    void clearSelection();
    void recomputePath();
    void updateLabels();
    void setStatus(const QString& text);

    std::optional<std::vector<cv::Vec3f>> shortestVertexPath(int start,
                                                             int end,
                                                             double* outDistance) const;

    SegmentationModule* _segmentationModule{nullptr};
    PatchGraphOverlayController* _overlay{nullptr};
    QCheckBox* _captureCheck{nullptr};
    QPushButton* _folderButton{nullptr};
    QLabel* _folderLabel{nullptr};
    QDoubleSpinBox* _linkDistanceSpin{nullptr};
    QDoubleSpinBox* _pickDistanceSpin{nullptr};
    QLabel* _startLabel{nullptr};
    QLabel* _endLabel{nullptr};
    QLabel* _statusLabel{nullptr};
    QPushButton* _clearButton{nullptr};
    QString _folderPath;
    std::vector<std::string> _patchNames;
    std::vector<SurfacePatchIndex::SurfacePtr> _surfaces;
    std::unordered_map<QuadSurface*, int> _surfaceToPatch;
    std::vector<std::vector<int>> _patchNodeIds;
    std::vector<GraphNode> _nodes;
    std::vector<std::vector<Edge>> _adjacency;
    SurfacePatchIndex _patchIndex;
    std::optional<int> _hoverNode;
    std::optional<Endpoint> _start;
    std::optional<Endpoint> _end;
    bool _editingEnabled{false};
};
