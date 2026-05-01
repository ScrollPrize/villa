#include "PatchGraphWidget.hpp"

#include "overlays/PatchGraphOverlayController.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QDebug>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <queue>
#include <utility>

namespace
{
struct QueueItem {
    double distance{0.0};
    int node{-1};

    bool operator>(const QueueItem& other) const
    {
        return distance > other.distance;
    }
};

double distance3d(const cv::Vec3f& a, const cv::Vec3f& b)
{
    const double dx = static_cast<double>(a[0]) - static_cast<double>(b[0]);
    const double dy = static_cast<double>(a[1]) - static_cast<double>(b[1]);
    const double dz = static_cast<double>(a[2]) - static_cast<double>(b[2]);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

QString endpointText(const std::optional<PatchGraphWidget::Endpoint>& endpoint)
{
    if (!endpoint) {
        return QStringLiteral("-");
    }
    return QStringLiteral("node %1").arg(endpoint->node);
}

bool isTifxyzDir(const std::filesystem::path& dir)
{
    return std::filesystem::is_directory(dir) &&
           std::filesystem::exists(dir / "meta.json") &&
           std::filesystem::exists(dir / "x.tif") &&
           std::filesystem::exists(dir / "y.tif") &&
           std::filesystem::exists(dir / "z.tif");
}

bool isValidVertex(const cv::Mat_<cv::Vec3f>& points, int row, int col)
{
    return row >= 0 && row < points.rows &&
           col >= 0 && col < points.cols &&
           points(row, col)[0] != -1.0f;
}
} // namespace

PatchGraphWidget::PatchGraphWidget(SegmentationModule* segmentationModule,
                                   PatchGraphOverlayController* overlay,
                                   QWidget* parent)
    : QWidget(parent)
    , _segmentationModule(segmentationModule)
    , _overlay(overlay)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(10, 10, 10, 10);
    layout->setSpacing(8);

    auto* folderRow = new QHBoxLayout();
    folderRow->setContentsMargins(0, 0, 0, 0);
    _folderButton = new QPushButton(tr("Folder"), this);
    _folderButton->setToolTip(tr("Choose a folder containing tifxyz patch directories."));
    folderRow->addWidget(_folderButton);
    _folderLabel = new QLabel(tr("No folder selected"), this);
    _folderLabel->setWordWrap(true);
    folderRow->addWidget(_folderLabel, 1);
    layout->addLayout(folderRow);

    auto* options = new QGridLayout();
    options->setContentsMargins(0, 0, 0, 0);
    options->setHorizontalSpacing(8);
    options->setVerticalSpacing(4);
    _linkDistanceSpin = new QDoubleSpinBox(this);
    _linkDistanceSpin->setRange(0.0, 1000.0);
    _linkDistanceSpin->setDecimals(2);
    _linkDistanceSpin->setValue(5.0);
    _linkDistanceSpin->setSingleStep(1.0);
    _linkDistanceSpin->setToolTip(tr("Maximum 3D distance for links between vertices from different patches."));
    options->addWidget(new QLabel(tr("Patch link"), this), 0, 0);
    options->addWidget(_linkDistanceSpin, 0, 1);
    _pickDistanceSpin = new QDoubleSpinBox(this);
    _pickDistanceSpin->setRange(0.1, 1000.0);
    _pickDistanceSpin->setDecimals(2);
    _pickDistanceSpin->setValue(8.0);
    _pickDistanceSpin->setSingleStep(1.0);
    _pickDistanceSpin->setToolTip(tr("Maximum 3D distance from cursor to selectable graph vertex."));
    options->addWidget(new QLabel(tr("Pick radius"), this), 1, 0);
    options->addWidget(_pickDistanceSpin, 1, 1);
    layout->addLayout(options);

    _captureCheck = new QCheckBox(tr("Capture hover clicks"), this);
    _captureCheck->setToolTip(tr("Hover and click graph vertices without enabling mesh editing."));
    layout->addWidget(_captureCheck);

    auto* grid = new QGridLayout();
    grid->setContentsMargins(0, 0, 0, 0);
    grid->setHorizontalSpacing(8);
    grid->setVerticalSpacing(4);
    grid->addWidget(new QLabel(tr("Start"), this), 0, 0);
    _startLabel = new QLabel(QStringLiteral("-"), this);
    grid->addWidget(_startLabel, 0, 1);
    grid->addWidget(new QLabel(tr("End"), this), 1, 0);
    _endLabel = new QLabel(QStringLiteral("-"), this);
    grid->addWidget(_endLabel, 1, 1);
    layout->addLayout(grid);

    auto* buttonRow = new QHBoxLayout();
    buttonRow->setContentsMargins(0, 0, 0, 0);
    _clearButton = new QPushButton(tr("Clear"), this);
    buttonRow->addWidget(_clearButton);
    buttonRow->addStretch(1);
    layout->addLayout(buttonRow);

    _statusLabel = new QLabel(tr("Choose a patch folder, then capture two vertices."), this);
    _statusLabel->setWordWrap(true);
    layout->addWidget(_statusLabel);

    connect(_clearButton, &QPushButton::clicked, this, &PatchGraphWidget::clearSelection);
    connect(_folderButton, &QPushButton::clicked, this, &PatchGraphWidget::chooseFolder);
    connect(_linkDistanceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double) { rebuildGraph(); });
    connect(_captureCheck, &QCheckBox::toggled, this, [this](bool enabled) {
        emit captureActiveChanged(enabled);
        if (!enabled && _overlay) {
            _overlay->setHoverPoint(std::nullopt);
            _hoverNode.reset();
        } else if (enabled) {
            setStatus(_nodes.empty()
                          ? tr("Choose a patch folder before capturing.")
                          : tr("Hover a graph vertex and click for start, then end."));
        }
    });

    updateLabels();
}

void PatchGraphWidget::setEditingEnabled(bool enabled)
{
    _editingEnabled = enabled;
    Q_UNUSED(_editingEnabled);
}

void PatchGraphWidget::onViewerMouseMove(const cv::Vec3f& worldPos,
                                         Qt::MouseButtons buttons,
                                         Qt::KeyboardModifiers /*modifiers*/)
{
    if (!_captureCheck || !_captureCheck->isChecked() || buttons != Qt::NoButton) {
        return;
    }
    auto node = nearestNode(worldPos, _pickDistanceSpin ? _pickDistanceSpin->value() : 8.0);
    if (node == _hoverNode) {
        return;
    }
    _hoverNode = node;
    if (_overlay) {
        _overlay->setHoverPoint(node ? std::optional<cv::Vec3f>(_nodes[*node].world) : std::nullopt);
    }
}

void PatchGraphWidget::onViewerMouseRelease(const cv::Vec3f& /*worldPos*/,
                                            Qt::MouseButton button,
                                            Qt::KeyboardModifiers modifiers)
{
    if (!_captureCheck || !_captureCheck->isChecked()) {
        return;
    }
    if (button != Qt::LeftButton || modifiers.testFlag(Qt::ControlModifier) ||
        modifiers.testFlag(Qt::AltModifier)) {
        return;
    }
    if (_nodes.empty()) {
        setStatus(tr("Choose a patch folder before selecting vertices."));
        return;
    }
    if (!_hoverNode) {
        setStatus(tr("No graph vertex under the cursor."));
        return;
    }

    Endpoint endpoint{*_hoverNode, _nodes[*_hoverNode].world};
    if (!_start || (_start && _end)) {
        _start = endpoint;
        _end.reset();
        if (_overlay) {
            _overlay->setPath({});
        }
        setStatus(tr("Start selected. Click the end vertex."));
    } else {
        _end = endpoint;
        recomputePath();
    }
    updateLabels();
}

void PatchGraphWidget::chooseFolder()
{
    const QString folder = QFileDialog::getExistingDirectory(
        this,
        tr("Choose Patch Folder"),
        _folderPath.isEmpty() ? QString() : _folderPath);
    if (folder.isEmpty()) {
        return;
    }
    _folderPath = folder;
    if (_folderLabel) {
        _folderLabel->setText(QFileInfo(folder).fileName().isEmpty() ? folder : QFileInfo(folder).fileName());
    }
    rebuildGraph();
}

void PatchGraphWidget::rebuildGraph()
{
    _nodes.clear();
    _adjacency.clear();
    _patchNames.clear();
    _surfaces.clear();
    _surfaceToPatch.clear();
    _patchNodeIds.clear();
    _patchIndex.clear();
    _hoverNode.reset();
    _start.reset();
    _end.reset();
    if (_overlay) {
        _overlay->clearPath();
    }
    updateLabels();

    if (_folderPath.isEmpty()) {
        setStatus(tr("Choose a patch folder."));
        return;
    }

    const std::filesystem::path root = _folderPath.toStdString();
    if (!std::filesystem::is_directory(root)) {
        setStatus(tr("Patch folder does not exist."));
        return;
    }

    std::vector<std::filesystem::path> patchDirs;
    if (isTifxyzDir(root)) {
        patchDirs.push_back(root);
    } else {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
            if (entry.is_directory() && isTifxyzDir(entry.path())) {
                patchDirs.push_back(entry.path());
            }
        }
    }
    std::sort(patchDirs.begin(), patchDirs.end());

    for (const auto& path : patchDirs) {
        try {
            auto surface = load_quad_from_tifxyz(path.string());
            if (!surface) {
                continue;
            }
            surface->ensureLoaded();
            _surfaceToPatch[surface.get()] = static_cast<int>(_surfaces.size());
            _patchNames.push_back(path.filename().string());
            _surfaces.push_back(std::move(surface));
        } catch (const std::exception& e) {
            qWarning("Failed to load patch graph tifxyz %s: %s", path.string().c_str(), e.what());
        }
    }

    if (_surfaces.empty()) {
        setStatus(tr("No tifxyz patches found in folder."));
        return;
    }

    _patchIndex.rebuild(_surfaces);

    for (int patch = 0; patch < static_cast<int>(_surfaces.size()); ++patch) {
        addPatchNodes(patch);
    }
    for (int patch = 0; patch < static_cast<int>(_surfaces.size()); ++patch) {
        addPatchGridEdges(patch);
    }
    addCrossPatchEdges(_linkDistanceSpin ? _linkDistanceSpin->value() : 5.0);

    std::size_t edgeCount = 0;
    for (const auto& edges : _adjacency) {
        edgeCount += edges.size();
    }
    setStatus(tr("Loaded %1 patch(es), %2 vertices, %3 edges.")
                  .arg(_surfaces.size())
                  .arg(_nodes.size())
                  .arg(edgeCount / 2));
}

void PatchGraphWidget::addPatchNodes(int patchIndex)
{
    if (patchIndex < 0 || patchIndex >= static_cast<int>(_surfaces.size())) {
        return;
    }
    auto surface = _surfaces[patchIndex];
    const cv::Mat_<cv::Vec3f>* points = surface ? surface->rawPointsPtr() : nullptr;
    if (!points || points->empty()) {
        return;
    }

    std::vector<int> localIds(static_cast<std::size_t>(points->rows) * points->cols, -1);

    auto localIndex = [points](int row, int col) {
        return row * points->cols + col;
    };

    for (int row = 0; row < points->rows; ++row) {
        for (int col = 0; col < points->cols; ++col) {
            if (!isValidVertex(*points, row, col)) {
                continue;
            }
            const int node = static_cast<int>(_nodes.size());
            const cv::Vec3f world = (*points)(row, col);
            localIds[localIndex(row, col)] = node;
            _nodes.push_back({world, patchIndex, row, col});
            _adjacency.emplace_back();
        }
    }
    _patchNodeIds.push_back(std::move(localIds));
}

void PatchGraphWidget::addPatchGridEdges(int patchIndex)
{
    if (patchIndex < 0 || patchIndex >= static_cast<int>(_surfaces.size()) ||
        patchIndex >= static_cast<int>(_patchNodeIds.size())) {
        return;
    }
    auto surface = _surfaces[patchIndex];
    const cv::Mat_<cv::Vec3f>* points = surface ? surface->rawPointsPtr() : nullptr;
    if (!points || points->empty()) {
        return;
    }

    const auto& localIds = _patchNodeIds[patchIndex];
    auto localIndex = [points](int row, int col) {
        return row * points->cols + col;
    };

    static constexpr int kDRow[] = {-1, 0, 1, 0};
    static constexpr int kDCol[] = {0, 1, 0, -1};
    for (int row = 0; row < points->rows; ++row) {
        for (int col = 0; col < points->cols; ++col) {
            const int node = localIds[localIndex(row, col)];
            if (node < 0) {
                continue;
            }
            for (int i = 0; i < 4; ++i) {
                const int nRow = row + kDRow[i];
                const int nCol = col + kDCol[i];
                if (nRow < 0 || nRow >= points->rows || nCol < 0 || nCol >= points->cols) {
                    continue;
                }
                const int neighbor = localIds[localIndex(nRow, nCol)];
                if (neighbor >= 0 && neighbor > node) {
                    addEdge(node, neighbor, distance3d(_nodes[node].world, _nodes[neighbor].world));
                }
            }
        }
    }
}

void PatchGraphWidget::addCrossPatchEdges(double linkDistance)
{
    if (linkDistance <= 0.0 || _patchIndex.empty()) {
        return;
    }
    const double linkDistanceSq = linkDistance * linkDistance;
    for (int node = 0; node < static_cast<int>(_nodes.size()); ++node) {
        SurfacePatchIndex::PointQuery query;
        query.worldPoint = _nodes[node].world;
        query.tolerance = static_cast<float>(linkDistance);
        const auto hits = _patchIndex.locateAll(query);
        for (const auto& hit : hits) {
            auto patchIt = _surfaceToPatch.find(hit.surface.get());
            if (patchIt == _surfaceToPatch.end() || patchIt->second == _nodes[node].patch) {
                continue;
            }
            auto other = nodeNearSurfaceHit(hit, _nodes[node].world);
            if (!other || *other == node) {
                continue;
            }
            const double dist = distance3d(_nodes[node].world, _nodes[*other].world);
            if (dist * dist <= linkDistanceSq) {
                addEdge(node, *other, dist);
            }
        }
    }
}

void PatchGraphWidget::addEdge(int from, int to, double weight)
{
    if (from < 0 || to < 0 || from >= static_cast<int>(_adjacency.size()) ||
        to >= static_cast<int>(_adjacency.size()) || from == to) {
        return;
    }
    _adjacency[from].push_back({to, weight});
    _adjacency[to].push_back({from, weight});
}

std::optional<int> PatchGraphWidget::nearestNode(const cv::Vec3f& worldPos, double maxDistance) const
{
    if (_nodes.empty() || maxDistance <= 0.0 || _patchIndex.empty()) {
        return std::nullopt;
    }

    SurfacePatchIndex::PointQuery query;
    query.worldPoint = worldPos;
    query.tolerance = static_cast<float>(maxDistance);
    const auto hits = _patchIndex.locateAll(query);

    double bestDistance = maxDistance;
    std::optional<int> bestNode;
    for (const auto& hit : hits) {
        auto node = nodeNearSurfaceHit(hit, worldPos);
        if (!node) {
            continue;
        }
        const double dist = distance3d(worldPos, _nodes[*node].world);
        if (dist <= bestDistance) {
            bestDistance = dist;
            bestNode = node;
        }
    }
    return bestNode;
}

std::optional<int> PatchGraphWidget::nodeNearSurfaceHit(const SurfacePatchIndex::LookupResult& hit,
                                                        const cv::Vec3f& referencePoint) const
{
    auto patchIt = _surfaceToPatch.find(hit.surface.get());
    if (patchIt == _surfaceToPatch.end()) {
        return std::nullopt;
    }
    const int patchIndex = patchIt->second;
    if (patchIndex < 0 || patchIndex >= static_cast<int>(_surfaces.size()) ||
        patchIndex >= static_cast<int>(_patchNodeIds.size())) {
        return std::nullopt;
    }
    const auto surface = _surfaces[patchIndex];
    const cv::Mat_<cv::Vec3f>* points = surface ? surface->rawPointsPtr() : nullptr;
    if (!points || points->empty()) {
        return std::nullopt;
    }

    const cv::Vec2f grid = surface->ptrToGrid(hit.ptr);
    const int centerCol = static_cast<int>(std::lround(grid[0]));
    const int centerRow = static_cast<int>(std::lround(grid[1]));
    const auto& ids = _patchNodeIds[patchIndex];
    auto localIndex = [points](int row, int col) {
        return row * points->cols + col;
    };

    double bestDist = std::numeric_limits<double>::infinity();
    std::optional<int> best;
    for (int row = centerRow - 1; row <= centerRow + 1; ++row) {
        for (int col = centerCol - 1; col <= centerCol + 1; ++col) {
            if (row < 0 || row >= points->rows || col < 0 || col >= points->cols) {
                continue;
            }
            const int node = ids[localIndex(row, col)];
            if (node < 0) {
                continue;
            }
            const double dist = distance3d(referencePoint, _nodes[node].world);
            if (dist < bestDist) {
                bestDist = dist;
                best = node;
            }
        }
    }
    return best;
}

void PatchGraphWidget::clearSelection()
{
    _start.reset();
    _end.reset();
    if (_overlay) {
        _overlay->clearPath();
    }
    setStatus(tr("Selection cleared."));
    updateLabels();
}

void PatchGraphWidget::recomputePath()
{
    if (!_start || !_end) {
        return;
    }

    if (_nodes.empty()) {
        setStatus(tr("Choose a patch folder before computing a path."));
        return;
    }

    double distance = 0.0;
    auto path = shortestVertexPath(_start->node, _end->node, &distance);
    if (!path || path->empty()) {
        if (_overlay) {
            _overlay->clearPath();
        }
        setStatus(tr("No graph path found between selected vertices."));
        return;
    }

    if (_overlay) {
        _overlay->setPath(*path);
    }
    setStatus(tr("Path: %1 vertices, distance %2")
                  .arg(path->size())
                  .arg(distance, 0, 'f', 3));
}

void PatchGraphWidget::updateLabels()
{
    if (_startLabel) {
        _startLabel->setText(endpointText(_start));
    }
    if (_endLabel) {
        _endLabel->setText(endpointText(_end));
    }
}

void PatchGraphWidget::setStatus(const QString& text)
{
    if (_statusLabel) {
        _statusLabel->setText(text);
    }
}

std::optional<std::vector<cv::Vec3f>>
PatchGraphWidget::shortestVertexPath(int start,
                                     int end,
                                     double* outDistance) const
{
    if (start < 0 || end < 0 ||
        start >= static_cast<int>(_nodes.size()) ||
        end >= static_cast<int>(_nodes.size())) {
        return std::nullopt;
    }

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> dist(_nodes.size(), inf);
    std::vector<int> parent(_nodes.size(), -1);
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> queue;

    dist[start] = 0.0;
    queue.push({0.0, start});

    while (!queue.empty()) {
        const QueueItem current = queue.top();
        queue.pop();
        if (current.distance != dist[current.node]) {
            continue;
        }
        if (current.node == end) {
            break;
        }

        for (const Edge& edge : _adjacency[current.node]) {
            const int neighbor = edge.to;
            const double candidate = current.distance + edge.weight;
            if (candidate < dist[neighbor]) {
                dist[neighbor] = candidate;
                parent[neighbor] = current.node;
                queue.push({candidate, neighbor});
            }
        }
    }

    if (!std::isfinite(dist[end])) {
        return std::nullopt;
    }
    if (outDistance) {
        *outDistance = dist[end];
    }

    std::vector<int> nodePath;
    for (int node = end; node >= 0; node = parent[node]) {
        nodePath.push_back(node);
        if (node == start) {
            break;
        }
    }
    std::reverse(nodePath.begin(), nodePath.end());

    std::vector<cv::Vec3f> path;
    path.reserve(nodePath.size());
    for (int node : nodePath) {
        path.push_back(_nodes[node].world);
    }
    return path;
}
