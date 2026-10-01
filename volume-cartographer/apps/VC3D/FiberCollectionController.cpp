#include "FiberCollectionController.hpp"
#include "CState.hpp"
#include "ViewerManager.hpp"
#include "OpenDataCoordinateIdentity.hpp"
#include "VCSettings.hpp"
#include "LineAnnotationController.hpp"
#include "LineAnnotationDialog.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/PlaneSurface.hpp"

#include <QCheckBox>
#include <QCryptographicHash>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QGraphicsView>
#include <QGuiApplication>
#include <QFrame>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMouseEvent>
#include <QPainter>
#include <QPolygonF>
#include <QPushButton>
#include <QSaveFile>
#include <QScopeGuard>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QStyleHints>
#include <QTimer>
#include <QtConcurrent/QtConcurrentRun>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

using vc::fibers::Bounds;
using vc::fibers::FiberCollection;
using vc::fibers::Point;
using Json = nlohmann::json;
namespace
{
cv::Vec3f cvpoint(const Point& p, double scale = 1)
{
    return {float(p[0] * scale), float(p[1] * scale), float(p[2] * scale)};
}
Point point(const cv::Vec3f& p, double scale = 1)
{
    return {p[0] * scale, p[1] * scale, p[2] * scale};
}
Point sub(const Point& a, const Point& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
double dot(const Point& a, const Point& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Point normalized(Point p)
{
    const double n = std::sqrt(dot(p, p));
    if (!(n > 0))
        throw std::runtime_error("Degenerate slice frame");
    for (auto& x : p)
        x /= n;
    return p;
}
bool contains(const Bounds& outer, const Bounds& inner)
{
    for (size_t i = 0; i < 3; ++i)
        if (inner.lower[i] < outer.lower[i] || inner.upper[i] > outer.upper[i])
            return false;
    return true;
}
struct RegionResult {
    vc::fibers::RegionPage page;
    Bounds coverage;
};
template <class T>
struct Outcome {
    T value{};
    QString error;
};
template <class T, class Work, class Done>
void background(QObject* parent, Work work, Done done)
{
    auto* watcher = new QFutureWatcher<Outcome<T>>(parent);
    QObject::connect(watcher, &QFutureWatcher<Outcome<T>>::finished, parent, [watcher, done]() {
        auto result = watcher->future().takeResult();
        watcher->deleteLater();
        done(std::move(result));
    });
    watcher->setFuture(QtConcurrent::run([work]() {
        Outcome<T> result;
        try {
            result.value = work();
        } catch (const std::exception& error) {
            result.error = QString::fromUtf8(error.what());
        }
        return result;
    }));
}
}  // namespace

struct FiberCollectionController::View {
    // One in-flight query per view; successive workers reuse its connection/LRU.
    // Capturing this context keeps it alive if the collection/view is detached.
    struct Reader { std::unique_ptr<FiberCollection> collection; };
    std::shared_ptr<Reader> reader;
    std::optional<Slice> requested, displayed, coverageSlice;
    std::optional<Bounds> coverage, pendingBounds;
    std::vector<vc::fibers::Block> blocks;
    std::shared_ptr<std::atomic_bool> cancelled;
    bool busy{false};
    bool complete{true};
    bool dirty{true};
    QRectF displayedRect;
    qreal displayedWidth{}, displayedDpr{};
    bool displayedRainbow{};
    QString error;
    struct Hit {
        QPointF a, b;
        int64_t fiber;
    };
    std::vector<Hit> hits;
};

FiberCollectionController::FiberCollectionController(CState* state, ViewerManager* manager, QMainWindow* window, LineAnnotationController* annotations)
    : ViewerOverlayControllerBase("fiber-collection", window), state_(state), manager_(manager), annotations_(annotations)
{
    dock_ = new QDockWidget(tr("Automated Fiber Volume"), window);
    dock_->setObjectName("fiberCollectionsDock");
    auto* body = new QWidget(dock_);
    auto* layout = new QVBoxLayout(body);
    layout->setSpacing(10);
    auto addSection = [body, layout](const QString& title, int stretch = 0, QWidget** separatorOut = nullptr) {
        if (layout->count()) {
            auto* separator = new QFrame(body);
            separator->setFrameShape(QFrame::HLine);
            separator->setFrameShadow(QFrame::Plain);
            separator->setForegroundRole(QPalette::Mid);
            layout->addWidget(separator);
            if (separatorOut) *separatorOut = separator;
        }
        auto* section = new QWidget(body);
        auto* contents = new QVBoxLayout(section);
        contents->setContentsMargins(0, 0, 0, 0);
        contents->setSpacing(8);
        auto* heading = new QLabel(title, section);
        auto font = heading->font();
        font.setBold(true);
        heading->setFont(font);
        contents->addWidget(heading);
        layout->addWidget(section, stretch);
        return contents;
    };
    auto* collectionLayout = addSection(tr("Volume"));
    auto* collectionActions = new QHBoxLayout;
    auto* open = new QPushButton(tr("Open volume…"), body);
    detach_ = new QPushButton(tr("Detach volume"), body);
    detach_->setEnabled(false);
    collectionActions->addWidget(open, 1);
    collectionActions->addWidget(detach_);
    collectionLayout->addLayout(collectionActions);
    visible_ = new QCheckBox(tr("Show volume · read only"), body);
    visible_->setChecked(true);
    collectionLayout->addWidget(visible_);
    status_ = new QLabel(tr("Open an Automated Fiber Volume (.afv)."), body);
    status_->setObjectName("fiberCollectionStatus");
    status_->setWordWrap(true);
    collectionLayout->addWidget(status_);
    auto* filtersLayout = addSection(tr("Filters"));
    auto* filterForm = new QFormLayout;
    filterForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    filterForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    auto* displayLayout = addSection(tr("Display"));
    auto* form = new QFormLayout;
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    minLength_ = new QDoubleSpinBox(body);
    minLength_->setObjectName("fiberCollectionMinLengthMm");
    minLength_->setRange(0, 1000000);
    minLength_->setDecimals(2);
    minLength_->setSingleStep(1);
    minLength_->setKeyboardTracking(false);
    minLength_->setToolTip(tr("Minimum length of the complete fiber in millimeters, inclusive. 0 shows all lengths. Applies to the catalog and visible fibers before the display limit."));
    distance_ = new QDoubleSpinBox(body);
    distance_->setRange(0.1, 1000);
    distance_->setValue(3);
    lineWidth_ = new QDoubleSpinBox(body);
    lineWidth_->setObjectName("fiberCollectionLineWidth");
    lineWidth_->setRange(0.5, 12);
    lineWidth_->setDecimals(1);
    lineWidth_->setSingleStep(0.5);
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    minLength_->setValue(settings.value("fiberCollections/minLengthMm", 0.0).toDouble());
    maxDisplayed_ = new QSpinBox(body);
    maxDisplayed_->setObjectName("fiberCollectionMaxDisplayed");
    maxDisplayed_->setRange(0, 10000000);
    maxDisplayed_->setSpecialValueText(tr("All"));
    maxDisplayed_->setSingleStep(100);
    maxDisplayed_->setValue(settings.value("fiberCollections/maxDisplayed", 1000).toInt());
    maxDisplayed_->setToolTip(tr("Maximum visible fibers in each viewport, sampled only among fibers intersecting its current slice. Stable random priorities. 0 shows all. A visible selected fiber is included within this limit."));
    lineWidth_->setValue(settings.value("fiberCollections/lineWidth", 3.0).toDouble());
    filterForm->addRow(tr("Minimum length (mm)"), minLength_);
    filtersLayout->addLayout(filterForm);
    form->addRow(tr("Max displayed fibers"), maxDisplayed_);
    form->addRow(tr("Slice distance ± (L0 vx)"), distance_);
    form->addRow(tr("Line width (px)"), lineWidth_);
    displayLayout->addLayout(form);
    horizontal_ = new QCheckBox(tr("Horizontal"), body);
    horizontal_->setObjectName("fiberCollectionHorizontal");
    horizontal_->setChecked(settings.value("fiberCollections/showHorizontal", true).toBool());
    vertical_ = new QCheckBox(tr("Vertical"), body);
    vertical_->setObjectName("fiberCollectionVertical");
    vertical_->setChecked(settings.value("fiberCollections/showVertical", true).toBool());
    auto* orientations = new QHBoxLayout;
    orientations->addWidget(horizontal_);
    orientations->addWidget(vertical_);
    filtersLayout->addLayout(orientations);
    rainbow_ = new QCheckBox(tr("Rainbow colors"), body);
    rainbow_->setObjectName("fiberCollectionRainbow");
    rainbow_->setChecked(settings.value("fiberCollections/rainbowColors", false).toBool());
    displayLayout->addWidget(rainbow_);
    auto* fibersLayout = addSection(tr("Fibers · longest first"), 1);
    list_ = new QListWidget(body);
    list_->setObjectName("fiberCollectionList");
    list_->setMinimumHeight(120);
    list_->setToolTip(tr("Click a fiber to select it and focus the main views."));
    fibersLayout->addWidget(list_, 1);
    next_ = new QPushButton(tr("Next page"), body);
    next_->setObjectName("fiberCollectionNextPage");
    auto* pages = new QHBoxLayout;
    auto* first = new QPushButton(tr("First page"), body);
    first->setObjectName("fiberCollectionFirstPage");
    pages->addWidget(first);
    pages->addWidget(next_);
    fibersLayout->addLayout(pages);
    connect(first, &QPushButton::clicked, this, [this]() { listPage(true); });
    idInput_ = new QLineEdit(body);
    idInput_->setObjectName("fiberCollectionIdInput");
    idInput_->setPlaceholderText(tr("Fiber ID · Enter to select"));
    fibersLayout->addWidget(idInput_);
    openAnnotation_ = new QPushButton(tr("Open in Line Annotation"), body);
    openAnnotation_->setObjectName("fiberCollectionOpenAnnotation");
    openAnnotation_->setEnabled(false);
    fibersLayout->addWidget(openAnnotation_);
    annotationStatus_ = new QLabel(body);
    annotationStatus_->setObjectName("fiberCollectionAnnotationStatus");
    annotationStatus_->setWordWrap(true);
    annotationStatus_->hide();
    fibersLayout->addWidget(annotationStatus_);
    auto* selectionLayout = addSection(tr("Selection"), 0, &selectionSeparator_);
    selectionSection_ = selectionLayout->parentWidget();
    selectionSection_->setObjectName("fiberCollectionSelectionSection");
    selection_ = new QLabel(tr("Click a trace to select it in the list."), body);
    selection_->setObjectName("fiberCollectionSelection");
    selection_->setWordWrap(true);
    selectionLayout->addWidget(selection_);
    along_ = new QSlider(Qt::Horizontal, body);
    along_->setEnabled(false);
    selectionLayout->addWidget(along_);
    dock_->setWidget(body);
    window->addDockWidget(Qt::RightDockWidgetArea, dock_);
    dock_->hide();
    connect(open, &QPushButton::clicked, this, [this]() {
        // macOS hides the native file-type selector when there is only one filter.
        const auto path = QFileDialog::getOpenFileName(
            dock_, tr("Open Automated Fiber Volume (.afv)"), {},
            tr("Automated Fiber Volume (*.afv);;All files (*)"));
        if (!path.isEmpty())
            openCollection(path);
    });
    connect(visible_, &QCheckBox::toggled, this, [this]() { invalidate(); });
    connect(minLength_, &QDoubleSpinBox::valueChanged, this, [this](double length) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue("fiberCollections/minLengthMm", length);
        invalidate();
        listPage(true);
    });
    auto updateFamilies = [this]() {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue("fiberCollections/showHorizontal", horizontal_->isChecked());
        settings.setValue("fiberCollections/showVertical", vertical_->isChecked());
        invalidate();
        listPage(true);
    };
    connect(horizontal_, &QCheckBox::toggled, this, updateFamilies);
    connect(vertical_, &QCheckBox::toggled, this, updateFamilies);
    connect(maxDisplayed_, &QSpinBox::valueChanged, this, [this](int maximum) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue("fiberCollections/maxDisplayed", maximum);
        invalidate();
    });
    connect(distance_, &QDoubleSpinBox::valueChanged, this, [this]() { invalidate(); });
    connect(lineWidth_, &QDoubleSpinBox::valueChanged, this, [this](double width) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue("fiberCollections/lineWidth", width);
        refreshAll();
    });
    connect(rainbow_, &QCheckBox::toggled, this, [this](bool checked) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue("fiberCollections/rainbowColors", checked);
        refreshAll();
    });
    connect(next_, &QPushButton::clicked, this, [this]() { listPage(false); });
    connect(list_, &QListWidget::itemClicked, this, [this](QListWidgetItem* item) {
        const auto id = item->data(Qt::UserRole).toLongLong();
        const bool open = lineAnnotationActive_ || annotationBusy_ || pendingAnnotation_;
        selectFiber(id, !open, open);
    });
    connect(openAnnotation_, &QPushButton::clicked, this, [this]() {
        if (selected_ > 0)
            showInLineAnnotation(selected_);
    });
    connect(idInput_, &QLineEdit::returnPressed, this, [this]() {
        bool ok = false;
        const auto id = idInput_->text().toLongLong(&ok);
        if (ok)
            selectFiber(id, !lineAnnotationActive_, lineAnnotationActive_);
    });
    connect(along_, &QSlider::valueChanged, this, &FiberCollectionController::navigate);
    connect(detach_, &QPushButton::clicked, this, [this]() {
        const auto file = attachmentPath();
        clear();
        QFile::remove(file);
    });
    connect(state_, &CState::volumeChanged, this, [this]() { volumeChanged(); });
    connect(state_, &CState::volumeClosing, this, [this]() { clear(); });
    bindToViewerManager(manager);
}
FiberCollectionController::~FiberCollectionController()
{
    for (auto& [v, s] : views_)
        if (s->cancelled)
            s->cancelled->store(true);
}
void FiberCollectionController::setLineAnnotationActive(bool active)
{
    lineAnnotationActive_ = active;
    list_->setToolTip(active
        ? tr("Click a fiber to open its flattened view in Line Annotation.")
        : tr("Click a fiber to select it and focus the main views."));
    openAnnotation_->setVisible(!active);
    selectionSection_->setVisible(!active);
    selectionSeparator_->setVisible(!active);
}
void FiberCollectionController::clear()
{
    ++revision_;
    ++selectionRevision_;
    ++navigationRevision_;
    ++annotationRevision_;
    for (const auto& connection : annotationRenderConnections_)
        disconnect(connection);
    annotationRenderConnections_.clear();
    annotationStatus_->clear();
    annotationStatus_->hide();
    press_.reset();
    click_.reset();
    enabled_ = false;
    detach_->setEnabled(false);
    openAnnotation_->setEnabled(false);
    selected_ = 0;
    pendingAnnotation_ = 0;
    path_.clear();
    attachment_.clear();
    uuid_.clear();
    coordinateSpace_.clear();
    for (auto& [v, s] : views_) {
        if (s->cancelled)
            s->cancelled->store(true);
        s->blocks.clear();
        s->hits.clear();
        s->requested.reset();
        s->displayed.reset();
        s->coverage.reset();
        s->coverageSlice.reset();
        s->pendingBounds.reset();
        s->reader.reset();
        s->dirty = true;
    }
    catalogPending_ = false;
    pendingIndex_.reset();
    list_->clear();
    along_->setEnabled(false);
    status_->setText(tr("No Automated Fiber Volume open."));
    selection_->clear();
    refreshAll();
}
QString FiberCollectionController::attachmentPath() const
{
    const auto path = state_->vpkg() ? QString::fromStdString(state_->vpkg()->path().string()) : QString{};
    if (path.isEmpty())
        return {};
    return QFileInfo(path).isDir() ? path + "/fiber-collection.json" : path + ".fiber-collection.json";
}
void FiberCollectionController::projectChanged()
{
    const auto attached = attachmentPath();
    clear();
    QFile file(attached);
    if (file.open(QIODevice::ReadOnly)) {
        try {
            const auto data = Json::parse(file.readAll().toStdString());
            if (data.at("version").get<int>() != 1)
                throw std::runtime_error("Unsupported Automated Fiber Volume attachment version");
            const auto storedPath = QString::fromStdString(data.at("path").get<std::string>());
            // Preserve absolute network paths (including Windows UNC shares).
            // Only relative attachments are resolved against the project folder.
            const auto path = QDir::isAbsolutePath(storedPath)
                ? storedPath : QFileInfo(attached).dir().absoluteFilePath(storedPath);
            openCollection(path, false, data.at("uuid").get<std::string>());
        } catch (const std::exception& e) {
            status_->setText(QString::fromUtf8(e.what()));
        }
    }
}
void FiberCollectionController::volumeChanged()
{
    // Another volume of the same scan keeps the open collection, its selection
    // and caches; only the native-to-viewer factor may differ.
    if (enabled_ && state_->currentVolume() && state_->vpkg() && attachment_ == attachmentPath()) {
        const auto identity = vc3d::opendata::coordinateIdentityForVolume(*state_->vpkg(), state_->currentVolumeId());
        if (identity && identity->coordinateSpace == coordinateSpace_ && identity->sourceOriginalResolution == sourceResolution_) {
            const double factor = 1.0 / double(identity->sourceCoordinateScaleFactor);
            if (factor != nativeToViewer_) {
                nativeToViewer_ = factor;
                invalidate();
            }
            return;
        }
    }
    projectChanged();
}
void FiberCollectionController::openCollection(const QString& path, bool persist, const std::string& expectedUuid)
{
    clear();
    // Restoring the project's attachment leaves the dock as the user left it.
    if (persist)
        dock_->show();
    if (!state_->currentVolume() || !state_->vpkg()) {
        status_->setText(tr("Open the CT volume first."));
        return;
    }
    const auto identity = vc3d::opendata::coordinateIdentityForVolume(*state_->vpkg(), state_->currentVolumeId());
    if (!identity) {
        status_->setText(tr("This volume has no verified coordinate identity. Attach a volume with coordinate metadata first."));
        return;
    }
    path_ = QFileInfo(path).absoluteFilePath();
    const auto file = path_.toStdString();
    const auto rev = revision_;
    status_->setText(tr("Opening volume…"));
    background<Json>(
        this,
        [file, identity, expectedUuid]() {
            FiberCollection c(file);
            Json info;
            for (const auto* k : {"uuid", "frame", "fiber_count", "point_count"})
                info[k] = Json::parse(c.metadata(k));
            const auto uuid = info.at("uuid").get<std::string>();
            if (!expectedUuid.empty() && expectedUuid != uuid)
                throw std::runtime_error(
                    "The Automated Fiber Volume has been replaced (UUID mismatch). Open the new file explicitly.");
            if (uuid.empty() || info.at("fiber_count").get<int64_t>() < 0)
                throw std::runtime_error("Invalid Automated Fiber Volume metadata");
            const auto& frame = info.at("frame");
            if (frame.at("vc_open_data_coordinate_space").get<std::string>() != identity->coordinateSpace ||
                frame.value("vc_open_data_source_coordinate_level", 0) != 0 || frame.value("vc_open_data_source_coordinate_scale_factor", 1) != 1)
                throw std::runtime_error("Automated Fiber Volume and CT coordinate frames do not match.");
            return info;
        },
        [this, rev, identity, persist](const Outcome<Json>& result) {
            if (rev != revision_)
                return;
            if (!result.error.isEmpty()) {
                status_->setText(result.error);
                return;
            }
            nativeToViewer_ = 1.0 / double(identity->sourceCoordinateScaleFactor);
            // Collection lengths are L0 voxels; the identity resolution is in
            // micrometres, independent of the displayed pyramid/zoom level.
            nativeVoxelMm_ = identity->sourceOriginalResolution / 1000.0;
            coordinateSpace_ = identity->coordinateSpace;
            sourceResolution_ = identity->sourceOriginalResolution;
            attachment_ = attachmentPath();
            uuid_ = result.value.at("uuid").get<std::string>();
            totalFibers_ = result.value.at("fiber_count").get<int64_t>();
            enabled_ = true;
            detach_->setEnabled(true);
            status_->setToolTip(path_ + "\n" + QString::fromStdString(uuid_));
            status_->setText(tr("%1 fibers · native geometry · read only").arg(result.value.at("fiber_count").get<int64_t>()));
            if (persist && !attachmentPath().isEmpty()) {
                const auto base = std::filesystem::path(attachmentPath().toStdString()).parent_path();
                std::error_code ec;
                auto relative = std::filesystem::relative(path_.toStdString(), base, ec);
                Json attachment = {{"version", 1}, {"path", ec ? path_.toStdString() : relative.string()}, {"uuid", uuid_}};
                QSaveFile out(attachmentPath());
                if (!out.open(QIODevice::WriteOnly) || out.write(attachment.dump().c_str()) < 0 || !out.commit())
                    status_->setText(tr("Volume opened; its attachment to the project could not be saved."));
            }
            listPage(true);
            invalidate();
        });
}
void FiberCollectionController::invalidate()
{
    if (enabled_)
        status_->setText(maxDisplayed_->value()
            ? tr("%1 fibers in volume · up to %2 visible per view").arg(totalFibers_).arg(maxDisplayed_->value())
            : tr("%1 fibers in volume · all visible fibers").arg(totalFibers_));
    for (auto& [v, s] : views_) {
        if (s->cancelled)
            s->cancelled->store(true);
        s->requested.reset();
        s->displayed.reset();
        s->coverage.reset();
        s->coverageSlice.reset();
        s->pendingBounds.reset();
        s->blocks.clear();
        s->hits.clear();
        s->dirty = true;
    }
    refreshAll();
}
vc::fibers::FamilyFilter FiberCollectionController::familyFilter() const
{
    using vc::fibers::FamilyFilter;
    if (horizontal_->isChecked() && vertical_->isChecked()) return FamilyFilter::All;
    if (horizontal_->isChecked()) return FamilyFilter::Horizontal;
    if (vertical_->isChecked()) return FamilyFilter::Vertical;
    return FamilyFilter::None;
}
void FiberCollectionController::listPage(bool reset)
{
    if (!enabled_)
        return;
    if (reset) {
        lastRow_ = 0;
        lastLength_ = std::numeric_limits<double>::infinity();
        list_->clear();
    }
    if (catalogBusy_) {
        catalogPending_ = true;
        return;
    }
    const auto after = lastRow_;
    const auto beforeLength = lastLength_;
    const auto rev = revision_;
    const double minMm = minLength_->value();
    const double min = minMm / nativeVoxelMm_;
    const auto family = familyFilter();
    const auto path = path_.toStdString();
    catalogBusy_ = true;
    next_->setEnabled(false);
    background<std::vector<vc::fibers::Summary>>(
        this,
        [path, after, beforeLength, min, family]() { return FiberCollection(path).catalogByLength(beforeLength, after, 200, min, family); },
        [this, rev, minMm, after, beforeLength, family](const Outcome<std::vector<vc::fibers::Summary>>& result) {
            catalogBusy_ = false;
            if (catalogPending_) {
                catalogPending_ = false;
                listPage(false);
                return;
            }
            if (rev != revision_ || minMm != minLength_->value() || family != familyFilter() || after != lastRow_ || beforeLength != lastLength_)
                return;
            if (!result.error.isEmpty()) {
                status_->setText(result.error);
                return;
            }
            list_->clear();
            for (const auto& s : result.value) {
                auto* row =
                    new QListWidgetItem(tr("%1 mm · #%2 · %3").arg(s.length * nativeVoxelMm_, 0, 'f', 2).arg(s.id).arg(QString::fromStdString(s.family)), list_);
                row->setData(Qt::UserRole, qlonglong(s.id));
                row->setToolTip(tr("%1 points").arg(s.pointCount));
                lastRow_ = s.id;
                lastLength_ = s.length;
            }
            highlightListFiber(selected_);
            next_->setEnabled(result.value.size() == 200);
        });
}
bool FiberCollectionController::highlightListFiber(int64_t id)
{
    for (int i = 0; i < list_->count(); ++i) {
        auto* row = list_->item(i);
        if (row->data(Qt::UserRole).toLongLong() != id)
            continue;
        // Programmatic selection must never open an annotation.
        const QSignalBlocker block(list_);
        list_->setCurrentItem(row);
        list_->scrollToItem(row);
        return true;
    }
    return false;
}
void FiberCollectionController::revealListFiber(const vc::fibers::Summary& summary)
{
    if (highlightListFiber(summary.id) && !catalogBusy_ && !catalogPending_)
        return;
    // Seek directly to this fiber in the length-sorted catalog. The first
    // row is inclusive; Next page continues from the actual final row.
    lastLength_ = summary.length;
    lastRow_ = summary.id - 1;
    listPage(false);
}
void FiberCollectionController::selectFiber(int64_t id, bool focusView, bool openAnnotation)
{
    if (!enabled_ || id <= 0)
        return;
    selected_ = id;
    ++navigationRevision_;
    pendingIndex_.reset();
    ++annotationRevision_;
    pendingAnnotation_ = 0;
    for (const auto& connection : annotationRenderConnections_)
        disconnect(connection);
    annotationRenderConnections_.clear();
    annotationStatus_->setVisible(openAnnotation);
    if (openAnnotation)
        annotationStatus_->setText(tr("Loading fiber #%1…").arg(id));
    highlightListFiber(id);
    const auto token = ++selectionRevision_;
    const auto rev = revision_;
    along_->setEnabled(false);
    openAnnotation_->setEnabled(false);
    selection_->setText(tr("Loading fiber #%1…").arg(id));
    const auto path = path_.toStdString();
    background<vc::fibers::Summary>(
        this,
        [path, id]() { return FiberCollection(path).summary(id); },
        [this, token, rev, focusView, openAnnotation](const Outcome<vc::fibers::Summary>& result) {
            if (rev != revision_ || token != selectionRevision_)
                return;
            if (!result.error.isEmpty()) {
                selection_->setText(result.error);
                if (openAnnotation) annotationStatus_->setText(result.error);
                return;
            }
            const auto& s = result.value;
            revealListFiber(s);
            selection_->setToolTip(QString::fromStdString(s.name));
            selection_->setText(
                tr("#%1 · %2 points · %3 mm\nFollow the complete stored fiber with the slider.").arg(s.id).arg(s.pointCount).arg(s.length * nativeVoxelMm_, 0, 'f', 2));
            const QSignalBlocker block(along_);
            along_->setRange(0, int(std::min<int64_t>(s.pointCount - 1, std::numeric_limits<int>::max())));
            const int centerIndex = along_->maximum() / 2;
            along_->setValue(centerIndex);
            along_->setEnabled(true);
            openAnnotation_->setEnabled(annotations_ != nullptr);
            openAnnotation_->setToolTip(tr("Open fiber #%1 in Line Annotation.").arg(s.id));
            invalidate();
            if (focusView)
                navigate(centerIndex);
            refreshAll();
            if (openAnnotation)
                showInLineAnnotation(s.id);
        });
}
void FiberCollectionController::navigate(int index)
{
    if (!selected_ || !enabled_)
        return;
    ++navigationRevision_;
    pendingIndex_ = index;
    if (navigationBusy_)
        return;
    navigationBusy_ = true;
    pendingIndex_.reset();
    const auto id = selected_;
    const auto rev = revision_;
    const auto token = navigationRevision_;
    const auto path = path_.toStdString();
    background<Point>(
        this,
        [path, id, index]() { return FiberCollection(path).point(id, index); },
        [this, rev, token](const Outcome<Point>& result) {
            navigationBusy_ = false;
            if (pendingIndex_) {
                const int next = *pendingIndex_;
                pendingIndex_.reset();
                navigate(next);
                return;
            }
            if (rev != revision_ || token != navigationRevision_)
                return;
            if (!result.error.isEmpty()) {
                selection_->setText(result.error);
                return;
            }
            manager_->centerFocusAt(cvpoint(result.value, nativeToViewer_), {0, 0, 1}, "fiber-collection");
        });
}
std::optional<FiberCollectionController::Slice> FiberCollectionController::slice(VolumeViewerBase* viewer) const
{
    if (!dynamic_cast<PlaneSurface*>(viewer->currentSurface()))
        return std::nullopt;
    const auto rect = visibleSceneRect(viewer);
    if (rect.isEmpty())
        return std::nullopt;
    Slice s;
    s.origin = point(viewer->sceneToVolume(rect.topLeft()), 1 / nativeToViewer_);
    const auto x = point(viewer->sceneToVolume(rect.topRight()), 1 / nativeToViewer_);
    const auto y = point(viewer->sceneToVolume(rect.bottomLeft()), 1 / nativeToViewer_);
    try {
        s.u = normalized(sub(x, s.origin));
        s.v = normalized(sub(y, s.origin));
        s.n = normalized({s.u[1] * s.v[2] - s.u[2] * s.v[1], s.u[2] * s.v[0] - s.u[0] * s.v[2], s.u[0] * s.v[1] - s.u[1] * s.v[0]});
    } catch (...) {
        return std::nullopt;
    }
    const double d = distance_->value();
    s.local = {{0, 0, -d}, {std::sqrt(dot(sub(x, s.origin), sub(x, s.origin))), std::sqrt(dot(sub(y, s.origin), sub(y, s.origin))), d}};
    s.box.lower.fill(std::numeric_limits<double>::infinity());
    s.box.upper.fill(-std::numeric_limits<double>::infinity());
    for (double a : {0.0, s.local.upper[0]})
        for (double b : {0.0, s.local.upper[1]})
            for (double c : {-d, d})
                for (int i = 0; i < 3; ++i) {
                    const double p = s.origin[i] + a * s.u[i] + b * s.v[i] + c * s.n[i];
                    s.box.lower[i] = std::min(s.box.lower[i], p);
                    s.box.upper[i] = std::max(s.box.upper[i], p);
                }
    return s;
}
bool FiberCollectionController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && enabled_ && visible_->isChecked();
}
bool FiberCollectionController::needsOverlayRebuild(VolumeViewerBase* viewer) const
{
    const auto found = views_.find(viewer);
    if (found == views_.end()) return true;
    const auto& v = *found->second;
    const auto current = slice(viewer);
    return v.dirty || v.displayed != current || v.displayedRect != visibleSceneRect(viewer) ||
        v.displayedWidth != lineWidth_->value() || v.displayedRainbow != rainbow_->isChecked() ||
        v.displayedDpr != viewer->graphicsView()->viewport()->devicePixelRatioF();
}
void FiberCollectionController::request(VolumeViewerBase* viewer, const Slice& s)
{
    auto& ptr = views_[viewer];
    if (!ptr) {
        ptr = std::make_unique<View>();
        viewer->graphicsView()->viewport()->installEventFilter(this);
    }
    auto& v = *ptr;
    const bool limited = maxDisplayed_->value() > 0;
    if (!limited && v.complete && v.coverageSlice) {
        // A complete region already contains all segments needed for a zoom
        // or pan inside its prism; AABB containment is unsafe for oblique views.
        if (v.coverageSlice->contains(s)) {
            if (v.busy && v.cancelled) v.cancelled->store(true);
            return;
        }
    }
    if (v.requested && *v.requested == s)
        return;
    if (v.busy) {
        // Let a query covering the new layer finish instead of restarting it.
        if (v.cancelled && (limited || !v.pendingBounds || !contains(*v.pendingBounds, s.box)))
            v.cancelled->store(true);
        return;
    }
    v.busy = true;
    v.dirty = true;
    v.requested = s;
    v.error.clear();
    auto cancel = std::make_shared<std::atomic_bool>(false);
    v.cancelled = cancel;
    const auto rev = revision_;
    const auto path = path_.toStdString();
    const double min = minLength_->value() / nativeVoxelMm_;
    const auto family = familyFilter();
    const auto maximum = maxDisplayed_->value();
    const auto selected = selected_;
    if (!v.reader) v.reader = std::make_shared<View::Reader>();
    const auto reader = v.reader;
    v.pendingBounds = s.box;
    background<RegionResult>(
        this,
        [path, reader, s, min, cancel, maximum, selected, family]() {
            if (!reader->collection)
                reader->collection = std::make_unique<FiberCollection>(path, 32 * 1024 * 1024);
            return RegionResult{reader->collection->viewRegion(s, 2, size_t(maximum), selected,
                8 * 1024 * 1024, cancel.get(), min, family), s.box};
        },
        [this, viewer, rev, cancel, s](Outcome<RegionResult> result) {
            auto found = views_.find(viewer);
            if (found == views_.end() || found->second->cancelled != cancel)
                return;
            auto& v = *found->second;
            v.busy = false;
            v.dirty = true;
            v.pendingBounds.reset();
            if (rev != revision_ || cancel->load()) {
                v.requested.reset();
                refreshViewer(viewer);
                return;
            }
            v.error = result.error;
            if (result.error.isEmpty()) {
                v.blocks = std::move(result.value.page.blocks);
                v.complete = result.value.page.complete;
                v.coverage = result.value.coverage;
                v.coverageSlice = s;
            }
            refreshViewer(viewer);
        });
}
void FiberCollectionController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    const auto current = slice(viewer);
    if (!current) {
        // The base clears this layer when no planar projection is supported.
        // Returning to the same former plane must rebuild it, not reuse a stamp.
        if (auto found = views_.find(viewer); found != views_.end()) {
            found->second->displayed.reset();
            found->second->dirty = true;
            found->second->hits.clear();
        }
        return;
    }
    request(viewer, *current);
    auto& v = *views_.at(viewer);
    v.dirty = false;
    v.displayed = *current;
    v.displayedRect = visibleSceneRect(viewer);
    v.displayedWidth = lineWidth_->value();
    v.displayedDpr = viewer->graphicsView()->viewport()->devicePixelRatioF();
    v.displayedRainbow = rainbow_->isChecked();
    v.hits.clear();
    OverlayStyle info;
    info.penColor = QColor(255, 210, 100);
    info.z = 60;
    const auto rect = visibleSceneRect(viewer);
    if (!v.error.isEmpty()) {
        builder.addText(rect.topLeft() + QPointF(8, 20), v.error, QFont{}, info);
        if (!v.coverage)
            return;
    }
    if (!v.coverage) {
        builder.addText(rect.topLeft() + QPointF(8, 20), tr("Loading fiber region…"), QFont{}, info);
        return;
    }
    // Reclip cached geometry to the current slice while replacement data is
    // fetched. Do not replace the whole overlay with a blank/loading frame.
    v.displayed = *current;
    using Runs = std::vector<QPolygonF>;
    Runs horizontal, vertical, other, selected;
    std::array<Runs, 32> rainbowPaths;
    const bool useRainbow = rainbow_->isChecked();
    std::unordered_set<int64_t> visibleIds;
    for (const auto& b : v.blocks) {
        // Hash the logical fiber ID, not its storage block or catalog row, so
        // every block of a fiber keeps its color across region/filter changes.
        const auto colorIndex = ((uint64_t(b.fiberId) * 0x9e3779b97f4a7c15ULL) >> 32) % rainbowPaths.size();
        auto& path = b.fiberId == selected_ ? selected
                     : useRainbow           ? rainbowPaths[colorIndex]
                     : b.family == "H"      ? horizontal
                     : b.family == "V"      ? vertical
                                            : other;
        if (b.points.empty()) continue;
        QPolygonF run;
        auto flush = [&]() {
            if (!run.isEmpty()) { path.push_back(std::move(run)); run = {}; }
        };
        auto previous = current->project(b.points.front());
        for (size_t i = 1; i < b.points.size(); ++i) {
            const auto a = previous, z = current->project(b.points[i]);
            previous = z;
            double lo, hi;
            if (!vc::fibers::clipSegment(a, z, current->local, lo, hi)) {
                flush();
                continue;
            }
            Point p, q;
            for (int j = 0; j < 3; ++j) {
                const double d = b.points[i][j] - b.points[i - 1][j];
                p[j] = b.points[i - 1][j] + lo * d;
                q[j] = b.points[i - 1][j] + hi * d;
            }
            const auto pa = viewer->volumeToScene(cvpoint(p, nativeToViewer_)), pb = viewer->volumeToScene(cvpoint(q, nativeToViewer_));
            // Stroke contiguous segments together. Independent round caps on
            // every tiny segment make Qt's compound-path rasterizer expensive
            // in dense views. Keep every vertex and every actual slice break.
            if (!run.isEmpty() && run.back() != pa) flush();
            if (run.isEmpty()) run.push_back(pa);
            run.push_back(pb);
            v.hits.push_back({pa, pb, b.fiberId});
            visibleIds.insert(b.fiberId);
        }
        flush();
    }
    // Rasterize this layer once per actual geometry/view change. CT tile
    // arrivals can then repaint it with one blit instead of stroking thousands
    // of segments again.
    const qreal dpr = viewer->graphicsView()->viewport()->devicePixelRatioF();
    const QSize pixels(qCeil(rect.width() * dpr), qCeil(rect.height() * dpr));
    QImage raster;
    if (pixels.width() > 0 && pixels.height() > 0 &&
        qint64(pixels.width()) * pixels.height() * 4 <= 32 * 1024 * 1024) {
        raster = QImage(pixels, QImage::Format_ARGB32_Premultiplied);
        raster.setDevicePixelRatio(dpr);
        raster.fill(Qt::transparent);
    }
    QPainter painter;
    if (!raster.isNull()) {
        painter.begin(&raster);
        painter.setRenderHints(viewer->graphicsView()->renderHints());
        painter.translate(-rect.topLeft());
        painter.setBrush(Qt::NoBrush);
    }
    auto draw = [&](const Runs& runs, QColor color, qreal width) {
        if (painter.isActive()) {
            painter.setPen(QPen(color, width, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
            for (const auto& run : runs) painter.drawPolyline(run);
            return;
        }
        OverlayStyle style;
        style.penColor = color;
        style.penWidth = width;
        style.z = 45;
        QPainterPath path;
        for (const auto& run : runs) {
            path.moveTo(run.front());
            for (qsizetype i = 1; i < run.size(); ++i) path.lineTo(run[i]);
        }
        builder.addPainterPath(path, style);
    };
    const qreal width = lineWidth_->value();
    if (useRainbow) {
        for (size_t i = 0; i < rainbowPaths.size(); ++i)
            draw(rainbowPaths[i], QColor::fromHsvF(double(i) / rainbowPaths.size(), 0.7, 1.0), width);
    } else {
        draw(horizontal, QColor("#40a5ff"), width);
        draw(vertical, QColor("#ffad43"), width);
        draw(other, QColor("#bbbbbb"), width);
    }
    draw(selected, QColor("#ffff66"), width + 2);
    if (painter.isActive()) {
        painter.end();
        builder.addImage(raster, QPixmap::fromImage(raster), rect.topLeft(), 1, 1, 1, 45);
    }
    const QString count = maxDisplayed_->value()
        ? tr("%1 visible fibers · max %2").arg(visibleIds.size()).arg(maxDisplayed_->value())
        : tr("%1 visible fibers").arg(visibleIds.size());
    builder.addText(rect.topLeft() + QPointF(8, 40), count, QFont{}, info);
    if (v.busy && (!v.requested || *v.requested != *current || maxDisplayed_->value() || !contains(*v.coverage, current->box)))
        builder.addText(rect.topLeft() + QPointF(8, 20), tr("Updating fiber region…"), QFont{}, info);
    else if (!v.complete)
        builder.addText(rect.topLeft() + QPointF(8, 20), tr("Partial fiber view · zoom in or increase minimum length"), QFont{}, info);
}
bool FiberCollectionController::eventFilter(QObject* obj, QEvent* event)
{
    // Only observe the mouse here, so native tools receive every press, drag
    // and release. Picking waits for handleVolumeClick().
    if (event->type() != QEvent::MouseButtonPress && event->type() != QEvent::MouseButtonRelease)
        return false;
    const auto* mouse = static_cast<QMouseEvent*>(event);
    if (mouse->button() != Qt::LeftButton)
        return false;
    if (event->type() == QEvent::MouseButtonPress) {
        press_ = Click{obj, mouse->position()};
        click_.reset();
    } else {
        const bool clicked = press_ && press_->viewport == obj &&
                             (mouse->position() - press_->position).manhattanLength() <= QGuiApplication::styleHints()->startDragDistance();
        click_ = clicked ? std::optional<Click>(Click{obj, mouse->position()}) : std::nullopt;
        press_.reset();
    }
    return false;
}
bool FiberCollectionController::handleVolumeClick(Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    const auto clicked = std::exchange(click_, std::nullopt);
    if (button != Qt::LeftButton || modifiers != Qt::NoModifier || !clicked || !clicked->viewport || !enabled_ ||
        !visible_->isChecked())
        return false;
    for (auto& [viewer, v] : views_) {
        if (viewer->graphicsView()->viewport() != clicked->viewport)
            continue;
        const auto* chunked = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject());
        if (chunked && chunked->lineAnnotationPlacementPreviewEnabled())
            return false;
        const auto current = slice(viewer);
        if (!current || !v->displayed || *current != *v->displayed)
            return false;
        const auto* view = viewer->graphicsView();
        const auto click = clicked->position;
        double best = 8 * 8;
        int64_t id = 0;
        for (const auto& hit : v->hits) {
            const QPointF a = view->mapFromScene(hit.a), b = view->mapFromScene(hit.b), d = b - a;
            const double len = dot({d.x(), d.y(), 0}, {d.x(), d.y(), 0});
            const double t = len ? std::clamp(((click.x() - a.x()) * d.x() + (click.y() - a.y()) * d.y()) / len, 0.0, 1.0) : 0;
            const auto delta = click - (a + t * d);
            const double dist = delta.x() * delta.x() + delta.y() * delta.y();
            if (dist < best) {
                best = dist;
                id = hit.fiber;
            }
        }
        if (!id)
            return false;
        selectFiber(id, false);
        dock_->show();
        dock_->raise();
        return true;
    }
    return false;
}
void FiberCollectionController::showInLineAnnotation(int64_t id)
{
    if (!annotations_ || !enabled_)
        return;
    pendingAnnotation_ = id;
    ++annotationRevision_;
    for (const auto& connection : annotationRenderConnections_)
        disconnect(connection);
    annotationRenderConnections_.clear();
    annotationStatus_->setText(tr("Loading fiber #%1…").arg(id));
    annotationStatus_->show();
    // Paint the requested ID before the native importer/view construction runs.
    QTimer::singleShot(0, this, &FiberCollectionController::startPendingAnnotation);
}
void FiberCollectionController::startPendingAnnotation()
{
    if (annotationBusy_ || !pendingAnnotation_ || !enabled_)
        return;
    const auto id = std::exchange(pendingAnnotation_, 0);
    const auto token = annotationRevision_;
    const auto digest = QCryptographicHash::hash(QByteArray::fromStdString(uuid_), QCryptographicHash::Sha256).toHex().left(32);
    const auto filename = QString("collection_%1_%2.json").arg(QString::fromLatin1(digest)).arg(id);
    const auto existing = annotations_->fiberIdForFileName(filename.toStdString());
    annotationBusy_ = true;
    const auto path = path_.toStdString();
    const auto rev = revision_;
    background<Json>(this, [path, id, existing]() {
        if (existing) return Json{};
        FiberCollection collection(path);
        auto entry = Json::parse(collection.annotation(id));
        const auto summary = collection.summary(id);
        auto points = Json::array();
        int64_t next = 0;
        while (next < summary.pointCount-1) {
            auto blocks = collection.fiberBlocks(id, next);
            if (blocks.empty())
                throw std::runtime_error("Incomplete fiber geometry");
            for (const auto& b : blocks) {
                if (b.firstSegment != next)
                    throw std::runtime_error("Non-contiguous fiber geometry");
                for (size_t i = next ? 1 : 0; i < b.points.size(); ++i)
                    points.push_back(b.points[i]);
                next += int64_t(b.points.size())-1;
            }
        }
        if (points.size() != size_t(summary.pointCount))
            throw std::runtime_error("Incomplete fiber geometry");
        entry["line_points"] = std::move(points);
        // Coordinate metadata may live at collection level. Native annotation
        // files must be self-contained because they outlive this attachment.
        const auto root = Json::parse(collection.metadata("root"));
        for (const auto* key : {"coordinate_base_shape_zyx", "vc_open_data_coordinate_space", "vc_open_data_source_path",
             "vc_open_data_source_coordinate_level", "vc_open_data_source_coordinate_scale_factor", "vc_open_data_source_original_resolution"})
            if (!entry.contains(key) && root.contains(key))
                entry[key] = root.at(key);
        if (!entry.contains("control_points"))
            entry["control_points"] = Json::array({entry["line_points"].front(), entry["line_points"].back()});
        return entry;
    }, [this, rev, token, filename, id, existing](const Outcome<Json>& result) {
        // Keep the guard through native open too: saving an old annotation can
        // run a nested event loop. New clicks replace pending work, never recurse.
        const auto finish = qScopeGuard([this]() { finishAnnotationRequest(); });
        if (rev != revision_ || token != annotationRevision_)
            return;
        if (!result.error.isEmpty()) {
            annotationStatus_->setText(tr("Fiber #%1: %2").arg(id).arg(result.error));
            return;
        }
        uint64_t nativeId = existing;
        if (!nativeId) {
            // Only the requested fiber enters the native editable store.
            QString error;
            nativeId = annotations_->importFiberJson(result.value, filename.toStdString(), &error);
            if (!nativeId) {
                annotationStatus_->setText(error.isEmpty() ? tr("Could not open fiber #%1.").arg(id) : error);
                return;
            }
        }
        annotationStatus_->setText(tr("Building flattened views for fiber #%1…").arg(id));
        QString error;
        auto* dialog = annotations_->openCollectionFiber(nativeId, &error);
        if (rev != revision_ || token != annotationRevision_)
            return;
        if (!dialog) {
            annotationStatus_->setText(tr("Fiber #%1: %2").arg(id).arg(error));
            return;
        }
        watchAnnotationRender(dialog, id, token);
    });
}
void FiberCollectionController::finishAnnotationRequest()
{
    annotationBusy_ = false;
    if (pendingAnnotation_)
        QTimer::singleShot(0, this, &FiberCollectionController::startPendingAnnotation);
}
void FiberCollectionController::watchAnnotationRender(LineAnnotationDialog* dialog, int64_t id, uint64_t token)
{
    annotationStatus_->setText(tr("Loading CT for fiber #%1…").arg(id));
    auto waiting = std::make_shared<std::unordered_set<CChunkedVolumeViewer*>>();
    for (const auto& pane : dialog->panes()) {
        const QString name = QString::fromStdString(pane.surfaceName);
        if (pane.viewer && (name.endsWith("_line_surface") || name.endsWith("_line_side_slice")))
            waiting->insert(pane.viewer.data());
    }
    const auto ready = [this, token, id, waiting](CChunkedVolumeViewer* viewer) {
        if (token != annotationRevision_ || !enabled_)
            return;
        if (!viewer->hasDisplayedRenderFrame() ||
            viewer->displayedSurfaceGeometryEpoch() != viewer->surfaceGeometryEpoch())
            return;
        waiting->erase(viewer);
        if (waiting->empty()) {
            annotationStatus_->setText(tr("Fiber #%1 displayed.").arg(id));
            for (const auto& connection : annotationRenderConnections_)
                disconnect(connection);
            annotationRenderConnections_.clear();
        }
    };
    const auto strips = *waiting;
    for (auto* viewer : strips) {
        annotationRenderConnections_.push_back(connect(viewer, &CChunkedVolumeViewer::renderFrameCompleted,
            this, [ready, viewer](std::uint64_t, double) { ready(viewer); }));
    }
    for (auto* viewer : strips)
        ready(viewer);
    annotationRenderConnections_.push_back(connect(dialog, &QObject::destroyed, this, [this, token]() {
        if (token == annotationRevision_) {
            annotationStatus_->clear();
            annotationStatus_->hide();
        }
    }));
}
void FiberCollectionController::detachViewer(VolumeViewerBase* viewer)
{
    if (auto it = views_.find(viewer); it != views_.end()) {
        if (it->second->cancelled)
            it->second->cancelled->store(true);
        views_.erase(it);
    }
    ViewerOverlayControllerBase::detachViewer(viewer);
}
