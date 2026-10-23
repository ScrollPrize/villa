#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <bit>
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

#include "ObjectStoreCoordinates.hpp"
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
#include <set>

using vc::fibers::Bounds;

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
struct CatalogEntry {
    vc::objects::PinnedObject pin;
    vc::fibers::Summary summary;
    double factor = -1, sourceLength = 0;
    size_t group = 0;
};
struct RegionResult {
    vc::fibers::RegionPage page;
    Bounds coverage;
    std::map<std::string, std::pair<vc::objects::PinnedObject, double>> fibers;
    std::vector<std::string> blockUuids;
};
std::optional<double> groupScale(const vc::objects::ObjectCoordinateGroup& group, const std::string& volume)
{
    if (volume.empty()) return std::nullopt;
    return vc::objects::geometryScaleToVolume(Json{{"coordinates", Json::parse(group.coordinates)}}, volume);
}
vc::objects::PinnedObject pinRevision(const vc::objects::AttachmentIdentity& attachment,
                                     const vc::objects::ObjectRevision& revision)
{
    return {attachment, {attachment.storeUuid, revision.objectUuid}, revision.uuid, revision.geometryRevision};
}
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
    dock_ = new QDockWidget(tr("Object Store Fibers"), window);
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
    visible_ = new QCheckBox(tr("Show stored fibers"), body);
    visible_->setChecked(true);
    collectionLayout->addWidget(visible_);
    status_ = new QLabel(tr("Open an Object Store (.vcobj)."), body);
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
    maxDisplayed_->setSingleStep(500);
    maxDisplayed_->setKeyboardTracking(false);
    maxDisplayed_->setValue(settings.value("fiberCollections/maxDisplayed", 5000).toInt());
    maxDisplayed_->setToolTip(tr("Maximum visible fibers in each viewport, sampled only among fibers intersecting its current slice. Stable UUID order. 0 shows all. A visible selected fiber is included within this limit."));
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
            dock_, tr("Open Object Store (.vcobj)"), {},
            tr("Object Store (*.vcobj);;All files (*)"));
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
        state_->detachObjectStore();
    });
    connect(state_, &CState::objectStoreChanged, this, [this](const vc::objects::StoreSessionEvent& event) {
        projectChanged(event.kind == vc::objects::StoreSessionEvent::Kind::Committed);
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
    uuid_.clear();
    pins_.clear(); coordinateGroups_.clear(); coordinateFactors_.clear(); catalogCursors_.clear();
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
        s->dirty = true;
    }
    catalogPending_ = false;
    pendingIndex_.reset();
    list_->clear();
    along_->setEnabled(false);
    status_->setText(tr("No Object Store Fibers open."));
    selection_->clear();
    refreshAll();
}
void FiberCollectionController::projectChanged(bool preserveSelection)
{
    if (!preserveSelection) clear();
    else ++revision_;
    const auto connection = state_->objectStoreSession();
    const auto attachment = connection->attachment();
    const auto volume = state_->currentVolume();
    if (!attachment || !annotations_) return;
    uuid_ = attachment->storeUuid;
    coordinateGroups_.clear();
    coordinateFactors_.clear();
    catalogCursors_.clear();
    enabled_ = false;
    invalidate();
    status_->setText(tr("Reading fiber catalog metadata…"));
    detach_->setEnabled(true);
    nativeToViewer_ = 1;
    nativeVoxelMm_ = volume ? volume->voxelSize()/1000. : 1.;
    const auto rev = revision_;
    background<std::vector<vc::objects::ObjectCoordinateGroup>>(this,
        [connection, attachment=*attachment] {
            std::vector<vc::objects::ObjectCoordinateGroup> groups;
            connection->read(attachment, [&](const auto& db, const auto& txn, const auto& cancelled) {
                groups = db.objectCoordinateGroups(txn, "fiber", &cancelled);
            });
            return groups;
        },
        [this, rev, preserveSelection](Outcome<std::vector<vc::objects::ObjectCoordinateGroup>> result) {
            if (rev != revision_) return;
            if (!result.error.isEmpty()) { status_->setText(result.error); return; }
            coordinateGroups_ = std::move(result.value);
            totalFibers_ = 0;
            for (const auto& group : coordinateGroups_) totalFibers_ += group.count;
            enabled_ = true;
            listPage(true);
            invalidate();
            if (preserveSelection && selected_ > 0) selectFiber(selected_, false, false);
        });
}
void FiberCollectionController::volumeChanged()
{
    projectChanged();
}
void FiberCollectionController::openCollection(const QString& path)
{
    try {
        state_->attachObjectStore(path,qEnvironmentVariable("USER",qEnvironmentVariable("USERNAME","vc3d")));
        dock_->show();
    } catch(const std::exception& e) { status_->setText(QString::fromUtf8(e.what())); }

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
    if (!enabled_) return;
    if (reset) { catalogCursors_.assign(coordinateGroups_.size(), std::nullopt); list_->clear(); }
    if (catalogBusy_) { catalogPending_ = true; return; }
    const auto rev = revision_;
    const double minMm = minLength_->value(), min = minMm / nativeVoxelMm_;
    const auto family = familyFilter();
    const auto volume = state_->currentVolume();
    const auto volumeId = volume ? volume->id() : std::string{};
    catalogBusy_ = true;
    next_->setEnabled(false);
    background<std::vector<CatalogEntry>>(this,
        [connection=state_->objectStoreSession(), attachment=*state_->objectStoreSession()->attachment(),
         groups=coordinateGroups_, cursors=catalogCursors_, volumeId, min, family] {
            std::vector<CatalogEntry> rows;
            if (family == vc::fibers::FamilyFilter::None) return rows;
            connection->read(attachment, [&](const auto& db, const auto& txn, const auto& cancelled) {
                for (size_t g = 0; g < groups.size(); ++g) {
                    if (cancelled.load()) return;
                    const auto factor = groupScale(groups[g], volumeId);
                    if (!factor && min > 0) continue;
                    vc::objects::ObjectSelection selection;
                    selection.kind = "fiber"; selection.coordinates = groups[g].coordinates;
                    selection.limit = 200; selection.orderBy = factor ? "length_desc" : "uuid";
                    if (factor) selection.minLength = min / *factor;
                    if (family != vc::fibers::FamilyFilter::All)
                        selection.family = family == vc::fibers::FamilyFilter::Horizontal ? "H" : "V";
                    if (cursors[g]) {
                        selection.afterUuid = cursors[g]->second;
                        if (factor) selection.afterLength = cursors[g]->first;
                    }
                    for (const auto& object : db.selectObjects(txn, selection, std::vector<std::string>{"name", "fiber"})) {
                        const auto revision = db.currentObjectRevision(txn, object.uuid);
                        if (!revision) continue;
                        CatalogEntry row; row.pin = pinRevision(attachment, *revision); row.group = g;
                        row.factor = factor.value_or(-1); row.summary.name = object.name;
                        if (object.fiber) {
                            row.summary.family = object.fiber->family;
                            row.sourceLength = object.fiber->length;
                            row.summary.pointCount = object.fiber->linePointCount ? object.fiber->linePointCount : object.fiber->controlPointCount;
                        }
                        row.summary.length = factor ? row.sourceLength * *factor : -1;
                        rows.push_back(std::move(row));
                    }
                }
            });
            std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
                return a.summary.length != b.summary.length ? a.summary.length > b.summary.length : a.pin.object.objectUuid < b.pin.object.objectUuid;
            });
            if (rows.size() > 200) rows.resize(200);
            return rows;
        },
        [this, rev, minMm, family](Outcome<std::vector<CatalogEntry>> result) {
            catalogBusy_ = false;
            if (catalogPending_) { catalogPending_ = false; listPage(false); return; }
            if (rev != revision_ || minMm != minLength_->value() || family != familyFilter()) return;
            if (!result.error.isEmpty()) { status_->setText(result.error); return; }
            list_->clear();
            for (auto& entry : result.value) {
                const auto id = annotations_->registerStoreFiber(entry.pin);
                pins_.insert_or_assign(id, entry.pin);
                if (entry.factor > 0) coordinateFactors_[id] = entry.factor;
                entry.summary.id = id;
                const auto& row = entry.summary;
                auto* item = new QListWidgetItem(row.length < 0
                    ? tr("Unbound · #%1 · %2").arg(id).arg(QString::fromStdString(row.family))
                    : tr("%1 mm · #%2 · %3").arg(row.length * nativeVoxelMm_, 0, 'f', 2).arg(id).arg(QString::fromStdString(row.family)), list_);
                item->setData(Qt::UserRole, qlonglong(id));
                item->setToolTip(tr("%1 points").arg(row.pointCount));
                catalogCursors_[entry.group] = std::pair{entry.sourceLength, entry.pin.object.objectUuid};
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
    if (highlightListFiber(summary.id)) return;
    auto* item = new QListWidgetItem(tr("Selected · #%1 · %2").arg(summary.id).arg(QString::fromStdString(summary.family)));
    item->setData(Qt::UserRole, qlonglong(summary.id));
    list_->insertItem(0, item);
    highlightListFiber(summary.id);
}
void FiberCollectionController::selectFiber(int64_t id, bool focusView, bool openAnnotation)
{
    if (!enabled_ || id <= 0 || !pins_.contains(id))
        return;
    const std::optional<int> previousIndex = selected_ == id && !focusView ? std::optional(along_->value()) : std::nullopt;
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
    const auto volume = state_->currentVolume();
    const auto volumeId = volume ? volume->id() : std::string{};
    background<CatalogEntry>(
        this,
        [connection=state_->objectStoreSession(),pin=pins_.at(id),id,volumeId]() {
            CatalogEntry entry; entry.summary.id=id;
            connection->read(pin.attachment,[&](const auto& db,const auto& txn,const auto&) {
                const auto revision=db.currentObjectRevision(txn,pin.object.objectUuid);
                if (!revision || revision->record.retired || !revision->record.fiber)
                    throw std::runtime_error("Fiber is no longer available");
                entry.pin=pinRevision(pin.attachment,*revision);
                const auto& object=revision->record;
                auto& row=entry.summary;
                row.name=object.name; row.family=object.fiber->family;
                const auto factor=volumeId.empty() ? std::nullopt : vc::objects::geometryScaleToVolume(Json::parse(object.meta),volumeId);
                entry.factor=factor.value_or(-1);
                row.length=factor ? object.fiber->length * *factor : -1;
                row.pointCount=object.fiber->linePointCount ? object.fiber->linePointCount : object.fiber->controlPointCount;
            });
            return entry;
        },
        [this, token, rev, focusView, openAnnotation, previousIndex](const Outcome<CatalogEntry>& result) {
            if (rev != revision_ || token != selectionRevision_) return;
            if (!result.error.isEmpty()) {
                selection_->setText(result.error);
                if (openAnnotation) annotationStatus_->setText(result.error);
                return;
            }
            const auto& s = result.value.summary;
            annotations_->registerStoreFiber(result.value.pin);
            pins_.insert_or_assign(s.id,result.value.pin);
            if (result.value.factor > 0) coordinateFactors_[s.id]=result.value.factor;
            else coordinateFactors_.erase(s.id);
            revealListFiber(s);
            selection_->setToolTip(QString::fromStdString(s.name));
            selection_->setText(s.length<0 ? tr("#%1 · %2 points · coordinate binding required").arg(s.id).arg(s.pointCount) :
                tr("#%1 · %2 points · %3 mm\nFollow the complete stored fiber with the slider.").arg(s.id).arg(s.pointCount).arg(s.length * nativeVoxelMm_, 0, 'f', 2));
            const QSignalBlocker block(along_);
            along_->setRange(0, int(std::min<int64_t>(s.pointCount - 1, std::numeric_limits<int>::max())));
            const int centerIndex = previousIndex ? std::clamp(*previousIndex, 0, along_->maximum()) : along_->maximum() / 2;
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
    if (!selected_ || !enabled_ || !pins_.contains(selected_)) return;
    const auto factor = coordinateFactors_.contains(selected_) ? std::optional(coordinateFactors_.at(selected_))
        : annotations_->fiberCoordinateScale(selected_,true);
    if(!factor) return;
    coordinateFactors_[selected_]=*factor;
    ++navigationRevision_;
    pendingIndex_ = index;
    if (navigationBusy_)
        return;
    navigationBusy_ = true;
    pendingIndex_.reset();
    const auto id = selected_;
    const auto rev = revision_;
    const auto token = navigationRevision_;
    background<Point>(
        this,
        [connection=state_->objectStoreSession(),pin=pins_.at(id),index]() {
            Point point{};
            connection->read(pin.attachment,[&](const auto& db,const auto& txn,const auto&) {
                if(!pin.geometryRevision) throw std::runtime_error("Fiber has no geometry");
                const auto geometry=db.geometryRevision(txn,*pin.geometryRevision);
                const auto lines=std::find_if(geometry->arrays.begin(),geometry->arrays.end(),[](const auto& a){return a.name=="line_points";});
                const auto name=lines!=geometry->arrays.end() && lines->shape[0]>0 ? "line_points" : "control_points";
                const auto bytes=db.readGeometryArray(txn,*pin.geometryRevision,name,int64_t(index)*24,24);
                for(size_t axis=0;axis<3;++axis) { uint64_t bits=0;for(int byte=0;byte<8;++byte) bits|=uint64_t(bytes[axis*8+byte])<<(8*byte);point[axis]=std::bit_cast<double>(bits); }
            });return point;
        },
        [this, rev, token, factor=*factor](const Outcome<Point>& result) {
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
            manager_->centerFocusAt(cvpoint(result.value, factor), {0, 0, 1}, "fiber-collection");
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
    const double min = minLength_->value() / nativeVoxelMm_;
    const auto family = familyFilter();
    const auto maximum = maxDisplayed_->value();
    const auto selected = selected_;
    v.pendingBounds = s.box;
    background<RegionResult>(
        this,
        [connection=state_->objectStoreSession(),attachment=*state_->objectStoreSession()->attachment(),
         groups=coordinateGroups_, volume=state_->currentVolume() ? state_->currentVolume()->id() : std::string{},
         selectedUuid=pins_.contains(selected) ? pins_.at(selected).object.objectUuid : std::string{}, s,min,cancel,maximum,family]() {
            RegionResult result; result.coverage=s.box;
            if (family == vc::fibers::FamilyFilter::None) return result;
            struct Visible { vc::objects::VisibleRun run; double factor; };
            std::vector<Visible> visible;
            connection->read(attachment,[&](const auto& db,const auto& txn,const auto& cancelled) {
                for (const auto& group : groups) {
                    if (cancel->load() || cancelled.load()) return;
                    const auto scale = groupScale(group, volume);
                    if (!scale) continue;
                    const auto factor = *scale;
                    vc::objects::RegionSlice slice; slice.origin=s.origin; slice.u=s.u; slice.v=s.v; slice.normal=s.n;
                    for (auto& coordinate : slice.origin) coordinate /= factor;
                    slice.localBounds={s.local.lower[0]/factor,s.local.lower[1]/factor,s.local.lower[2]/factor,s.local.upper[0]/factor,s.local.upper[1]/factor,s.local.upper[2]/factor};
                    vc::objects::RegionOptions options;
                    options.kind="fiber"; options.maximumObjects=maximum; options.minLength=min/factor; options.coordinates=group.coordinates;
                    if (!selectedUuid.empty()) options.selected=selectedUuid;
                    if (result.page.decodedBytes >= options.outputByteBudget) { result.page.complete=false; break; }
                    options.outputByteBudget-=result.page.decodedBytes;
                    if (family != vc::fibers::FamilyFilter::All) options.family=family==vc::fibers::FamilyFilter::Horizontal ? "H" : "V";
                    auto region=db.geometryRegion(txn,slice,options,[&](const auto&) { return cancel->load() || cancelled.load(); });
                    result.page.complete = result.page.complete && region.complete();
                    result.page.decodedBytes += region.stats.outputBytes;
                    for (auto& run : region.runs) visible.push_back({std::move(run),factor});
                }
                // Each group was bounded before geometry decoding. Merge the
                // groups in the same stable UUID order, with selected priority.
                std::stable_sort(visible.begin(),visible.end(),[&](const auto& a,const auto& b) {
                    if ((a.run.objectUuid==selectedUuid)!=(b.run.objectUuid==selectedUuid)) return a.run.objectUuid==selectedUuid;
                    return a.run.objectUuid < b.run.objectUuid;
                });
                for (auto& item : visible) {
                    const auto& uuid = item.run.objectUuid;
                    if (!result.fibers.contains(uuid)) {
                        if (maximum > 0 && result.fibers.size() >= size_t(maximum)) { result.page.complete=false; break; }
                        const auto revision = db.currentObjectRevision(txn,uuid);
                        if (!revision) continue;
                        result.fibers.emplace(uuid,std::pair{pinRevision(attachment,*revision),item.factor});
                    }
                    for (auto& point : item.run.points) for (auto& coordinate : point) coordinate *= item.factor;
                    result.blockUuids.push_back(uuid);
                    result.page.blocks.push_back({0,0,item.run.firstSegment,item.run.family,std::move(item.run.points)});
                }
            });
            return result;
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
                std::unordered_map<std::string, uint64_t> ids;
                for (const auto& [uuid, entry] : result.value.fibers) {
                    const auto id = annotations_->registerStoreFiber(entry.first);
                    ids.emplace(uuid, id);
                    pins_.insert_or_assign(id, entry.first);
                    coordinateFactors_[id] = entry.second;
                }
                for (size_t i = 0; i < result.value.page.blocks.size(); ++i)
                    result.value.page.blocks[i].fiberId = ids.at(result.value.blockUuids[i]);
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
    if(annotationBusy_ || !pendingAnnotation_ || !enabled_) return;
    const auto id=std::exchange(pendingAnnotation_,0);
    if(!pins_.contains(id)) return;
    annotationBusy_=true;
    const auto finish=qScopeGuard([this] { finishAnnotationRequest(); });
    QString error;
    auto* dialog=annotations_->openCollectionFiber(id,&error);
    if(dialog) watchAnnotationRender(dialog,id,annotationRevision_);
    else annotationStatus_->setText(error);

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
