#include "WrapAnnotationWidget.hpp"

#include <algorithm>

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QTreeView>
#include <QVBoxLayout>
#include <QWidget>

namespace {
enum SameWrapAnnotationColumn {
    kNameColumn = 0,
    kCountColumn = 1,
    kDirectionColumn = 2,
    kPositionColumn = 3
};

bool isSameWrapCollection(const VCCollection::Collection& collection)
{
    return collection.name.rfind("same_wrap", 0) == 0;
}

QString sameWrapDirectionText(const VCCollection::Collection& collection)
{
    const auto it = collection.tags.find("same_wrap_direction");
    if (it == collection.tags.end()) {
        return {};
    }
    return QString::fromStdString(it->second);
}

QString pointPositionText(const cv::Vec3f& point)
{
    return QString("{%1, %2, %3}").arg(point[0]).arg(point[1]).arg(point[2]);
}

QStandardItem* readOnlyItem(const QString& text = {})
{
    QStandardItem* item = new QStandardItem(text);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    return item;
}
} // namespace

WrapAnnotationWidget::WrapAnnotationWidget(VCCollection* collection, QWidget* parent)
    : QDockWidget("Wrap Annotation", parent)
    , _pointCollection(collection)
{
    setupUi();

    if (_pointCollection) {
        connect(_pointCollection, &VCCollection::collectionsAdded,
                this, &WrapAnnotationWidget::refreshSameWrapTree);
        connect(_pointCollection, &VCCollection::collectionChanged,
                this, &WrapAnnotationWidget::refreshSameWrapTree);
        connect(_pointCollection, &VCCollection::collectionRemoved,
                this, &WrapAnnotationWidget::refreshSameWrapTree);
        connect(_pointCollection, &VCCollection::pointAdded,
                this, &WrapAnnotationWidget::refreshSameWrapTree);
        connect(_pointCollection, &VCCollection::pointChanged,
                this, &WrapAnnotationWidget::refreshSameWrapTree);
        connect(_pointCollection, &VCCollection::pointRemoved,
                this, &WrapAnnotationWidget::refreshSameWrapTree);
    }

    refreshSameWrapTree();
}

void WrapAnnotationWidget::setupUi()
{
    auto* mainWidget = new QWidget(this);
    auto* layout = new QVBoxLayout(mainWidget);

    auto* sameWrapGroup = new QGroupBox("Same-wrap Annotation", mainWidget);
    auto* sameWrapLayout = new QVBoxLayout(sameWrapGroup);

    _chkSameWrapAnnotation = new QCheckBox("Same-wrap annotation mode", sameWrapGroup);
    _chkSameWrapAnnotation->setToolTip("Shift-click or Shift-drag in an active volume viewer to preview same-wrap annotation points. Shift+E commits; Ctrl+Z clears the preview or undoes the last committed collection.");
    sameWrapLayout->addWidget(_chkSameWrapAnnotation);

    _chkSameWrapMerge = new QCheckBox("Manual same-wrap merge", sameWrapGroup);
    _chkSameWrapMerge->setToolTip("Shift-right-click a point in one same-wrap collection, then Shift-right-click a point in another same-wrap collection to merge them.");
    sameWrapLayout->addWidget(_chkSameWrapMerge);

    auto* sameWrapPathTypeLayout = new QHBoxLayout();
    sameWrapPathTypeLayout->addWidget(new QLabel("Path type:"));
    _sameWrapPathTypeCombo = new QComboBox(sameWrapGroup);
    _sameWrapPathTypeCombo->addItem("Connected components", 0);
    _sameWrapPathTypeCombo->addItem("Shortest path", 1);
    _sameWrapPathTypeCombo->addItem("Manual", 2);
    _sameWrapPathTypeCombo->setToolTip("Choose whether Shift input selects a skeleton component, two shortest-path endpoints, or a manually drawn path.");
    sameWrapPathTypeLayout->addWidget(_sameWrapPathTypeCombo);
    sameWrapPathTypeLayout->addStretch();
    sameWrapLayout->addLayout(sameWrapPathTypeLayout);

    auto* sameWrapFilterLayout = new QHBoxLayout();
    sameWrapFilterLayout->addWidget(new QLabel("Filter:"));
    _sameWrapFilterTypeCombo = new QComboBox(sameWrapGroup);
    _sameWrapFilterTypeCombo->addItem("None", 0);
    _sameWrapFilterTypeCombo->addItem("Median", 1);
    _sameWrapFilterTypeCombo->addItem("Gaussian", 2);
    _sameWrapFilterTypeCombo->setToolTip("Optionally filter the source image before thresholding and skeleton tracing.");
    sameWrapFilterLayout->addWidget(_sameWrapFilterTypeCombo);
    sameWrapFilterLayout->addWidget(new QLabel("Kernel:"));
    _sameWrapFilterKernelSpinbox = new QSpinBox(sameWrapGroup);
    _sameWrapFilterKernelSpinbox->setRange(3, 99);
    _sameWrapFilterKernelSpinbox->setSingleStep(2);
    _sameWrapFilterKernelSpinbox->setValue(3);
    _sameWrapFilterKernelSpinbox->setSuffix(" px");
    _sameWrapFilterKernelSpinbox->setMaximumWidth(80);
    _sameWrapFilterKernelSpinbox->setEnabled(false);
    _sameWrapFilterKernelSpinbox->setToolTip("Odd blur kernel size applied before connected components or shortest-path tracing.");
    sameWrapFilterLayout->addWidget(_sameWrapFilterKernelSpinbox);
    sameWrapFilterLayout->addStretch();
    sameWrapLayout->addLayout(sameWrapFilterLayout);

    auto* sameWrapSpacingLayout = new QHBoxLayout();
    sameWrapSpacingLayout->addWidget(new QLabel("Spacing:"));
    _sameWrapSpacingSpinbox = new QDoubleSpinBox(sameWrapGroup);
    _sameWrapSpacingSpinbox->setRange(1.0, 1000.0);
    _sameWrapSpacingSpinbox->setDecimals(1);
    _sameWrapSpacingSpinbox->setSingleStep(1.0);
    _sameWrapSpacingSpinbox->setValue(20.0);
    _sameWrapSpacingSpinbox->setSuffix(" vx");
    _sameWrapSpacingSpinbox->setMaximumWidth(90);
    _sameWrapSpacingSpinbox->setToolTip("Distance between generated same-wrap annotation points in surface voxels.");
    sameWrapSpacingLayout->addWidget(_sameWrapSpacingSpinbox);
    sameWrapSpacingLayout->addWidget(new QLabel("Merge tol.:"));
    _sameWrapMergeToleranceSpinbox = new QDoubleSpinBox(sameWrapGroup);
    _sameWrapMergeToleranceSpinbox->setRange(0.0, 1000.0);
    _sameWrapMergeToleranceSpinbox->setDecimals(2);
    _sameWrapMergeToleranceSpinbox->setSingleStep(0.25);
    _sameWrapMergeToleranceSpinbox->setValue(1.0);
    _sameWrapMergeToleranceSpinbox->setSuffix(" vx");
    _sameWrapMergeToleranceSpinbox->setMaximumWidth(90);
    _sameWrapMergeToleranceSpinbox->setToolTip("Reserved for same-wrap merge tools.");
    sameWrapSpacingLayout->addWidget(_sameWrapMergeToleranceSpinbox);
    sameWrapSpacingLayout->addStretch();
    sameWrapLayout->addLayout(sameWrapSpacingLayout);

    auto* sameWrapPolylineLayout = new QHBoxLayout();
    sameWrapPolylineLayout->addWidget(new QLabel("Polyline opacity:"));
    _sameWrapPolylineOpacitySpinbox = new QDoubleSpinBox(sameWrapGroup);
    _sameWrapPolylineOpacitySpinbox->setRange(0.0, 1.0);
    _sameWrapPolylineOpacitySpinbox->setDecimals(2);
    _sameWrapPolylineOpacitySpinbox->setSingleStep(0.05);
    _sameWrapPolylineOpacitySpinbox->setValue(0.75);
    _sameWrapPolylineOpacitySpinbox->setMaximumWidth(90);
    _sameWrapPolylineOpacitySpinbox->setToolTip("Opacity of same-wrap point collection guide polylines.");
    sameWrapPolylineLayout->addWidget(_sameWrapPolylineOpacitySpinbox);
    sameWrapPolylineLayout->addStretch();
    sameWrapLayout->addLayout(sameWrapPolylineLayout);

    _clearSameWrapAnnotationButton = new QPushButton("Clear Same-wrap Preview", sameWrapGroup);
    _clearSameWrapAnnotationButton->setToolTip("Clear the current same-wrap annotation preview without committing it.");
    sameWrapLayout->addWidget(_clearSameWrapAnnotationButton);

    sameWrapLayout->addWidget(new QLabel("Current annotations:", sameWrapGroup));
    _sameWrapTreeView = new QTreeView(sameWrapGroup);
    _sameWrapModel = new QStandardItemModel(this);
    _sameWrapTreeView->setModel(_sameWrapModel);
    _sameWrapTreeView->setSelectionBehavior(QAbstractItemView::SelectRows);
    _sameWrapTreeView->setSelectionMode(QAbstractItemView::SingleSelection);
    _sameWrapTreeView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _sameWrapTreeView->setToolTip("Committed same-wrap annotation collections.");
    sameWrapLayout->addWidget(_sameWrapTreeView);

    layout->addWidget(sameWrapGroup);

    _relWindingAnnotationButton = new QPushButton("Start Rel Winding Annotation", mainWidget);
    _relWindingAnnotationButton->setCheckable(true);
    _relWindingAnnotationButton->setToolTip("Draw a line to auto-create a new collection with relative winding labels. Hold Shift for decreasing order.");
    layout->addWidget(_relWindingAnnotationButton);

    connect(_chkSameWrapAnnotation, &QCheckBox::toggled, this, &WrapAnnotationWidget::sameWrapAnnotationToggled);
    connect(_chkSameWrapMerge, &QCheckBox::toggled, this, &WrapAnnotationWidget::sameWrapAnnotationMergeToggled);
    connect(_sameWrapPathTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        emit sameWrapAnnotationPathTypeChanged(sameWrapAnnotationPathType());
    });
    connect(_sameWrapFilterTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        const bool filterEnabled = sameWrapAnnotationFilterType() != 0;
        _sameWrapFilterKernelSpinbox->setEnabled(filterEnabled);
        emit sameWrapAnnotationFilterTypeChanged(sameWrapAnnotationFilterType());
    });
    connect(_sameWrapFilterKernelSpinbox, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if ((value % 2) == 0) {
            const QSignalBlocker blocker(_sameWrapFilterKernelSpinbox);
            _sameWrapFilterKernelSpinbox->setValue(value + 1);
        }
        emit sameWrapAnnotationFilterKernelSizeChanged(sameWrapAnnotationFilterKernelSize());
    });
    connect(_sameWrapSpacingSpinbox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &WrapAnnotationWidget::sameWrapAnnotationSpacingChanged);
    connect(_sameWrapMergeToleranceSpinbox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &WrapAnnotationWidget::sameWrapAnnotationMergeToleranceChanged);
    connect(_sameWrapPolylineOpacitySpinbox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &WrapAnnotationWidget::sameWrapAnnotationPolylineOpacityChanged);
    connect(_clearSameWrapAnnotationButton, &QPushButton::clicked,
            this, &WrapAnnotationWidget::sameWrapAnnotationClearRequested);
    connect(_relWindingAnnotationButton, &QPushButton::toggled, this, [this](bool checked) {
        _relWindingAnnotationButton->setText(checked ? "Stop Rel Winding Annotation" : "Start Rel Winding Annotation");
        emit relWindingAnnotationToggled(checked);
    });

    layout->addStretch();
    setWidget(mainWidget);
}

void WrapAnnotationWidget::setRelWindingAnnotationChecked(bool checked)
{
    if (!_relWindingAnnotationButton || _relWindingAnnotationButton->isChecked() == checked) {
        return;
    }

    const QSignalBlocker blocker(_relWindingAnnotationButton);
    _relWindingAnnotationButton->setChecked(checked);
    _relWindingAnnotationButton->setText(checked ? "Stop Rel Winding Annotation" : "Start Rel Winding Annotation");
}

bool WrapAnnotationWidget::sameWrapAnnotationEnabled() const
{
    return _chkSameWrapAnnotation && _chkSameWrapAnnotation->isChecked();
}

double WrapAnnotationWidget::sameWrapAnnotationSpacing() const
{
    return _sameWrapSpacingSpinbox ? _sameWrapSpacingSpinbox->value() : 20.0;
}

double WrapAnnotationWidget::sameWrapAnnotationMergeTolerance() const
{
    return _sameWrapMergeToleranceSpinbox ? _sameWrapMergeToleranceSpinbox->value() : 1.0;
}

double WrapAnnotationWidget::sameWrapAnnotationPolylineOpacity() const
{
    return _sameWrapPolylineOpacitySpinbox ? _sameWrapPolylineOpacitySpinbox->value() : 0.75;
}

bool WrapAnnotationWidget::sameWrapAnnotationMergeEnabled() const
{
    return _chkSameWrapMerge && _chkSameWrapMerge->isChecked();
}

int WrapAnnotationWidget::sameWrapAnnotationPathType() const
{
    return _sameWrapPathTypeCombo ? _sameWrapPathTypeCombo->currentData().toInt() : 0;
}

int WrapAnnotationWidget::sameWrapAnnotationFilterType() const
{
    return _sameWrapFilterTypeCombo ? _sameWrapFilterTypeCombo->currentData().toInt() : 0;
}

int WrapAnnotationWidget::sameWrapAnnotationFilterKernelSize() const
{
    const int kernelSize = _sameWrapFilterKernelSpinbox ? _sameWrapFilterKernelSpinbox->value() : 3;
    return std::max(3, kernelSize | 1);
}

void WrapAnnotationWidget::refreshSameWrapTree()
{
    if (!_sameWrapModel) {
        return;
    }

    _sameWrapModel->blockSignals(true);
    _sameWrapModel->clear();
    _sameWrapModel->setHorizontalHeaderLabels({"Name", "Points", "Direction", "Position"});

    if (_pointCollection) {
        std::vector<VCCollection::Collection> collections;
        for (const auto& [_, collection] : _pointCollection->getAllCollections()) {
            if (isSameWrapCollection(collection)) {
                collections.push_back(collection);
            }
        }

        std::sort(collections.begin(), collections.end(),
                  [](const VCCollection::Collection& a, const VCCollection::Collection& b) {
                      return a.name < b.name;
                  });

        for (const VCCollection::Collection& collection : collections) {
            QStandardItem* nameItem = readOnlyItem(QString::fromStdString(collection.name));
            const QColor color(collection.color[0] * 255,
                               collection.color[1] * 255,
                               collection.color[2] * 255);
            nameItem->setData(QBrush(color), Qt::DecorationRole);
            nameItem->setData(QVariant::fromValue(collection.id));

            QStandardItem* countItem = readOnlyItem(QString::number(collection.points.size()));
            QStandardItem* directionItem = readOnlyItem(sameWrapDirectionText(collection));
            QStandardItem* positionItem = readOnlyItem();
            _sameWrapModel->appendRow({nameItem, countItem, directionItem, positionItem});

            std::vector<ColPoint> points;
            points.reserve(collection.points.size());
            for (const auto& [_, point] : collection.points) {
                points.push_back(point);
            }
            std::sort(points.begin(), points.end(),
                      [](const ColPoint& a, const ColPoint& b) {
                          return a.id < b.id;
                      });

            for (const ColPoint& point : points) {
                QStandardItem* pointItem = readOnlyItem(QString::number(point.id));
                pointItem->setData(QVariant::fromValue(point.id));
                QStandardItem* emptyCountItem = readOnlyItem();
                QStandardItem* pointDirectionItem = readOnlyItem(sameWrapDirectionText(collection));
                QStandardItem* pointPositionItem = readOnlyItem(pointPositionText(point.p));
                nameItem->appendRow({pointItem, emptyCountItem, pointDirectionItem, pointPositionItem});
            }
        }
    }

    _sameWrapModel->blockSignals(false);

    if (_sameWrapTreeView) {
        _sameWrapTreeView->expandAll();
        _sameWrapTreeView->resizeColumnToContents(kNameColumn);
        _sameWrapTreeView->resizeColumnToContents(kCountColumn);
        _sameWrapTreeView->resizeColumnToContents(kDirectionColumn);
    }
}
