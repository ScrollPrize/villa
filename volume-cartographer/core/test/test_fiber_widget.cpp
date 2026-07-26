#include "CFiberWidget.hpp"
#include "FiberNameDisplay.hpp"

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QHeaderView>
#include <QLineEdit>
#include <QMetaObject>
#include <QPushButton>
#include <QScrollBar>
#include <QStandardItemModel>
#include <QTreeView>
#include <QThreadPool>

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

void ensureApplication(int& argc, char** argv, std::unique_ptr<QApplication>& app)
{
    if (!QApplication::instance()) {
        app = std::make_unique<QApplication>(argc, argv);
    }
}

void require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}

bool sameIds(const std::vector<uint64_t>& actual, const std::vector<uint64_t>& expected)
{
    return actual == expected;
}

CFiberWidget::FiberEntry makeFiber(uint64_t id,
                                   const std::string& fileName,
                                   int controlPoints,
                                   int linePoints,
                                   double length,
                                   std::initializer_list<std::string> tags = {})
{
    CFiberWidget::FiberEntry fiber;
    fiber.id = id;
    fiber.fileName = fileName;
    fiber.controlPointCount = controlPoints;
    fiber.linePointCount = linePoints;
    fiber.lengthVx = length;
    fiber.tags = tags;
    return fiber;
}

CFiberWidget::FiberEntry::AlignmentMetrics makeMetric(double mean, double max, int samples)
{
    CFiberWidget::FiberEntry::AlignmentMetrics metric;
    metric.available = true;
    metric.meanErrorDegrees = mean;
    metric.maxErrorDegrees = max;
    metric.sampleCount = samples;
    return metric;
}

} // namespace

int main(int argc, char** argv)
{
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    std::unique_ptr<QApplication> app;
    ensureApplication(argc, argv, app);
    QThreadPool::globalInstance()->setMaxThreadCount(1);

    CFiberWidget widget;
    auto first = makeFiber(1, "aa_20260605T184821587_000001.json", 2, 20, 12.0, {"source-a"});
    first.spans.push_back({0, 0, 1, 2, 20, 12.0, makeMetric(8.0, 12.0, 38)});
    auto second = makeFiber(2, "kb_20260605T184821587_000002.json", 3, 30, 24.0, {"review"});
    second.alignment = makeMetric(25.0, 50.0, 58);
    second.linkedFiberCount = 2;
    second.pendingLinkCount = 1;
    second.spans.push_back({0, 0, 1, 2, 14, 11.0, makeMetric(7.0, 20.0, 26)});
    second.spans.push_back({1, 1, 2, 2, 17, 13.0, makeMetric(31.0, 52.0, 32)});
    auto third = makeFiber(3, "zz_20260605T184821587_000003.json", 4, 40, 36.0);

    widget.setFibers({first, second, third});
    widget.setKnownTags({"review", "source-a", "todo"});

    auto* treeView = widget.findChild<QTreeView*>(QStringLiteral("fiberTreeView"));
    require(treeView != nullptr, "Fiber tree view was not found");
    require(treeView->model() != nullptr, "Fiber tree view model was not found");
    const QFontMetrics metrics(treeView->font());
    const QString adaptedName = vc3d::adaptFiberNameToWidth(
        QStringLiteral("kb_20260605T184821587_000002"),
        metrics,
        metrics.horizontalAdvance(QStringLiteral("kb_..._000002")));
    require(adaptedName.endsWith(QStringLiteral("_000002")),
            "Fiber name elision should preserve the sequence suffix");
    require(treeView->model()->columnCount() == 10, "Fiber tree should expose ten columns");
    require(treeView->header()->sectionResizeMode(0) == QHeaderView::Interactive,
            "Fiber tree header should allow changing column widths");
    require(treeView->model()->index(1, 0).data().toString() ==
                QStringLiteral("kb_20260605T184821587_000002"),
            "Fiber tree row should keep the full JSON stem without the internal ID");
    require(!treeView->model()->index(1, 0).data().toString().contains(QStringLiteral(".json")),
            "Fiber tree row should not show the JSON extension");
    require(!treeView->model()->index(1, 0).data().toString().startsWith(QStringLiteral("2 ")),
            "Fiber tree row should not show the runtime fiber ID");
    require(treeView->model()->rowCount(treeView->model()->index(1, 0)) == 2,
            "Fiber tree row did not expose span children");
    require(treeView->model()->index(1, 2).data().toString() == QStringLiteral("2"),
            "Fiber tree link column did not show the linked fiber count");
    require(treeView->model()->index(1, 3).data().toString() == QStringLiteral("1"),
            "Fiber tree pending column did not show the pending link count");
    require(treeView->model()->index(0, 3).data().toString().isEmpty(),
            "Fiber tree pending column should be blank when no links are pending");
    require(treeView->model()->index(1, 7).data().toString() == QStringLiteral("review"),
            "Fiber tree tags column did not show fiber tags");
    require(treeView->model()->index(1, 8).data().toString() == QStringLiteral("-"),
            "Metrics should be hidden before Calc metrics is enabled");
    require(widget.orderedFiberIds() == std::vector<uint64_t>({1, 2, 3}),
            "Initial fiber order should match the sorted list order");

    int spanOpenRequests = 0;
    uint64_t spanOpenFiberId = 0;
    int spanOpenFirstControl = -1;
    int spanOpenSecondControl = -1;
    QObject::connect(&widget,
                     &CFiberWidget::fiberSpanOpenRequested,
                     &widget,
                     [&](uint64_t fiberId, int firstControlIndex, int secondControlIndex) {
                         ++spanOpenRequests;
                         spanOpenFiberId = fiberId;
                         spanOpenFirstControl = firstControlIndex;
                         spanOpenSecondControl = secondControlIndex;
                     });
    const QModelIndex secondParentBeforeMetrics = treeView->model()->index(1, 0);
    const QModelIndex secondSpanIndex = treeView->model()->index(1, 0, secondParentBeforeMetrics);
    QMetaObject::invokeMethod(treeView,
                              "doubleClicked",
                              Q_ARG(QModelIndex, secondSpanIndex));
    require(spanOpenRequests == 1, "Double-clicking a span did not emit one span-open request");
    require(spanOpenFiberId == 2, "Span double-click emitted the wrong fiber ID");
    require(spanOpenFirstControl == 1 && spanOpenSecondControl == 2,
            "Span double-click emitted the wrong control point range");

    int metricRequests = 0;
    std::vector<uint64_t> metricRequestOrder;
    QObject::connect(&widget,
                     &CFiberWidget::metricsCalculationRequested,
                     &widget,
                     [&](std::vector<uint64_t> orderedFiberIds) {
                         ++metricRequests;
                         metricRequestOrder = std::move(orderedFiberIds);
                     });
    auto* calcMetrics = widget.findChild<QCheckBox*>(QStringLiteral("fiberCalcMetricsCheckBox"));
    require(calcMetrics != nullptr, "Calc metrics checkbox was not found");
    auto* showFibers = widget.findChild<QCheckBox*>(QStringLiteral("fiberShowFibersCheckBox"));
    require(showFibers != nullptr, "Show fibers checkbox was not found");
    require(!showFibers->isEnabled(), "Show fibers should start disabled");
    require(!widget.showFibersChecked(), "Show fibers should start unchecked");
    auto* fiberViewDistance =
        widget.findChild<QDoubleSpinBox*>(QStringLiteral("fiberViewDistanceSpinBox"));
    require(fiberViewDistance != nullptr, "Fiber view-distance spinbox was not found");
    require(widget.fiberViewDistance() == 10.0,
            "Fiber view distance should default to 10 voxels");
    int fiberViewDistanceRequests = 0;
    double requestedFiberViewDistance = 0.0;
    QObject::connect(&widget,
                     &CFiberWidget::fiberViewDistanceChanged,
                     &widget,
                     [&](double distance) {
                         ++fiberViewDistanceRequests;
                         requestedFiberViewDistance = distance;
                     });
    widget.setFiberViewDistance(25.0);
    require(widget.fiberViewDistance() == 25.0,
            "Programmatic fiber view-distance sync did not update the spinbox");
    require(fiberViewDistanceRequests == 0,
            "Programmatic fiber view-distance sync should not emit a request");
    fiberViewDistance->setValue(30.0);
    require(fiberViewDistanceRequests == 1 && requestedFiberViewDistance == 30.0,
            "User fiber view-distance change did not emit the requested distance");
    int showFiberRequests = 0;
    bool requestedShowFibers = false;
    QObject::connect(&widget,
                     &CFiberWidget::showFibersToggled,
                     &widget,
                     [&](bool checked) {
                         ++showFiberRequests;
                         requestedShowFibers = checked;
                     });
    widget.setShowFibersAvailable(true);
    require(showFibers->isEnabled(), "Loaded fibers should enable Show fibers");
    widget.setShowFibersChecked(true);
    require(widget.showFibersChecked(), "Programmatic Show fibers sync did not update the checkbox");
    require(showFiberRequests == 0,
            "Programmatic Show fibers sync should not emit a visibility request");
    showFibers->setChecked(false);
    require(showFiberRequests == 1 && !requestedShowFibers,
            "User Show fibers toggle did not emit the requested state");
    showFibers->setChecked(true);
    require(showFiberRequests == 2 && requestedShowFibers,
            "User Show fibers enable did not emit the requested state");
    widget.setShowFibersAvailable(false);
    require(!showFibers->isEnabled(), "Unavailable fibers should disable Show fibers");
    require(!widget.showFibersChecked(), "Unavailable fibers should reset Show fibers");
    require(showFiberRequests == 2,
            "Availability reset should not emit a duplicate visibility request");
    widget.setShowFibersAvailable(true);
    auto* model = qobject_cast<QStandardItemModel*>(treeView->model());
    require(model != nullptr, "Fiber tree should use a standard item model");
    QMetaObject::invokeMethod(treeView->header(),
                              "sectionClicked",
                              Q_ARG(int, 4));
    QMetaObject::invokeMethod(treeView->header(),
                              "sectionClicked",
                              Q_ARG(int, 4));
    require(widget.orderedFiberIds() == std::vector<uint64_t>({3, 2, 1}),
            "Fiber order should track the current sorted list order");
    QStandardItem* secondItemBeforeMetrics = model->item(1, 0);
    calcMetrics->setChecked(true);
    require(metricRequests == 1, "Calc metrics checkbox did not emit one request");
    require(metricRequestOrder == std::vector<uint64_t>({3, 2, 1}),
            "Calc metrics request should use the current fiber list order");
    require(model->item(1, 0) == secondItemBeforeMetrics,
            "Toggling Calc metrics should update cells without rebuilding the fiber rows");
    require(treeView->model()->index(1, 8).data().toString() == QStringLiteral("25.0"),
            "Mean alignment error metric was not displayed");
    require(treeView->model()->index(1, 9).data().toString() == QStringLiteral("50.0"),
            "Max alignment error metric was not displayed");
    require(treeView->model()->index(1, 9).data(Qt::BackgroundRole).isValid(),
            "Fiber row with max alignment error > 45 deg was not highlighted");
    const QModelIndex secondParent = treeView->model()->index(1, 0);
    require(treeView->model()->index(1, 9, secondParent).data().toString() == QStringLiteral("52.0"),
            "Span max alignment error metric was not displayed");
    require(treeView->model()->index(1, 9, secondParent).data(Qt::BackgroundRole).isValid(),
            "Span row with max alignment error > 45 deg was not highlighted");
    widget.setAlignmentMetricsPending(true);
    require(model->item(1, 0) == secondItemBeforeMetrics,
            "Marking metrics pending should update cells without rebuilding the fiber rows");
    require(treeView->model()->index(1, 8).data().toString() == QStringLiteral("..."),
            "Pending metric state should be displayed in-place");
    widget.updateAlignmentMetrics(
        2,
        makeMetric(26.0, 51.0, 60),
        {makeMetric(9.0, 21.0, 27), makeMetric(32.0, 53.0, 33)});
    require(model->item(1, 0) == secondItemBeforeMetrics,
            "Live metric update should update cells without rebuilding the fiber rows");
    require(treeView->model()->index(1, 8).data().toString() == QStringLiteral("26.0"),
            "Live fiber mean alignment metric was not displayed");
    require(treeView->model()->index(1, 9).data().toString() == QStringLiteral("51.0"),
            "Live fiber max alignment metric was not displayed");
    require(treeView->model()->index(1, 9, secondParent).data().toString() == QStringLiteral("53.0"),
            "Live span alignment metric was not displayed");
    QMetaObject::invokeMethod(treeView->header(),
                              "sectionClicked",
                              Q_ARG(int, 4));
    QMetaObject::invokeMethod(treeView->header(),
                              "sectionClicked",
                              Q_ARG(int, 4));
    require(treeView->model()->index(0, 0).data().toString() ==
                QStringLiteral("zz_20260605T184821587_000003"),
            "Sorting by length descending should reorder only top-level fibers");
    const QModelIndex sortedSecondParent = treeView->model()->index(1, 0);
    require(treeView->model()->index(0, 0, sortedSecondParent).data().toString().contains(
                QStringLiteral("span 1")),
            "Sorting top-level fibers should preserve child span order");
    widget.setFibers({first, second, third});
    require(metricRequests == 2,
            "Refreshing the fiber list while metrics are checked should request metrics again");
    require(metricRequestOrder == std::vector<uint64_t>({3, 2, 1}),
            "Refresh-time metric request should preserve the current fiber list order");

    auto* deleteButton = widget.findChild<QPushButton*>(QStringLiteral("fiberDeleteButton"));
    require(deleteButton != nullptr, "Fiber delete button was not found");
    auto* importButton = widget.findChild<QPushButton*>(QStringLiteral("fiberImportButton"));
    require(importButton != nullptr, "Fiber import button was not found");
    auto* exportButton = widget.findChild<QPushButton*>(QStringLiteral("fiberExportButton"));
    require(exportButton != nullptr, "Fiber export button was not found");
    require(!deleteButton->isEnabled(), "Delete button should start disabled");
    require(!widget.canDeleteSelection(), "Empty selection should not allow delete");
    require(!widget.canCreateAtlasFromSelection(), "Empty selection should not allow atlas creation");
    require(!widget.canShowFiberSlice(), "Empty selection should not allow fiber slice");
    require(!widget.canRenameFiberFile(), "Empty selection should not allow JSON rename");

    widget.selectFiber(2);
    require(widget.selectedFiberId() == 2, "Single selection did not set selectedFiberId");
    require(sameIds(widget.selectedFiberIds(), {2}), "Single selection IDs are wrong");
    require(deleteButton->isEnabled(), "Delete button should enable for a single selection");
    require(widget.canDeleteSelection(), "Single selection should allow delete");
    require(widget.canCreateAtlasFromSelection(), "Single selection should allow atlas creation");
    require(widget.canShowFiberSlice(), "Single selection should allow fiber slice");
    require(widget.canRenameFiberFile(), "Single selection should allow JSON rename");

    int sliceRequests = 0;
    int renameRequests = 0;
    int tagRequests = 0;
    uint64_t requestedSliceFiberId = 0;
    uint64_t requestedRenameFiberId = 0;
    uint64_t requestedTagFiberId = 0;
    QString requestedTag;
    bool requestedTagEnabled = false;
    QObject::connect(&widget,
                     &CFiberWidget::fiberSliceRequested,
                     &widget,
                     [&](uint64_t fiberId) {
                         ++sliceRequests;
                         requestedSliceFiberId = fiberId;
                     });
    QObject::connect(&widget,
                     &CFiberWidget::renameFiberFileRequested,
                     &widget,
                     [&](uint64_t fiberId) {
                         ++renameRequests;
                         requestedRenameFiberId = fiberId;
                     });
    QObject::connect(&widget,
                     &CFiberWidget::fiberTagChanged,
                     &widget,
                     [&](uint64_t fiberId, QString tag, bool enabled) {
                         ++tagRequests;
                         requestedTagFiberId = fiberId;
                         requestedTag = tag;
                         requestedTagEnabled = enabled;
                     });
    auto* showSliceAction = widget.createShowFiberSliceAction(&widget);
    require(showSliceAction->isEnabled(), "Single selection should enable Show fiber slice action");
    showSliceAction->trigger();
    require(sliceRequests == 1, "Show fiber slice action did not emit one request");
    require(requestedSliceFiberId == 2, "Show fiber slice emitted the wrong fiber ID");
    auto* renameAction = widget.createRenameFiberFileAction(&widget);
    require(renameAction->isEnabled(), "Single selection should enable Rename JSON file action");
    renameAction->trigger();
    require(renameRequests == 1, "Rename JSON file action did not emit one request");
    require(requestedRenameFiberId == 2, "Rename JSON file emitted the wrong fiber ID");

    auto tagCheckboxes = widget.findChildren<QCheckBox*>(QStringLiteral("fiberTagCheckBox"));
    require(tagCheckboxes.size() == 3, "Known tags did not create three checkboxes");
    QCheckBox* reviewCheckbox = nullptr;
    QCheckBox* todoCheckbox = nullptr;
    for (auto* checkbox : tagCheckboxes) {
        if (checkbox->text() == QStringLiteral("review")) {
            reviewCheckbox = checkbox;
        } else if (checkbox->text() == QStringLiteral("todo")) {
            todoCheckbox = checkbox;
        }
    }
    require(reviewCheckbox != nullptr, "Review tag checkbox was not found");
    require(todoCheckbox != nullptr, "Todo tag checkbox was not found");
    require(reviewCheckbox->isChecked(), "Selected fiber tag should be checked");
    require(!todoCheckbox->isChecked(), "Unchecked known tag should not be checked");
    todoCheckbox->setChecked(true);
    QApplication::processEvents();
    require(tagRequests == 1, "Checking a tag did not emit one tag request");
    require(requestedTagFiberId == 2, "Tag check emitted the wrong fiber ID");
    require(requestedTag == QStringLiteral("todo"), "Tag check emitted the wrong tag");
    require(requestedTagEnabled, "Tag check should enable the tag");

    auto* newTagEdit = widget.findChild<QLineEdit*>(QStringLiteral("fiberNewTagEdit"));
    auto* addTagButton = widget.findChild<QPushButton*>(QStringLiteral("fiberAddTagButton"));
    require(newTagEdit != nullptr, "New tag text field was not found");
    require(addTagButton != nullptr, "Add tag button was not found");
    newTagEdit->setText(QStringLiteral("needs-proofread"));
    addTagButton->click();
    QApplication::processEvents();
    require(tagRequests == 2, "Adding a tag did not emit a second tag request");
    require(requestedTagFiberId == 2, "Added tag emitted the wrong fiber ID");
    require(requestedTag == QStringLiteral("needs-proofread"), "Added tag emitted the wrong tag");
    require(requestedTagEnabled, "Added tag should enable the tag");

    widget.selectFibers({1, 3});
    require(widget.selectedFiberId() == 0, "Multi-selection should not expose a single selected fiber");
    require(sameIds(widget.selectedFiberIds(), {1, 3}), "Multi-selection IDs are wrong");
    require(deleteButton->isEnabled(), "Delete button should enable for multi-selection");
    require(widget.canDeleteSelection(), "Multi-selection should allow delete");
    require(!widget.canCreateAtlasFromSelection(), "Multi-selection should gray out atlas creation");
    require(!widget.canShowFiberSlice(), "Multi-selection should gray out fiber slice");
    require(!widget.canRenameFiberFile(), "Multi-selection should gray out JSON rename");
    auto* multiShowSliceAction = widget.createShowFiberSliceAction(&widget);
    require(!multiShowSliceAction->isEnabled(), "Multi-selection should disable Show fiber slice action");
    auto* multiRenameAction = widget.createRenameFiberFileAction(&widget);
    require(!multiRenameAction->isEnabled(), "Multi-selection should disable Rename JSON file action");
    require(!newTagEdit->isEnabled(), "Multi-selection should disable new tag text field");
    require(!addTagButton->isEnabled(), "Multi-selection should disable add tag button");

    CFiberWidget scrollWidget;
    std::vector<CFiberWidget::FiberEntry> manyFibers;
    manyFibers.reserve(80);
    for (uint64_t id = 1; id <= 80; ++id) {
        auto fiber = makeFiber(id,
                               "fiber_" + std::to_string(id) + ".json",
                               2,
                               20,
                               static_cast<double>(id));
        if (id % 2 == 0) {
            fiber.tags.push_back("review");
        }
        if (id % 3 == 0) {
            fiber.tags.push_back("source-a");
        }
        manyFibers.push_back(std::move(fiber));
    }
    scrollWidget.setFibers(manyFibers);
    scrollWidget.setKnownTags({"review", "source-a", "todo", "needs-proofread"});
    scrollWidget.resize(520, 300);
    scrollWidget.show();
    QApplication::processEvents();
    auto* scrollTree = scrollWidget.findChild<QTreeView*>(QStringLiteral("fiberTreeView"));
    require(scrollTree != nullptr, "Scrollable fiber tree view was not found");
    auto* scrollBar = scrollTree->verticalScrollBar();
    require(scrollBar != nullptr, "Fiber tree vertical scrollbar was not found");
    require(scrollBar->maximum() > 0, "Fiber tree should have a vertical scroll range");
    const int preservedScroll = scrollBar->maximum() / 2;
    scrollBar->setValue(preservedScroll);
    scrollWidget.selectFiber(70);
    require(scrollBar->value() == preservedScroll,
            "Programmatic fiber selection should not scroll the fiber list");

    const auto tagCheckboxesBefore =
        scrollWidget.findChildren<QCheckBox*>(QStringLiteral("fiberTagCheckBox"));
    require(tagCheckboxesBefore.size() == 4, "Scrollable fiber widget should expose four tag checkboxes");
    scrollBar->setValue(scrollBar->maximum());
    QApplication::processEvents();
    const int directSelectionScroll = scrollBar->value();
    const int directSelectionMaximum = scrollBar->maximum();
    const QRect directSelectionViewport = scrollTree->viewport()->geometry();
    const QModelIndex directSelectionIndex = scrollTree->model()->index(
        scrollTree->model()->rowCount() - 2,
        0);
    require(directSelectionIndex.isValid(), "Direct selection target was not valid");
    scrollTree->selectionModel()->select(directSelectionIndex,
                                         QItemSelectionModel::ClearAndSelect |
                                         QItemSelectionModel::Rows);
    QApplication::processEvents();
    require(scrollBar->value() == directSelectionScroll,
            "Tree selection changes should not move the fiber list scrollbar");
    require(scrollBar->maximum() == directSelectionMaximum,
            "Tree selection changes should not resize the fiber list scroll range");
    require(scrollTree->viewport()->geometry() == directSelectionViewport,
            "Tree selection changes should not resize the fiber list viewport");
    const auto tagCheckboxesAfter =
        scrollWidget.findChildren<QCheckBox*>(QStringLiteral("fiberTagCheckBox"));
    require(tagCheckboxesAfter == tagCheckboxesBefore,
            "Tree selection changes should update tag checkboxes in place");

    int confirmations = 0;
    int batchDeletes = 0;
    int importRequests = 0;
    int exportRequests = 0;
    std::vector<uint64_t> confirmedIds;
    std::vector<uint64_t> deletedIds;
    QObject::connect(&widget,
                     &CFiberWidget::importFibersRequested,
                     &widget,
                     [&]() {
                         ++importRequests;
                     });
    QObject::connect(&widget,
                     &CFiberWidget::exportFibersRequested,
                     &widget,
                     [&]() {
                         ++exportRequests;
                     });
    QObject::connect(&widget,
                     &CFiberWidget::deleteFibersRequested,
                     &widget,
                     [&](std::vector<uint64_t> ids) {
                         ++batchDeletes;
                         deletedIds = std::move(ids);
                     });
    importButton->click();
    require(importRequests == 1, "Import button did not emit one import request");
    exportButton->click();
    require(exportRequests == 1, "Export button did not emit one export request");

    widget.setDeleteConfirmationForTesting([&](const std::vector<uint64_t>& ids) {
        ++confirmations;
        confirmedIds = ids;
        return false;
    });
    deleteButton->click();
    require(confirmations == 1, "Delete did not ask for confirmation");
    require(sameIds(confirmedIds, {1, 3}), "Confirmation did not receive selected IDs");
    require(batchDeletes == 0, "Canceled delete should not emit delete request");

    widget.setDeleteConfirmationForTesting([&](const std::vector<uint64_t>& ids) {
        ++confirmations;
        confirmedIds = ids;
        return true;
    });
    deleteButton->click();
    require(confirmations == 2, "Confirmed delete did not ask for confirmation");
    require(batchDeletes == 1, "Confirmed delete did not emit one batch delete request");
    require(sameIds(deletedIds, {1, 3}), "Batch delete request IDs are wrong");

    QThreadPool::globalInstance()->waitForDone();
    return 0;
}
