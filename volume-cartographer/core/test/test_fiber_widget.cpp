#include "CFiberWidget.hpp"

#include <QApplication>
#include <QPushButton>

#include <cstdlib>
#include <iostream>
#include <memory>
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

} // namespace

int main(int argc, char** argv)
{
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    std::unique_ptr<QApplication> app;
    ensureApplication(argc, argv, app);

    CFiberWidget widget;
    widget.setFibers({
        {1, 2, 20, 12.0},
        {2, 3, 30, 24.0},
        {3, 4, 40, 36.0},
    });

    auto* deleteButton = widget.findChild<QPushButton*>(QStringLiteral("fiberDeleteButton"));
    require(deleteButton != nullptr, "Fiber delete button was not found");
    require(!deleteButton->isEnabled(), "Delete button should start disabled");
    require(!widget.canDeleteSelection(), "Empty selection should not allow delete");
    require(!widget.canCreateAtlasFromSelection(), "Empty selection should not allow atlas creation");

    widget.selectFiber(2);
    require(widget.selectedFiberId() == 2, "Single selection did not set selectedFiberId");
    require(sameIds(widget.selectedFiberIds(), {2}), "Single selection IDs are wrong");
    require(deleteButton->isEnabled(), "Delete button should enable for a single selection");
    require(widget.canDeleteSelection(), "Single selection should allow delete");
    require(widget.canCreateAtlasFromSelection(), "Single selection should allow atlas creation");

    widget.selectFibers({1, 3});
    require(widget.selectedFiberId() == 0, "Multi-selection should not expose a single selected fiber");
    require(sameIds(widget.selectedFiberIds(), {1, 3}), "Multi-selection IDs are wrong");
    require(deleteButton->isEnabled(), "Delete button should enable for multi-selection");
    require(widget.canDeleteSelection(), "Multi-selection should allow delete");
    require(!widget.canCreateAtlasFromSelection(), "Multi-selection should gray out atlas creation");

    int confirmations = 0;
    int batchDeletes = 0;
    std::vector<uint64_t> confirmedIds;
    std::vector<uint64_t> deletedIds;
    QObject::connect(&widget,
                     &CFiberWidget::deleteFibersRequested,
                     &widget,
                     [&](std::vector<uint64_t> ids) {
                         ++batchDeletes;
                         deletedIds = std::move(ids);
                     });

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

    return 0;
}
