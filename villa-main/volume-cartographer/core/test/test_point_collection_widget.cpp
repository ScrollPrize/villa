#include "CPointCollectionWidget.hpp"

#include <QApplication>
#include <QCoreApplication>
#include <QItemSelectionModel>
#include <QPushButton>
#include <QStandardItemModel>
#include <QThreadPool>
#include <QTreeView>

#include <cstdlib>
#include <iostream>
#include <memory>

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

} // namespace

int main(int argc, char** argv)
{
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    std::unique_ptr<QApplication> app;
    ensureApplication(argc, argv, app);
    QThreadPool::globalInstance()->setMaxThreadCount(1);

    VCCollection collection;
    collection.addPoint("a", {1, 2, 3});
    collection.addPoint("a", {4, 5, 6});

    CPointCollectionWidget widget(&collection);

    auto* treeView = widget.findChild<QTreeView*>(QStringLiteral("pointCollectionTreeView"));
    require(treeView != nullptr, "Point collection tree view was not found");
    auto* model = qobject_cast<QStandardItemModel*>(treeView->model());
    require(model != nullptr, "Point collection tree model was not found");
    require(model->rowCount() == 1, "Expected one point collection row before clear");
    require(model->item(0, 0)->rowCount() == 2, "Expected two point rows before clear");

    const QModelIndex firstPoint = model->item(0, 0)->child(0, 0)->index();
    treeView->selectionModel()->select(firstPoint, QItemSelectionModel::Select | QItemSelectionModel::Rows);
    treeView->selectionModel()->setCurrentIndex(firstPoint, QItemSelectionModel::NoUpdate);

    auto* clearAllButton = widget.findChild<QPushButton*>(QStringLiteral("pointCollectionClearAllButton"));
    require(clearAllButton != nullptr, "Clear All Points button was not found");
    clearAllButton->click();
    QCoreApplication::processEvents();

    require(collection.getAllCollections().empty(), "Backing collection was not cleared");
    require(model->rowCount() == 0, "Point collection tree model was not cleared");
    require(treeView->selectionModel()->selectedIndexes().isEmpty(), "Point collection selection was not cleared");

    return 0;
}
