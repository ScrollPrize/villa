#include "CPointCollectionWidget.hpp"

#include <QApplication>
#include <QCoreApplication>
#include <QItemSelectionModel>
#include <QPushButton>
#include <QStandardItemModel>
#include <QThreadPool>
#include <QTemporaryDir>
#include <QTreeView>

#include <cstdlib>
#include <iostream>
#include <fstream>
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

void testCorrectionResults()
{
    QTemporaryDir tmp;
    require(tmp.isValid(), "Could not create correction test directory");
    const auto pointsPath = std::filesystem::path(tmp.path().toStdString()) / "points.json";
    const auto resultsPath = pointsPath.parent_path() / "results.json";
    {
        std::ofstream out(pointsPath);
        out << R"({"vc_pointcollections_json_version":"1","collections":{
            "0":{"name":"a","points":{"0":{"p":[1,2,3]},"5":{"p":[1,2,3]}}},
            "7":{"name":"b","points":{"0":{"p":[1,2,3]}}}
        }})";
    }
    VCCollection collection;
    require(collection.loadFromJSON(pointsPath.string()), "Could not load duplicate point IDs");
    CPointCollectionWidget widget(&collection);
    auto* tree = widget.findChild<QTreeView*>(QStringLiteral("pointCollectionTreeView"));
    require(tree != nullptr, "Correction tree missing");
    auto* model = qobject_cast<QStandardItemModel*>(tree->model());
    require(model != nullptr, "Correction model missing");
    const auto cell = [model](vc::PointRef ref, int column) -> QString {
        for (int i = 0; i < model->rowCount(); ++i) {
            auto* parent = model->item(i);
            if (parent->data().toULongLong() != ref.collectionId) continue;
            for (int j = 0; j < parent->rowCount(); ++j) {
                if (parent->child(j)->data().toULongLong() == ref.pointId)
                    return parent->child(j, column)->text();
            }
        }
        require(false, "Correction point row missing");
        return {};
    };
    const auto load = [&](const char* json) {
        { std::ofstream out(resultsPath); out << json; }
        widget.loadCorrPointsResults(resultsPath);
    };
    load(R"({"points_list":[
        {"collection_id":0,"point_id":0,"p":[1,2,3],"winding_obs":1,"winding_err":0.1},
        {"collection_id":7,"point_id":0,"p":[1,2,3],"winding_obs":2,"winding_err":0.2}
    ],"points":{"0":{"collection_id":7,"p":[1,2,3],"winding_obs":99}}})");
    const auto checkQualified = [&] {
        require(cell({0,0},2) == "1.000", "First collection lost its winding result");
        require(cell({7,0},2) == "2.000", "Second collection got the wrong winding result");
        require(cell({0,0},3) == "0.100", "First collection got the wrong error");
        require(cell({7,0},3) == "0.200", "Second collection got the wrong error");
    };
    checkQualified(); // refreshTree
    auto point = collection.getPoint({0,0}).value();
    collection.updatePoint(point); // onPointChanged
    checkQualified();
    // Replay removal/addition notifications to exercise the incremental row path.
    collection.pointRemoved({0,0});
    collection.pointAdded(point); // onPointAdded
    checkQualified();
    point.p = {10,20,30};
    collection.updatePoint(point);
    require(cell({0,0},2).isEmpty() && cell({0,0},3).isEmpty(), "Moved point kept stale results");
    require(cell({7,0},2) == "2.000", "Moving one collection affected another");
    point.p = {1,2,3};
    collection.updatePoint(point);

    load(R"({"points":{"0":{"collection_id":7,"p":[1,2,3],"winding_obs":3}}})");
    require(cell({0,0},2).isEmpty(), "Qualified legacy result leaked to another collection");
    require(cell({7,0},2) == "3.000", "Qualified legacy result was lost");
    load(R"({"points":{
        "0":{"p":[1,2,3],"winding_obs":4},
        "5":{"p":[1,2,3],"winding_obs":5}
    }})");
    require(cell({0,0},2).isEmpty() && cell({7,0},2).isEmpty(), "Ambiguous legacy ID was accepted");
    require(cell({0,5},2) == "5.000", "Unambiguous legacy result was lost");
    load(R"({"points_list":[],"points":{"5":{"p":[1,2,3],"winding_obs":99}}})");
    require(cell({0,5},2).isEmpty(), "Empty qualified list fell back to stale legacy results");
    load(R"({"points_list":[
        {"collection_id":-1,"point_id":0,"p":[1,2,3],"winding_obs":9},
        {"collection_id":0,"point_id":0.5,"p":[1,2,3],"winding_obs":9},
        {"point_id":0,"p":[1,2,3],"winding_obs":9},
        {"collection_id":0,"point_id":5,"p":[1,2,3],"winding_obs":6}
    ]})");
    require(cell({0,0},2).isEmpty(), "Malformed qualified IDs were accepted");
    require(cell({0,5},2) == "6.000", "Valid result after malformed IDs was lost");
    widget.clearCorrPointsResults();
    require(cell({0,5},2).isEmpty(), "Clearing correction results left stale values");
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

    testCorrectionResults();
    return 0;
}
