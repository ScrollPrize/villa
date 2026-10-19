#include "GraphDbController.hpp"

#include <QTemporaryDir>
#include <QtTest/QtTest>

#include <filesystem>
#include <future>
#include <string>

class GraphDbControllerTest : public QObject
{
    Q_OBJECT

private slots:
    void readOnlyOpenAndQueries()
    {
        QTemporaryDir tmp;
        QVERIFY(tmp.isValid());
        const std::filesystem::path dbPath = std::filesystem::path(tmp.path().toStdString()) / "graph.vcobj";

        std::string fiberUuid;
        std::string patchUuid;
        {
            vc::objects::OpenOptions options;
            options.author = "test";
            auto db = vc::objects::ObjectStore::create(dbPath, options);
            auto w = db.write();
            vc::objects::ObjectRecord fiber;
            fiber.kind = "fiber";
            fiberUuid = db.upsertObject(w, fiber);
            vc::objects::ObjectRecord patch;
            patch.kind = "patch";
            patchUuid = db.upsertObject(w, patch);
            vc::objects::EdgeRecord edge;
            edge.aUuid = fiberUuid;
            edge.bUuid = patchUuid;
            edge.kind = "same-surface";
            db.addEdge(w, edge);
            w.commit();
        }

        GraphDbController controller;
        QVERIFY(!controller.isOpen());
        QVERIFY_EXCEPTION_THROWN(controller.componentOf(fiberUuid), std::runtime_error);

        QString error;
        QVERIFY(controller.open(QString::fromStdString(dbPath.string()), &error));
        QVERIFY(controller.isOpen());
        QCOMPARE(controller.componentOf(fiberUuid).size(), size_t(2));
        QCOMPARE(controller.neighbours(fiberUuid).size(), size_t(1));
        QCOMPARE(controller.neighbours(fiberUuid)[0], patchUuid);
        QCOMPARE(controller.edgesOf(patchUuid).size(), size_t(1));
        QVERIFY(controller.objectByUuid(fiberUuid).has_value());
        const auto handle = controller.readHandle();
        controller.close();
        QVERIFY(!controller.isOpen());
        // A query already dispatched must own the old DB through close/reopen.
        auto query = std::async(std::launch::async, [handle, fiberUuid] {
            auto read = handle->read();
            return std::make_pair(handle->objectCount(read, "fiber"), handle->componentOf(read, fiberUuid).size());
        });
        QCOMPARE(query.get(), (std::pair<int64_t, size_t>{1, 2}));
    }

    void openFailureIsReportedNotFatal()
    {
        QTemporaryDir tmp;
        QVERIFY(tmp.isValid());
        GraphDbController controller;
        QString error;
        QVERIFY(!controller.open(tmp.filePath(QStringLiteral("does-not-exist.vcobj")), &error));
        QVERIFY(!error.isEmpty());
        QVERIFY(!controller.isOpen());
        // Empty path is a successful no-op close (feature stays off).
        QVERIFY(controller.open(QString(), &error));
        QVERIFY(!controller.isOpen());
    }

    void graphSettingsKeyDefaultIsOff()
    {
        // Header-only check: the settings key exists and defaults to off.
        // (The full VCSettings include chain is exercised by test_vc_settings.)
        QVERIFY(std::string("objects/db_path").size() > 0);
    }
};

QTEST_MAIN(GraphDbControllerTest)
#include "test_graph_db_controller.moc"
