#include "ObjectStoreRegions.hpp"
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/Uuid.hpp>
#include <QTemporaryDir>
#include <QtTest/QtTest>

class StoreRegionRequestsTest : public QObject {
    Q_OBJECT
private slots:
    void recordedGraphClosureKeepsOffscreenPeersAndExplicitPins()
    {
        using namespace vc::objects;
        QTemporaryDir directory;
        OpenOptions options; options.author = "region-test";
        auto db = ObjectStore::create(directory.filePath("graph.vcobj").toStdString(), options);
        const auto a = util::newUuid(), b = util::newUuid(), c = util::newUuid();
        auto link = [&](auto& txn, const auto& first, const auto& second) {
            EdgeRecord edge; edge.kind="related"; edge.aUuid=first; edge.bUuid=second;
            edge.aAnchor.wholeGeometry=edge.bAnchor.wholeGeometry=true;
            db.addEdge(txn,edge);
        };
        {
            auto txn = db.write();
            for (const auto& uuid : {a,b,c}) {
                ObjectRecord record; record.uuid=uuid; record.kind="fiber";
                db.upsertObject(txn,record);
                vc::geometry::Fiber shape; shape.linePoints={{0,0,0},{1,0,0}};
                db.putGeometry(txn,uuid,vc::geometry::encode(shape));
            }
            link(txn,a,b);
            txn.commit();
        }
        ObjectVersion av, oldB, cv;
        { auto txn=db.read(); av=db.currentObjectRevision(txn,a)->reference(); oldB=db.currentObjectRevision(txn,b)->reference(); }
        { auto txn=db.write(); link(txn,c,b); txn.commit(); }
        auto txn=db.read();
        cv=db.currentObjectRevision(txn,c)->reference();
        LocalStoreReader reader(db,txn);
        EdgeSelection edges; edges.kind="related";
        const auto graph=reader.graphReadViewAround({av,cv},edges);
        QCOMPARE(graph.objectUuids.size(),size_t{3});
        QCOMPARE(graph.edgeUuids.size(),size_t{2});
        QCOMPARE(graph.componentOf(a).size(),size_t{3});
        // Explicit historical selections must not be silently advanced by the
        // later recorded endpoint of the second edge.
        const auto pinned=reader.graphReadViewAround({av,oldB,cv},edges);
        QCOMPARE(pinned.edgeUuids.size(),size_t{1});
        for (const auto& object : *pinned.selectedObjects)
            if (object.objectUuid==b) QVERIFY(object.reference()==oldB);
    }
    void changedStoreOrFrameCannotInstallOldCuts()
    {
        using vc3d::store::RegionRequests;
        const auto view = vc3d::store::viewRegion({0,0,0}, {10,0,0}, {0,10,0}, 1).value();
        for (const bool detach : {false, true}) {
            RegionRequests requests;
            const auto query = requests.start(view).value();
            const auto token = requests.cancelled;
            if (detach) requests.reset(); else requests.refresh();
            QVERIFY(token->load());
            QVERIFY(requests.finish(token, view, query, true, {}) == RegionRequests::Finished::Retry);
            QVERIFY(!requests.loaded);
            const auto fresh = requests.start(view).value();
            QVERIFY(requests.finish(token, view, query, true, {}) == RegionRequests::Finished::Stale);
            QVERIFY(requests.finish(requests.cancelled, view, fresh, true, {}) == RegionRequests::Finished::Install);
            QVERIFY(requests.loaded);
        }
    }
};
QTEST_APPLESS_MAIN(StoreRegionRequestsTest)
#include "test_store_region_requests.moc"
