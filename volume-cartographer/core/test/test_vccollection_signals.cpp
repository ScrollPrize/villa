// VCCollection is now a thin QObject shim that emits its signals from
// PointCollections's protected on*() hooks. Verify each mutation still
// fires the matching signal via QSignalSpy.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <QSignalSpy>

#include "vc/ui/VCCollection.hpp"

TEST_CASE("mutations emit the expected signals")
{
    VCCollection c;

    QSignalSpy added(&c, &VCCollection::pointAdded);
    QSignalSpy changed(&c, &VCCollection::pointChanged);
    QSignalSpy removed(&c, &VCCollection::pointRemoved);
    QSignalSpy colsAdded(&c, &VCCollection::collectionsAdded);
    QSignalSpy colChanged(&c, &VCCollection::collectionChanged);
    QSignalSpy colRemoved(&c, &VCCollection::collectionRemoved);

    auto p = c.addPoint("a", {0, 0, 0});
    CHECK(added.count() == 1);
    CHECK(colsAdded.count() == 1);  // first point creates the collection

    p.p = {1, 1, 1};
    c.updatePoint(p);
    CHECK(changed.count() == 1);

    const uint64_t cid = c.getCollectionId("a");
    c.renameCollection(cid, "b");
    CHECK(colChanged.count() >= 1);

    c.removePoint(p.id);
    CHECK(removed.count() == 1);

    c.clearAll();
    CHECK(colRemoved.count() >= 1);
}
