#include "SpiralUpdatedDrafts.hpp"

#include <QtTest/QtTest>

// Which saved drafts an input update promoted to the fit's inputs
// (SpiralWorkspace drops them as drawings afterwards).
class SpiralUpdatedDraftsTest : public QObject
{
    Q_OBJECT

    static int64_t version(const QString& id) { return id == QStringLiteral("resaved") ? 2 : 1; }
    static bool unchanged(const QString& id) { return id != QStringLiteral("edited"); }
    const QHash<QString, int64_t> saved{{QStringLiteral("a"), 1}, {QStringLiteral("b"), 1},
                                        {QStringLiteral("resaved"), 1}, {QStringLiteral("edited"), 1}};

private slots:
    void aPositionedUpdateIncludesTheDraftsSavedBeforeIt()
    {
        auto ids = vc3d::spiralDraftsIncludedByUpdate({}, 7, saved, version, unchanged);
        ids.sort();
        // One saved again meanwhile, or whose drawing changed, stays a drawing.
        QCOMPARE(ids, (QStringList{QStringLiteral("a"), QStringLiteral("b")}));
    }

    void aFailedUpdateIncludesNone()
    {
        QVERIFY(vc3d::spiralDraftsIncludedByUpdate(QStringLiteral("refused"), 7, saved, version, unchanged).isEmpty());
    }

    void anUpdateOfAnotherStoreIncludesNone()
    {
        // Drafts saved to the attached store, then an update of a service
        // whose store is not attached here (no position named): the drafts
        // never reached that store and must stay drawings.
        QVERIFY(vc3d::spiralDraftsIncludedByUpdate({}, -1, saved, version, unchanged).isEmpty());
    }
};

QTEST_APPLESS_MAIN(SpiralUpdatedDraftsTest)
#include "test_spiral_updated_drafts.moc"
