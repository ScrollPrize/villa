#include <QtTest/QtTest>
#include "LineAnnotationFiberDeletion.hpp"

class TestLineAnnotationFiberDeletion : public QObject {
    Q_OBJECT
private slots:
    void outcomesUseObjectIdentity() {
        vc3d::line_annotation::FiberDeleteOutcome outcome;
        outcome.requested = {{2, "uuid-a"}, {3, "uuid-b"}};
        outcome.deleted = {{17, "uuid-a"}};
        QVERIFY(outcome.deletedRequested(2));
        QVERIFY(!outcome.deletedRequested(3));
        QVERIFY(!outcome.deletedRequested(17));
    }
    void unnamedObjectsNeverMatch() {
        vc3d::line_annotation::FiberDeleteOutcome outcome;
        outcome.requested = {{2, ""}};
        outcome.deleted = {{2, ""}};
        QVERIFY(!outcome.deletedRequested(2));
    }
};
QTEST_APPLESS_MAIN(TestLineAnnotationFiberDeletion)
#include "test_line_annotation_fiber_deletion.moc"
