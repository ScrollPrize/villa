#include <QtTest/QtTest>
#include <QPlainTextEdit>
#include "../ConstraintGraphActivity.hpp"

class ConstraintGraphActivityTest : public QObject {
    Q_OBJECT
private slots:
    void independentOperationsAndHistory() {
        ConstraintGraphActivity activity;activity.show();
        activity.begin(ConstraintGraphActivity::Operation,"Refitting entire graph",true);
        activity.update(ConstraintGraphActivity::Operation,"Fitting flattened layout: iteration 12");
        activity.begin(ConstraintGraphActivity::Discovery,"Attachment search");
        activity.update(ConstraintGraphActivity::Discovery,"Searched 20 / 100 patches");
        auto* operation=activity.findChild<QLabel*>("graphActivity0");
        QVERIFY(operation->text().contains("iteration 12"));
        QVERIFY(activity.stopButton()->isEnabled());
        activity.finish(ConstraintGraphActivity::Discovery,"300 proposals found");
        QVERIFY(operation->text().contains("iteration 12"));
        activity.finish(ConstraintGraphActivity::Operation,"Cancelled");
        activity.update(ConstraintGraphActivity::Operation,"Late worker update");
        QVERIFY(operation->text().contains("Cancelled"));
        QVERIFY(!activity.stopButton()->isEnabled());
        auto log=activity.findChild<QPlainTextEdit*>()->toPlainText();
        QVERIFY(log.contains("iteration 12"));QVERIFY(log.contains("300 proposals found"));QVERIFY(log.contains("Cancelled"));
        activity.begin(ConstraintGraphActivity::Operation,"Saving graph",false);
        QVERIFY(!activity.stopButton()->isEnabled());
        activity.finish(ConstraintGraphActivity::Operation,"Completed");
        QVERIFY(operation->text().contains("Saving graph: Completed"));
        // A screenshot is also useful for reviewing the isolated real widget.
        if(qEnvironmentVariableIsSet("VC3D_ACTIVITY_SCREENSHOT"))activity.grab().save(qEnvironmentVariable("VC3D_ACTIVITY_SCREENSHOT"));
    }
};
QTEST_MAIN(ConstraintGraphActivityTest)
#include "test_constraint_graph_activity.moc"
