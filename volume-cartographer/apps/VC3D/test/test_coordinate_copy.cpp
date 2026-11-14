#include "CoordinateCopy.hpp"
#include "Keybinds.hpp"

#include <QDoubleSpinBox>
#include <QGraphicsProxyWidget>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QKeySequence>
#include <QLineEdit>
#include <QListWidget>
#include <QRegularExpression>
#include <QStringList>
#include <QtTest/QtTest>

#include <cmath>
#include <utility>

class CoordinateCopyTest : public QObject
{
    Q_OBJECT

private slots:
    void triplesAreWholeVoxels()
    {
        QCOMPARE(vc3d::coordinateTripleText(cv::Vec3f(1234.4f, 5678.6f, 91.0f)),
                 QStringLiteral("1234, 5679, 91"));
        QCOMPARE(vc3d::coordinateTripleText(cv::Vec3f(0.0f, 7.2f, 3999.9f)),
                 QStringLiteral("0, 7, 4000"));
        // Off-volume positions keep their sign, but nothing prints as "-0".
        QCOMPARE(vc3d::coordinateTripleText(cv::Vec3f(-0.3f, -2.6f, 12.0f)),
                 QStringLiteral("0, -3, 12"));
    }

    void triplesPasteBackIntoTheFocusField()
    {
        const QRegularExpression focusField(QString::fromLatin1(vc3d::kFocusFieldPattern));
        QVERIFY(focusField.match(vc3d::coordinateTripleText(cv::Vec3f(812.2f, 3301.7f, 10455.4f))).hasMatch());
        QVERIFY(focusField.match(vc3d::coordinateTripleText(cv::Vec3f(-0.4f, 0.0f, 0.4f))).hasMatch());
        QVERIFY(!focusField.match(vc3d::coordinateTripleText(cv::Vec3f(-0.5f, 0.0f, 0.0f))).hasMatch());
    }

    void onlyPositionsInsideTheVolumeCount()
    {
        // A 100 x 200 x 50 volume: voxels 0..99, 0..199, 0..49.
        QVERIFY(vc3d::roundsInsideVolume(cv::Vec3f(0.0f, 0.0f, 0.0f), 100, 200, 50));
        QVERIFY(vc3d::roundsInsideVolume(cv::Vec3f(99.4f, 199.4f, 49.4f), 100, 200, 50));
        QVERIFY(vc3d::roundsInsideVolume(cv::Vec3f(-0.4f, 10.0f, 10.0f), 100, 200, 50));
        // Exact halves round away from zero: -0.5 is written "-1" and 99.5 "100".
        QVERIFY(!vc3d::roundsInsideVolume(cv::Vec3f(-0.5f, 10.0f, 10.0f), 100, 200, 50));
        QVERIFY(!vc3d::roundsInsideVolume(cv::Vec3f(99.5f, 10.0f, 10.0f), 100, 200, 50));
        QVERIFY(!vc3d::roundsInsideVolume(cv::Vec3f(-0.6f, 10.0f, 10.0f), 100, 200, 50));
        QVERIFY(!vc3d::roundsInsideVolume(cv::Vec3f(10.0f, 200.0f, 10.0f), 100, 200, 50));
        QVERIFY(!vc3d::roundsInsideVolume(cv::Vec3f(10.0f, 10.0f, 49.5f), 100, 200, 50));
        QVERIFY(!vc3d::roundsInsideVolume(cv::Vec3f(10.0f, std::nanf(""), 10.0f), 100, 200, 50));
    }

    void ctrlCIsLeftToWidgetsThatCopied()
    {
        QVERIFY(!vc3d::focusedWidgetHandlesCopy(nullptr));
        QWidget plain;
        QVERIFY(!vc3d::focusedWidgetHandlesCopy(&plain));
        QListWidget list;
        QVERIFY(vc3d::focusedWidgetHandlesCopy(&list));

        // A viewer passes every key on, also Ctrl+C that a field in its scene
        // used to copy selected text.
        QGraphicsScene scene;
        QGraphicsView view(&scene);
        QVERIFY(!vc3d::focusedWidgetHandlesCopy(&view));
        auto* spin = new QDoubleSpinBox;
        spin->setValue(12.0);
        QGraphicsProxyWidget* proxy = scene.addWidget(spin);
        view.show();
        QVERIFY(QTest::qWaitForWindowExposed(&view));
        view.activateWindow();
        scene.setFocusItem(proxy);
        QCOMPARE(scene.focusItem(), proxy);
        spin->findChild<QLineEdit*>()->deselect();
        QVERIFY(!vc3d::focusedWidgetHandlesCopy(&view));
        spin->selectAll();
        QVERIFY(vc3d::focusedWidgetHandlesCopy(&view));
    }

    void keysAreCtrlCAndCtrlShiftC()
    {
        using namespace vc3d::keybinds;
        QCOMPARE(sequenceFor(keypress::CopyCursorCoordinates), QKeySequence(Qt::CTRL | Qt::Key_C));
        QCOMPARE(sequenceFor(shortcuts::CopyFocusCoordinates),
                 QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_C));
    }

    void helpListsEachKeyOnceAndNoOtherBindingUsesIt()
    {
        using namespace vc3d::keybinds;
        const QStringList lines = buildKeybindsHelpText().split(QLatin1Char('\n'));
        const std::pair<QKeySequence, const char*> bindings[] = {
            {sequenceFor(keypress::CopyCursorCoordinates), keypress::CopyCursorCoordinates.description},
            {sequenceFor(shortcuts::CopyFocusCoordinates), shortcuts::CopyFocusCoordinates.description},
        };
        for (const auto& [keys, description] : bindings) {
            const QString keyText = keys.toString(QKeySequence::NativeText);
            QVERIFY(!keyText.isEmpty());
            const QString expected = keyText + QStringLiteral(": ") + QString::fromUtf8(description);
            QCOMPARE(lines.count(expected), 1);
            int sameKey = 0;
            for (const QString& line : lines) {
                if (line.startsWith(keyText + QStringLiteral(": "))) {
                    ++sameKey;
                }
            }
            QCOMPARE(sameKey, 1);
        }
    }
};

QTEST_MAIN(CoordinateCopyTest)
#include "test_coordinate_copy.moc"
