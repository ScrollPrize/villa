#include <QApplication>
#include <QScrollBar>
#include <QSignalSpy>
#include <QTest>
#include <QWheelEvent>

#include "volume_viewers/CVolumeViewerView.hpp"

#include <memory>
#include <cstdlib>
#include <iostream>

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

QPoint tiltHandleCenter(const CVolumeViewerView& view)
{
    return QPoint(view.viewport()->width() - 37,
                  view.viewport()->height() - 37);
}

} // namespace

int main(int argc, char** argv)
{
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    std::unique_ptr<QApplication> app;
    ensureApplication(argc, argv, app);

    CVolumeViewerView view;
    view.resize(200, 160);
    view.show();
    (void)QTest::qWaitForWindowExposed(&view);

    const QPoint pos(80, 70);

    QSignalSpy contextSpy(&view, &CVolumeViewerView::sendAnnotationContextMenuRequested);
    QSignalSpy panSpy(&view, &CVolumeViewerView::sendPanStart);
    QSignalSpy panReleaseSpy(&view, &CVolumeViewerView::sendPanRelease);
    QSignalSpy pressSpy(&view, &CVolumeViewerView::sendMousePress);
    QSignalSpy moveSpy(&view, &CVolumeViewerView::sendMouseMove);
    QSignalSpy clickSpy(&view, &CVolumeViewerView::sendVolumeClicked);
    QSignalSpy releaseSpy(&view, &CVolumeViewerView::sendMouseRelease);
    QSignalSpy leaveSpy(&view, &CVolumeViewerView::sendMouseLeftView);
    QSignalSpy keyPressSpy(&view, &CVolumeViewerView::sendKeyPress);
    QSignalSpy keyReleaseSpy(&view, &CVolumeViewerView::sendKeyRelease);
    QSignalSpy tiltChangedSpy(&view, &CVolumeViewerView::sendTiltHandleChanged);
    QSignalSpy tiltResetSpy(&view, &CVolumeViewerView::sendTiltHandleReset);
    QSignalSpy zoomSpy(&view, &CVolumeViewerView::sendZoom);

    QTest::mousePress(view.viewport(), Qt::RightButton, Qt::ControlModifier, pos);
    require(contextSpy.count() == 1 && panSpy.count() == 0 && pressSpy.count() == 0,
            "Ctrl+Right did not route exclusively to annotation context menu");
    QTest::mouseRelease(view.viewport(), Qt::RightButton, Qt::ControlModifier, pos);

    QTest::mousePress(view.viewport(), Qt::RightButton, Qt::NoModifier, pos);
    require(panSpy.count() == 1 && contextSpy.count() == 1 && pressSpy.count() == 0,
            "Plain Right did not start pan without annotation/tool forwarding");
    QTest::mouseRelease(view.viewport(), Qt::RightButton, Qt::NoModifier, pos);
    require(panReleaseSpy.count() == 1,
            "Plain Right pan did not emit pan release");

    QTest::mousePress(view.viewport(), Qt::RightButton, Qt::ShiftModifier, pos);
    QTest::mouseMove(view.viewport(), pos + QPoint(4, 3), Qt::ShiftModifier);
    QTest::mouseRelease(view.viewport(), Qt::RightButton, Qt::ShiftModifier, pos + QPoint(4, 3));
    require(pressSpy.count() == 1 && moveSpy.count() >= 1 && releaseSpy.count() == 1 &&
                contextSpy.count() == 1 && panSpy.count() == 1,
            "Shift+Right did not forward tool mouse press/move/release without starting pan");

    QTest::mousePress(view.viewport(), Qt::RightButton, Qt::NoModifier, pos);
    QTest::mouseRelease(view.viewport(), Qt::RightButton, Qt::NoModifier, pos);
    require(pressSpy.count() == 1 && panSpy.count() == 2 && panReleaseSpy.count() == 2,
            "Right-button forwarding should clear after Shift+Right release");

    QTest::mousePress(view.viewport(), Qt::LeftButton, Qt::NoModifier, pos);
    QTest::mouseMove(view.viewport(), pos + QPoint(3, 2));
    QTest::mouseRelease(view.viewport(), Qt::LeftButton, Qt::NoModifier, pos + QPoint(3, 2));
    require(pressSpy.count() == 2 && clickSpy.count() == 1 && releaseSpy.count() == 2 &&
                moveSpy.count() >= 2,
            "Left press/move/release did not emit tool press/move/click/release");

    QTest::mousePress(view.viewport(), Qt::MiddleButton, Qt::NoModifier, pos);
    QTest::mouseRelease(view.viewport(), Qt::MiddleButton, Qt::NoModifier, pos);
    require(panSpy.count() == 3 && panReleaseSpy.count() == 3,
            "Middle button pan did not emit pan start/release while enabled");

    view.setMiddleButtonPanEnabled(false);
    QTest::mousePress(view.viewport(), Qt::MiddleButton, Qt::NoModifier, pos);
    QTest::mouseRelease(view.viewport(), Qt::MiddleButton, Qt::NoModifier, pos);
    require(pressSpy.count() == 3 && releaseSpy.count() == 3 && panSpy.count() == 3,
            "Disabled middle-button pan should forward middle mouse events");

    view.setScrollPanDisabled(true);
    QKeyEvent keyPress(QEvent::KeyPress, Qt::Key_Left, Qt::ShiftModifier);
    QApplication::sendEvent(&view, &keyPress);
    QKeyEvent keyRelease(QEvent::KeyRelease, Qt::Key_Left, Qt::ShiftModifier);
    QApplication::sendEvent(&view, &keyRelease);
    require(keyPressSpy.count() == 1 && keyReleaseSpy.count() == 1,
            "Scroll-pan-disabled arrow keys should be forwarded");

    view.setTiltHandle(CVolumeViewerView::TiltHandleMode::Square, QPointF(0.0, 0.0));
    const QPoint handle = tiltHandleCenter(view);
    QTest::mousePress(view.viewport(), Qt::LeftButton, Qt::NoModifier, handle);
    QTest::mouseMove(view.viewport(), handle + QPoint(8, 4));
    QTest::mouseRelease(view.viewport(), Qt::LeftButton, Qt::NoModifier, handle + QPoint(8, 4));
    require(tiltChangedSpy.count() >= 3,
            "Tilt handle drag should emit changed values");
    QTest::mouseDClick(view.viewport(), Qt::LeftButton, Qt::NoModifier, handle);
    require(tiltResetSpy.count() == 1,
            "Tilt handle double-click should emit reset");

    view.setTiltHandle(CVolumeViewerView::TiltHandleMode::Hidden, QPointF());
    QPointF wheelPos(pos);
    QPointF globalWheelPos = view.viewport()->mapToGlobal(pos);
    QWheelEvent smallWheel(wheelPos, globalWheelPos, QPoint(), QPoint(0, 60),
                           Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
    QApplication::sendEvent(view.viewport(), &smallWheel);
    require(zoomSpy.count() == 0,
            "Sub-threshold wheel delta should not emit zoom");
    QWheelEvent secondSmallWheel(wheelPos, globalWheelPos, QPoint(), QPoint(0, 60),
                                Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
    QApplication::sendEvent(view.viewport(), &secondSmallWheel);
    require(zoomSpy.count() == 1,
            "Accumulated wheel delta should emit zoom after threshold");

    QEvent leaveEvent(QEvent::Leave);
    QApplication::sendEvent(view.viewport(), &leaveEvent);
    require(leaveSpy.count() == 1, "Leave event did not emit mouse-left-view signal");

    return 0;
}
