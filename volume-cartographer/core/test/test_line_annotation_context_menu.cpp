#include <QApplication>
#include <QSignalSpy>
#include <QTest>

#include "volume_viewers/CVolumeViewerView.hpp"

#include <memory>

namespace {

void ensureApplication(int& argc, char** argv, std::unique_ptr<QApplication>& app)
{
    if (!QApplication::instance()) {
        app = std::make_unique<QApplication>(argc, argv);
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

    CVolumeViewerView view;
    view.resize(200, 160);
    view.show();
    (void)QTest::qWaitForWindowExposed(&view);

    const QPoint pos(80, 70);

    QSignalSpy contextSpy(&view, &CVolumeViewerView::sendAnnotationContextMenuRequested);
    QSignalSpy panSpy(&view, &CVolumeViewerView::sendPanStart);
    QSignalSpy pressSpy(&view, &CVolumeViewerView::sendMousePress);

    QTest::mousePress(view.viewport(), Qt::RightButton, Qt::ControlModifier, pos);
    if (contextSpy.count() != 1 || panSpy.count() != 0 || pressSpy.count() != 0) {
        qWarning("Ctrl+Right did not route exclusively to annotation context menu");
        return 1;
    }
    QTest::mouseRelease(view.viewport(), Qt::RightButton, Qt::ControlModifier, pos);

    QTest::mousePress(view.viewport(), Qt::RightButton, Qt::NoModifier, pos);
    if (panSpy.count() != 1 || contextSpy.count() != 1 || pressSpy.count() != 0) {
        qWarning("Plain Right did not start pan without annotation/tool forwarding");
        return 1;
    }
    QTest::mouseRelease(view.viewport(), Qt::RightButton, Qt::NoModifier, pos);

    QTest::mousePress(view.viewport(), Qt::RightButton, Qt::ShiftModifier, pos);
    if (pressSpy.count() != 1 || contextSpy.count() != 1 || panSpy.count() != 1) {
        qWarning("Shift+Right did not preserve mouse forwarding");
        return 1;
    }
    QTest::mouseRelease(view.viewport(), Qt::RightButton, Qt::ShiftModifier, pos);

    return 0;
}
