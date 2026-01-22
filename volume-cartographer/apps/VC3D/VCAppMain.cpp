#include <qapplication.h>
#include <QCommandLineParser>

#include "CWindow.hpp"
#include "vc/core/Version.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <opencv2/core.hpp>
#include <iostream>
#include <thread>


auto main(int argc, char* argv[]) -> int
{
    cv::setNumThreads(std::thread::hardware_concurrency());

    // Workaround for Qt dock widget issues on Wayland (QTBUG-87332)
    // Floating dock widgets become unmovable after initial drag on Wayland.
    // Force XCB (X11/XWayland) platform to restore full functionality.
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        const char* waylandDisplay = qgetenv("WAYLAND_DISPLAY").constData();
        if (waylandDisplay && *waylandDisplay) {
            qputenv("QT_QPA_PLATFORM", "xcb");
        }
    }

    QApplication app(argc, argv);
    QApplication::setOrganizationName("Vesuvius Challenge");
    QApplication::setApplicationName("VC3D");
    QApplication::setWindowIcon(QIcon(":/images/logo.png"));
    QApplication::setApplicationVersion(QString::fromStdString(ProjectInfo::VersionString()));
    std::cout << "VC3D commit: " << ProjectInfo::RepositoryHash() << std::endl;

    QCommandLineParser parser;
    parser.setApplicationDescription("VC3D - Volume Cartographer 3D Viewer");
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption skipShapeCheckOption(
        "skip-shape-check",
        "Skip validation of zarr shape against meta.json dimensions");
    parser.addOption(skipShapeCheckOption);

    QCommandLineOption loadFirstOption(
        "load-first",
        "Load segmentations from the specified directory first and defer others (e.g. paths or traces).",
        "dir");
    parser.addOption(loadFirstOption);

    parser.process(app);

    if (parser.isSet(skipShapeCheckOption)) {
        Volume::skipShapeCheck = true;
    }
    if (parser.isSet(loadFirstOption)) {
        QString loadFirstDir = parser.value(loadFirstOption).trimmed().toLower();
        if (!loadFirstDir.isEmpty()) {
            VolumePkg::setLoadFirstSegmentationDirectory(loadFirstDir.toStdString());
        }
    }

    CWindow aWin;
    aWin.show();
    return QApplication::exec();
}
