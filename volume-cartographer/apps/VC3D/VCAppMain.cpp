#include <qapplication.h>

#include "CWindow.hpp"
#include "vc/core/Version.hpp"

#include <opencv2/core.hpp>
#include <thread>





auto main(int argc, char* argv[]) -> int
{
    cv::setNumThreads(std::thread::hardware_concurrency());
    
    QApplication app(argc, argv);
    QApplication::setOrganizationName("EduceLab");
    QApplication::setApplicationName("VC3D");
    QApplication::setWindowIcon(QIcon(":/images/logo.png"));
    QApplication::setApplicationVersion(QString::fromStdString(ProjectInfo::VersionString()));

    CWindow aWin;
    aWin.show();
    return QApplication::exec();
}
