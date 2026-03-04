#include <qapplication.h>
#include <QCommandLineParser>

#include "CWindow.hpp"
#include "VCSettings.hpp"
#include "vc/core/Version.hpp"
#include <QSettings>
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <opencv2/core.hpp>
#include <iostream>
#include <thread>
#include <omp.h>
#include <blosc.h>
#include <cstdlib>
#ifndef _WIN32
#include <dlfcn.h>
#endif

// Runs before main() AND before all shared-library constructors.
// .preinit_array is processed by the dynamic linker before any .init_array,
// so env vars are visible when OpenBLAS/OpenMP create their thread pools.
static void setThreadPoliciesEarly()
{
    // Force passive wait policy so OpenMP threads sleep instead of
    // spin-waiting with sched_yield.  overwrite=1 is intentional —
    // spin-waiting on 500+ OMP threads kills the machine.
    setenv("OMP_WAIT_POLICY", "passive", 1);
    setenv("OMP_NUM_THREADS", "1", 0);       // limit OpenMP parallelism
    setenv("KMP_BLOCKTIME", "0", 1);         // LLVM/Intel OpenMP: sleep immediately
    setenv("KMP_AFFINITY", "disabled", 0);   // skip sched_setaffinity per fork/join
    setenv("OPENBLAS_NUM_THREADS", "1", 0);
    setenv("GOTO_NUM_THREADS", "1", 0);      // legacy name for OpenBLAS
    setenv("MKL_NUM_THREADS", "1", 0);       // Intel MKL
}
#ifdef __linux__
__attribute__((section(".preinit_array"), used))
static auto preinitFn = &setThreadPoliciesEarly;
#endif

auto main(int argc, char* argv[]) -> int
{
#ifndef __linux__
    // On non-Linux, preinit_array is unavailable so set env vars at start of main.
    // This may be too late for some libraries that init in static constructors.
    setThreadPoliciesEarly();
#endif

#ifndef _WIN32
    // LLVM/Intel OpenMP: set blocktime=0 so threads sleep immediately after
    // parallel regions. dlsym avoids weak-symbol issues under LTO.
    if (auto fn = reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT, "kmp_set_blocktime")))
        fn(0);

    // Also call openblas_set_num_threads(1) at runtime in case the env var
    // was too late for this particular build's init order.
    if (auto fn = reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads")))
        fn(1);

    // Kill OpenBLAS spin-waiting thread pool entirely. The pthreads build
    // creates N threads at init that busy-wait even when set to 1 thread.
    // blas_shutdown() terminates all pool threads. If OpenBLAS is needed
    // later, blas_thread_init() will be called automatically.
    if (auto fn = reinterpret_cast<void(*)()>(dlsym(RTLD_DEFAULT, "blas_shutdown")))
        fn();
#endif

    omp_set_num_threads(1);  // All parallelism is explicit (QThreadPool, IOPool); OMP threads just spin-wait
    cv::setNumThreads(1);
    blosc_set_nthreads(1);  // We parallelize at tile level; blosc internal threads just spin-wait

    // Workaround for Qt dock widget issues on Wayland (QTBUG-87332)
    // Floating dock widgets become unmovable after initial drag on Wayland.
    // Force XCB (X11/XWayland) platform to restore full functionality.
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        if (!qEnvironmentVariableIsEmpty("WAYLAND_DISPLAY")) {
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

    QCommandLineOption cacheSizeOption(
        "cache-size",
        QString("Set the chunk cache size in gigabytes (default: %1 GB).")
            .arg(CHUNK_CACHE_SIZE_GB),
        "GB",
        QString::number(CHUNK_CACHE_SIZE_GB));
    parser.addOption(cacheSizeOption);

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

    // RAM cache size: CLI flag > QSettings > CMake default
    size_t cacheSizeGB = CHUNK_CACHE_SIZE_GB;
    {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        cacheSizeGB = settings.value(perf::RAM_CACHE_SIZE_GB, perf::RAM_CACHE_SIZE_GB_DEFAULT).toULongLong();
    }
    if (parser.isSet(cacheSizeOption)) {
        bool ok = false;
        const qulonglong parsed = parser.value(cacheSizeOption).toULongLong(&ok);
        if (!ok || parsed == 0) {
            std::cerr << "Error: Invalid cache size. Must be a positive integer (GB)." << std::endl;
            return 1;
        }
        if (parsed > 256) {
            std::cerr << "Warning: Cache size " << parsed
                      << " GB is very large. Ensure sufficient system memory." << std::endl;
        }
        cacheSizeGB = static_cast<size_t>(parsed);
    }

    CWindow aWin(cacheSizeGB);
    aWin.show();
    return QApplication::exec();
}
