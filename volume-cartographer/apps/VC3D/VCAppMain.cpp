// Override global new/delete with mimalloc if available. Must be included
// in exactly ONE translation unit; the linker picks up the override symbols.
#if defined(VC_HAVE_MIMALLOC)
#include <mimalloc-new-delete.h>
#include <mimalloc.h>
#endif

#include <qapplication.h>
#include <QCommandLineParser>
#include <QElapsedTimer>
#include <QEvent>
#include <QObject>

#include "CWindow.hpp"
#include "VCSettings.hpp"
#include "vc/core/Version.hpp"
#include <QSettings>
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/CrashHandler.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>
#include <iostream>
#include <string_view>
#include <thread>
#include <omp.h>
#include <blosc.h>
#include <cstdlib>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace
{
QString eventTypeName(QEvent::Type type)
{
    switch (type) {
    case QEvent::None: return QStringLiteral("None");
    case QEvent::Timer: return QStringLiteral("Timer");
    case QEvent::MouseButtonPress: return QStringLiteral("MouseButtonPress");
    case QEvent::MouseButtonRelease: return QStringLiteral("MouseButtonRelease");
    case QEvent::MouseButtonDblClick: return QStringLiteral("MouseButtonDblClick");
    case QEvent::MouseMove: return QStringLiteral("MouseMove");
    case QEvent::KeyPress: return QStringLiteral("KeyPress");
    case QEvent::KeyRelease: return QStringLiteral("KeyRelease");
    case QEvent::FocusIn: return QStringLiteral("FocusIn");
    case QEvent::FocusOut: return QStringLiteral("FocusOut");
    case QEvent::Paint: return QStringLiteral("Paint");
    case QEvent::Move: return QStringLiteral("Move");
    case QEvent::Resize: return QStringLiteral("Resize");
    case QEvent::Close: return QStringLiteral("Close");
    case QEvent::Show: return QStringLiteral("Show");
    case QEvent::Hide: return QStringLiteral("Hide");
    case QEvent::Wheel: return QStringLiteral("Wheel");
    case QEvent::DeferredDelete: return QStringLiteral("DeferredDelete");
    case QEvent::UpdateRequest: return QStringLiteral("UpdateRequest");
    case QEvent::UpdateLater: return QStringLiteral("UpdateLater");
    case QEvent::MetaCall: return QStringLiteral("MetaCall");
    default:
        return QStringLiteral("type:%1").arg(static_cast<int>(type));
    }
}

class VCApplication : public QApplication
{
public:
    using QApplication::QApplication;

    bool notify(QObject* receiver, QEvent* event) override
    {
        QElapsedTimer timer;
        timer.start();
        const bool handled = QApplication::notify(receiver, event);
        const qint64 elapsedMs = timer.elapsed();
        if (elapsedMs >= 25 && receiver && event) {
            Logger()->info("[vc3d-event] slow notify ms={} event='{}' receiverClass='{}' receiverName='{}'",
                           elapsedMs,
                           eventTypeName(event->type()).toStdString(),
                           receiver->metaObject() ? receiver->metaObject()->className() : "<unknown>",
                           receiver->objectName().toStdString());
        }
        return handled;
    }
};
}
#ifndef _WIN32
#include <dlfcn.h>
#include <sys/resource.h>
#endif
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
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

__attribute__((visibility("default")))
auto main(int argc, char* argv[]) -> int
{
    vc::crash::install();

#ifndef __linux__
    // On non-Linux, preinit_array is unavailable so set env vars at start of main.
    // This may be too late for some libraries that init in static constructors.
    setThreadPoliciesEarly();
#endif

#if defined(__GLIBC__) && !defined(VC_HAVE_MIMALLOC)
    // Tune glibc's malloc to give freed pages back to the OS more aggressively.
    // Lower M_MMAP_THRESHOLD pushes bigger allocations through mmap (returned
    // independently on free), reducing main-heap fragmentation. Lower
    // M_TRIM_THRESHOLD runs sbrk-trim more often. Only takes effect when
    // mimalloc isn't overriding malloc.
    ::mallopt(M_MMAP_THRESHOLD, 128 * 1024);
    ::mallopt(M_TRIM_THRESHOLD, 128 * 1024);
#endif

#if defined(VC_HAVE_MIMALLOC)
    // Return freed pages to the OS immediately rather than holding them in
    // mimalloc's page cache. Matters for VC3D's allocation pattern: big
    // transient buffers (decoded chunks, render scratch, shard reads) are
    // freed quickly but the default delay keeps their pages committed,
    // inflating RSS during bulk-download workloads on RAM-constrained
    // machines. purge_decommits=1 actually decommits (not just reset);
    // purge_delay=0 skips the decommit-queue timeout; arena_eager_commit=0
    // avoids pre-committing arenas that never see writes.
    mi_option_set(mi_option_purge_decommits, 1);
    mi_option_set(mi_option_purge_delay, 0);
    mi_option_set(mi_option_arena_eager_commit, 0);
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

    // VC3D's interactive renderer performs better without BlockPipeline's
    // per-frame fetchInteractive dedup. Keep this app default scoped to VC3D,
    // while allowing users to set VC_DISABLE_FETCHINTERACTIVE_DEDUP=0 to
    // compare or debug the dedup path.
    if (qEnvironmentVariableIsEmpty("VC_DISABLE_FETCHINTERACTIVE_DEDUP")) {
        qputenv("VC_DISABLE_FETCHINTERACTIVE_DEDUP", "1");
    }

    auto hasCommandLineFlag = [argc, argv](std::string_view flag) {
        for (int i = 1; i < argc; ++i) {
            if (argv[i] && std::string_view(argv[i]) == flag) {
                return true;
            }
        }
        return false;
    };
    if (hasCommandLineFlag("--replay-offscreen-4k")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    // Workaround for Qt dock widget issues on Wayland (QTBUG-87332)
    // Floating dock widgets become unmovable after initial drag on Wayland.
    // Force XCB (X11/XWayland) platform to restore full functionality.
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        if (!qEnvironmentVariableIsEmpty("WAYLAND_DISPLAY")) {
            qputenv("QT_QPA_PLATFORM", "xcb");
        }
    }

    // VC3D uses traditional QWidget painting for the shell UI. Avoid Qt's
    // RHI-backed widget flushing path unless the user explicitly opts in;
    // it can route ordinary exposes through GLX and crash in some NVIDIA
    // driver/Qt combinations before any project code is on the stack.
    if (qEnvironmentVariableIsEmpty("QT_WIDGETS_RHI")) {
        qputenv("QT_WIDGETS_RHI", "0");
    }

    VCApplication app(argc, argv);
    QApplication::setOrganizationName("Vesuvius Challenge");
    QApplication::setApplicationName("VC3D");
    QApplication::setWindowIcon(QIcon(":/images/logo.png"));
    QApplication::setApplicationVersion(QString::fromStdString(ProjectInfo::VersionString()));
    std::cout << "VC3D commit: " << ProjectInfo::RepositoryHash() << std::endl;
    std::cout << "creating remote volume cache at "
              << vc3d::remoteCachePath().toStdString() << std::endl;

    QCommandLineParser parser;
    parser.setApplicationDescription("VC3D - Volume Cartographer 3D Viewer");
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption skipShapeCheckOption(
        "skip-shape-check",
        "Skip validation of zarr shape against meta.json dimensions");
    parser.addOption(skipShapeCheckOption);

    QCommandLineOption cacheSizeOption(
        "cache-size",
        QString("Set the chunk cache size in gigabytes (default: %1 GB).")
            .arg(CHUNK_CACHE_SIZE_GB),
        "GB",
        QString::number(CHUNK_CACHE_SIZE_GB));
    parser.addOption(cacheSizeOption);

    QCommandLineOption loadFirstOption(
        "load-first",
        "Load the named segmentation folder first instead of loading all segmentation folders.",
        "folder");
    parser.addOption(loadFirstOption);

    QCommandLineOption debugOption(
        "debug",
        "Enable verbose diagnostic logging while loading surfaces.");
    parser.addOption(debugOption);

    QCommandLineOption profileOption(
        "profile",
        "Enable VC3D render profiling logs.");
    parser.addOption(profileOption);

    QCommandLineOption recordOption(
        "record",
        "Record a navigation camera-state timeline to the given JSON file.",
        "file");
    parser.addOption(recordOption);

    QCommandLineOption replayOption(
        "replay",
        "Replay a recorded navigation timeline, then exit.",
        "file");
    parser.addOption(replayOption);

    QCommandLineOption replayWarmOption(
        "replay-warm",
        "With --replay, run a discarded warm-up pass before the timed pass.");
    parser.addOption(replayWarmOption);

    QCommandLineOption replayOffscreen4kOption(
        "replay-offscreen-4k",
        "With --replay, use Qt offscreen platform and a 3840x2160 replay window.");
    parser.addOption(replayOffscreen4kOption);

    parser.process(app);

    if (parser.isSet(debugOption)) {
        SetDebugLoggingEnabled(true);
        SetLogLevel("debug");
    }

    RenderBenchOptions benchOptions;
    benchOptions.recordPath = parser.value(recordOption).trimmed();
    benchOptions.replayPath = parser.value(replayOption).trimmed();
    benchOptions.replayWarm = parser.isSet(replayWarmOption);
    if (parser.isSet(replayOffscreen4kOption)) {
        benchOptions.replayWindowSize = QSize(3840, 2160);
    }

    if (parser.isSet(profileOption)) {
        SetProfileLoggingEnabled(true);
        Logger()->info("[vc3d-profile] enabled");
    }

    if (parser.isSet(skipShapeCheckOption)) {
        Volume::skipShapeCheck = true;
    }

    // RAM cache size: CLI flag > QSettings > CMake default
    size_t cacheSizeGB = CHUNK_CACHE_SIZE_GB;
    {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        cacheSizeGB = settings.value(perf::RAM_CACHE_SIZE_GB, perf::RAM_CACHE_SIZE_GB_DEFAULT).toULongLong();

        // Per-segment rotating-backup count -> core (used by saveOverwrite/growth).
        QuadSurface::setBackupCount(
            settings.value(backup::SEGMENT_COUNT, backup::SEGMENT_COUNT_DEFAULT).toInt());
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

    if (parser.isSet(loadFirstOption)) {
        const QString loadFirstDir = parser.value(loadFirstOption).trimmed();
        if (!loadFirstDir.isEmpty()) {
            VolumePkg::setLoadFirstSegmentationDirectory(loadFirstDir.toStdString());
        }
    }

    CWindow aWin(cacheSizeGB, benchOptions);
    aWin.show();
    return QApplication::exec();
}
