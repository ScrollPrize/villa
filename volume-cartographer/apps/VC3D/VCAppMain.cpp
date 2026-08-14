// Override global new/delete with mimalloc if available. Must be included
// in exactly ONE translation unit; the linker picks up the override symbols.
#if defined(VC_HAVE_MIMALLOC)
#include <mimalloc-new-delete.h>
#include <mimalloc.h>
#endif

#include <qapplication.h>
#include <QAbstractSpinBox>
#include <QCommandLineParser>
#include <QComboBox>
#include <QEvent>
#include <QImageReader>

#include "CWindow.hpp"
#include "agent_bridge/AgentBridgeServer.hpp"
#include "VCSettings.hpp"
#include "vc/core/Version.hpp"
#include <QSettings>
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/CrashHandler.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <thread>
#include <omp.h>
#include <blosc.h>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/resource.h>
#endif
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

// POSIX setenv with an MSVCRT/UCRT fallback (Windows has no setenv).
static void vcSetEnv(const char* name, const char* value, int overwrite)
{
#if defined(_WIN32)
    if (!overwrite && std::getenv(name)) return;
    _putenv_s(name, value);
#else
    ::setenv(name, value, overwrite);
#endif
}

// Runs before main() AND before all shared-library constructors.
// .preinit_array is processed by the dynamic linker before any .init_array,
// so env vars are visible when OpenBLAS/OpenMP create their thread pools.
static void setThreadPoliciesEarly()
{
    // Force passive wait policy so OpenMP threads sleep instead of
    // spin-waiting with sched_yield.  overwrite=1 is intentional —
    // spin-waiting on 500+ OMP threads kills the machine.
    vcSetEnv("OMP_WAIT_POLICY", "passive", 1);
    vcSetEnv("OMP_NUM_THREADS", "1", 0);       // limit OpenMP parallelism
    vcSetEnv("KMP_BLOCKTIME", "0", 1);         // LLVM/Intel OpenMP: sleep immediately
    vcSetEnv("KMP_AFFINITY", "disabled", 0);   // skip sched_setaffinity per fork/join
    vcSetEnv("OPENBLAS_NUM_THREADS", "1", 0);
    vcSetEnv("GOTO_NUM_THREADS", "1", 0);      // legacy name for OpenBLAS
    vcSetEnv("MKL_NUM_THREADS", "1", 0);       // Intel MKL
}
#ifdef __linux__
__attribute__((section(".preinit_array"), used))
static auto preinitFn = &setThreadPoliciesEarly;
#endif

static bool hasCliFlag(int argc, char* argv[], const char* flag)
{
    for (int i = 1; i < argc; ++i) {
        if (argv[i] && std::strcmp(argv[i], flag) == 0)
            return true;
    }
    return false;
}

static std::optional<vc::render::ChunkCacheService::AdaptiveDownloadState>
loadAdaptiveDownloadState()
{
    using namespace vc3d::settings::perf;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    if (settings.value(REMOTE_DOWNLOAD_STATE_VERSION, 0).toInt() !=
        REMOTE_DOWNLOAD_STATE_VERSION_CURRENT) {
        return std::nullopt;
    }

    bool admissionOk = false;
    const auto admission = settings.value(REMOTE_DOWNLOAD_SETTLED_ADMISSION)
        .toULongLong(&admissionOk);
    if (!admissionOk || admission == 0 ||
        admission > std::numeric_limits<std::size_t>::max())
        return std::nullopt;

    vc::render::ChunkCacheService::AdaptiveDownloadState state;
    state.settledAdmissionLimit = static_cast<std::size_t>(admission);
    state.longTermBytesPerSecond = settings.value(
        REMOTE_DOWNLOAD_LONG_TERM_BYTES_PER_SECOND, 0.0).toDouble();
    const auto maximumSaturated = settings.value(
        REMOTE_DOWNLOAD_MAX_SATURATED_PARALLELISM, 0).toULongLong();
    state.maximumSaturatedParallelism = static_cast<std::size_t>(std::min(
        maximumSaturated,
        static_cast<qulonglong>(std::numeric_limits<std::size_t>::max())));
    state.saturatedBytesPerSecondPerWorker = settings.value(
        REMOTE_DOWNLOAD_SATURATED_BYTES_PER_SECOND_PER_WORKER, 0.0).toDouble();
    if (!std::isfinite(state.longTermBytesPerSecond) ||
        state.longTermBytesPerSecond < 0.0) {
        state.longTermBytesPerSecond = 0.0;
    }
    if (!std::isfinite(state.saturatedBytesPerSecondPerWorker) ||
        state.saturatedBytesPerSecondPerWorker <= 0.0) {
        state.maximumSaturatedParallelism = 0;
        state.saturatedBytesPerSecondPerWorker = 0.0;
    }
    return state;
}

static void saveAdaptiveDownloadState(
    const std::optional<vc::render::ChunkCacheService::AdaptiveDownloadState>& state)
{
    if (!state || state->settledAdmissionLimit == 0)
        return;
    using namespace vc3d::settings::perf;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(REMOTE_DOWNLOAD_STATE_VERSION,
                      REMOTE_DOWNLOAD_STATE_VERSION_CURRENT);
    settings.setValue(REMOTE_DOWNLOAD_SETTLED_ADMISSION,
                      static_cast<qulonglong>(state->settledAdmissionLimit));
    settings.setValue(REMOTE_DOWNLOAD_LONG_TERM_BYTES_PER_SECOND,
                      state->longTermBytesPerSecond);
    settings.setValue(
        REMOTE_DOWNLOAD_MAX_SATURATED_PARALLELISM,
        static_cast<qulonglong>(state->maximumSaturatedParallelism));
    settings.setValue(REMOTE_DOWNLOAD_SATURATED_BYTES_PER_SECOND_PER_WORKER,
                      state->saturatedBytesPerSecondPerWorker);
    settings.sync();
}

class WheelFocusFilter final : public QObject
{
protected:
    bool eventFilter(QObject* watched, QEvent* event) override
    {
        // Polish fires once per widget as it is styled, so this catches every
        // combo/spinner app-wide, including ones created later. StrongFocus
        // (vs the default WheelFocus) stops the wheel from focusing the
        // widget as the cursor passes over it.
        if (event->type() == QEvent::Polish) {
            auto* widget = qobject_cast<QWidget*>(watched);
            if (widget && (qobject_cast<QAbstractSpinBox*>(widget) ||
                           qobject_cast<QComboBox*>(widget))) {
                widget->setFocusPolicy(Qt::StrongFocus);
            }
            return QObject::eventFilter(watched, event);
        }

        if (event->type() != QEvent::Wheel)
            return QObject::eventFilter(watched, event);

        auto* widget = qobject_cast<QWidget*>(watched);
        while (widget) {
            if (auto* spinBox = qobject_cast<QAbstractSpinBox*>(widget)) {
                if (!spinBox->hasFocus()) {
                    event->ignore();
                    return true;
                }
                break;
            }
            if (auto* comboBox = qobject_cast<QComboBox*>(widget)) {
                if (!comboBox->hasFocus()) {
                    event->ignore();
                    return true;
                }
                break;
            }
            widget = widget->parentWidget();
        }

        return QObject::eventFilter(watched, event);
    }
};

#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
auto main(int argc, char* argv[]) -> int
{
#ifdef _WIN32
    // GUI-subsystem exe: stdout/stderr are detached by default. When launched
    // from a terminal, reattach them so logs and --version/--help output are
    // visible; double-click launches still get no console window.
    if (::AttachConsole(ATTACH_PARENT_PROCESS)) {
        std::freopen("CONOUT$", "w", stdout);
        std::freopen("CONOUT$", "w", stderr);
    }
#endif

    vc::crash::install();

#ifndef __linux__
    // On non-Linux, preinit_array is unavailable so set env vars at start of main.
    // This may be too late for some libraries that init in static constructors.
    setThreadPoliciesEarly();
#endif

#if defined(__GLIBC__) && !defined(VC_HAVE_MIMALLOC)
    // Tune glibc's malloc to give freed pages back to the OS more aggressively.
    // Bound per-thread arenas before VC3D creates its worker pools. Smaller
    // volume chunks increasingly use arena allocations rather than independent
    // mmaps; allowing glibc's default (up to 8 * CPU count) let OpenMP/thread
    // pool multiplication retain tens of GiB in 128-MiB arena mappings even
    // after the cache references were released.
    ::mallopt(M_ARENA_MAX, 4);
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

    // Qt selects its platform plugin while QApplication is constructed, before
    // QCommandLineParser runs. Pre-scan this replay flag so the flag alone is
    // enough for headless offscreen benchmarking.
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM") &&
        hasCliFlag(argc, argv, "--replay-offscreen-4k")) {
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

    QApplication app(argc, argv);
    // Wide surface-aligned Spiral overlays can legitimately exceed Qt's
    // 128-MiB default image-I/O allocation limit. Set the Qt runtime value
    // after QApplication construction because Qt may cache the environment
    // setting before VC3D's pre-main hook runs. Preserve an explicit user
    // override and retain a finite allocation guard.
    if (qEnvironmentVariableIsEmpty("QT_IMAGEIO_MAXALLOC")) {
        QImageReader::setAllocationLimit(512);
    }
    WheelFocusFilter wheelFocusFilter;
    app.installEventFilter(&wheelFocusFilter);
    QApplication::setOrganizationName("Vesuvius Challenge");
    QApplication::setApplicationName("VC3D");
    QApplication::setWindowIcon(QIcon(":/images/logo.png"));
    QApplication::setApplicationVersion(QString::fromStdString(ProjectInfo::VersionString()));

    // Handle this before constructing the main window. QCommandLineParser's
    // built-in version option normally exits from process(), but keeping the
    // probe explicit makes the packaged GUI-subsystem executable a reliable
    // non-interactive smoke test on Windows.
    if (hasCliFlag(argc, argv, "--version")) {
        std::cout << QApplication::applicationName().toStdString() << ' '
                  << QApplication::applicationVersion().toStdString() << std::endl;
        std::cout.flush();
        std::cerr.flush();
        std::_Exit(0);
    }

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

    QCommandLineOption volumePackageOption(
        "volpkg",
        "Open a volume package at startup.",
        "path");
    parser.addOption(volumePackageOption);

    QCommandLineOption debugOption(
        "debug",
        "Enable verbose diagnostic logging while loading surfaces.");
    parser.addOption(debugOption);

    QCommandLineOption profileOption(
        "profile",
        "Enable VC3D render profiling logs.");
    parser.addOption(profileOption);

    QCommandLineOption debugDownloadQueueOption(
        "debug-download-queue",
        "Overlay currently downloading remote chunks in every slice view.");
    parser.addOption(debugDownloadQueueOption);

    QCommandLineOption recordOption(
        "record",
        "Record a navigation camera-state timeline to the given JSON file.",
        "file");
    parser.addOption(recordOption);

    QCommandLineOption replayOption(
        "replay",
        "Replay a recorded navigation timeline (implies --profile), then exit.",
        "file");
    parser.addOption(replayOption);

    QCommandLineOption replayWarmOption(
        "replay-warm",
        "With --replay, run a discarded warm-up pass before the timed pass.");
    parser.addOption(replayWarmOption);

    QCommandLineOption replayOffscreen4kOption(
        "replay-offscreen-4k",
        "With --replay, force the replay viewport to 3840x2160.");
    parser.addOption(replayOffscreen4kOption);

    QCommandLineOption replayLimitOption(
        "replay-limit",
        "With --replay, replay only the first N recorded keyframes (0 = all).",
        "frames",
        "0");
    parser.addOption(replayLimitOption);

    QCommandLineOption replaySkipChunkCompleteOption(
        "replay-skip-chunk-complete",
        "With --replay, advance after the first full render instead of waiting for chunk-complete quiet settle.");
    parser.addOption(replaySkipChunkCompleteOption);

    QCommandLineOption replaySkipFastRenderOption(
        "replay-skip-fast-render",
        "With --replay, skip any fast-render phase and submit the full render directly.");
    parser.addOption(replaySkipFastRenderOption);

    QCommandLineOption replayTimedProfileOption(
        "replay-timed-profile",
        "With --replay, switch frames on a fixed timer and print one row per paint.");
    parser.addOption(replayTimedProfileOption);

    QCommandLineOption replayTimedProfilePeriodOption(
        "replay-timed-profile-period-ms",
        "With --replay-timed-profile, milliseconds between frame switches.",
        "ms",
        "200");
    parser.addOption(replayTimedProfilePeriodOption);

    QCommandLineOption agentBridgeOption(
        "agent-bridge",
        "Enable the agent bridge (JSON-RPC over a local socket) on the default "
        "socket name vc3d-agent-<pid>.");
    parser.addOption(agentBridgeOption);

    QCommandLineOption agentBridgeNameOption(
        "agent-bridge-name",
        "Enable the agent bridge on an explicit QLocalServer name (implies "
        "--agent-bridge).",
        "name");
    parser.addOption(agentBridgeNameOption);

    parser.process(app);

    if (parser.isSet(debugOption)) {
        SetDebugLoggingEnabled(true);
        SetLogLevel("debug");
    }

    RenderBenchOptions benchOptions;
    benchOptions.recordPath = parser.value(recordOption).trimmed();
    benchOptions.replayPath = parser.value(replayOption).trimmed();
    benchOptions.replayWarm = parser.isSet(replayWarmOption);
    benchOptions.replayOffscreen4k = parser.isSet(replayOffscreen4kOption);
    benchOptions.replaySkipChunkComplete = parser.isSet(replaySkipChunkCompleteOption);
    benchOptions.replaySkipFastRender = parser.isSet(replaySkipFastRenderOption);
    benchOptions.replayTimedProfile = parser.isSet(replayTimedProfileOption);
    benchOptions.debugDownloadQueue = parser.isSet(debugDownloadQueueOption);
    bool limitOk = false;
    const int replayLimit = parser.value(replayLimitOption).toInt(&limitOk);
    benchOptions.replayLimit = (limitOk && replayLimit > 0) ? replayLimit : 0;
    bool periodOk = false;
    const int timedPeriod = parser.value(replayTimedProfilePeriodOption).toInt(&periodOk);
    benchOptions.replayTimedProfilePeriodMs = (periodOk && timedPeriod > 0) ? timedPeriod : 200;

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

        // Remote-volume disk-cache compression. Applied as a process-wide
        // default so every ChunkCache picks it up, including the core-created
        // ones used by blocking readers.
        vc::render::ChunkCache::setPersistentCompressionDefault(
            settings.value(perf::REMOTE_CACHE_COMPRESSION,
                           perf::REMOTE_CACHE_COMPRESSION_DEFAULT).toBool());
        vc::render::ChunkCache::setPersistentQuantizationDefault(
            settings.value(perf::REMOTE_CACHE_QUANTIZATION,
                           perf::REMOTE_CACHE_QUANTIZATION_DEFAULT).toInt());
        constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
        vc::render::PersistentZarrCacheBudget::Limits limits;
        const auto maximumGiB = settings.value(
            perf::REMOTE_CACHE_MAX_GIB, perf::REMOTE_CACHE_MAX_GIB_DEFAULT).toULongLong();
        if (maximumGiB > 0)
            limits.maximumBytes = maximumGiB * gib;
        limits.minimumFreeBytes = settings.value(
            perf::REMOTE_CACHE_MIN_FREE_GIB,
            perf::REMOTE_CACHE_MIN_FREE_GIB_DEFAULT).toULongLong() * gib;
        const auto cacheRoot = vc3d::remoteCachePath(
            settings.value(viewer::REMOTE_CACHE_DIR).toString()).toStdString();
        vc::render::PersistentZarrCacheBudget::configure(cacheRoot, limits);
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

    int rc = 0;
    std::shared_ptr<vc::render::ChunkCacheService> chunkCacheService;
    {
        const std::size_t cacheSizeBytes =
            cacheSizeGB * 1024ULL * 1024ULL * 1024ULL;
        chunkCacheService = std::make_shared<vc::render::ChunkCacheService>(
            cacheSizeBytes,
            std::shared_ptr<vc::render::DecodedChunkCacheBudget>{},
            loadAdaptiveDownloadState());
        CWindow aWin(cacheSizeGB, benchOptions, chunkCacheService);

        if (parser.isSet(volumePackageOption)) {
            QString errorMessage;
            if (!aWin.openVolumePackage(parser.value(volumePackageOption).trimmed(),
                                        false,
                                        &errorMessage)) {
                std::cerr << "Error: " << errorMessage.toStdString() << std::endl;
                std::cout.flush();
                std::cerr.flush();
                std::_Exit(2);
            }
        }

        // Agent bridge (opt-in, off by default). Constructed only when a bridge
        // flag is present, so normal runs pay zero cost and open no socket.
        std::unique_ptr<AgentBridgeServer> agentBridge;
        const bool bridgeRequested =
            parser.isSet(agentBridgeOption) || parser.isSet(agentBridgeNameOption);
        if (bridgeRequested) {
            QString bridgeName = parser.value(agentBridgeNameOption).trimmed();
            if (bridgeName.isEmpty()) {
                bridgeName = QStringLiteral("vc3d-agent-%1")
                                 .arg(QCoreApplication::applicationPid());
            }
            agentBridge = std::make_unique<AgentBridgeServer>(&aWin);
            if (!agentBridge->listen(bridgeName)) {
                std::cerr << "Error: agent bridge failed to listen on '"
                          << bridgeName.toStdString() << "'." << std::endl;
                std::cout.flush();
                std::cerr.flush();
                // Exit like the std::_Exit path below rather than returning: a plain
                // return runs DSO finalizers (gnutls/libtasn1 free through mimalloc
                // post-teardown) and segfaults in _dl_fini. Nothing has loaded yet.
                std::_Exit(2);
            }
            std::cout << "VC3D-AGENT-BRIDGE: listening name="
                      << agentBridge->serverName().toStdString()
                      << " path=" << agentBridge->fullServerName().toStdString()
                      << std::endl;
            std::cout.flush();
        }

        aWin.show();
        rc = QApplication::exec();
    }
    saveAdaptiveDownloadState(chunkCacheService->adaptiveDownloadState());
    chunkCacheService.reset();
    // Skip DSO finalizers: gnutls/libtasn1 destructors free through mimalloc after
    // its own teardown, segfaulting in _dl_fini on every otherwise-clean exit.
    // CWindow (above scope) has already run its real cleanup.
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(rc);
}
