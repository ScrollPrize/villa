#include "bench_session.hpp"

#include "CState.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QApplication>
#include <QSettings>
#include <QTemporaryDir>
#include <QThread>
#include <QVBoxLayout>
#include <QPixmap>
#include <QWidget>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace vcbench {

// ---------------------------------------------------------------------------
// Session

Session::Session(int& argc, char** argv, Fixture& fixture) : fixture_(fixture)
{
    // Isolated settings: nothing from the user's VC3D.ini leaks into a run.
    configDir_ = std::make_unique<QTemporaryDir>();
    if (!configDir_->isValid())
        throw std::runtime_error("cannot create a settings folder");
    qputenv("VC3D_CONFIG_DIR", configDir_->path().toUtf8());
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM"))
        qputenv("QT_QPA_PLATFORM", "offscreen");
    const auto ini = vc3d::settingsFilePath();
    const int maxResolution = fixture.integer("max_resolution", 0);
    {
        QSettings settings(ini, QSettings::IniFormat);
        // VC3D's "max displayed resolution": 0 (the default) lets views reach level 0.
        settings.setValue(vc3d::settings::viewer::MAX_DISPLAYED_RESOLUTION, maxResolution);
        settings.sync();
    }
    pipeline::writeSettings(fixture, ini.toStdString());

    app_ = std::make_unique<QApplication>(argc, argv);
    pipeline::initProcess(fixture);
    state_ = pipeline::makeState(fixture);
    points_ = std::make_unique<VCCollection>();
    manager_ = std::make_unique<ViewerManager>(state_.get(), points_.get());

    prefetch_ = fixture.prefetching();
    margin_ = prefetch_ ? fixture.num("prefetch_margin", 1.5) : 1.0;
    fetchMaxBytes_ = std::uint64_t(fixture.num("fetch_max_gb", 100) * double(1ull << 30));
    expectCached_ = prefetch_ && fixture.flag("expect_cached");
    // Before any VC3D object reads VC3D_CONFIG_DIR's cache.json.
    const auto source = fixture.volumeSource(configDir_->path().toStdString());
    remote_ = source.remote;
    const auto start = Clock::now();
    volume_ = pipeline::openVolume(source);
    if (prefetch_)
        config_["prefetch_margin"] = std::to_string(margin_);
    config_["max_resolution"] = std::to_string(maxResolution);
    config_["volume_open_ms"] = std::to_string(msSince(start));
}

std::map<std::string, std::string> Session::config() const
{
    auto out = fixture_.config();
    for (const auto& [key, value] : config_)
        out[key] = value;
    return out;
}

Session::~Session()
{
    if (state_)
        state_->setCurrentVolume(nullptr);
    manager_.reset();
    points_.reset();
    state_.reset();
    volume_.reset();
    app_.reset();
}

void Session::showVolume()
{
    state_->setCurrentVolume(volume_);
}

void Session::noteCaches()
{
    for (const auto& [key, value] : pipeline::cacheSummary(*state_, *volume_))
        config_[key] = value;
    if (expectCached_) {
        if (const auto fetched = pipeline::fetchedBytes(*state_); fetched > 0)
            std::fprintf(stderr,
                         "warning: %s/%s still fetched %llu MB: the cache does not hold everything it reads yet\n",
                         fixture_.benchmark().c_str(), fixture_.caseName().c_str(),
                         static_cast<unsigned long long>(fetched >> 20));
    }
}

void Session::checkFetchBudget()
{
    if (!remote_ || Clock::now() - lastBudgetCheck_ < std::chrono::milliseconds(100))
        return;
    lastBudgetCheck_ = Clock::now();
    const auto fetched = pipeline::fetchedBytes(*state_);
    if (fetched > fetchMaxBytes_)
        throw std::runtime_error("stopped: fetched " + std::to_string(fetched >> 20) +
                                 " MB, over fetch_max_gb");
}

void Session::processEvents()
{
    QCoreApplication::sendPostedEvents();
    QCoreApplication::processEvents(QEventLoop::AllEvents);
}

// ---------------------------------------------------------------------------
// Viewer

Viewer::Viewer(Session& session, const std::string& surfaceName, bool annotationRole, int width, int height)
    : surfaceName_(surfaceName)
{
    host_ = std::make_unique<QWidget>();
    auto* layout = new QVBoxLayout(host_.get());
    layout->setContentsMargins(0, 0, 0, 0);
    auto* base = session.manager().createViewerInWidget(
        surfaceName, host_.get(),
        annotationRole ? ViewerManager::ViewerRole::Annotation : ViewerManager::ViewerRole::Standard);
    viewer_ = dynamic_cast<CChunkedVolumeViewer*>(base);
    if (!viewer_)
        throw std::runtime_error("ViewerManager did not create a CChunkedVolumeViewer");
    layout->addWidget(viewer_);
    QObject::connect(viewer_, &CChunkedVolumeViewer::renderFrameCompleted, viewer_,
                     [this](std::uint64_t, double workerMs) {
                         ++frames_;
                         workerMs_.push_back(workerMs);
                     });
    host_->resize(width, height);
    host_->show();
}

Viewer::~Viewer()
{
    host_.reset();
}

void Viewer::save(const std::filesystem::path& file)
{
    host_->grab().save(QString::fromStdString(file.string()));
}

std::vector<double> Viewer::takeWorkerMs()
{
    std::vector<double> out;
    out.swap(workerMs_);
    return out;
}

StepTiming waitSettled(Session& session, const std::vector<Viewer*>& viewers, Clock::time_point start,
                       double timeoutMs)
{
    StepTiming step;
    std::vector<std::uint64_t> framesAtStart;
    for (auto* v : viewers)
        framesAtStart.push_back(v->framesCompleted());
    const auto framesNow = [&]() {
        std::uint64_t frames = 0;
        for (std::size_t i = 0; i < viewers.size(); ++i)
            frames += viewers[i]->framesCompleted() - framesAtStart[i];
        return frames;
    };
    const auto allSettled = [&]() {
        return std::all_of(viewers.begin(), viewers.end(), [](Viewer* v) { return v->get().renderSettled(); });
    };
    for (;;) {
        session.processEvents();
        const auto frames = framesNow();
        if (frames > 0 && step.firstFrameMs < 0)
            step.firstFrameMs = msSince(start);
        if (allSettled()) {
            // A data-ready kick or a frame completion can still be queued: drain the
            // queue and require the viewers to stay settled with no new frame.
            session.processEvents();
            if (allSettled() && framesNow() == frames) {
                step.settledMs = msSince(start);
                break;
            }
        }
        if (msSince(start) > timeoutMs) {
            step.timedOut = true;
            step.settledMs = msSince(start);
            break;
        }
        session.checkFetchBudget();
        QThread::usleep(200);
    }
    step.frames = framesNow();
    for (auto* v : viewers) {
        auto ms = v->takeWorkerMs();
        step.workerMs.insert(step.workerMs.end(), ms.begin(), ms.end());
    }
    return step;
}

void record(Report& report, const std::string& group, const StepTiming& step)
{
    if (step.firstFrameMs >= 0)
        report.add(group, "first_frame_ms", step.firstFrameMs);
    report.add(group, "settled_ms", step.settledMs);
    report.add(group, "frames", double(step.frames));
    for (const double ms : step.workerMs)
        report.add(group, "worker_ms", ms);
    if (step.timedOut) {
        report.add(group, "timeouts", 1);
        std::fprintf(stderr, "warning: %s did not settle within the timeout\n", group.c_str());
    }
}

} // namespace vcbench
