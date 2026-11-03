#pragma once

// The viewer benchmarks' shared VC3D pieces, on top of the Fixture: the headless
// VC3D session (CState, ViewerManager and real CChunkedVolumeViewers on the
// offscreen Qt platform) and waiting for viewers to settle.
//
// The same sources build against origin/main (VC's ChunkCache pipeline) and
// against vc_reader trees; only bench_pipeline_*.cpp differ per tree.

#include "bench_fixture.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

class CChunkedVolumeViewer;
class CState;
class QApplication;
class QTemporaryDir;
class QWidget;
class QuadSurface;
class Volume;
class VCCollection;
class ViewerManager;

namespace vcbench {

// Process-wide setup for one tree's render pipeline (bench_pipeline_*.cpp).
namespace pipeline {
// Writes the pipeline's VC3D settings before any VC3D object reads them.
void writeSettings(Fixture& fixture, const std::string& iniPath);
// Called once the QApplication exists, before the CState.
void initProcess(Fixture& fixture);
std::unique_ptr<CState> makeState(Fixture& fixture);
// Opens the timed volume from Fixture::volumeSource: a local OME-Zarr folder, or
// (vc_reader trees only) its URL through the cache root VC3D's cache.json names.
std::shared_ptr<Volume> openVolume(const VolumeSource& source);
// Decoded cache occupancy and chunk traffic, for the report.
std::map<std::string, std::string> cacheSummary(CState& state, Volume& volume);
// Bytes fetched from remote sources so far (0 for local volumes).
std::uint64_t fetchedBytes(CState& state);
} // namespace pipeline

// The headless application: settings, Qt, CState, ViewerManager and the volume.
class Session {
public:
    Session(int& argc, char** argv, Fixture& fixture);
    ~Session();

    CState& state() { return *state_; }
    ViewerManager& manager() { return *manager_; }
    const std::shared_ptr<Volume>& volume() const { return volume_; }
    Fixture& fixture() { return fixture_; }
    const std::string& pipelineName() const { return fixture_.pipelineName(); }
    // The fixture's facts and the session's notes, for the report.
    std::map<std::string, std::string> config() const;
    void note(const std::string& key, const std::string& value) { config_[key] = value; }
    // --prefetch: views grow by --prefetch-margin so the cache covers what any
    // pipeline's level and fallback choices read around them.
    bool prefetching() const { return prefetch_; }
    int viewSize(int pixels) const { return int(pixels * margin_); }
    // Adds the pipeline's cache and traffic counters to the config. With --prefetch
    // and --expect-cached, warns when the run still fetched data.
    void noteCaches();
    // Remote volumes (prefetch, cold, partial): throws once more than fetch_max_gb
    // (100) has been fetched.
    void checkFetchBudget();
    // Makes the volume current: viewers created before this see it arrive as in VC3D.
    void showVolume();
    // Runs posted events (frame completions, data-ready kicks) without waiting.
    void processEvents();

private:
    Fixture& fixture_;
    std::unique_ptr<QTemporaryDir> configDir_;
    std::unique_ptr<QApplication> app_;
    std::unique_ptr<CState> state_;
    std::unique_ptr<VCCollection> points_;
    std::unique_ptr<ViewerManager> manager_;
    std::shared_ptr<Volume> volume_;
    std::map<std::string, std::string> config_;
    bool prefetch_ = false, remote_ = false, expectCached_ = false;
    double margin_ = 1.0;
    std::uint64_t fetchMaxBytes_ = 0;
    Clock::time_point lastBudgetCheck_{};
};

// A real CChunkedVolumeViewer of one registered surface, in a shown host widget.
class Viewer {
public:
    Viewer(Session& session, const std::string& surfaceName, bool annotationRole, int width, int height);
    ~Viewer();

    CChunkedVolumeViewer& get() { return *viewer_; }
    const std::string& surfaceName() const { return surfaceName_; }
    std::uint64_t framesCompleted() const { return frames_; }
    // Worker milliseconds of the frames completed since the last call.
    std::vector<double> takeWorkerMs();
    // Saves what the viewer shows (PNG), for checking that frames hold data.
    void save(const std::filesystem::path& file);

private:
    std::string surfaceName_;
    std::unique_ptr<QWidget> host_;
    CChunkedVolumeViewer* viewer_ = nullptr;
    std::uint64_t frames_ = 0;
    std::vector<double> workerMs_;
};

// What one navigation step cost, from the input to the last viewer settling.
struct StepTiming {
    double firstFrameMs = -1;  // first frame any viewer presented after the input
    double settledMs = -1;     // every viewer showing its final frame, no reads left
    std::uint64_t frames = 0;  // frames presented across the viewers
    std::vector<double> workerMs;
    bool timedOut = false;
};

// Processes events until every viewer is settled (twice in a row, with the
// event queue drained in between) or `timeoutMs` passes.
StepTiming waitSettled(Session& session, const std::vector<Viewer*>& viewers, Clock::time_point start,
                       double timeoutMs);

// Records a step under `group` (first_frame_ms, settled_ms, frames, worker_ms).
void record(Report& report, const std::string& group, const StepTiming& step);

} // namespace vcbench
