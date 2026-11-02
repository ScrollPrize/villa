#pragma once

// Shared pieces of the workflow benchmarks: argument parsing, the headless
// VC3D session (CState, ViewerManager and real CChunkedVolumeViewers on the
// offscreen Qt platform), waiting for viewers to settle, and statistics.
//
// The same sources build against origin/main (VC's ChunkCache pipeline) and
// against vc_reader trees; only bench_pipeline_*.cpp differ per tree.

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

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

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start);

// --key value / --flag arguments; unknown keys are an error at the end of setup.
class Args {
public:
    Args(int argc, char** argv);
    std::string str(const std::string& key, const std::string& fallback = {}) const;
    double num(const std::string& key, double fallback) const;
    bool flag(const std::string& key) const;
    // Fails when an argument was given that no str/num/flag call asked for.
    void rejectUnused() const;
    const std::string& program() const { return program_; }

private:
    std::string program_;
    std::map<std::string, std::string> values_;
    mutable std::map<std::string, bool> used_;
};

struct Summary {
    std::size_t n = 0;
    double mean = 0, min = 0, p50 = 0, p95 = 0, max = 0, total = 0;
};
Summary summarize(std::vector<double> values);

// One row of results: a scenario step type with its samples (ms).
class Report {
public:
    Report(std::string benchmark, std::string pipeline, std::map<std::string, std::string> config);
    void add(const std::string& group, const std::string& metric, double value);
    void setConfig(std::map<std::string, std::string> config) { config_ = std::move(config); }
    // Prints a table and, when `jsonPath` is set, appends one JSON line per metric.
    void finish(const std::string& jsonPath) const;

private:
    std::string benchmark_, pipeline_;
    std::map<std::string, std::string> config_;
    std::vector<std::string> order_;
    std::map<std::string, std::vector<double>> samples_;  // "group|metric"
};

// Process-wide setup for one tree's render pipeline (bench_pipeline_*.cpp).
namespace pipeline {
// "chunkcache" on origin/main; "vc_reader-cpu" / "vc_reader-cuda" with vc_reader.
std::string name(const Args& args);
// Writes the pipeline's VC3D settings before any VC3D object reads them.
void writeSettings(const Args& args, const std::string& iniPath);
// Called once the QApplication exists, before the CState.
void initProcess(const Args& args);
std::unique_ptr<CState> makeState(const Args& args);
// Opens the timed volume: the local OME-Zarr folder. With --prefetch (vc_reader
// trees only) the same volume is opened as its s3:// source with the cache folder's
// root as VC3D's cache root, so replaying a scenario fills the cache it reads.
std::shared_ptr<Volume> openVolume(const Args& args, const std::string& localPath, const std::string& configDir);
// Decoded cache occupancy and chunk traffic, for the report.
std::map<std::string, std::string> cacheSummary(CState& state, Volume& volume);
// Bytes fetched from remote sources so far (0 for local volumes).
std::uint64_t fetchedBytes(CState& state);
} // namespace pipeline

// The headless application: settings, Qt, CState, ViewerManager and the volume.
class Session {
public:
    Session(int& argc, char** argv, Args& args);
    ~Session();

    CState& state() { return *state_; }
    ViewerManager& manager() { return *manager_; }
    const std::shared_ptr<Volume>& volume() const { return volume_; }
    const std::string& pipelineName() const { return pipeline_; }
    std::map<std::string, std::string> config() const { return config_; }
    void note(const std::string& key, const std::string& value) { config_[key] = value; }
    // --prefetch: views grow by --prefetch-margin so the cache covers what any
    // pipeline's level and fallback choices read around them.
    bool prefetching() const { return prefetch_; }
    int viewSize(int pixels) const { return int(pixels * margin_); }
    // Adds the pipeline's cache and traffic counters to the config.
    void noteCaches();
    // --prefetch: throws once more than --prefetch-max-gb (100) has been fetched.
    void checkPrefetchBudget();
    // Makes the volume current: viewers created before this see it arrive as in VC3D.
    void showVolume();
    // Runs posted events (frame completions, data-ready kicks) without waiting.
    void processEvents();

private:
    std::unique_ptr<QTemporaryDir> configDir_;
    std::unique_ptr<QApplication> app_;
    std::unique_ptr<CState> state_;
    std::unique_ptr<VCCollection> points_;
    std::unique_ptr<ViewerManager> manager_;
    std::shared_ptr<Volume> volume_;
    std::string pipeline_;
    std::map<std::string, std::string> config_;
    bool prefetch_ = false;
    double margin_ = 1.0;
    std::uint64_t prefetchMaxBytes_ = 0;
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

// Loads a tifxyz and scales its vertices into the volume's level-0 voxels.
std::shared_ptr<QuadSurface> loadSurface(const std::filesystem::path& tifxyz, float coordinateScale);
// A valid vertex near the grid centre (volume coordinates).
cv::Vec3f surfaceCenter(QuadSurface& surface);
// Valid vertices sampled evenly over the grid, in a fixed order.
std::vector<cv::Vec3f> surfaceSamplePoints(QuadSurface& surface, int count);

} // namespace vcbench
