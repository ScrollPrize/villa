#include "vc/tracer/SheetGrowth.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SheetNormalField.hpp"
#include "vc/core/util/SheetSeed.hpp"
#include "vc/core/util/SurfaceArea.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace {
// The heartbeat runs independently of CT reads, so a slow chunk fetch still
// shows the active stage and elapsed time. All output is flushed for QProcess.
class SheetGrowthProgress
{
public:
    explicit SheetGrowthProgress(std::chrono::steady_clock::time_point started)
        : started_(started), worker_([this](std::stop_token stop) {
            std::unique_lock lock(mutex_);
            while (!stop.stop_requested()) {
                if (wake_.wait_for(lock, std::chrono::seconds(1), [&] { return stop.stop_requested(); }))
                    break;
                print();
            }
        }) {}

    ~SheetGrowthProgress()
    {
        worker_.request_stop();
        wake_.notify_all();
        worker_.join();
    }

    void update(vc::sheet::SeedStage stage, std::size_t completed, std::size_t total)
    {
        using Stage = vc::sheet::SeedStage;
        const char* label = "";
        switch (stage) {
        case Stage::SeedNormal: label = "Estimating seed normal"; break;
        case Stage::CentralColumn: label = "Growing central column"; break;
        case Stage::RowDirections: label = "Preparing row directions"; break;
        case Stage::Rows: label = "Growing rows"; break;
        }
        std::lock_guard lock(mutex_);
        label_ = label;
        unit_ = stage == Stage::CentralColumn || stage == Stage::Rows ? "vertices" : "points";
        completed_ = completed;
        total_ = total;
        if (completed == 0 || completed == total) print();
    }

    void status(const char* label)
    {
        std::lock_guard lock(mutex_);
        label_ = label;
        total_ = 0;
        print();
    }

private:
    void print() const // mutex_ is held
    {
        if (!label_) return;
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_).count();
        std::ostringstream line;
        line << "[Sheet growth] " << label_;
        if (total_ > 0)
            line << ": " << completed_ << "/" << total_ << " " << unit_ << " ("
                 << std::fixed << std::setprecision(1) << 100.0 * completed_ / total_ << "% of stage)";
        line << "; elapsed " << std::fixed << std::setprecision(1) << elapsed << " s";
        std::cout << line.str() << std::endl;
    }

    std::chrono::steady_clock::time_point started_;
    std::mutex mutex_;
    std::condition_variable wake_;
    const char* label_{nullptr};
    const char* unit_{"points"};
    std::size_t completed_{0};
    std::size_t total_{0};
    std::jthread worker_;
};
} // namespace

QuadSurface* growSheetPatch(Volume& volume, const cv::Vec3f& seed,
                            const utils::Json& params, float voxelSize,
                            const utils::Json& metadata)
{
    const auto started = std::chrono::steady_clock::now();
    vc::sheet::SeedOptions options;
    options.generations = params.value("generations", 20);
    options.spacing = params.value("step_size", 20.0);
    const int workers = params.value("thread_limit", 1);
    if (workers < 1 || workers > 256)
        throw std::invalid_argument("Sheet growth thread_limit must be between 1 and 256.");
    options.workers = static_cast<unsigned>(workers);
    options.walk.maxStep = params.value("walk_step_size", 2.0);
    const auto method = params.value("growth_method", std::string("default"));
    if (method != "projection" && method != "streamline")
        throw std::invalid_argument("Sheet growth requires projection or streamline.");
    options.walk.mode = method == "projection" ? vc::sheet::SheetWalkMode::Projection
                                               : vc::sheet::SheetWalkMode::Streamline;
    const auto integrator = params.value("walk_integrator", std::string("midpoint"));
    if (integrator == "midpoint") options.walk.integrator = vc::sheet::SheetIntegrator::Midpoint;
    else if (integrator == "heun") options.walk.integrator = vc::sheet::SheetIntegrator::Heun;
    else if (integrator == "rk4") options.walk.integrator = vc::sheet::SheetIntegrator::RK4;
    else throw std::invalid_argument("walk_integrator must be midpoint, heun or rk4.");

    std::cout << "[Sheet growth] Starting " << method
              << (method == "streamline" ? " (" + integrator + ")" : "")
              << ": " << options.generations << " cells per side, spacing " << options.spacing
              << " voxels, max integration step " << options.walk.maxStep
              << " voxels, " << workers << " workers" << std::endl;
    SheetGrowthProgress progress(started);

    // The field cannot outlive this call; Volume remains owned by the caller.
    SheetNormalField field(std::shared_ptr<Volume>(&volume, [](Volume*) {}));
    const auto shape = volume.shape(0);
    const auto checkBounds = [&](const cv::Vec3f& point) {
        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(point[axis]) || point[axis] < 0 || point[axis] > shape[2 - axis] - 1)
                throw std::runtime_error("Sheet growth left the CT volume; reduce generations or mesh spacing.");
    };
    // Automatic seeds require reliable normals. Manual growth retains its
    // existing straight-line fallback, but no unsupported guess is saved here.
    const vc::sheet::SheetNormalFn normal = [&](const cv::Vec3f& point) {
        checkBounds(point);
        const auto n = field.normalAt(point, options.walk.continuousNormals());
        if (!n || !std::isfinite(cv::norm(*n)) || cv::norm(*n) <= 0.5)
            throw std::runtime_error("No usable CT sheet normal during seed growth; reduce the patch size or choose another seed.");
        return n;
    };
    auto points = vc::sheet::growSheetSeed(seed, normal, options,
        [&](vc::sheet::SeedStage stage, std::size_t completed, std::size_t total) {
            progress.update(stage, completed, total);
        });
    progress.status("Preparing surface for saving");
    for (const auto& point : points) checkBounds(point);
    auto surface = std::make_unique<QuadSurface>(points,
        cv::Vec2f(1.0f / options.spacing, 1.0f / options.spacing));
    cv::Mat_<uint16_t> generations(points.size());
    const int center = options.generations;
    for (int row = 0; row < points.rows; ++row)
        for (int col = 0; col < points.cols; ++col)
            generations(row, col) = static_cast<uint16_t>(std::max(1,
                std::max(std::abs(row - center), std::abs(col - center))));
    surface->setChannel("generations", generations);
    surface->meta = utils::Json::parse(metadata.dump());
    surface->meta["growth_method"] = method;
    surface->meta["max_gen"] = options.generations;
    auto seedJson = utils::Json::array();
    for (int axis = 0; axis < 3; ++axis) seedJson.push_back(seed[axis]);
    surface->meta["seed"] = seedJson;
    surface->meta["elapsed_time_s"] = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    vc::surface::storeAreaMeta(surface->meta,
        vc::surface::computeSurfaceAreaVox2(*surface), static_cast<double>(voxelSize));
    progress.status("Surface ready; saving output");
    return surface.release();
}
