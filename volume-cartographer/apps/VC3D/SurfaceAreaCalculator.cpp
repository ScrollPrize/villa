#include "SurfaceAreaCalculator.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/DateTime.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <filesystem>
#include <omp.h>

namespace {

static constexpr bool kDeactivateWhenZero = true;
static constexpr double kTauDeactivate = 0.50;

static inline bool isFinite3(const cv::Vec3d& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

static inline double tri_area3D(const cv::Vec3d& a, const cv::Vec3d& b, const cv::Vec3d& c)
{
    return 0.5 * cv::norm((b - a).cross(c - a));
}

static int choose_largest_page(const std::vector<cv::Mat>& pages)
{
    int bestIdx = -1;
    size_t bestPix = 0;
    for (int i = 0; i < (int)pages.size(); ++i) {
        const size_t pix = (size_t)pages[i].rows * (size_t)pages[i].cols;
        if (pix > bestPix) { bestPix = pix; bestIdx = i; }
    }
    return bestIdx;
}

static void binarize_mask(const cv::Mat& srcAnyDepth, cv::Mat1b& mask01)
{
    cv::Mat m;
    if (srcAnyDepth.channels() != 1) {
        cv::Mat gray;
        cv::cvtColor(srcAnyDepth, gray, cv::COLOR_BGR2GRAY);
        m = gray;
    } else {
        m = srcAnyDepth;
    }

    if (m.type() != CV_8U) {
        double minv, maxv;
        cv::minMaxLoc(m, &minv, &maxv);
        if (std::abs(maxv - minv) < 1e-12) {
            mask01 = cv::Mat1b(m.size(), 0);
            return;
        }
        cv::Mat m8;
        m.convertTo(m8, CV_8U, 255.0 / (maxv - minv), (-minv) * 255.0 / (maxv - minv));
        m = m8;
    }

    int nz = cv::countNonZero(m);
    if (nz == 0) { mask01 = cv::Mat1b(m.size(), 0); return; }
    if (nz == m.rows * m.cols) { mask01 = cv::Mat1b(m.size(), 1); return; }

    cv::Mat1b tmp;
    cv::threshold(m, tmp, 0, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(m != tmp) == 0) {
        mask01 = (tmp > 0) / 255;
        return;
    }

    cv::Mat1b otsu;
    cv::threshold(m, otsu, 0, 1, cv::THRESH_BINARY | cv::THRESH_OTSU);
    mask01 = otsu;
}

static bool load_tif_as_float(const std::filesystem::path& file, cv::Mat1f& out)
{
    cv::Mat raw = cv::imread(file.string(), cv::IMREAD_UNCHANGED);
    if (raw.empty() || raw.channels() != 1) return false;
    switch (raw.type()) {
        case CV_32FC1: out = raw; return true;
        case CV_64FC1: raw.convertTo(out, CV_32F); return true;
        default: raw.convertTo(out, CV_32F); return true;
    }
}

static inline double sumRect01d(const cv::Mat1d& ii, int x0, int y0, int x1, int y1)
{
    return ii(y1, x1) - ii(y0, x1) - ii(y1, x0) + ii(y0, x0);
}

static double area_from_mesh_and_mask(const cv::Mat1f& X, const cv::Mat1f& Y, const cv::Mat1f& Z,
                                      const cv::Mat1b& mask01)
{
    const int Hq = X.rows, Wq = X.cols;
    if (Hq < 2 || Wq < 2) return 0.0;

    const int Hm = mask01.rows, Wm = mask01.cols;
    if (Hm <= 0 || Wm <= 0) return 0.0;

    cv::Mat1b deact;
    if (kDeactivateWhenZero) deact = (mask01 == 0);
    else deact = (mask01 != 0);

    cv::Mat1d ii;
    cv::integral(deact, ii, CV_64F);

    const double sx = static_cast<double>(Wm) / static_cast<double>(Wq - 1);
    const double sy = static_cast<double>(Hm) / static_cast<double>(Hq - 1);

    double total = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:total) schedule(static)
#endif
    for (int qy = 0; qy < Hq - 1; ++qy) {
        for (int qx = 0; qx < Wq - 1; ++qx) {
            int x0 = (int)std::floor(qx * sx);
            int y0 = (int)std::floor(qy * sy);
            int x1 = (int)std::ceil((qx + 1) * sx);
            int y1 = (int)std::ceil((qy + 1) * sy);

            x0 = std::clamp(x0, 0, Wm - 1);
            y0 = std::clamp(y0, 0, Hm - 1);
            x1 = std::clamp(x1, x0 + 1, Wm);
            y1 = std::clamp(y1, y0 + 1, Hm);

            const int rectPix = (x1 - x0) * (y1 - y0);
            if (rectPix <= 0) continue;

            const double deactCount = sumRect01d(ii, x0, y0, x1, y1);
            const double fracDeact = deactCount / (double)rectPix;

            if (fracDeact >= kTauDeactivate) continue;

            const cv::Vec3d A(X(qy, qx), Y(qy, qx), Z(qy, qx));
            const cv::Vec3d B(X(qy, qx + 1), Y(qy, qx + 1), Z(qy, qx + 1));
            const cv::Vec3d C(X(qy + 1, qx), Y(qy + 1, qx), Z(qy + 1, qx));
            const cv::Vec3d D(X(qy + 1, qx + 1), Y(qy + 1, qx + 1), Z(qy + 1, qx + 1));
            if (!isFinite3(A) || !isFinite3(B) || !isFinite3(C) || !isFinite3(D))
                continue;

            total += tri_area3D(A, B, D) + tri_area3D(A, D, C);
        }
    }
    return total;
}

} // anonymous namespace

std::vector<AreaResult> SurfaceAreaCalculator::calculateAreas(
    const std::shared_ptr<VolumePkg>& vpkg,
    const std::shared_ptr<Volume>& volume,
    const std::vector<std::string>& ids)
{
    std::vector<AreaResult> results;
    results.reserve(ids.size());

    float voxelsize = 1.0f;
    try {
        if (volume) {
            voxelsize = static_cast<float>(volume->voxelSize());
        }
    } catch (...) { voxelsize = 1.0f; }
    if (!std::isfinite(voxelsize) || voxelsize <= 0.f) voxelsize = 1.0f;

    for (const auto& id : ids) {
        AreaResult result;
        result.segmentId = id;

        auto sm = vpkg->getSurface(id);
        if (!sm) {
            result.errorReason = "missing surface";
            results.push_back(result);
            continue;
        }

        const std::filesystem::path maskPath = sm->path / "mask.tif";
        if (!std::filesystem::exists(maskPath)) {
            result.errorReason = "no mask.tif";
            results.push_back(result);
            continue;
        }

        cv::Mat1b mask01;
        {
            std::vector<cv::Mat> pages;
            if (cv::imreadmulti(maskPath.string(), pages, cv::IMREAD_UNCHANGED) && !pages.empty()) {
                int best = choose_largest_page(pages);
                if (best < 0) { result.errorReason = "mask pages invalid"; results.push_back(result); continue; }
                binarize_mask(pages[best], mask01);
            } else {
                cv::Mat m = cv::imread(maskPath.string(), cv::IMREAD_UNCHANGED);
                if (m.empty()) { result.errorReason = "mask read error"; results.push_back(result); continue; }
                binarize_mask(m, mask01);
            }
        }
        if (mask01.empty()) {
            result.errorReason = "empty mask";
            results.push_back(result);
            continue;
        }

        cv::Mat1f X, Y, Z;
        if (!load_tif_as_float(sm->path / "x.tif", X) ||
            !load_tif_as_float(sm->path / "y.tif", Y) ||
            !load_tif_as_float(sm->path / "z.tif", Z)) {
            result.errorReason = "bad or missing x/y/z.tif";
            results.push_back(result);
            continue;
        }
        if (X.size() != Y.size() || X.size() != Z.size() || X.rows < 2 || X.cols < 2) {
            result.errorReason = "xyz size mismatch";
            results.push_back(result);
            continue;
        }

        double area_vx2 = 0.0;
        try {
            area_vx2 = area_from_mesh_and_mask(X, Y, Z, mask01);
        } catch (...) {
            result.errorReason = "area compute error";
            results.push_back(result);
            continue;
        }
        if (!std::isfinite(area_vx2)) {
            result.errorReason = "non-finite area";
            results.push_back(result);
            continue;
        }

        const double area_cm2 = area_vx2 * double(voxelsize) * double(voxelsize) / 1e8;
        if (!std::isfinite(area_cm2)) {
            result.errorReason = "non-finite cm²";
            results.push_back(result);
            continue;
        }

        // Persist to meta
        try {
            auto* surf = sm.get();
            if (!surf->meta) surf->meta = std::make_unique<nlohmann::json>();
            (*surf->meta)["area_vx2"] = area_vx2;
            (*surf->meta)["area_cm2"] = area_cm2;
            (*surf->meta)["date_last_modified"] = get_surface_time_str();
            surf->save_meta();
            result.areaVx2 = area_vx2;
            result.areaCm2 = area_cm2;
            result.success = true;
        } catch (...) {
            result.errorReason = "meta save failed";
        }

        results.push_back(result);
    }

    return results;
}
