// vc_transform_geom.cpp
// Small utility to apply an affine and optional uniform scales before/after
// OBJ geometry, a single TIFXYZ mesh, or a directory of TIFXYZ subfolders,
// writing the transformed result.

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/TransformSurface.hpp"

#include <boost/program_options.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace po = boost::program_options;
using Affine = vcr::catalog::Affine;

static int count_valid_points(const cv::Mat_<cv::Vec3f>& points)
{
    int valid_count = 0;
    for (int j = 0; j < points.rows; ++j) {
        for (int i = 0; i < points.cols; ++i) {
            const auto& p = points(j, i);
            if (p[0] == -1.f) continue;
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
            ++valid_count;
        }
    }
    return valid_count;
}

static inline float clamp_to_float(double value, float fallback, int& replacement_count)
{
    constexpr double k_float_max = std::numeric_limits<float>::max();
    if (!std::isfinite(value)) {
        ++replacement_count;
        return fallback;
    }
    if (value > k_float_max || value < -k_float_max) {
        ++replacement_count;
        return fallback;
    }
    return static_cast<float>(value);
}

static inline cv::Vec3f cast_to_finite_float(const cv::Vec3d& value,
                                            const cv::Vec3f& fallback,
                                            int& replacement_count)
{
    return {
        clamp_to_float(value[0], fallback[0], replacement_count),
        clamp_to_float(value[1], fallback[1], replacement_count),
        clamp_to_float(value[2], fallback[2], replacement_count)
    };
}

static inline cv::Vec3d cast_to_double(const cv::Vec3f& value)
{
    return {static_cast<double>(value[0]), static_cast<double>(value[1]), static_cast<double>(value[2])};
}

static bool is_tifxyz_dir(const std::filesystem::path& p) {
    return std::filesystem::is_directory(p)
        && std::filesystem::exists(p/"x.tif")
        && std::filesystem::exists(p/"y.tif")
        && std::filesystem::exists(p/"z.tif");
}

static int run_tifxyz(const std::filesystem::path& inDir,
                      const std::filesystem::path& outDir,
                      const std::optional<Affine>& matrix,
                      double scale_before_affine,
                      double scale_after_affine,
                      bool noResample = false,
                      std::optional<std::array<int, 3>> targetShape = std::nullopt)
{
    if (noResample) {
        try {
            vc::core::util::scaleTifxyzWithoutResampling(inDir, outDir,
                scale_before_affine * scale_after_affine, targetShape);
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "failed to scale tifxyz: " << e.what() << std::endl;
            return 4;
        }
    }
    std::unique_ptr<QuadSurface> surf;
    try { surf = load_quad_from_tifxyz(inDir.string()); }
    catch (const std::exception& e) {
        std::cerr << "failed to load tifxyz: " << e.what() << std::endl; return 3;
    }

    if (matrix && !matrix->similarityScale(2e-4)) {
        std::cerr << "Warning: affine contains non-uniform scaling or shear; "
                  << "applying best-effort axis-wise tifxyz regridding from measured "
                  << "neighbor spacing. tifxyz cannot encode shear/off-axis metric exactly."
                  << std::endl;
    }

    vc::core::util::transformSurfacePoints(surf.get(), scale_before_affine, matrix, scale_after_affine);

    cv::Mat_<cv::Vec3f>* P = surf->rawPointsPtr();
    int final_valid_count = count_valid_points(*P);
    if (final_valid_count == 0) {
        std::cerr << "No valid points remain after transform; aborting save" << std::endl;
        return 7;
    }

    // Points were modified in-place; invalidate cached derived geometry so bbox
    // and other cached quantities are recomputed from transformed points.
    surf->invalidateCache();

    // Keep voxel-space area metadata consistent with transformed geometry.
    // area_cm2 intentionally remains unchanged for cross-volume registration
    // workflows where physical scale is defined by the target volume context.
    const double area_vx2 = vc::surface::computeSurfaceAreaVox2(*P);
    surf->meta["area_vx2"] = area_vx2;

    try {
        surf->saveEdited(inDir, outDir);
    } catch (const std::exception& e) {
        std::cerr << "failed to save tifxyz: " << e.what() << std::endl; return 4;
    }
    return 0;
}

static int run_tifxyz_batch(const std::filesystem::path& inRoot,
                            const std::filesystem::path& outRoot,
                            const std::optional<Affine>& matrix,
                            double scale_before_affine,
                            double scale_after_affine,
                            bool noResample = false,
                            std::optional<std::array<int, 3>> targetShape = std::nullopt)
{
    if (std::filesystem::exists(outRoot)) {
        std::cerr << "output directory already exists: " << outRoot << std::endl;
        return 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(outRoot, ec);
    if (ec) {
        std::cerr << "failed to create output directory: " << outRoot << std::endl;
        return 1;
    }

    int found = 0;
    int ok = 0;
    for (const auto& entry : std::filesystem::directory_iterator(inRoot)) {
        if (!entry.is_directory()) continue;
        const auto& sub = entry.path();
        if (!is_tifxyz_dir(sub)) continue;

        found++;
        const auto out = outRoot / sub.filename();
        const int rc = run_tifxyz(sub, out, matrix, scale_before_affine, scale_after_affine, noResample, targetShape);
        if (rc == 0) {
            ok++;
            std::cout << "[ok] " << sub.filename() << std::endl;
        } else {
            std::cerr << "[fail] " << sub.filename() << " (exit " << rc << ")" << std::endl;
        }
    }

    if (found == 0) {
        std::cerr << "No tifxyz subfolders found under: " << inRoot << std::endl;
        return 1;
    }

    std::cout << "Processed " << ok << "/" << found << " tifxyz folders." << std::endl;
    return ok == found ? 0 : 1;
}

static bool starts_with(const std::string& s, const char* pfx) {
    return s.rfind(pfx, 0) == 0;
}

static int run_obj(const std::filesystem::path& inFile,
                   const std::filesystem::path& outFile,
                   const std::optional<Affine>& matrix,
                   double scale_before_affine,
                   double scale_after_affine)
{
    std::ifstream in(inFile);
    if (!in.is_open()) { std::cerr << "cannot open OBJ: " << inFile << std::endl; return 5; }
    // Ensure output directory exists
    {
        const auto parent = outFile.parent_path();
        if (!parent.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(parent, ec);
        }
    }
    std::ofstream out(outFile);
    if (!out.is_open()) { std::cerr << "cannot open output OBJ: " << outFile << std::endl; return 6; }

    std::string line;
    while (std::getline(in, line)) {
        // Preserve exact content if not v/vn lines
        if (starts_with(line, "v ")) {
            std::istringstream ss(line);
            char c; ss >> c; // 'v'
            double x, y, z; ss >> x >> y >> z;
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                out << "v " << x << " " << y << " " << z << "\n";
                continue;
            }

            int dummy = 0;
            cv::Vec3d p = {x * scale_before_affine, y * scale_before_affine, z * scale_before_affine};
            cv::Vec3f pre_scale = cast_to_finite_float(p, cv::Vec3f(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)), dummy);
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) p = cast_to_double(pre_scale);

            cv::Vec3f pre_affine = pre_scale;
            if (matrix) {
                const auto q = matrix->apply({p[0], p[1], p[2]});
                if (std::isfinite(q[0]) && std::isfinite(q[1]) && std::isfinite(q[2])) {
                    p = {q[0], q[1], q[2]};
                } else {
                    p = cast_to_double(pre_affine);
                }
                pre_affine = cast_to_finite_float(p, pre_scale, dummy);
            }

            p[0] *= scale_after_affine;
            p[1] *= scale_after_affine;
            p[2] *= scale_after_affine;
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) ) {
                p = cast_to_double(pre_affine);
            }
            const cv::Vec3f out_p = cast_to_finite_float(p, pre_affine, dummy);
            out << std::setprecision(9) << "v " << out_p[0] << " " << out_p[1] << " " << out_p[2] << "\n";
        } else if (starts_with(line, "vn ")) {
            std::istringstream ss(line);
            std::string tag; ss >> tag; // "vn"
            double nx, ny, nz; ss >> nx >> ny >> nz;
            cv::Vec3f n = {static_cast<float>(nx), static_cast<float>(ny), static_cast<float>(nz)};
            if (matrix && std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
                const auto m = matrix->mapNormal({n[0], n[1], n[2]});
                if (std::isfinite(m[0]) && std::isfinite(m[1]) && std::isfinite(m[2]))
                    n = {static_cast<float>(m[0]), static_cast<float>(m[1]), static_cast<float>(m[2])};
            }
            // uniform scaling before/after affine has no effect on normalized normals
            out << std::setprecision(9) << "vn " << n[0] << " " << n[1] << " " << n[2] << "\n";
        } else {
            out << line << "\n";
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        po::options_description desc("Options");
        desc.add_options()
            ("help,h", "Show help")
            ("input,i",  po::value<std::string>()->required(), "Input path: OBJ file, TIFXYZ dir, or dir containing TIFXYZ subfolders")
            ("output,o", po::value<std::string>()->required(), "Output path (required): OBJ file, TIFXYZ dir, or output root for batch")
            ("affine,a", po::value<std::string>(), "Affine JSON with 'transformation_matrix'")
            ("invert",   po::bool_switch()->default_value(false), "Invert the affine")
            ("scale-before-affine", po::value<double>()->default_value(1.0), "Uniform scale applied before affine")
            ("scale-after-affine", po::value<double>()->default_value(1.0), "Uniform scale applied after affine")
            ("scale-segmentation", po::value<double>(), "[Deprecated] Alias for --scale-before-affine")
            ("no-resample", po::bool_switch(), "TIFXYZ scale only: stream XYZ without changing grid dimensions; scale the grid spacing too")
            ("target-shape-zyx", po::value<std::vector<int>>()->multitoken(), "Actual destination volume shape Z Y X for --no-resample (required if input records base_shape_zyx)")
        ;

        po::variables_map vm;
        try {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if (vm.count("help")) { std::cout << desc << std::endl; return 0; }
            po::notify(vm);
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n" << desc << std::endl; return 1;
        }

        const std::filesystem::path inPath(vm["input"].as<std::string>());
        const std::filesystem::path outPath(vm["output"].as<std::string>());
        double scale_before_affine = vm["scale-before-affine"].as<double>();
        const double scale_after_affine = vm["scale-after-affine"].as<double>();
        const bool invert = vm["invert"].as<bool>();
        const bool noResample = vm["no-resample"].as<bool>();
        std::optional<std::array<int, 3>> targetShape;
        if (vm.count("target-shape-zyx")) {
            const auto shape = vm["target-shape-zyx"].as<std::vector<int>>();
            if (!noResample || shape.size() != 3 || std::any_of(shape.begin(), shape.end(), [](int n) { return n <= 0; }))
                throw std::invalid_argument("--target-shape-zyx requires --no-resample and three positive sizes");
            targetShape = std::array<int, 3>{shape[0], shape[1], shape[2]};
        }
        if (noResample && (vm.count("affine") || invert || !std::filesystem::is_directory(inPath)))
            throw std::invalid_argument("--no-resample supports TIFXYZ uniform scaling only, without --affine/--invert");

        if (vm.count("scale-segmentation")) {
            const double legacy_scale = vm["scale-segmentation"].as<double>();
            if (!vm["scale-before-affine"].defaulted() && scale_before_affine != legacy_scale) {
                std::cerr << "Conflicting values for --scale-before-affine and deprecated --scale-segmentation" << std::endl;
                return 1;
            }
            scale_before_affine = legacy_scale;
            std::cerr << "Warning: --scale-segmentation is deprecated; use --scale-before-affine" << std::endl;
        }

        std::optional<Affine> matrix;
        if (vm.count("affine")) {
            matrix = Affine::loadTransformJson(vm["affine"].as<std::string>());
            if (invert) matrix = matrix->inverse();
        } else if (invert) {
            throw std::invalid_argument("--invert needs --affine");
        }

        // Determine input type and route:
        // 1) single TIFXYZ dir, 2) batch TIFXYZ root dir, 3) OBJ file.
        if (is_tifxyz_dir(inPath)) {
            if (std::filesystem::exists(outPath)) {
                std::cerr << "output directory already exists: " << outPath << std::endl; return 1;
            }
            return run_tifxyz(inPath, outPath, matrix, scale_before_affine, scale_after_affine, noResample, targetShape);
        }

        if (std::filesystem::is_directory(inPath)) {
            return run_tifxyz_batch(inPath, outPath, matrix, scale_before_affine, scale_after_affine, noResample, targetShape);
        }

        if (inPath.extension() == ".obj") {
            if (outPath.extension() != ".obj") {
                std::cerr << "output should have .obj extension for OBJ input" << std::endl; return 1;
            }
            return run_obj(inPath, outPath, matrix, scale_before_affine, scale_after_affine);
        }

        std::cerr << "Unknown input type. Provide a .obj file, a TIFXYZ directory, or a directory containing TIFXYZ subfolders." << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl; return 1;
    }
}
