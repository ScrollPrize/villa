// vc_transform_geom.cpp
// Small utility to apply an affine and optional uniform scales before/after
// OBJ geometry, a single TIFXYZ mesh, or a directory of TIFXYZ subfolders,
// writing the transformed result.

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "vc/core/util/Surface.hpp"

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace po = boost::program_options;
using json = nlohmann::json;

struct AffineTransform {
    cv::Mat_<double> M; // 4x4
    AffineTransform() { M = cv::Mat_<double>::eye(4, 4); }
};

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

static AffineTransform load_affine_json(const std::string& filename) {
    AffineTransform t;
    const std::filesystem::path p(filename);
    if (!std::filesystem::exists(p))
        throw std::runtime_error("affine path does not exist: " + filename);
    if (!std::filesystem::is_regular_file(p))
        throw std::runtime_error("affine path is not a regular file: " + filename);

    std::ifstream f(filename);
    if (!f.is_open()) throw std::runtime_error("cannot open affine file: " + filename);
    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to parse affine file '" + filename + "': " + e.what());
    }
    if (!j.contains("transformation_matrix")) return t; // identity
    auto mat = j["transformation_matrix"];
    if (mat.size() != 3 && mat.size() != 4) throw std::runtime_error("affine must be 3x4 or 4x4");
    for (int r = 0; r < (int)mat.size(); ++r) {
        if (mat[r].size() != 4) throw std::runtime_error("affine rows must have 4 cols");
        for (int c = 0; c < 4; ++c) t.M(r,c) = mat[r][c].get<double>();
    }
    if (mat.size() == 4) {
        const double a30 = t.M(3,0), a31 = t.M(3,1), a32 = t.M(3,2), a33 = t.M(3,3);
        if (std::abs(a30) > 1e-12 || std::abs(a31) > 1e-12 || std::abs(a32) > 1e-12 || std::abs(a33 - 1.0) > 1e-12)
            throw std::runtime_error("bottom row must be [0,0,0,1]");
    }
    return t;
}



static bool invert_affine_in_place(AffineTransform& T) {
    cv::Mat A(3, 3, CV_64F), Ainv;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            A.at<double>(r, c) = T.M(r, c);

    const cv::SVD svd(A, cv::SVD::FULL_UV);
    const auto* w = svd.w.ptr<double>(0);
    if (!std::isfinite(w[0]) || !std::isfinite(w[2])) return false;

    const double smax = w[0];
    const double smin = w[2];
    if (smax <= 0.0) return false;

    const double rcond = smin / smax;
    if (!std::isfinite(rcond) || rcond < std::numeric_limits<double>::epsilon()) return false;

    if (cv::invert(A, Ainv, cv::DECOMP_SVD) <= 0.0) return false;

    const double inv_residual = cv::norm(A * Ainv - cv::Mat::eye(3, 3, CV_64F), cv::NORM_L2);
    const double ref_scale = cv::norm(A, cv::NORM_L2) * cv::norm(Ainv, cv::NORM_L2);
    if (!std::isfinite(inv_residual) || inv_residual > 1e-12 * (ref_scale + 1.0)) return false;

    cv::Matx33d Ai;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            Ai(r, c) = Ainv.at<double>(r, c);

    cv::Vec3d t(T.M(0,3), T.M(1,3), T.M(2,3));
    cv::Vec3d ti = -(Ai * t);

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T.M(r, c) = Ai(r, c);
        T.M(r, 3) = ti(r);
    }
    T.M(3,0) = T.M(3,1) = T.M(3,2) = 0.0;
    T.M(3,3) = 1.0;
    return true;
}

static inline bool apply_affine_point(const cv::Vec3d& p, const AffineTransform& A, cv::Vec3d& out) {
    const double x = p[0], y = p[1], z = p[2];
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;

    const double nx = A.M(0,0) * x + A.M(0,1) * y + A.M(0,2) * z + A.M(0,3);
    const double ny = A.M(1,0) * x + A.M(1,1) * y + A.M(1,2) * z + A.M(1,3);
    const double nz = A.M(2,0) * x + A.M(2,1) * y + A.M(2,2) * z + A.M(2,3);

    if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz)) return false;
    out = {nx, ny, nz};
    return true;
}

static inline cv::Vec3f transform_normal(const cv::Vec3f& n, const AffineTransform& A) {
    if (!std::isfinite(n[0]) || !std::isfinite(n[1]) || !std::isfinite(n[2])) return n;

    const double det = A.M(0,0)*(A.M(1,1)*A.M(2,2) - A.M(1,2)*A.M(2,1))
                     - A.M(0,1)*(A.M(1,0)*A.M(2,2) - A.M(1,2)*A.M(2,0))
                     + A.M(0,2)*(A.M(1,0)*A.M(2,1) - A.M(1,1)*A.M(2,0));
    if (!std::isfinite(det) || std::abs(det) < std::numeric_limits<double>::epsilon()) return n;

    // Proper normal transform: n' ∝ (A^{-1})^T * n (ignore uniform pre-scale)
    cv::Matx33d Lin(
        A.M(0,0), A.M(0,1), A.M(0,2),
        A.M(1,0), A.M(1,1), A.M(1,2),
        A.M(2,0), A.M(2,1), A.M(2,2)
    );
    cv::Matx33d invAT = Lin.inv().t();
    const double nx = invAT(0,0)*n[0] + invAT(0,1)*n[1] + invAT(0,2)*n[2];
    const double ny = invAT(1,0)*n[0] + invAT(1,1)*n[1] + invAT(1,2)*n[2];
    const double nz = invAT(2,0)*n[0] + invAT(2,1)*n[1] + invAT(2,2)*n[2];
    if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz)) return n;
    const double L2 = nx*nx + ny*ny + nz*nz;
    if (L2 > 0) {
        if (!std::isfinite(L2)) return n;
        const double invL = 1.0 / std::sqrt(L2);
        return {static_cast<float>(nx*invL), static_cast<float>(ny*invL), static_cast<float>(nz*invL)};
    }
    return n;
}

static bool is_tifxyz_dir(const std::filesystem::path& p) {
    return std::filesystem::is_directory(p)
        && std::filesystem::exists(p/"x.tif")
        && std::filesystem::exists(p/"y.tif")
        && std::filesystem::exists(p/"z.tif");
}

static int run_tifxyz(const std::filesystem::path& inDir,
                      const std::filesystem::path& outDir,
                      const AffineTransform* A,
                      bool invert,
                      double scale_before_affine,
                      double scale_after_affine)
{
    std::unique_ptr<AffineTransform> AA;
    if (A) {
        AA = std::make_unique<AffineTransform>(*A);
        if (invert && !invert_affine_in_place(*AA)) {
            std::cerr << "non-invertible affine" << std::endl; return 2;
        }
    }

    std::unique_ptr<QuadSurface> surf;
    try { surf = load_quad_from_tifxyz(inDir.string()); }
    catch (const std::exception& e) {
        std::cerr << "failed to load tifxyz: " << e.what() << std::endl; return 3;
    }

    int sanitize_replacements = 0;
    cv::Mat_<cv::Vec3f>* P = surf->rawPointsPtr();
    for (int j = 0; j < P->rows; ++j) {
        for (int i = 0; i < P->cols; ++i) {
            cv::Vec3f& p = (*P)(j,i);
            if (p[0] == -1) continue; // keep invalids
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {
                p = cv::Vec3f(-1.f, -1.f, -1.f);
                continue;
            }

            cv::Vec3d q = cast_to_double(p);
            cv::Vec3f pre_scale = cast_to_finite_float(q, p, sanitize_replacements);
            q[0] *= scale_before_affine;
            q[1] *= scale_before_affine;
            q[2] *= scale_before_affine;
            if (!std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[2])) {
                p = pre_scale;
                q = cast_to_double(p);
            }

            const cv::Vec3f pre_affine = cast_to_finite_float(q, pre_scale, sanitize_replacements);
            if (AA) {
                cv::Vec3d q_after;
                if (apply_affine_point(q, *AA, q_after)) {
                    q = q_after;
                } else {
                    q = cast_to_double(pre_affine);
                }
            }

            const cv::Vec3f post_affine = cast_to_finite_float(q, pre_affine, sanitize_replacements);
            q[0] *= scale_after_affine;
            q[1] *= scale_after_affine;
            q[2] *= scale_after_affine;
            if (!std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[2])) {
                q = cast_to_double(post_affine);
            }
            p = cast_to_finite_float(q, post_affine, sanitize_replacements);
        }
    }

    if (sanitize_replacements > 0) {
        std::cout << "Replaced " << sanitize_replacements
                  << " non-finite transform components with finite fallbacks." << std::endl;
    }

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
    (*surf->meta)["area_vx2"] = area_vx2;

    // QuadSurface exposes points directly, but not a writable raw normal grid API.
    // Keep TIFXYZ point transforms here and leave normal recomputation to downstream tools.

    try {
        std::filesystem::path out = outDir;
        surf->save(out);
    } catch (const std::exception& e) {
        std::cerr << "failed to save tifxyz: " << e.what() << std::endl; return 4;
    }
    return 0;
}

static int run_tifxyz_batch(const std::filesystem::path& inRoot,
                            const std::filesystem::path& outRoot,
                            const AffineTransform* A,
                            bool invert,
                            double scale_before_affine,
                            double scale_after_affine)
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
        const int rc = run_tifxyz(sub, out, A, invert, scale_before_affine, scale_after_affine);
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
                   const AffineTransform* A,
                   bool invert,
                   double scale_before_affine,
                   double scale_after_affine)
{
    std::unique_ptr<AffineTransform> AA;
    if (A) {
        AA = std::make_unique<AffineTransform>(*A);
        if (invert && !invert_affine_in_place(*AA)) {
            std::cerr << "non-invertible affine" << std::endl; return 2;
        }
    }

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
            if (AA) {
                cv::Vec3d q;
                if (apply_affine_point(p, *AA, q)) {
                    p = q;
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
            if (AA) n = transform_normal(n, *AA);
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

        if (vm.count("scale-segmentation")) {
            const double legacy_scale = vm["scale-segmentation"].as<double>();
            if (!vm["scale-before-affine"].defaulted() && scale_before_affine != legacy_scale) {
                std::cerr << "Conflicting values for --scale-before-affine and deprecated --scale-segmentation" << std::endl;
                return 1;
            }
            scale_before_affine = legacy_scale;
            std::cerr << "Warning: --scale-segmentation is deprecated; use --scale-before-affine" << std::endl;
        }

        std::unique_ptr<AffineTransform> A;
        if (vm.count("affine")) {
            A = std::make_unique<AffineTransform>(load_affine_json(vm["affine"].as<std::string>()));
        }

        // Determine input type and route:
        // 1) single TIFXYZ dir, 2) batch TIFXYZ root dir, 3) OBJ file.
        if (is_tifxyz_dir(inPath)) {
            if (std::filesystem::exists(outPath)) {
                std::cerr << "output directory already exists: " << outPath << std::endl; return 1;
            }
            return run_tifxyz(inPath, outPath, A.get(), invert, scale_before_affine, scale_after_affine);
        }

        if (std::filesystem::is_directory(inPath)) {
            return run_tifxyz_batch(inPath, outPath, A.get(), invert, scale_before_affine, scale_after_affine);
        }

        if (inPath.extension() == ".obj") {
            if (outPath.extension() != ".obj") {
                std::cerr << "output should have .obj extension for OBJ input" << std::endl; return 1;
            }
            return run_obj(inPath, outPath, A.get(), invert, scale_before_affine, scale_after_affine);
        }

        std::cerr << "Unknown input type. Provide a .obj file, a TIFXYZ directory, or a directory containing TIFXYZ subfolders." << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl; return 1;
    }
}
