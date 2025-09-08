// vc_transform_geom.cpp
// Small utility to apply an affine (and optional scale-segmentation) to
// either OBJ or TIFXYZ geometry, writing the transformed result.

#include "vc/core/util/Surface.hpp"

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>

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

static AffineTransform load_affine_json(const std::string& filename) {
    AffineTransform t;
    std::ifstream f(filename);
    if (!f.is_open()) throw std::runtime_error("cannot open affine file: " + filename);
    json j; f >> j;
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

static inline cv::Vec3f apply_affine_point(const cv::Vec3f& p, const AffineTransform& A) {
    const double x = p[0], y = p[1], z = p[2];
    const double nx = A.M(0,0)*x + A.M(0,1)*y + A.M(0,2)*z + A.M(0,3);
    const double ny = A.M(1,0)*x + A.M(1,1)*y + A.M(1,2)*z + A.M(1,3);
    const double nz = A.M(2,0)*x + A.M(2,1)*y + A.M(2,2)*z + A.M(2,3);
    return {static_cast<float>(nx), static_cast<float>(ny), static_cast<float>(nz)};
}

static inline cv::Vec3f transform_normal(const cv::Vec3f& n, const AffineTransform& A) {
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
    const double L = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (L > 0) return {static_cast<float>(nx/L), static_cast<float>(ny/L), static_cast<float>(nz/L)};
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
                      double scale_seg)
{
    std::unique_ptr<AffineTransform> AA;
    if (A) {
        AA = std::make_unique<AffineTransform>(*A);
        if (invert) {
            cv::Mat inv = cv::Mat(AA->M).inv();
            if (inv.empty()) { std::cerr << "non-invertible affine" << std::endl; return 2; }
            inv.copyTo(AA->M);
        }
    }

    QuadSurface* surf = nullptr;
    try { surf = load_quad_from_tifxyz(inDir.string()); }
    catch (const std::exception& e) {
        std::cerr << "failed to load tifxyz: " << e.what() << std::endl; return 3;
    }

    cv::Mat_<cv::Vec3f>* P = surf->rawPointsPtr();
    for (int j = 0; j < P->rows; ++j) {
        for (int i = 0; i < P->cols; ++i) {
            cv::Vec3f& p = (*P)(j,i);
            if (p[0] == -1) continue; // keep invalids
            cv::Vec3f q = p * static_cast<float>(scale_seg);
            if (AA) q = apply_affine_point(q, *AA);
            p = q;
        }
    }

    try {
        std::filesystem::path out = outDir;
        surf->save(out);
    } catch (const std::exception& e) {
        std::cerr << "failed to save tifxyz: " << e.what() << std::endl; delete surf; return 4;
    }
    delete surf;
    return 0;
}

static bool starts_with(const std::string& s, const char* pfx) {
    return s.rfind(pfx, 0) == 0;
}

static int run_obj(const std::filesystem::path& inFile,
                   const std::filesystem::path& outFile,
                   const AffineTransform* A,
                   bool invert,
                   double scale_seg)
{
    std::unique_ptr<AffineTransform> AA;
    if (A) {
        AA = std::make_unique<AffineTransform>(*A);
        if (invert) {
            cv::Mat inv = cv::Mat(AA->M).inv();
            if (inv.empty()) { std::cerr << "non-invertible affine" << std::endl; return 2; }
            inv.copyTo(AA->M);
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
            cv::Vec3f p = {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
            p *= static_cast<float>(scale_seg);
            if (AA) p = apply_affine_point(p, *AA);
            out << std::setprecision(9) << "v " << p[0] << " " << p[1] << " " << p[2] << "\n";
        } else if (starts_with(line, "vn ")) {
            std::istringstream ss(line);
            std::string tag; ss >> tag; // "vn"
            double nx, ny, nz; ss >> nx >> ny >> nz;
            cv::Vec3f n = {static_cast<float>(nx), static_cast<float>(ny), static_cast<float>(nz)};
            if (AA) n = transform_normal(n, *AA);
            // scale_seg is uniform -> no effect on normalized normals
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
            ("input,i",  po::value<std::string>()->required(), "Input path: OBJ file or TIFXYZ dir")
            ("output,o", po::value<std::string>()->required(), "Output: OBJ file or TIFXYZ dir (must not exist)")
            ("affine,a", po::value<std::string>(), "Affine JSON with 'transformation_matrix'")
            ("invert",   po::bool_switch()->default_value(false), "Invert the affine")
            ("scale-segmentation", po::value<double>()->default_value(1.0), "Pre-scale applied to coordinates (uniform)")
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
        const double scale_seg = vm["scale-segmentation"].as<double>();
        const bool invert = vm["invert"].as<bool>();

        std::unique_ptr<AffineTransform> A;
        if (vm.count("affine")) {
            A = std::make_unique<AffineTransform>(load_affine_json(vm["affine"].as<std::string>()));
        }

        // Determine input type and route
        if (is_tifxyz_dir(inPath)) {
            if (std::filesystem::exists(outPath)) {
                std::cerr << "output directory already exists: " << outPath << std::endl; return 1;
            }
            return run_tifxyz(inPath, outPath, A.get(), invert, scale_seg);
        }

        if (inPath.extension() == ".obj") {
            if (outPath.extension() != ".obj") {
                std::cerr << "output should have .obj extension for OBJ input" << std::endl; return 1;
            }
            return run_obj(inPath, outPath, A.get(), invert, scale_seg);
        }

        std::cerr << "Unknown input type. Provide a .obj file or a TIFXYZ directory (containing x.tif,y.tif,z.tif)." << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl; return 1;
    }
}
