#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/StreamOperators.hpp"

#include "z5/factory.hxx"
#include <nlohmann/json.hpp>

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <boost/program_options.hpp>
#include <tiffio.h>
#include <mutex>
#include <cmath>
#include <set>

namespace po = boost::program_options;

using json = nlohmann::json;

/**
 * @brief Structure to hold affine transform data
 */
struct AffineTransform {
    cv::Mat_<double> matrix;  // 4x4 matrix in XYZ format
    
    AffineTransform() {
        matrix = cv::Mat_<double>::eye(4, 4);
    }
};

/**
 * @brief Invert an affine transform in-place.
 *        M = [A | t; 0 0 0 1]  ->  M^{-1} = [A^{-1} | -A^{-1} t; 0 0 0 1]
 * @return bool True on success, false if A is non-invertible.
 */
static inline bool invertAffineInPlace(AffineTransform& T)
{
    cv::Mat A_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            A_cv.at<double>(r, c) = T.matrix(r, c);

    cv::Mat Ainv_cv;
    double det = cv::invert(A_cv, Ainv_cv, cv::DECOMP_LU);
    if (det < 1e-10) {
        return false;
    }
    cv::Matx33d Ainv;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            Ainv(r, c) = Ainv_cv.at<double>(r, c);
    const cv::Vec3d t(T.matrix(0,3), T.matrix(1,3), T.matrix(2,3));
    const cv::Vec3d tinv = -(Ainv * t);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T.matrix(r, c) = Ainv(r, c);
        T.matrix(r, 3) = tinv(r);
    }
    T.matrix(3,0) = 0.0; T.matrix(3,1) = 0.0; T.matrix(3,2) = 0.0; T.matrix(3,3) = 1.0;
    return true;
}


/**
 * @brief Load affine transform from file (JSON)
 * 
 * @param filename Path to affine transform file
 * @return AffineTransform Loaded transform data
 */
AffineTransform loadAffineTransform(const std::string& filename) {
    AffineTransform transform;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open affine transform file: " + filename);
    }
    
    try {
        json j;
        file >> j;
        
        if (j.contains("transformation_matrix")) {
            auto mat = j["transformation_matrix"];
            if (mat.size() != 3 && mat.size() != 4) {
                throw std::runtime_error("Affine matrix must have 3 or 4 rows");
            }

            for (int row = 0; row < (int)mat.size(); row++) {
                if (mat[row].size() != 4) {
                    throw std::runtime_error("Each row of affine matrix must have 4 elements");
                }
                for (int col = 0; col < 4; col++) {
                    transform.matrix.at<double>(row, col) = mat[row][col].get<double>();
                }
            }
            // If 3x4 provided, bottom row remains [0 0 0 1] from identity ctor.
            if (mat.size() == 4) {
                // Optional: sanity-check bottom row is [0 0 0 1] within tolerance
                const double a30 = transform.matrix(3,0);
                const double a31 = transform.matrix(3,1);
                const double a32 = transform.matrix(3,2);
                const double a33 = transform.matrix(3,3);
                if (std::abs(a30) > 1e-12 || std::abs(a31) > 1e-12 ||
                    std::abs(a32) > 1e-12 || std::abs(a33 - 1.0) > 1e-12)
                    throw std::runtime_error("Bottom affine row must be [0,0,0,1]");
            }
        }
    } catch (json::parse_error&) {
        throw std::runtime_error("Error parsing affine transform file: " + filename);
    }

    return transform;
}

/**
 * @brief Compose two affines: result = B * A (apply A first, then B)
 */
static inline AffineTransform composeAffine(const AffineTransform& A, const AffineTransform& B)
{
    AffineTransform R;
    cv::Mat tmp = B.matrix * A.matrix; // left-multiply: row-vector convention used in code
    tmp.copyTo(R.matrix);
    return R;
}

/**
 * @brief Pretty-print a 4x4 matrix to stdout.
 */
static inline void printMat4x4(const cv::Mat_<double>& M, const char* header)
{
    if (header) std::cout << header << "\n";
    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(6);
    for (int r = 0; r < 4; ++r) {
        std::cout << "  [";
        for (int c = 0; c < 4; ++c) {
            std::cout << std::setw(12) << M(r,c);
            if (c < 3) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout.unsetf(std::ios::floatfield);
}

/**
 * @brief Parse an affine spec string possibly ending with an inversion hint.
 *        Accepted suffixes: :inv, :invert, :i  (e.g., "path/to/A.json:inv").
 *        Only the trailing token is interpreted, so Windows "C:\..." paths are safe.
 */

static inline std::pair<std::string, bool> parseAffineSpec(const std::string& spec)
{
    std::string path = spec;
    bool inv = false;
    const std::vector<std::string> suffixes = {":inv", ":invert", ":i"};
    for (const auto& suffix : suffixes) {
        if (spec.size() > suffix.size() &&
            spec.substr(spec.size() - suffix.size()) == suffix) {
            inv = true;
            path = spec.substr(0, spec.size() - suffix.size());
            break;
        }
    }
    return {path, inv};
}


/**
 * @brief Apply affine transform to a single point
 * 
 * @param point Point to transform
 * @param transform Affine transform to apply
 * @return cv::Vec3f Transformed point
 */
cv::Vec3f applyAffineTransformToPoint(const cv::Vec3f& point, const AffineTransform& transform) {
    const double ptx = static_cast<double>(point[0]);
    const double pty = static_cast<double>(point[1]);
    const double ptz = static_cast<double>(point[2]);
    
    // Apply affine transform (note: matrix is in XYZ format)
    const double ptx_new = transform.matrix(0, 0) * ptx + transform.matrix(0, 1) * pty + transform.matrix(0, 2) * ptz + transform.matrix(0, 3);
    const double pty_new = transform.matrix(1, 0) * ptx + transform.matrix(1, 1) * pty + transform.matrix(1, 2) * ptz + transform.matrix(1, 3);
    const double ptz_new = transform.matrix(2, 0) * ptx + transform.matrix(2, 1) * pty + transform.matrix(2, 2) * ptz + transform.matrix(2, 3);
    
    return cv::Vec3f(
        static_cast<float>(ptx_new),
        static_cast<float>(pty_new),
        static_cast<float>(ptz_new));
}

/**
 * @brief Apply affine transform to points and normals
 * 
 * @param points Points to transform (modified in-place)
 * @param normals Normals to transform (modified in-place)
 * @param transform Affine transform to apply
 */
void applyAffineTransform(cv::Mat_<cv::Vec3f>& points, 
                         cv::Mat_<cv::Vec3f>& normals, 
                         const AffineTransform& transform) {
    // Precompute linear part A and its inverse-transpose for proper normal transform
    const cv::Matx33d A(
        transform.matrix(0,0), transform.matrix(0,1), transform.matrix(0,2),
        transform.matrix(1,0), transform.matrix(1,1), transform.matrix(1,2),
        transform.matrix(2,0), transform.matrix(2,1), transform.matrix(2,2)
    );
    // Use double precision for inversion; normals will be renormalized afterwards.
    const cv::Matx33d invAT = A.inv().t();

    // Apply transform to each point
    for (int y = 0; y < points.rows; y++) {
        for (int x = 0; x < points.cols; x++) {
            cv::Vec3f& pt = points(y, x);
            
            // Skip NaN points
            if (std::isnan(pt[0]) || std::isnan(pt[1]) || std::isnan(pt[2])) {
                continue;
            }

            pt = applyAffineTransformToPoint(pt, transform);
        }
    }
    
    // Apply correct normal transform: n' ∝ (A^{-1})^T * n (then normalize)
    for (int y = 0; y < normals.rows; y++) {
        for (int x = 0; x < normals.cols; x++) {
            cv::Vec3f& n = normals(y, x);
            if (std::isnan(n[0]) || std::isnan(n[1]) || std::isnan(n[2])) {
                continue;
            }

            const double nx_new =
                invAT(0,0) * static_cast<double>(n[0]) + invAT(0,1) * static_cast<double>(n[1]) + invAT(0,2) * static_cast<double>(n[2]);
            const double ny_new =
                invAT(1,0) * static_cast<double>(n[0]) + invAT(1,1) * static_cast<double>(n[1]) + invAT(1,2) * static_cast<double>(n[2]);
            const double nz_new =
                invAT(2,0) * static_cast<double>(n[0]) + invAT(2,1) * static_cast<double>(n[1]) + invAT(2,2) * static_cast<double>(n[2]);

            const double norm = std::sqrt(nx_new * nx_new + ny_new * ny_new + nz_new * nz_new);
            if (norm > 0.0) {
                n[0] = static_cast<float>(nx_new / norm);
                n[1] = static_cast<float>(ny_new / norm);
                n[2] = static_cast<float>(nz_new / norm);
            }
        }
    }
}


/**
 * @brief Calculate the centroid of valid 3D points in the mesh
 *
 * @param points Matrix of 3D points (cv::Mat_<cv::Vec3f>)
 * @return cv::Vec3f The centroid of all valid points
 */
cv::Vec3f calculateMeshCentroid(const cv::Mat_<cv::Vec3f>& points)
{
    cv::Vec3f centroid(0, 0, 0);
    int count = 0;

    for (int y = 0; y < points.rows; y++) {
        for (int x = 0; x < points.cols; x++) {
            const cv::Vec3f& pt = points(y, x);
            if (!std::isnan(pt[0]) && !std::isnan(pt[1]) && !std::isnan(pt[2])) {
                centroid += pt;
                count++;
            }
        }
    }

    if (count > 0) {
        centroid /= static_cast<float>(count);
    }
    return centroid;
}

/**
 * @brief Determine if normals should be flipped based on a reference point
 *
 * @param points Matrix of 3D points (cv::Mat_<cv::Vec3f>)
 * @param normals Matrix of normal vectors
 * @param referencePoint The reference point to orient normals towards/away from
 * @return bool True if normals should be flipped, false otherwise
 */
bool shouldFlipNormals(
    const cv::Mat_<cv::Vec3f>& points,
    const cv::Mat_<cv::Vec3f>& normals,
    const cv::Vec3f& referencePoint)
{
    size_t pointingToward = 0;
    size_t pointingAway = 0;

    for (int y = 0; y < points.rows; y++) {
        for (int x = 0; x < points.cols; x++) {
            const cv::Vec3f& pt = points(y, x);
            const cv::Vec3f& n = normals(y, x);

            if (std::isnan(pt[0]) || std::isnan(pt[1]) || std::isnan(pt[2]) ||
                std::isnan(n[0]) || std::isnan(n[1]) || std::isnan(n[2])) {
                continue;
            }

            // Calculate direction from point to reference
            cv::Vec3f toRef = referencePoint - pt;

            // Check if normal points toward or away from reference
            float dotProduct = toRef.dot(n);
            if (dotProduct > 0) {
                pointingToward++;
            } else {
                pointingAway++;
            }
        }
    }

    // Flip if majority point away from reference
    return pointingAway > pointingToward;
}

/**
 * @brief Apply normal flipping decision to a set of normals
 *
 * @param normals Matrix of normal vectors to potentially flip (modified in-place)
 * @param shouldFlip Whether to flip the normals
 */
void applyNormalOrientation(cv::Mat_<cv::Vec3f>& normals, bool shouldFlip)
{
    if (shouldFlip) {
        for (int y = 0; y < normals.rows; y++) {
            for (int x = 0; x < normals.cols; x++) {
                cv::Vec3f& n = normals(y, x);
                if (!std::isnan(n[0]) && !std::isnan(n[1]) && !std::isnan(n[2])) {
                    n = -n;
                }
            }
        }
    }
}

/**
 * @brief Apply flip transformation to an image
 *
 * @param img Image to flip (modified in-place)
 * @param flipType Flip type: 0=Vertical, 1=Horizontal, 2=Both
 */
void flipImage(cv::Mat& img, int flipType)
{
    if (flipType < 0 || flipType > 2) {
        return; // Invalid flip type
    }

    if (flipType == 0) {
        // Vertical flip (flip around horizontal axis)
        cv::flip(img, img, 0);
    } else if (flipType == 1) {
        // Horizontal flip (flip around vertical axis)
        cv::flip(img, img, 1);
    } else if (flipType == 2) {
        // Both (flip around both axes)
        cv::flip(img, img, -1);
    }
}

static inline int normalizeQuadrantRotation(double angleDeg, double tolDeg = 0.5)
{
    // Map to [0, 360)
    double a = std::fmod(angleDeg, 360.0);
    if (a < 0) a += 360.0;
    // Find nearest multiple of 90
    static const double q[4] = {0.0, 90.0, 180.0, 270.0};
    int best = 0;
    double bestDiff = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; ++i) {
        double d = std::abs(a - q[i]);
        if (d < bestDiff) { bestDiff = d; best = i; }
    }
    return (bestDiff <= tolDeg) ? best : -1;
}

static inline void applyRightAngleRotation(cv::Mat& m, int quad)
{
    if (quad == 1)      cv::rotate(m, m, cv::ROTATE_90_COUNTERCLOCKWISE);
    else if (quad == 2) cv::rotate(m, m, cv::ROTATE_180);
    else if (quad == 3) cv::rotate(m, m, cv::ROTATE_90_CLOCKWISE);
}

// Convenience: apply optional right-angle rotation and optional flip in one place
static inline void rotateFlipIfNeeded(cv::Mat& m, int rotQuad, int flip_axis)
{
    if (rotQuad >= 0) applyRightAngleRotation(m, rotQuad);
    if (flip_axis >= 0) flipImage(m, flip_axis);
}

// Map source tile index (tx,ty) in a grid (tilesX,tilesY) to destination index
// after applying a 90-degree-multiple rotation followed by optional flip.
static inline void mapTileIndex(int tx, int ty,
                                int tilesX, int tilesY,
                                int quadRot, int flipType,
                                int& outTx, int& outTy,
                                int& outTilesX, int& outTilesY)
{
    const bool swap = (quadRot % 2) == 1;
    const int rTilesX = swap ? tilesY : tilesX;
    const int rTilesY = swap ? tilesX : tilesY;

    int rx = tx, ry = ty;
    switch (quadRot) {
        case 0: rx = tx;                ry = ty;                break;
        case 1: rx = ty;                ry = (tilesX - 1 - tx); break; // 90 CCW
        case 2: rx = (tilesX - 1 - tx); ry = (tilesY - 1 - ty); break; // 180
        case 3: rx = (tilesY - 1 - ty); ry = tx;                break; // 270 CCW
        default: rx = tx; ry = ty; break;
    }

    int fx = rx, fy = ry;
    if (flipType == 0) {
        // Vertical flip: flip rows
        fy = (rTilesY - 1 - ry);
    } else if (flipType == 1) {
        // Horizontal flip: flip columns
        fx = (rTilesX - 1 - rx);
    } else if (flipType == 2) {
        // Both
        fx = (rTilesX - 1 - rx);
        fy = (rTilesY - 1 - ry);
    }

    outTx = fx;
    outTy = fy;
    outTilesX = rTilesX;
    outTilesY = rTilesY;
}

// Normalize a matrix of 3D vectors in-place; skip NaNs and zero-length
static inline void normalizeNormals(cv::Mat_<cv::Vec3f>& nrm)
{
    for (int yy = 0; yy < nrm.rows; ++yy)
        for (int xx = 0; xx < nrm.cols; ++xx) {
            cv::Vec3f& v = nrm(yy, xx);
            if (std::isnan(v[0]) || std::isnan(v[1]) || std::isnan(v[2])) continue;
            float L = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            if (L > 0) v /= L;
        }
}

// Compute a global normal orientation flip decision using a small probe tile
static inline bool computeGlobalFlipDecision(
    QuadSurface* surf,
    int dx0,
    int dy0,
    float u0,
    float v0,
    float render_scale,
    float scale_seg,
    bool hasAffine,
    const AffineTransform& affineTransform,
    cv::Vec3f& outCentroid)
{
    cv::Mat_<cv::Vec3f> _tp, _tn;
    surf->gen(&_tp, &_tn,
              cv::Size(dx0, dy0),
              cv::Vec3f(0,0,0),
              render_scale,
              cv::Vec3f(u0, v0, 0.0f));

    _tp *= scale_seg;
    if (hasAffine) {
        applyAffineTransform(_tp, _tn, affineTransform);
    }
    outCentroid = calculateMeshCentroid(_tp);
    return shouldFlipNormals(_tp, _tn, outCentroid);
}

// Given raw tile points/normals, produce dataset-space base points and normalized step dirs
static inline void prepareBasePointsAndStepDirs(
    const cv::Mat_<cv::Vec3f>& tilePoints,
    const cv::Mat_<cv::Vec3f>& tileNormals,
    float scale_seg,
    float ds_scale,
    bool hasAffine,
    const AffineTransform& affineTransform,
    bool globalFlipDecision,
    cv::Mat_<cv::Vec3f>& basePointsOut,
    cv::Mat_<cv::Vec3f>& stepDirsOut)
{
    basePointsOut = tilePoints.clone();
    basePointsOut *= scale_seg;
    stepDirsOut = tileNormals.clone();
    if (hasAffine) {
        applyAffineTransform(basePointsOut, stepDirsOut, affineTransform);
    }
    applyNormalOrientation(stepDirsOut, globalFlipDecision);
    normalizeNormals(stepDirsOut);
    basePointsOut *= ds_scale;
}

// Compute canvas-centered origin (u0,v0) for given target size
static inline void computeCanvasOrigin(const cv::Size& size, float& u0, float& v0)
{
    u0 = -0.5f * (static_cast<float>(size.width)  - 1.0f);
    v0 = -0.5f * (static_cast<float>(size.height) - 1.0f);
}

// Compute per-tile origin by offsetting the canvas origin by (x0_src,y0_src)
static inline void computeTileOrigin(const cv::Size& fullSize, size_t x0_src, size_t y0_src, float& u0, float& v0)
{
    computeCanvasOrigin(fullSize, u0, v0);
    u0 += static_cast<float>(x0_src);
    v0 += static_cast<float>(y0_src);
}

// Thin wrapper around QuadSurface::gen with consistent parameters
static inline void genTile(
    QuadSurface* surf,
    const cv::Size& size,
    float render_scale,
    float u0, float v0,
    cv::Mat_<cv::Vec3f>& points,
    cv::Mat_<cv::Vec3f>& normals)
{
    surf->gen(&points, &normals, size, cv::Vec3f(0,0,0), render_scale, cv::Vec3f(u0, v0, 0.0f));
}

// Render one slice from base points and unit step directions at offset `off`
static inline void renderSliceFromBase(
    cv::Mat& out,
    z5::Dataset* ds,
    ChunkCache* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    float off,
    float ds_scale)
{
    cv::Mat_<cv::Vec3f> coords(basePoints.size());
    for (int yy = 0; yy < coords.rows; ++yy) {
        for (int xx = 0; xx < coords.cols; ++xx) {
            const cv::Vec3f& p = basePoints(yy, xx);
            const cv::Vec3f& d = stepDirs(yy, xx);
            coords(yy, xx) = p + off * d * static_cast<float>(ds_scale);
        }
    }
    cv::Mat_<uint8_t> tmp;
    readInterpolated3D(tmp, ds, coords, cache);
    out = tmp;
}



int main(int argc, char *argv[])
{
    // clang-format off
    po::options_description required("Required arguments");
    required.add_options()
        ("volume,v", po::value<std::string>()->required(),
            "Path to the OME-Zarr volume")
        ("output,o", po::value<std::string>()->required(),
            "Output path or name (Zarr: name without extension; TIF: filename or printf pattern)")
        ("scale", po::value<float>()->required(),
            "Pixels per level-g voxel (Pg)")
        ("group-idx,g", po::value<int>()->required(),
            "OME-Zarr group index");

    po::options_description optional("Optional arguments");
    optional.add_options()
        ("help,h", "Show this help message")
        ("segmentation,s", po::value<std::string>(),
            "Path to a single tifxyz segmentation folder (ignored if --render-folder is set)")
        ("render-folder", po::value<std::string>(),
            "Folder containing tifxyz segmentation folders to batch render")
        ("cache-gb", po::value<size_t>()->default_value(16),
            "Zarr chunk cache size in gigabytes (default: 16)")
        ("format", po::value<std::string>(),
            "When using --render-folder, choose 'zarr' or 'tif' output")
        ("num-slices,n", po::value<int>()->default_value(1),
            "Number of slices to render")
        ("crop-x", po::value<int>()->default_value(0),
            "Crop region X coordinate")
        ("crop-y", po::value<int>()->default_value(0),
            "Crop region Y coordinate")
        ("crop-width", po::value<int>()->default_value(0),
            "Crop region width (0 = no crop)")
        ("crop-height", po::value<int>()->default_value(0),
            "Crop region height (0 = no crop)")
        // Multi-affine interface (preferred):
        ("affine", po::value<std::vector<std::string>>()->multitoken()->composing(),
            "One or more affine JSON files, in application order (first listed applies first). "
            "You may append :inv / :invert / :i to a spec to invert that transform (e.g., --affine A.json:inv). "
            "Key in JSON: 'transformation_matrix' (3x4 or 4x4).")
        ("affine-invert", po::value<std::vector<int>>()->multitoken()->composing(),
            "0-based indices into --affine to invert before composing (e.g., --affine-invert 0 2).")
        // Backward-compatible single-affine options (deprecated):
        ("affine-transform", po::value<std::string>(),
            "[DEPRECATED] Single affine JSON; prefer --affine. Key 'transformation_matrix' (3x4 or 4x4).")
        ("invert-affine", po::bool_switch()->default_value(false),
            "[DEPRECATED] Invert the single --affine-transform.")
        ("scale-segmentation", po::value<float>()->default_value(1.0),
            "Scale segmentation to target scale")
        ("rotate", po::value<double>()->default_value(0.0),
            "Rotate output image by angle in degrees (counterclockwise)")
        ("flip", po::value<int>()->default_value(-1),
            "Flip output image. 0=Vertical, 1=Horizontal, 2=Both")
        ("include-tifs", po::bool_switch()->default_value(false),
            "If output is Zarr, also export per-Z TIFF slices to layers_{zarrname}");
    // clang-format on

    po::options_description all("Usage");
    all.add(required).add(optional);

    po::variables_map parsed;
    try {
        po::store(po::command_line_parser(argc, argv).options(all).run(), parsed);

        if (parsed.count("help") > 0 || argc < 2) {
            std::cout << "vc_render_tifxyz: Render volume data using segmentation surfaces\n\n";
            std::cout << all << '\n';
            return EXIT_SUCCESS;
        }
        
        po::notify(parsed);
    } catch (po::error& e) {
        std::cerr << "Error: " << e.what() << '\n';
        std::cerr << "Use --help for usage information\n";
        return EXIT_FAILURE;
    }

    std::filesystem::path vol_path = parsed["volume"].as<std::string>();
    std::string base_output_arg = parsed["output"].as<std::string>();
    const bool has_render_folder = parsed.count("render-folder") > 0;
    std::filesystem::path render_folder_path;
    std::string batch_format;
    if (has_render_folder) {
        render_folder_path = std::filesystem::path(parsed["render-folder"].as<std::string>());
        if (parsed.count("format") == 0) {
            std::cerr << "Error: --format is required when using --render-folder (zarr|tif).\n";
            return EXIT_FAILURE;
        }
        batch_format = parsed["format"].as<std::string>();
        std::transform(batch_format.begin(), batch_format.end(), batch_format.begin(), ::tolower);
        if (batch_format != "zarr" && batch_format != "tif") {
            std::cerr << "Error: --format must be 'zarr' or 'tif'.\n";
            return EXIT_FAILURE;
        }
        if (!std::filesystem::exists(render_folder_path) || !std::filesystem::is_directory(render_folder_path)) {
            std::cerr << "Error: --render-folder path is not a directory: " << render_folder_path << "\n";
            return EXIT_FAILURE;
        }
    }
    std::filesystem::path seg_path;
    if (!has_render_folder) {
        if (parsed.count("segmentation") == 0) {
            std::cerr << "Error: --segmentation is required unless --render-folder is used.\n";
            return EXIT_FAILURE;
        }
        seg_path = parsed["segmentation"].as<std::string>();
    }
    float tgt_scale = parsed["scale"].as<float>();
    int group_idx = parsed["group-idx"].as<int>();
    int num_slices = parsed["num-slices"].as<int>();
    // Downsample factor for this OME-Zarr pyramid level: g=0 -> 1, g=1 -> 0.5, ...
    const float ds_scale = std::ldexp(1.0f, -group_idx);  // 2^(-group_idx)
    float scale_seg = parsed["scale-segmentation"].as<float>();

    double rotate_angle = parsed["rotate"].as<double>();
    int flip_axis = parsed["flip"].as<int>();
    const bool include_tifs = parsed["include-tifs"].as<bool>();

    AffineTransform affineTransform;
    bool hasAffine = false;
    
    // --- New multi-affine loading & composition ---
    std::vector<std::pair<std::string,bool>> affineSpecs; // (path, invert?)
    if (parsed.count("affine") > 0) {
        for (const auto& s : parsed["affine"].as<std::vector<std::string>>()) {
            affineSpecs.emplace_back(parseAffineSpec(s));
        }
    }
    // Back-compat: single --affine-transform [--invert-affine]
    if (parsed.count("affine-transform") > 0) {
        const std::string singlePath = parsed["affine-transform"].as<std::string>();
        const bool singleInv = parsed["invert-affine"].as<bool>();
        affineSpecs.emplace_back(singlePath, singleInv);
        std::cout << "[deprecated] Using --affine-transform"
                  << (singleInv ? " (with inversion)" : "") << "; prefer --affine.\n";
    }
    // Optional index-based inversion flags for --affine
    if (parsed.count("affine-invert") > 0 && !affineSpecs.empty()) {
        std::set<int> idxInv;
        for (int idx : parsed["affine-invert"].as<std::vector<int>>()) {
            if (idx < 0 || idx >= static_cast<int>(affineSpecs.size())) {
                std::cerr << "Error: --affine-invert index " << idx
                          << " out of range [0.." << (affineSpecs.size()-1) << "].\n";
                return EXIT_FAILURE;
            }
            idxInv.insert(idx);
        }
        int k = 0;
        for (auto& spec : affineSpecs) {
            if (idxInv.count(k)) spec.second = true;
            ++k;
        }
    }
    // Load, optionally invert, then compose (first listed applies first)
    if (!affineSpecs.empty()) {
        AffineTransform composed; // identity
        int k = 0;
        for (const auto& [path, invertFlag] : affineSpecs) {
            try {
                AffineTransform T = loadAffineTransform(path);
                std::cout << "Loaded affine[" << k << "]: " << path
                          << (invertFlag ? " (invert)" : "") << std::endl;
                if (invertFlag) {
                    if (!invertAffineInPlace(T)) {
                        std::cerr << "Error: affine[" << k << "] has non-invertible linear part.\n";
                        return EXIT_FAILURE;
                    }
                }
                composed = composeAffine(composed, T);
                ++k;
            } catch (const std::exception& e) {
                std::cerr << "Error loading affine[" << k << "]: " << e.what() << std::endl;
                return EXIT_FAILURE;

            }
        }
        hasAffine = true;
        affineTransform = composed;
        printMat4x4(affineTransform.matrix, "Final composed affine (applied to points first):");
    }
    
    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, std::to_string(group_idx), json::parse(std::ifstream(vol_path/std::to_string(group_idx)/".zarray")).value<std::string>("dimension_separator","."));
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group " << group_idx << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;
    std::cout << "output argument: " << base_output_arg << std::endl;

    // Enforce 90-degree-increment rotations only
    int rotQuadGlobal = -1;
    if (std::abs(rotate_angle) > 1e-6) {
        rotQuadGlobal = normalizeQuadrantRotation(rotate_angle);
        if (rotQuadGlobal < 0) {
            std::cerr << "Error: only 0/90/180/270 degree rotations are supported." << std::endl;
            return EXIT_FAILURE;
        }
        rotate_angle = rotQuadGlobal * 90.0; // normalize
        std::cout << "Rotation: " << rotate_angle << " degrees" << std::endl;
    }
    if (flip_axis >= 0) {
        std::cout << "Flip: " << (flip_axis == 0 ? "Vertical" : flip_axis == 1 ? "Horizontal" : "Both") << std::endl;
    }

    std::filesystem::path output_path(base_output_arg);
    {
        const auto parent = output_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
    }

    const size_t cache_gb = parsed["cache-gb"].as<size_t>();
    const size_t cache_bytes = cache_gb * 1024ull * 1024ull * 1024ull;
    std::cout << "Chunk cache: " << cache_gb << " GB (" << cache_bytes << " bytes)" << std::endl;
    ChunkCache chunk_cache(cache_bytes);

    auto process_one = [&](const std::filesystem::path& seg_folder, const std::string& out_arg, bool force_zarr) -> void {
        std::filesystem::path output_path_local(out_arg);
        if (force_zarr) {
            // ensure .zarr extension
            if (output_path_local.extension() != ".zarr")
                output_path_local = output_path_local.string() + ".zarr";
        }
        bool output_is_zarr = force_zarr || (output_path_local.extension() == ".zarr");
        if (!output_is_zarr) {
            // May be a directory target (no printf pattern): create directory
            if (output_path_local.string().find('%') == std::string::npos) {
                std::filesystem::create_directories(output_path_local);
            } else {
                std::filesystem::create_directories(output_path_local.parent_path());
            }
        }

        std::cout << "Rendering segmentation: "
                  << seg_folder.string() << " -> "
                  << output_path_local.string()
                  << (output_is_zarr?" (zarr)":" (tif)")
                  << std::endl;

        QuadSurface *surf = nullptr;
        try {
            surf = load_quad_from_tifxyz(seg_folder);
        }
        catch (...) {
            std::cout << "error when loading: " << seg_folder << std::endl;
            return;
        }

    cv::Mat_<cv::Vec3f> *raw_points = surf->rawPointsPtr();
    for(int j=0;j<raw_points->rows;j++)
        for(int i=0;i<raw_points->cols;i++)
            if ((*raw_points)(j,i)[0] == -1)
                (*raw_points)(j,i) = {NAN,NAN,NAN};
    
    cv::Size full_size = raw_points->size();

    // Interpret --scale as Pg = pixels per level-g voxel.
    // Compute isotropic affine scale sA = cbrt(|det(A)|) (ignore shear/rot)
    // and the effective render scale used by surf->gen() and canvas sizing:
    //   render_scale = Pg / (scale_seg * sA * ds_scale)
    // This keeps pixels locked to level-g voxels while geometry is still
    // mapped to dataset index space by: scale_seg -> affine -> ds_scale.
    double sA = 1.0;
    if (hasAffine) {
        const cv::Matx33d A(
            affineTransform.matrix(0,0), affineTransform.matrix(0,1), affineTransform.matrix(0,2),
            affineTransform.matrix(1,0), affineTransform.matrix(1,1), affineTransform.matrix(1,2),
            affineTransform.matrix(2,0), affineTransform.matrix(2,1), affineTransform.matrix(2,2)
        );
        const double detA = cv::determinant(cv::Mat(A));
        if (std::isfinite(detA) && std::abs(detA) > 1e-18)
            sA = std::cbrt(std::abs(detA));
    }
    const double Pg = static_cast<double>(tgt_scale);
    const double render_scale = Pg * (static_cast<double>(scale_seg) * sA * static_cast<double>(ds_scale));

    // Canvas sizing depends ONLY on render_scale and the saved surface stride.
    {
        const double sx = render_scale / static_cast<double>(surf->_scale[0]);
        const double sy = render_scale / static_cast<double>(surf->_scale[1]);
        full_size.width  = std::max(1, static_cast<int>(std::lround(full_size.width  * sx)));
        full_size.height = std::max(1, static_cast<int>(std::lround(full_size.height * sy)));
    }
    
    // The uncropped, scaled canvas:
    cv::Size tgt_size = full_size;
    // 'crop' is expressed in the *uncropped* canvas coordinate system
    cv::Rect crop = {0, 0, full_size.width, full_size.height};
    
    std::cout << "downsample level " << group_idx
              << " (ds_scale=" << ds_scale << ", sA=" << sA
              << ", Pg=" << Pg << ", render_scale=" << render_scale << ")\n";

    // Handle crop parameters (clamped to canvas)
    const int crop_x = parsed["crop-x"].as<int>();
    const int crop_y = parsed["crop-y"].as<int>();
    const int crop_width  = parsed["crop-width"].as<int>();
    const int crop_height = parsed["crop-height"].as<int>();
    const cv::Rect canvasROI{0, 0, full_size.width, full_size.height};
    if (crop_width > 0 && crop_height > 0) {
        const cv::Rect req{crop_x, crop_y, crop_width, crop_height};
        crop = (req & canvasROI); // intersect with canvas
        if (crop.width <= 0 || crop.height <= 0) {
            std::cerr << "Error: crop rectangle " << req
                      << " lies outside the render canvas " << canvasROI << std::endl;
            delete surf;
            return;
        }
        tgt_size = crop.size();
    } else {
        crop = canvasROI;              // no crop requested
        tgt_size = crop.size();
    }
    
    std::cout << "rendering size " << tgt_size
              << " at scale " << tgt_scale
              << " crop " << crop << std::endl;    
              
    cv::Mat_<cv::Vec3f> points, normals;
    
    bool slice_gen = false;
    
    // Global normal orientation decision (for consistency across chunks)
    bool globalFlipDecision = false;
    bool orientationDetermined = false;
    cv::Vec3f meshCentroid;

    if ((tgt_size.width >= 10000 || tgt_size.height >= 10000) && num_slices > 1)
        slice_gen = true;
    else {
        // Origin must be computed in full (uncropped) canvas and shifted by crop origin.
        float u0, v0; computeCanvasOrigin(full_size, u0, v0);
        u0 += static_cast<float>(crop.x);
        v0 += static_cast<float>(crop.y);
        genTile(surf, tgt_size, static_cast<float>(render_scale), u0, v0, points, normals);
    }

    if (output_is_zarr) {
        const double render_scale_zarr = render_scale;

        cv::Mat_<cv::Vec3f> points, normals;
        float u0_base, v0_base;
        computeCanvasOrigin(full_size, u0_base, v0_base);
        u0_base += static_cast<float>(crop.x);
        v0_base += static_cast<float>(crop.y);
        const size_t CH = 128, CW = 128;
        const size_t baseZ = std::max(1, num_slices);
        const size_t CZ = baseZ;
        const int rotQuad = normalizeQuadrantRotation(rotate_angle);
        cv::Size zarr_xy_size = tgt_size;

        if (rotQuad >= 0) {
            if ((rotQuad % 2) == 1) std::swap(zarr_xy_size.width, zarr_xy_size.height);
        }
        const size_t baseY = static_cast<size_t>(zarr_xy_size.height);
        const size_t baseX = static_cast<size_t>(zarr_xy_size.width);

        z5::filesystem::handle::File outFile(output_path_local);
        z5::createFile(outFile, true);

        auto make_shape = [](size_t z, size_t y, size_t x){
            return std::vector<size_t>{z, y, x};
        };

        auto make_chunks = [](size_t z, size_t y, size_t x){
            return std::vector<size_t>{z, y, x};
        };

        std::vector<size_t> shape0 = make_shape(baseZ, baseY, baseX);
        std::vector<size_t> chunks0 = make_chunks(shape0[0], std::min(CH, shape0[1]), std::min(CW, shape0[2]));
        nlohmann::json compOpts0 = {
            {"cname",   "zstd"},
            {"clevel",  1},
            {"shuffle", 0}
        };
        auto dsOut0 = z5::createDataset(outFile, "0", "uint8", shape0, chunks0, std::string("blosc"), compOpts0);

        const size_t tilesY_src = (static_cast<size_t>(tgt_size.height) + CH - 1) / CH;
        const size_t tilesX_src = (static_cast<size_t>(tgt_size.width)  + CW - 1) / CW;
        const size_t totalTiles = tilesY_src * tilesX_src;
        std::atomic<size_t> tilesDone{0};

        bool globalFlipDecision = false;
        {
            const int dx0 = static_cast<int>(std::min(CW, shape0[2]));
            const int dy0 = static_cast<int>(std::min(CH, shape0[1]));
            const float u0 = u0_base;
            const float v0 = v0_base;
            globalFlipDecision = computeGlobalFlipDecision(
                surf, dx0, dy0, u0, v0,
                static_cast<float>(render_scale_zarr),
                scale_seg, hasAffine, affineTransform,
                meshCentroid);
        }

        // Iterate output chunks and render directly into them (parallel over XY tiles)
        for (size_t z0 = 0; z0 < shape0[0]; z0 += CZ) {
            const size_t dz = std::min(CZ, shape0[0] - z0);
            #pragma omp parallel for schedule(dynamic) collapse(2)
            for (long long ty = 0; ty < static_cast<long long>(tilesY_src); ++ty) {
                for (long long tx = 0; tx < static_cast<long long>(tilesX_src); ++tx) {
                    const size_t y0_src = static_cast<size_t>(ty) * CH;
                    const size_t x0_src = static_cast<size_t>(tx) * CW;
                    const size_t dy = std::min(static_cast<size_t>(CH), static_cast<size_t>(tgt_size.height) - y0_src);
                    const size_t dx = std::min(static_cast<size_t>(CW), static_cast<size_t>(tgt_size.width)  - x0_src);

                    float u0, v0;
                    computeTileOrigin(full_size,
                                      x0_src + static_cast<size_t>(crop.x),
                                      y0_src + static_cast<size_t>(crop.y),
                                      u0, v0);

                    cv::Mat_<cv::Vec3f> tilePoints, tileNormals;
                    genTile(surf, cv::Size(static_cast<int>(dx), static_cast<int>(dy)),
                            static_cast<float>(render_scale_zarr), u0, v0, tilePoints, tileNormals);

                    cv::Mat_<cv::Vec3f> basePoints, stepDirs;
                    prepareBasePointsAndStepDirs(
                        tilePoints, tileNormals,
                        scale_seg, ds_scale,
                        hasAffine, affineTransform,
                        globalFlipDecision,
                        basePoints, stepDirs);

                    const bool swapWH = (rotQuad % 2) == 1 && rotQuad >= 0;
                    const size_t dy_dst = swapWH ? dx : dy;
                    const size_t dx_dst = swapWH ? dy : dx;
                    xt::xarray<uint8_t> outChunk = xt::empty<uint8_t>({dz, dy_dst, dx_dst});

                    cv::Mat tileOut; // CV_8UC1
                    for (size_t zi = 0; zi < dz; ++zi) {
                        const size_t sliceIndex = z0 + zi;
                        const float off = static_cast<float>(static_cast<double>(sliceIndex) - 0.5 * (static_cast<double>(baseZ) - 1.0));
                        renderSliceFromBase(tileOut, ds.get(), &chunk_cache, basePoints, stepDirs, off, static_cast<float>(ds_scale));

                        if (rotQuad >= 0 || flip_axis >= 0) {
                            rotateFlipIfNeeded(tileOut, rotQuad, flip_axis);
                        }

                        // Copy into outChunk
                        const size_t cH = static_cast<size_t>(tileOut.rows);
                        const size_t cW = static_cast<size_t>(tileOut.cols);
                        for (size_t yy = 0; yy < cH; ++yy) {
                            for (size_t xx = 0; xx < cW; ++xx) {
                                outChunk(zi, yy, xx) = tileOut.at<uint8_t>(static_cast<int>(yy), static_cast<int>(xx));
                            }
                        }
                    }

                    int dstTx = static_cast<int>(tx), dstTy = static_cast<int>(ty);
                    int dstTilesX = static_cast<int>(tilesX_src), dstTilesY = static_cast<int>(tilesY_src);
                    if (rotQuad >= 0 || flip_axis >= 0) {
                        mapTileIndex(static_cast<int>(tx), static_cast<int>(ty),
                                     static_cast<int>(tilesX_src), static_cast<int>(tilesY_src),
                                     std::max(rotQuad, 0), flip_axis,
                                     dstTx, dstTy, dstTilesX, dstTilesY);
                    }
                    const size_t x0_dst = static_cast<size_t>(dstTx) * CW;
                    const size_t y0_dst = static_cast<size_t>(dstTy) * CH;

                    z5::types::ShapeType outOffset = {z0, y0_dst, x0_dst};
                    z5::multiarray::writeSubarray<uint8_t>(dsOut0, outChunk, outOffset.begin());

                    size_t done = ++tilesDone;
                    int pct = static_cast<int>(100.0 * double(done) / double(totalTiles));
                    #pragma omp critical(progress_print)
                    {
                        std::cout << "\r[render L0] tile " << done << "/" << totalTiles
                                  << " (" << pct << "%)" << std::flush;
                    }
                }
            }
        }

        // After finishing L0 tiles, add newline for the progress line
        std::cout << std::endl;

        // Build multi-resolution pyramid levels 1..5 by averaging 2x blocks in Z, Y, and X
        auto downsample_avg = [&](int targetLevel){
            auto src = z5::openDataset(outFile, std::to_string(targetLevel - 1));
            const auto& sShape = src->shape();
            // Downsample Z, Y and X by 2 (handle odd sizes)
            std::vector<size_t> dShape = {
                (sShape[0] + 1) / 2,
                (sShape[1] + 1) / 2,
                (sShape[2] + 1) / 2
            };
            // Chunk Z equals number of slices at this level (full Z), XY = 128
            std::vector<size_t> dChunks = make_chunks(dShape[0], std::min(CH, dShape[1]), std::min(CW, dShape[2]));
            nlohmann::json compOpts = {
                {"cname",   "zstd"},
                {"clevel",  1},
                {"shuffle", 0}
            };
            auto dst = z5::createDataset(outFile, std::to_string(targetLevel), "uint8", dShape, dChunks, std::string("blosc"), compOpts);

            const size_t tileZ = dShape[0], tileY = CH, tileX = CW;
            const size_t tilesY = (dShape[1] + tileY - 1) / tileY;
            const size_t tilesX = (dShape[2] + tileX - 1) / tileX;
            const size_t totalTiles = tilesY * tilesX;
            std::atomic<size_t> tilesDone{0};

            for (size_t z = 0; z < dShape[0]; z += tileZ) {
                const size_t lz = std::min(tileZ, dShape[0] - z);
                #pragma omp parallel for schedule(dynamic) collapse(2)
                for (long long y = 0; y < static_cast<long long>(dShape[1]); y += tileY) {
                    for (long long x = 0; x < static_cast<long long>(dShape[2]); x += tileX) {
                        const size_t ly = std::min(tileY, static_cast<size_t>(dShape[1] - y));
                        const size_t lx = std::min(tileX, static_cast<size_t>(dShape[2] - x));

                        const size_t sz = std::min<size_t>(2*lz, sShape[0] - 2*z);
                        const size_t sy = std::min<size_t>(2*ly, sShape[1] - y*2);
                        const size_t sx = std::min<size_t>(2*lx, sShape[2] - x*2);

                        xt::xarray<uint8_t> srcChunk = xt::empty<uint8_t>({sz, sy, sx});
                        {
                            z5::types::ShapeType sOff = {static_cast<size_t>(2*z), static_cast<size_t>(2*y), static_cast<size_t>(2*x)};
                            z5::multiarray::readSubarray<uint8_t>(src, srcChunk, sOff.begin());
                        }

                        xt::xarray<uint8_t> dstChunk = xt::empty<uint8_t>({lz, ly, lx});
                        for (size_t zz = 0; zz < lz; ++zz) {
                            for (size_t yy = 0; yy < ly; ++yy) {
                                for (size_t xx = 0; xx < lx; ++xx) {
                                    int sum = 0;
                                    int cnt = 0;
                                    for (int dz2 = 0; dz2 < 2 && (2*zz + dz2) < sz; ++dz2)
                                        for (int dy2 = 0; dy2 < 2 && (2*yy + dy2) < sy; ++dy2)
                                            for (int dx2 = 0; dx2 < 2 && (2*xx + dx2) < sx; ++dx2) {
                                                sum += srcChunk(2*zz + dz2, 2*yy + dy2, 2*xx + dx2);
                                                cnt += 1;
                                            }
                                    dstChunk(zz, yy, xx) = static_cast<uint8_t>((sum + (cnt/2)) / std::max(1, cnt));
                                }
                            }
                        }

                        z5::types::ShapeType dOff = {z, static_cast<size_t>(y), static_cast<size_t>(x)};
                        z5::multiarray::writeSubarray<uint8_t>(dst, dstChunk, dOff.begin());

                        size_t done = ++tilesDone;
                        int pct = static_cast<int>(100.0 * double(done) / double(totalTiles));
                        #pragma omp critical(progress_print)
                        {
                            std::cout << "\r[render L" << targetLevel << "] tile " << done << "/" << totalTiles
                                      << " (" << pct << "%)" << std::flush;
                        }
                    }
                }
            }
            std::cout << std::endl;
        };

        for (int level = 1; level <= 5; ++level) {
            downsample_avg(level);
        }

        nlohmann::json attrs;
        attrs["source_zarr"] = vol_path.string();
        attrs["source_group"] = group_idx;
        attrs["num_slices"] = baseZ;
        {
            cv::Size attr_xy = tgt_size;
            const int rotQuadAttr = normalizeQuadrantRotation(rotate_angle);
            if (rotQuadAttr >= 0 && (rotQuadAttr % 2) == 1) std::swap(attr_xy.width, attr_xy.height);
            attrs["canvas_size"] = {attr_xy.width, attr_xy.height};
        }
        attrs["chunk_size"] = {static_cast<int>(CZ), static_cast<int>(CH), static_cast<int>(CW)};
        attrs["note_axes_order"] = "ZYX (slice, row, col)";

        nlohmann::json multiscale;
        multiscale["version"] = "0.4";
        multiscale["name"] = "render";
        multiscale["axes"] = nlohmann::json::array({
            nlohmann::json{{"name","z"},{"type","space"}},
            nlohmann::json{{"name","y"},{"type","space"}},
            nlohmann::json{{"name","x"},{"type","space"}}
        });
        multiscale["datasets"] = nlohmann::json::array();
        for (int level = 0; level <= 5; ++level) {
            const double s = std::pow(2.0, level);
            nlohmann::json dset;
            dset["path"] = std::to_string(level);
            dset["coordinateTransformations"] = nlohmann::json::array({
                // Z, Y and X scale by 2^level
                nlohmann::json{{"type","scale"},{"scale", nlohmann::json::array({s, s, s})}},
                nlohmann::json{{"type","translation"},{"translation", nlohmann::json::array({0.0, 0.0, 0.0})}}
            });
            multiscale["datasets"].push_back(dset);
        }
        multiscale["metadata"] = nlohmann::json{{"downsampling_method","mean"}};
        attrs["multiscales"] = nlohmann::json::array({multiscale});

        z5::filesystem::writeAttributes(outFile, attrs);

        // Optionally export per-Z TIFFs from level 0 into layers_{zarrname}
        if (include_tifs) {
            try {
                auto dsL0 = z5::openDataset(outFile, "0");
                const auto& shape0_check = dsL0->shape(); // [Z, Y, X]
                const size_t Z = shape0_check[0];
                const int Y = static_cast<int>(shape0_check[1]);
                const int X = static_cast<int>(shape0_check[2]);

                std::string zname = output_path_local.stem().string();
                std::filesystem::path layers_dir = output_path_local.parent_path() / (std::string("layers_") + zname);
                std::filesystem::create_directories(layers_dir);

                int pad = 2;
                size_t maxIndex = (Z > 0) ? (Z - 1) : 0;
                while (maxIndex >= 100) { pad++; maxIndex /= 10; }

                bool all_exist = true;
                for (size_t z = 0; z < Z; ++z) {
                    std::ostringstream fn;
                    fn << std::setw(pad) << std::setfill('0') << z;
                    std::filesystem::path outPath = layers_dir / (fn.str() + std::string(".tif"));
                    if (!std::filesystem::exists(outPath)) { all_exist = false; break; }
                }
                if (all_exist) {
                    std::cout << "[tif export] all slices exist in " << layers_dir.string() << ", skipping." << std::endl;
                    // Nothing else to do
                    delete surf;
                    return;
                }

                const uint32_t tileW = static_cast<uint32_t>(CW);
                const uint32_t tileH = static_cast<uint32_t>(CH);
                const uint32_t tilesX_src = (static_cast<uint32_t>(X) + tileW - 1) / tileW;
                const uint32_t tilesY_src = (static_cast<uint32_t>(Y) + tileH - 1) / tileH;
                // Zarr L0 already has rotation/flip applied; TIFFs should match L0 exactly
                const uint32_t outW = static_cast<uint32_t>(X);
                const uint32_t outH = static_cast<uint32_t>(Y);
                const size_t totalTiles = static_cast<size_t>(tilesX_src) * static_cast<size_t>(tilesY_src);
                std::atomic<size_t> tilesDone{0};

                std::vector<TIFF*> tiffs(Z, nullptr);
                std::vector<std::mutex> tiffLocks(Z); // per-slice locks
                for (size_t z = 0; z < Z; ++z) {
                    std::ostringstream fn;
                    fn << std::setw(pad) << std::setfill('0') << z;
                    std::filesystem::path outPath = layers_dir / (fn.str() + std::string(".tif"));
                    TIFF* tf = TIFFOpen(outPath.string().c_str(), "w8");
                    if (!tf) throw std::runtime_error("failed to open TIFF for writing: " + outPath.string());
                    TIFFSetField(tf, TIFFTAG_IMAGEWIDTH, outW);
                    TIFFSetField(tf, TIFFTAG_IMAGELENGTH, outH);
                    TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
                    TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, 8);
                    TIFFSetField(tf, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
                    TIFFSetField(tf, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
                    TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
                    TIFFSetField(tf, TIFFTAG_PREDICTOR, 2);
                    TIFFSetField(tf, TIFFTAG_TILEWIDTH, tileW);
                    TIFFSetField(tf, TIFFTAG_TILELENGTH, tileH);
                    tiffs[z] = tf;
                }

                // Per-thread tile buffer padded to tile size
                #pragma omp parallel for schedule(dynamic) collapse(2)
                for (long long ty = 0; ty < static_cast<long long>(tilesY_src); ++ty) {
                    for (long long tx = 0; tx < static_cast<long long>(tilesX_src); ++tx) {
                        const uint32_t x0_src = static_cast<uint32_t>(tx) * tileW;
                        const uint32_t y0_src = static_cast<uint32_t>(ty) * tileH;
                        const uint32_t dx = std::min<uint32_t>(tileW, static_cast<uint32_t>(X) - x0_src);
                        const uint32_t dy = std::min<uint32_t>(tileH, static_cast<uint32_t>(Y) - y0_src);

                        xt::xarray<uint8_t> tile = xt::empty<uint8_t>({Z, static_cast<size_t>(dy), static_cast<size_t>(dx)});
                        {
                            z5::types::ShapeType off = {0, static_cast<size_t>(y0_src), static_cast<size_t>(x0_src)};
                            z5::multiarray::readSubarray<uint8_t>(dsL0, tile, off.begin());
                        }

                        std::vector<uint8_t> tileBuf(tileW * tileH, 0);
                        for (size_t z = 0; z < Z; ++z) {
                            cv::Mat srcTile(static_cast<int>(dy), static_cast<int>(dx), CV_8UC1);
                            for (uint32_t yy = 0; yy < dy; ++yy) {
                                uint8_t* dst = srcTile.ptr<uint8_t>(static_cast<int>(yy));
                                for (uint32_t xx = 0; xx < dx; ++xx) dst[xx] = tile(z, yy, xx);
                            }
                            // No additional rotate/flip here; L0 is final orientation
                            cv::Mat& dstTile = srcTile;
                            const uint32_t x0_dst = static_cast<uint32_t>(tx) * tileW;
                            const uint32_t y0_dst = static_cast<uint32_t>(ty) * tileH;

                            const uint32_t ddy = static_cast<uint32_t>(dstTile.rows);
                            const uint32_t ddx = static_cast<uint32_t>(dstTile.cols);
                            // Fill pad buffer
                            std::fill(tileBuf.begin(), tileBuf.end(), 0);
                            for (uint32_t yy = 0; yy < ddy; ++yy) {
                                const uint8_t* src = dstTile.ptr<uint8_t>(static_cast<int>(yy));
                                std::memcpy(tileBuf.data() + yy * tileW, src, ddx);
                            }

                            // Write tile to the corresponding TIFF (per-slice lock)
                            {
                                std::lock_guard<std::mutex> guard(tiffLocks[z]);
                                TIFF* tf = tiffs[z];
                                const uint32_t tileIndex = TIFFComputeTile(tf, x0_dst, y0_dst, 0, 0);
                                (void)tileIndex;
                                if (TIFFWriteEncodedTile(tf, tileIndex, tileBuf.data(), static_cast<tmsize_t>(tileBuf.size())) < 0) {
                                    // ignore individual tile write errors for now
                                }
                            }
                        }

                        size_t done = ++tilesDone;
                        int pct = static_cast<int>(100.0 * double(done) / double(totalTiles));
                        #pragma omp critical(progress_print)
                        {
                            std::cout << "\r[tif export] tiles " << done << "/" << totalTiles
                                      << " (" << pct << "%)" << std::flush;
                        }
                    }
                }

                for (auto* tf : tiffs) {
                    TIFFClose(tf);
                }

                std::cout << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[tif export] warning: failed to export TIFFs: " << e.what() << std::endl;
            }
        }

        delete surf;
        return;
    }

    {
        {
            try {
                const int rotQuad = normalizeQuadrantRotation(rotate_angle);
                if (std::abs(rotate_angle) > 1e-6 && rotQuad < 0) {
                    throw std::runtime_error("non-right-angle rotation not supported in tiled-TIFF path");
                }

                const int outW = ((rotQuad >= 0) && (rotQuad % 2 == 1)) ? tgt_size.height : tgt_size.width;
                const int outH = ((rotQuad >= 0) && (rotQuad % 2 == 1)) ? tgt_size.width  : tgt_size.height;

                const uint32_t tileW = 128;
                const uint32_t tileH = 128;

                std::vector<TIFF*> tiffs(static_cast<size_t>(num_slices), nullptr);
                std::vector<std::mutex> tiffLocks(static_cast<size_t>(num_slices));

                auto make_out_path = [&](int sliceIdx) -> std::filesystem::path {
                    if (output_path_local.string().find('%') == std::string::npos) {
                        //
                        int pad = 2; int v = std::max(0, num_slices-1);
                        while (v >= 100) { pad++; v /= 10; }
                        std::ostringstream fn;
                        fn << std::setw(pad) << std::setfill('0') << sliceIdx;
                        return output_path_local / (fn.str() + ".tif");
                    } else {
                        char buf[1024];
                        snprintf(buf, sizeof(buf), output_path_local.string().c_str(), sliceIdx);
                        return std::filesystem::path(buf);
                    }
                };

                // If all expected TIFFs exist, skip this segmentation
                {
                    bool all_exist = true;
                    for (int z = 0; z < num_slices; ++z) {
                        std::filesystem::path outPath = make_out_path(z);
                        if (!std::filesystem::exists(outPath)) { all_exist = false; break; }
                    }
                    if (all_exist) {
                        std::cout << "[tif tiled] all slices exist in " << output_path_local.string() << ", skipping." << std::endl;
                        delete surf;
                        return;
                    }
                }

                for (int z = 0; z < num_slices; ++z) {
                    std::filesystem::path outPath = make_out_path(z);
                    TIFF* tf = TIFFOpen(outPath.string().c_str(), "w8");
                    if (!tf) throw std::runtime_error(std::string("failed to open TIFF for writing: ") + outPath.string());
                    TIFFSetField(tf, TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(outW));
                    TIFFSetField(tf, TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(outH));
                    TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
                    TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, 8);
                    TIFFSetField(tf, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
                    TIFFSetField(tf, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
                    TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
                    TIFFSetField(tf, TIFFTAG_PREDICTOR, 2);
                    TIFFSetField(tf, TIFFTAG_TILEWIDTH, tileW);
                    TIFFSetField(tf, TIFFTAG_TILELENGTH, tileH);
                    tiffs[static_cast<size_t>(z)] = tf;
                }

                {
                    const int dx0 = std::min<int>(static_cast<int>(tileW), tgt_size.width);
                    const int dy0 = std::min<int>(static_cast<int>(tileH), tgt_size.height);
                    float u0, v0; computeCanvasOrigin(full_size, u0, v0);
                    u0 += static_cast<float>(crop.x);
                    v0 += static_cast<float>(crop.y);
                    globalFlipDecision = computeGlobalFlipDecision(
                        surf, dx0, dy0, u0, v0,
                        static_cast<float>(render_scale),
                        scale_seg, hasAffine, affineTransform,
                        meshCentroid);
                }

                const uint32_t tilesX_src = (static_cast<uint32_t>(tgt_size.width)  + tileW - 1) / tileW;
                const uint32_t tilesY_src = (static_cast<uint32_t>(tgt_size.height) + tileH - 1) / tileH;
                const size_t totalTiles = static_cast<size_t>(tilesX_src) * static_cast<size_t>(tilesY_src);
                std::atomic<size_t> tilesDone{0};

                #pragma omp parallel for schedule(dynamic) collapse(2)
                for (long long ty = 0; ty < static_cast<long long>(tilesY_src); ++ty) {
                    for (long long tx = 0; tx < static_cast<long long>(tilesX_src); ++tx) {
                        const uint32_t x0_src = static_cast<uint32_t>(tx) * tileW;
                        const uint32_t y0_src = static_cast<uint32_t>(ty) * tileH;
                        const uint32_t dx = std::min<uint32_t>(tileW, static_cast<uint32_t>(tgt_size.width)  - x0_src);
                        const uint32_t dy = std::min<uint32_t>(tileH, static_cast<uint32_t>(tgt_size.height) - y0_src);

                        // Generate base coordinates/normals for this tile once
                        float u0, v0;
                        computeTileOrigin(full_size,
                                          x0_src + static_cast<size_t>(crop.x),
                                          y0_src + static_cast<size_t>(crop.y),
                                          u0, v0);
                        cv::Mat_<cv::Vec3f> tilePoints, tileNormals;
                        genTile(surf, cv::Size(static_cast<int>(dx), static_cast<int>(dy)),
                                static_cast<float>(render_scale), u0, v0, tilePoints, tileNormals);

                        cv::Mat_<cv::Vec3f> basePoints, stepDirs;
                        prepareBasePointsAndStepDirs(
                            tilePoints, tileNormals,
                            scale_seg, ds_scale,
                            hasAffine, affineTransform,
                            globalFlipDecision,
                            basePoints, stepDirs);

                        std::vector<uint8_t> tileBuf(tileW * tileH, 0);
                        cv::Mat_<uint8_t> tileOut;
                        for (int zi = 0; zi < num_slices; ++zi) {
                            const float off = static_cast<float>(static_cast<double>(zi) - 0.5 * (static_cast<double>(num_slices) - 1.0));
                            renderSliceFromBase(tileOut, ds.get(), &chunk_cache, basePoints, stepDirs, off, static_cast<float>(ds_scale));

                            cv::Mat tileTransformed = tileOut;
                            rotateFlipIfNeeded(tileTransformed, rotQuad, flip_axis);

                            const uint32_t dty = static_cast<uint32_t>(tileTransformed.rows);
                            const uint32_t dtx = static_cast<uint32_t>(tileTransformed.cols);
                            std::fill(tileBuf.begin(), tileBuf.end(), 0);
                            for (uint32_t yy = 0; yy < dty; ++yy) {
                                const uint8_t* src = tileTransformed.ptr<uint8_t>(static_cast<int>(yy));
                                std::memcpy(tileBuf.data() + yy * tileW, src, dtx);
                            }

                            int dstTx, dstTy, rTilesX, rTilesY;
                            mapTileIndex(static_cast<int>(tx), static_cast<int>(ty),
                                         static_cast<int>(tilesX_src), static_cast<int>(tilesY_src),
                                         std::max(rotQuad, 0), flip_axis,
                                         dstTx, dstTy, rTilesX, rTilesY);
                            const uint32_t x0_dst = static_cast<uint32_t>(dstTx) * tileW;
                            const uint32_t y0_dst = static_cast<uint32_t>(dstTy) * tileH;

                            {
                                std::lock_guard<std::mutex> guard(tiffLocks[static_cast<size_t>(zi)]);
                                TIFF* tf = tiffs[static_cast<size_t>(zi)];
                                const uint32_t tileIndex = TIFFComputeTile(tf, x0_dst, y0_dst, 0, 0);
                                TIFFWriteEncodedTile(tf, tileIndex, tileBuf.data(), static_cast<tmsize_t>(tileBuf.size()));
                            }
                        }

                        size_t done = ++tilesDone;
                        int pct = static_cast<int>(100.0 * double(done) / double(totalTiles));
                        #pragma omp critical(progress_print)
                        {
                            std::cout << "\r[tif tiled] tiles " << done << "/" << totalTiles
                                      << " (" << pct << "%)" << std::flush;
                        }
                    }
                }

                for (auto* tf : tiffs) {
                    TIFFClose(tf);
                }
                std::cout << std::endl;


                delete surf;
                return;
            } catch (const std::exception& e) {
                std::cerr << "[tif tiled] error: " << e.what() << std::endl;
                delete surf;
                return;
            }
        }

        }

    delete surf;
    };


    if (has_render_folder) {
        // iterate through folders in render_folder_path
        for (const auto& entry : std::filesystem::directory_iterator(render_folder_path)) {
            if (!entry.is_directory()) continue;

            const std::string seg_name = entry.path().filename().string();
            const std::filesystem::path base(base_output_arg);

            std::filesystem::path out_arg_path;

            if (batch_format == "zarr") {
                // Treat -o as a DIRECTORY in batch-zarr mode so the wrapper can
                // find layers_* under that directory.
                // Write:  <-o>/<prefix>_<seg-name>.zarr
                // And TIFFs: <-o>/layers_<prefix>_<seg-name>
                std::filesystem::path base_dir;
                if ((std::filesystem::exists(base) && std::filesystem::is_directory(base)) || !base.has_extension()) {
                    base_dir = base;
                    std::error_code ec;
                    std::filesystem::create_directories(base_dir, ec); // best-effort
                } else {
                    // If -o looks like a file, fall back to its parent
                    base_dir = base.parent_path();
                    if (base_dir.empty()) base_dir = std::filesystem::current_path();
                }
                // Keep prior naming flavor by using the stem (or filename if empty) as prefix
                const std::string prefix = base.stem().string().empty()
                                             ? base.filename().string()
                                             : base.stem().string();
                out_arg_path = base_dir / (prefix + "_" + seg_name);
                std::cout << "[batch] writing Zarr under directory: "
                          << base_dir.string()
                          << "  name=" << (prefix + "_" + seg_name) << " (.zarr appended)\n";
            } else {
                // For TIFF, keep old behavior but handle absolute -o correctly
                out_arg_path = base.is_absolute() ? (base / seg_name)
                                                : (entry.path() / base);
            }

            process_one(entry.path(), out_arg_path.string(), batch_format == "zarr");
        }
    } else {
        process_one(seg_path, base_output_arg, false);
    }

    return EXIT_SUCCESS;
}
