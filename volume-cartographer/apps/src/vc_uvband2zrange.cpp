// vc_uvband_to_zrange.cpp
//
// Compute minZ/maxZ in the original volume (level-0 coordinates) for a UV-band
// in the rendered image, using the same geometry as vc_render_tifxyz.
//
// Build against the same includes/libs you use for vc_render_tifxyz.

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <set>
#include <cmath>

namespace po = boost::program_options;
using json = nlohmann::json;

// ---------------- Affine helpers (same conventions as your renderer) ----------------

struct AffineTransform {
    cv::Mat_<double> matrix;  // 4x4
    AffineTransform() { matrix = cv::Mat_<double>::eye(4,4); }
};

static inline bool invertAffineInPlace(AffineTransform& T)
{
    cv::Mat A_cv(3,3, CV_64F);
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) A_cv.at<double>(r,c) = T.matrix(r,c);

    cv::Mat Ainv_cv;
    double det = cv::invert(A_cv, Ainv_cv, cv::DECOMP_LU);
    if (det < 1e-10) return false;

    cv::Matx33d Ainv;
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) Ainv(r,c) = Ainv_cv.at<double>(r,c);
    const cv::Vec3d t(T.matrix(0,3), T.matrix(1,3), T.matrix(2,3));
    const cv::Vec3d tinv = -(Ainv * t);

    for (int r=0;r<3;++r) {
        for (int c=0;c<3;++c) T.matrix(r,c) = Ainv(r,c);
        T.matrix(r,3) = tinv(r);
    }
    T.matrix(3,0)=0; T.matrix(3,1)=0; T.matrix(3,2)=0; T.matrix(3,3)=1;
    return true;
}

static inline AffineTransform composeAffine(const AffineTransform& A, const AffineTransform& B)
{
    AffineTransform R;
    cv::Mat M = B.matrix * A.matrix; // apply A first, then B
    M.copyTo(R.matrix);
    return R;
}

static inline std::pair<std::string, bool> parseAffineSpec(const std::string& spec)
{
    std::string path = spec;
    bool inv = false;
    const std::vector<std::string> suffixes = {":inv",":invert",":i"};
    for (const auto& s : suffixes) {
        if (spec.size() > s.size() && spec.substr(spec.size()-s.size()) == s) {
            inv = true; path = spec.substr(0, spec.size()-s.size()); break;
        }
    }
    return {path, inv};
}

static AffineTransform loadAffineTransform(const std::string& filename)
{
    AffineTransform T;
    std::ifstream f(filename);
    if (!f.is_open()) throw std::runtime_error("Cannot open affine: " + filename);
    json j; f >> j;
    if (j.contains("transformation_matrix")) {
        auto mat = j["transformation_matrix"];
        if (mat.size()!=3 && mat.size()!=4) throw std::runtime_error("Affine rows must be 3 or 4");
        for (int r=0;r<(int)mat.size();++r) {
            if (mat[r].size()!=4) throw std::runtime_error("Affine columns must be 4");
            for (int c=0;c<4;++c) T.matrix(r,c) = mat[r][c].get<double>();
        }
        if (mat.size()==4) {
            if (std::abs(T.matrix(3,0))>1e-12 || std::abs(T.matrix(3,1))>1e-12 ||
                std::abs(T.matrix(3,2))>1e-12 || std::abs(T.matrix(3,3)-1.0)>1e-12) {
                throw std::runtime_error("Bottom row must be [0 0 0 1]");
            }
        }
    }
    return T;
}

static inline void applyAffineToPoints(cv::Mat_<cv::Vec3f>& pts, const AffineTransform& T)
{
    for (int y=0;y<pts.rows;++y) {
        for (int x=0;x<pts.cols;++x) {
            cv::Vec3f& p = pts(y,x);
            if (std::isnan(p[0]) || std::isnan(p[1]) || std::isnan(p[2])) continue;
            const double px=p[0], py=p[1], pz=p[2];
            const double nx = T.matrix(0,0)*px + T.matrix(0,1)*py + T.matrix(0,2)*pz + T.matrix(0,3);
            const double ny = T.matrix(1,0)*px + T.matrix(1,1)*py + T.matrix(1,2)*pz + T.matrix(1,3);
            const double nz = T.matrix(2,0)*px + T.matrix(2,1)*py + T.matrix(2,2)*pz + T.matrix(2,3);
            p = cv::Vec3f((float)nx,(float)ny,(float)nz);
        }
    }
}

// ------------- small helpers reused from your renderer ------------------

static inline int normalizeQuadrantRotation(double angleDeg, double tolDeg = 0.5)
{
    double a = std::fmod(angleDeg, 360.0); if (a<0) a+=360.0;
    static const double q[4] = {0.0, 90.0, 180.0, 270.0};
    int best=0; double bestDiff = std::numeric_limits<double>::infinity();
    for (int i=0;i<4;++i) { double d=std::abs(a-q[i]); if (d<bestDiff){bestDiff=d;best=i;} }
    return (bestDiff<=tolDeg)?best:-1;
}

static inline void computeCanvasOrigin(const cv::Size& size, float& u0, float& v0)
{
    u0 = -0.5f * (float(size.width)  - 1.0f);
    v0 = -0.5f * (float(size.height) - 1.0f);
}

static inline void computeTileOrigin(const cv::Size& fullSize, size_t x0, size_t y0, float& u0, float& v0)
{
    computeCanvasOrigin(fullSize, u0, v0);
    u0 += float(x0);
    v0 += float(y0);
}

static inline void genTile(
    QuadSurface* surf, const cv::Size& size,
    float render_scale, float u0, float v0,
    cv::Mat_<cv::Vec3f>& points)
{
    cv::Mat_<cv::Vec3f> dummyNormals;
    surf->gen(&points, nullptr, size, cv::Vec3f(0,0,0), render_scale, cv::Vec3f(u0,v0,0));
}

// Map a horizontal band [ymin..ymax] in the FINAL image (after rot/flip) back
// to a rectangle on the pre-rotation cropped canvas.
static cv::Rect destBandToPreRect(const cv::Size& preSize, int rotQuad, int flipType,
                                  int outW, int outH, int bandY0, int bandY1)
{
    if (bandY0 > bandY1) std::swap(bandY0, bandY1);
    // Undo flips (vertical affects y). Horizontal flip does not change full-width band.
    int y0f = bandY0, y1f = bandY1;
    if (flipType == 0 || flipType == 2) {
        y0f = outH - 1 - bandY1;
        y1f = outH - 1 - bandY0;
    }
    // Undo rotation
    switch (rotQuad) {
        case -1: // treat as 0
        case 0:  return cv::Rect(0, y0f, preSize.width, y1f - y0f + 1);
        case 1:  // 90 CCW: band in y' -> band in x (pre)
            return cv::Rect(preSize.width - 1 - y1f, 0,
                            (y1f - y0f + 1), preSize.height);
        case 2:  // 180
            return cv::Rect(0, preSize.height - 1 - y1f,
                            preSize.width, (y1f - y0f + 1));
        case 3:  // 270 CCW: band in y' -> band in x (pre)
            return cv::Rect(y0f, 0, (y1f - y0f + 1), preSize.height);
        default: return cv::Rect(0, y0f, preSize.width, y1f - y0f + 1);
    }
}

// ------------------------------ main ------------------------------------

int main(int argc, char** argv)
{
    // CLI
    po::options_description req("Required");
    req.add_options()
        ("segmentation,s", po::value<std::string>()->required(), "Path to tifxyz segmentation folder")
        ("scale", po::value<float>()->required(), "Pixels per level-g voxel (Pg) used for rendering")
        ("group-idx,g", po::value<int>()->required(), "OME-Zarr group index used for rendering")
        ("uv-ymin", po::value<int>()->required(), "Band ymin in FINAL rendered image (after rotate/flip)")
        ("uv-ymax", po::value<int>()->required(), "Band ymax in FINAL rendered image (after rotate/flip)");

    po::options_description opt("Optional (match your render settings)");
    opt.add_options()
        ("help,h", "Show help")
        // affine stack (same semantics as vc_render_tifxyz)
        ("affine", po::value<std::vector<std::string>>()->multitoken()->composing(), "Affine JSON(s) in application order; append :inv to invert one")
        ("affine-invert", po::value<std::vector<int>>()->multitoken()->composing(), "Indices into --affine to invert")
        ("affine-transform", po::value<std::string>(), "[DEPRECATED] single affine")
        ("invert-affine", po::bool_switch()->default_value(false), "[DEPRECATED] invert single affine")
        ("scale-segmentation", po::value<float>()->default_value(1.0f), "Scale segmentation to dataset scale used in render")
        // geometry/image ops
        ("rotate", po::value<double>()->default_value(0.0), "Rotate output by 0/90/180/270 (like render)")
        ("flip", po::value<int>()->default_value(-1), "Flip: 0=vertical, 1=horizontal, 2=both (like render)")
        ("crop-x", po::value<int>()->default_value(0), "Crop x used in render")
        ("crop-y", po::value<int>()->default_value(0), "Crop y used in render")
        ("crop-width",  po::value<int>()->default_value(0), "Crop width (0 = no crop)")
        ("crop-height", po::value<int>()->default_value(0), "Crop height (0 = no crop)");

    po::options_description all("Usage");
    all.add(req).add(opt);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(all).run(), vm);
        if (vm.count("help") || argc < 2) { std::cout << all << "\n"; return 0; }
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n\n" << all << "\n";
        return 1;
    }

    // Inputs
    const std::string seg_path = vm["segmentation"].as<std::string>();
    const float tgt_scale = vm["scale"].as<float>();
    const int   group_idx = vm["group-idx"].as<int>();
    const float ds_scale  = std::ldexp(1.0f, -group_idx); // 2^(-g)
    const float scale_seg = vm["scale-segmentation"].as<float>();
    const double rotate_angle_in = vm["rotate"].as<double>();
    const int flip_axis = vm["flip"].as<int>();
    const int bandY0 = vm["uv-ymin"].as<int>();
    const int bandY1 = vm["uv-ymax"].as<int>();

    // Compose affine stack (optional)
    AffineTransform affine = {};
    bool hasAffine = false;
    std::vector<std::pair<std::string,bool>> affineSpecs;

    if (vm.count("affine")>0) {
        for (const auto& s : vm["affine"].as<std::vector<std::string>>())
            affineSpecs.emplace_back(parseAffineSpec(s));
    }
    if (vm.count("affine-transform")>0) {
        affineSpecs.emplace_back(vm["affine-transform"].as<std::string>(),
                                 vm["invert-affine"].as<bool>());
    }
    if (vm.count("affine-invert")>0 && !affineSpecs.empty()) {
        std::set<int> idxInv;
        for (int i : vm["affine-invert"].as<std::vector<int>>()) {
            if (i>=0 && i < (int)affineSpecs.size()) idxInv.insert(i);
        }
        int k=0; for (auto& spec : affineSpecs) { if (idxInv.count(k)) spec.second = true; ++k; }
    }
    if (!affineSpecs.empty()) {
        AffineTransform composed;
        int k=0;
        try {
            for (auto& [path, inv] : affineSpecs) {
                AffineTransform T = loadAffineTransform(path);
                if (inv) {
                    if (!invertAffineInPlace(T))
                        throw std::runtime_error("Non-invertible affine[" + std::to_string(k) + "]");
                }
                composed = composeAffine(composed, T);
                ++k;
            }
            affine = composed; hasAffine = true;
        } catch (const std::exception& e) {
            std::cerr << "Affine load/compose error: " << e.what() << "\n";
            return 1;
        }
    }

    // Load segmentation
    QuadSurface* surf = nullptr;
    try {
        surf = load_quad_from_tifxyz(seg_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load segmentation: " << e.what() << "\n";
        return 1;
    }

    // Prepare canvas size same way as in vc_render_tifxyz
    cv::Mat_<cv::Vec3f>* raw_points = surf->rawPointsPtr();
    // normalize -1 to NaN to avoid accidental interpolation downstream
    for (int j=0;j<raw_points->rows;++j)
        for (int i=0;i<raw_points->cols;++i)
            if ((*raw_points)(j,i)[0] == -1) (*raw_points)(j,i) = {NAN,NAN,NAN};

    cv::Size full_size = raw_points->size();

    // Isotropic scale from affine (determinant's cbrt of linear part)
    double sA = 1.0;
    if (hasAffine) {
        const cv::Matx33d A(
            affine.matrix(0,0), affine.matrix(0,1), affine.matrix(0,2),
            affine.matrix(1,0), affine.matrix(1,1), affine.matrix(1,2),
            affine.matrix(2,0), affine.matrix(2,1), affine.matrix(2,2)
        );
        const double detA = cv::determinant(cv::Mat(A));
        if (std::isfinite(detA) && std::abs(detA) > 1e-18) sA = std::cbrt(std::abs(detA));
    }
    const double Pg = double(tgt_scale);
    // Keep the exact formula used in vc_render_tifxyz:
    const double render_scale = Pg * (double(scale_seg) * sA * double(ds_scale));

    // Scale the canvas
    {
        const double sx = render_scale / double(surf->_scale[0]);
        const double sy = render_scale / double(surf->_scale[1]);
        full_size.width  = std::max(1, int(std::lround(full_size.width  * sx)));
        full_size.height = std::max(1, int(std::lround(full_size.height * sy)));
    }

    // Crop (same semantics as renderer)
    int crop_x = vm["crop-x"].as<int>();
    int crop_y = vm["crop-y"].as<int>();
    int crop_w = vm["crop-width"].as<int>();
    int crop_h = vm["crop-height"].as<int>();
    const cv::Rect canvasROI{0,0, full_size.width, full_size.height};
    cv::Rect crop = canvasROI;
    if (crop_w > 0 && crop_h > 0) {
        crop = (cv::Rect{crop_x, crop_y, crop_w, crop_h} & canvasROI);
        if (crop.width <=0 || crop.height <=0) {
            std::cerr << "Crop rectangle out of canvas\n";
            delete surf; return 1;
        }
    }
    cv::Size tgt_size = crop.size();

    // Output size after rotation
    const int rotQuad = normalizeQuadrantRotation(rotate_angle_in);
    if (std::abs(rotate_angle_in) > 1e-6 && rotQuad < 0) {
        std::cerr << "Only 0/90/180/270 rotations supported.\n";
        delete surf; return 1;
    }
    int outW = tgt_size.width;
    int outH = tgt_size.height;
    if (rotQuad >=0 && (rotQuad % 2) == 1) std::swap(outW, outH);

    if (bandY0 < 0 || bandY1 < 0 || bandY0 >= outH || bandY1 >= outH) {
        std::cerr << "Band [uv-ymin,uv-ymax] must be within [0," << (outH-1) << "]\n";
        delete surf; return 1;
    }

    // Map the band in FINAL image back to a rectangle on the pre-rotate target canvas
    cv::Rect preRect = destBandToPreRect(tgt_size, (rotQuad<0?0:rotQuad), flip_axis, outW, outH, bandY0, bandY1);
    // Clamp the pre-rect to the target canvas (safety)
    preRect &= cv::Rect(0,0, tgt_size.width, tgt_size.height);
    if (preRect.width<=0 || preRect.height<=0) {
        std::cerr << "Mapped pre-rotation rectangle is empty.\n";
        delete surf; return 1;
    }

    // Compute base origin for the pre-rot/cropped canvas
    float u0_base, v0_base;
    computeCanvasOrigin(full_size, u0_base, v0_base);
    u0_base += float(crop.x);
    v0_base += float(crop.y);

    // We will iterate the pre-rectangle in tiles to keep memory small
    const int TILE_H = 2048;
    const int W = preRect.width;
    const int H = preRect.height;

    double minZ =  std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();
    size_t counted = 0;

    for (int yOff = 0; yOff < H; yOff += TILE_H) {
        const int hThis = std::min(TILE_H, H - yOff);
        const int x0 = preRect.x;
        const int y0 = preRect.y + yOff;

        float u0, v0;
        computeTileOrigin(full_size,
                          size_t(x0) + size_t(crop.x),
                          size_t(y0) + size_t(crop.y),
                          u0, v0);

        cv::Mat_<cv::Vec3f> pts;
        genTile(surf, cv::Size(W, hThis), float(render_scale), u0, v0, pts);

        // Map to original full-res dataset coordinates: scale_seg then affine (NO ds_scale!)
        if (scale_seg != 1.0f) pts *= scale_seg;
        if (hasAffine) applyAffineToPoints(pts, affine);

        // Collect Z range (component index 2 is Z from tifxyz: x=0,y=1,z=2)
        for (int yy=0; yy<hThis; ++yy) {
            const cv::Vec3f* row = pts.ptr<cv::Vec3f>(yy);
            for (int xx=0; xx<W; ++xx) {
                const cv::Vec3f& p = row[xx];
                if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])) {
                    const double z = (double)p[2];
                    if (z < minZ) minZ = z;
                    if (z > maxZ) maxZ = z;
                    ++counted;
                }
            }
        }

        // OPTIONAL_NORMAL_BAND:
        // If you want Z-range across a normal offset band [offMin, offMax], do:
        //   z(off) = (p.z + off * nz), where nz is the z-component of the unit
        //   normal after affine (NOT scaled by ds_scale since we want level-0).
        //   Per-pixel min is min(z(offMin), z(offMax)).
        // Hint: generate normals with surf->gen(..., normals=true), then
        //       transform normals with the correct inverse-transpose of the
        //       affine linear part (see your renderer), normalize them,
        //       and evaluate the two endpoints.
    }

    delete surf;

    if (!std::isfinite(minZ) || !std::isfinite(maxZ)) {
        std::cerr << "No valid points in the selected band.\n";
        return 2;
    }

    // Report
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(3);
    std::cout << "Selected band (FINAL image rows): [" << bandY0 << ", " << bandY1 << "]\n";
    std::cout << "Mapped pre-rotation rectangle: x=" << preRect.x << " y=" << preRect.y
              << " w=" << preRect.width << " h=" << preRect.height << "\n";
    std::cout << "Sampled points: " << counted << "\n\n";
    std::cout << "Z range in ORIGINAL (level-0) volume coordinates:\n";
    std::cout << "  minZ = " << minZ << "\n";
    std::cout << "  maxZ = " << maxZ << "\n";
    std::cout.unsetf(std::ios::floatfield);

    return 0;
}
