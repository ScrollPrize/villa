#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include <educelab/core/utils/Filesystem.hpp>
#include <educelab/core/utils/String.hpp>

#include <indicators/progress_bar.hpp>

#include <openMVG/cameras/Camera_Pinhole.hpp>
#include <openMVG/geometry/Similarity3.hpp>
#include <openMVG/multiview/triangulation_nview.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_data_io.hpp>
#include <openMVG/sfm/sfm_data_transform.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "RANSAC.hpp"

namespace ar = cv::aruco;
namespace el = educelab;
namespace fs = std::filesystem;
namespace po = boost::program_options;
using namespace openMVG;
using namespace indicators;
using namespace ransac;

namespace pgs {
/** Index number of image in SfM */
using ViewID = openMVG::IndexT;
/** List of a landmark's observations and the view where it was found */
using Observations = std::vector<std::pair<ViewID, Vec2>>;

/** Landmark: A collection of 2D observations and the triangulated 3D point */
struct Landmark {
  Landmark() = default;
  explicit Landmark(std::string id) : id{std::move(id)} {}
  std::string id;
  Observations obs;
  std::optional<Vec3> X;
};

/** Collection of landmarks indexed by ID */
using Landmarks = std::map<std::string, Landmark>;
} // namespace pgs

namespace {
// Known distance maps for the EduceLab Sample Square
/*
using IDPair = std::pair<int, int>;
using DistanceMap = std::map<IDPair, double>;

auto init_distance_map() -> DistanceMap {
    // relative marker positions in cm
    std::array<cv::Vec2d, 4> markerPos = {{
        {0.866666666666667, 0.2},
        {0.2, 0.866666666666667},
        {1.533333333333334, 0.866666666666667},
        {0.866666666666667, 1.533333333333334}
    }};

    DistanceMap res;
    for (std::size_t i = 0; i < 4; i++) {
        for (std::size_t j = 0; j < 4; j++) {
            if (i == j or res.contains({i, j})) {
                continue;
            }
            const auto d = cv::norm(markerPos[j] - markerPos[i]);
            res.insert({{i, j}, d});
            res.insert({{j, i}, d});
            res.insert({{i + 512, j + 512}, d});
            res.insert({{j + 512, i + 512}, d});
        }
    }
    return res;
}

auto DistanceMapCM = init_distance_map();
*/

/** List of ArUco IDs */
using IDList = std::vector<int>;
/** List of a single ArUco marker's corner locations */
using CornersList = std::vector<cv::Point2f>;
/** List of ArUco markers */
using CornersArray = std::vector<CornersList>;
/** Result from running an ArUco detection method */
struct DetectionResult {
    IDList charucoIDs;
    CornersList charucoCorners;
    IDList markerIds;
    CornersArray markerCorners;
    CornersArray rejected;
};

struct RansacObservation {
  using Cam = std::shared_ptr<cameras::IntrinsicBase>;
  RansacObservation() = default;
  RansacObservation(const Vec2 &obs, const Vec3 &pt, const Cam &cam,
                    const geometry::Pose3 &pose)
      : obs{obs}, pt{pt}, cam{cam}, pose{pose} {}
  Vec2 obs;
  Vec3 pt;
  Cam cam;
  geometry::Pose3 pose;
};

auto Triangulate(const std::vector<RansacObservation> &x)
    -> std::pair<bool, Vec3> {
  // Unzip
  std::vector<Vec3> pts;
  std::vector<Mat34> poses;
  pts.reserve(x.size());
  poses.reserve(x.size());
  for (const auto &ro : x) {
    pts.push_back(ro.pt);
    poses.push_back(ro.pose.asMatrix());
  }

  // Fit
  const Map<const Mat3X> mtx(pts[0].data(), 3, pts.size());
  Vec4 Xh;
  if (not TriangulateNViewAlgebraic(mtx, poses, &Xh)) {
    return {false, {}};
  }
  Vec3 X = Xh.hnormalized();

  if (X.hasNaN()) {
    return {false, {}};
  }

  // Test validity (in front of the cameras)
  for (const auto &ro : x) {
    auto chirality = ro.pt.dot(ro.pose(X)) > 0.0;
    if (not chirality) {
      return {false, {}};
    }
  }

  return {true, X};
}

auto EvalTriangulate(const std::vector<RansacObservation> &x, const Vec3 &X)
    -> RANSACResult<RansacObservation, double> {
  using Result = RANSACResult<RansacObservation, double>;
  Result result;
  result.error = 0.;
  constexpr double threshold = 0.1;
  for (const auto &ro : x) {
    // If any views fail chirality, it's a bad model
    const auto chirality = ro.pt.dot(ro.pose(X)) > 0.0;
    if (not chirality) {
      return Result{};
    }
    // Accumulate the residual error
    const auto err = ro.cam->residual(ro.pose(X), ro.obs).norm();
    if (err < threshold) {
      result.error += err;
      result.inliers.push_back(ro);
    }
  }

  // calculate fitness and rmse
  if (result.inliers.size() > 0) {
    result.fitness = static_cast<double>(result.inliers.size()) /
                     static_cast<double>(x.size());
    result.inlier_rmse =
        result.error / std::sqrt(static_cast<double>(result.inliers.size()));
  }
  result.success = true;
  return result;
}

auto TriangulateRansac(const std::vector<RansacObservation> &x)
    -> std::pair<bool, Vec3> {
  constexpr std::size_t nIters = 1000;
  constexpr std::size_t nSamples = 2;
  // fixed seed for reproducibility
  constexpr std::uint_fast32_t seed = 0;
  const auto [X, res] =
      RANSAC(x, Triangulate, EvalTriangulate, nSamples, nIters, seed);
  return {res.success, X};
}

/** Detect ArUco markers */
auto DetectMarkers(const cv::Mat &image, const ar::DetectorParameters &params)
    -> DetectionResult {
  const auto dict = ar::getPredefinedDictionary(ar::DICT_ARUCO_ORIGINAL);
  const ar::ArucoDetector detector(dict, params);

  DetectionResult res;
  detector.detectMarkers(image, res.markerCorners, res.markerIds, res.rejected);

  return res;
}

/** (EduceLab Sample Square only) Generate a ChArUco board */
auto GenerateBoard(int offset = 0) {
    auto dict = ar::getPredefinedDictionary(ar::DICT_ARUCO_ORIGINAL);
    dict.bytesList = dict.bytesList({offset, offset + 4}, cv::Range::all());
    auto board = ar::CharucoBoard({3, 3}, 10, 7, dict);
    return board;
}

/** Detect a ChArUco board */
auto DetectBoard(const cv::Mat &image, const ar::CharucoBoard &board,
                 const ar::DetectorParameters &params) -> DetectionResult {
  // Adjust detector scale relative to largest dimension

  ar::CharucoParameters charucoParams;
  charucoParams.tryRefineMarkers = true;

  // Detect the Aruco markers
  const ar::CharucoDetector detector(board, charucoParams, params);
  DetectionResult res;
  detector.detectBoard(image, res.charucoCorners, res.charucoIDs,
                       res.markerCorners, res.markerIds);

  return res;
}

/** Detect the EduceLab Sample Square */
auto DetectSampleSquare(const cv::Mat &image,
                        const ar::DetectorParameters &params)
    -> DetectionResult {
  static const auto boardTop = GenerateBoard();
  auto res = DetectBoard(image, boardTop, params);
  if (res.charucoIDs.empty()) {
    res = DetectionResult();
  }

  static const auto boardBot = GenerateBoard(512);
  const auto res2 = DetectBoard(image, boardBot, params);
  if (res2.charucoIDs.empty()) {
    return res;
  }

  // Merge landmarks and IDs
  for (std::size_t idx = 0; idx < res2.markerIds.size(); ++idx) {
    res.markerIds.push_back(res2.markerIds[idx] + 512);
    res.markerCorners.push_back(res2.markerCorners[idx]);
  }
  for (std::size_t idx = 0; idx < res2.charucoIDs.size(); ++idx) {
    res.charucoIDs.push_back(res2.charucoIDs[idx] + 512);
    res.charucoCorners.push_back(res2.charucoCorners[idx]);
  }

  return res;
}

/**
 * Helper function to build the ID for a specific ArUco marker corner.
 *
 * CornerID:
 *  - 0: TL
 *  - 1: TR
 *  - 2: BR
 *  - 3: BL
 */
auto GetLandmarkID(const int arucoID, const int cornerID) -> std::string {
  return std::to_string(arucoID) + "." + std::to_string(cornerID);
}

/** Undistort an image using cv::undistort */
auto UndistortImage(const cv::Mat &image, cameras::IntrinsicBase *cam)
    -> cv::Mat {
  // Only support pinhole cameras
  if (not cameras::isPinhole(cam->getType())) {
    std::cout << "WARNING: Unsupported camera type! Undistortion skipped\n";
    return image;
  }

  // Basic pinhole has no distortion
  if (cam->getType() == cameras::PINHOLE_CAMERA) {
    return image;
  }

  // Get the intrinsic matrix
  auto pCam = dynamic_cast<cameras::Pinhole_Intrinsic *>(cam);
  cv::Mat mtx;
  cv::eigen2cv(pCam->K(), mtx);

  // Get the distortion parameters
  auto dist = cam->getParams();
  if (cam->getType() == cameras::PINHOLE_CAMERA_RADIAL1) {
    dist = {dist[3], 0., 0., 0.};
  } else if (cam->getType() == cameras::PINHOLE_CAMERA_RADIAL3) {
    dist = {dist[3], dist[4], 0., 0., dist[5]};
  } else if (cam->getType() == cameras::PINHOLE_CAMERA_BROWN) {
    dist = {dist[3], dist[4], dist[6], dist[7], dist[5]};
  } else if (cam->getType() == cameras::PINHOLE_CAMERA_FISHEYE) {
    dist = {dist[3], dist[4], 0., 0., dist[5], dist[6]};
  }

  // Calculate the new matrix for cv::undistort
  cv::Size size(image.cols, image.rows);
  cv::Rect roi;
  mtx = cv::getOptimalNewCameraMatrix(mtx, dist, size, 0., size, &roi);
  cv::Mat result;
  cv::undistort(image, result, mtx, dist);

  // Crop to the ROI
  cv::Mat ret;
  result(roi).copyTo(ret);

  return ret;
}

void ScaleLandmarks(pgs::Landmarks &ldms, const double scale) {
  for (auto &[_, ldm] : ldms) {
    if (ldm.X) {
      ldm.X.value() *= scale;
    }
  }
}

static uint32_t swap32(uint32_t v) {
  return ((v >> 24) & 0x000000FF) |
         ((v >>  8) & 0x0000FF00) |
         ((v <<  8) & 0x00FF0000) |
         ((v << 24) & 0xFF000000);
}

void ScalePLYMesh(
    const std::filesystem::path &inPath,
    const std::filesystem::path &outPath,
    double scale)
{
  // -- 1) Open input in binary mode, read header
  std::ifstream fin(inPath, std::ios::binary);
  if (!fin) throw std::runtime_error("Cannot open input mesh: " + inPath.string());

  enum Format { ASCII, BIN_LE, BIN_BE } format = ASCII;
  size_t vertCount = 0, faceCount = 0;
  bool inVertexElement = false;
  struct Property { std::string name; size_t size; size_t offset; };
  std::vector<Property> vertexProps;
  std::vector<std::string> headerLines;

  std::string line;
  size_t currentOffset = 0;
  std::streampos bodyStart = 0;

  auto sizeOfType = [&](const std::string &t){
    if (t=="char"||t=="int8")   return size_t(1);
    if (t=="uchar"||t=="uint8") return size_t(1);
    if (t=="short"||t=="int16") return size_t(2);
    if (t=="ushort"||t=="uint16") return size_t(2);
    if (t=="int"||t=="int32")   return size_t(4);
    if (t=="uint"||t=="uint32") return size_t(4);
    if (t=="float"||t=="float32") return size_t(4);
    if (t=="double"||t=="float64") return size_t(8);
    throw std::runtime_error("Unsupported PLY property type: " + t);
  };

  while (std::getline(fin, line)) {
    headerLines.push_back(line);
    std::istringstream iss(line);
    std::string token;
    iss >> token;
    if (token=="format") {
      std::string fmt; iss >> fmt;
      if (fmt=="ascii")               format = ASCII;
      else if (fmt=="binary_little_endian") format = BIN_LE;
      else if (fmt=="binary_big_endian")    format = BIN_BE;
    }
    else if (token=="element") {
      std::string name; iss >> name;
      if (name=="vertex") {
        inVertexElement = true;
        iss >> vertCount;
      } else {
        // next element; stop collecting vertex props
        inVertexElement = false;
        if (name=="face") iss >> faceCount;
      }
    }
    else if (inVertexElement && token=="property") {
      std::string sub; iss >> sub;
      if (sub=="list") {
        // we do not support list‐typed vertex props
        throw std::runtime_error("Unsupported PLY: vertex property is a list");
      }
      // scalar property: <type> <name>
      std::string propName; iss >> propName;
      size_t sz = sizeOfType(sub);
      vertexProps.push_back({propName, sz, currentOffset});
      currentOffset += sz;
    }
    else if (line=="end_header") {
      bodyStart = fin.tellg();
      break;
    }
  }

  if (vertCount==0 || vertexProps.empty())
    throw std::runtime_error("No vertex properties or zero vertices found");

  // find offsets of x,y,z
  auto findOffset = [&](const std::string &n){
    for (auto &p : vertexProps)
      if (p.name==n) return p.offset;
    throw std::runtime_error("PLY has no \""+n+"\" property");
  };
  size_t offX = findOffset("x");
  size_t offY = findOffset("y");
  size_t offZ = findOffset("z");
  size_t recordSize = currentOffset;

  // -- 2) Open output in binary mode, write header
  std::ofstream fout(outPath, std::ios::binary);
  if (!fout) throw std::runtime_error("Cannot open output mesh: " + outPath.string());
  for (auto &h : headerLines) {
    fout << h << "\n";
  }

  // -- 3) Seek input to body
  fin.clear();
  fin.seekg(bodyStart);

  if (format==ASCII) {
    // --- ASCII branch ---
    for (size_t i = 0; i < vertCount; ++i) {
      std::getline(fin, line);
      if (!fin) throw std::runtime_error("Unexpected EOF in ASCII body");
      std::istringstream iss(line);
      double x,y,z;
      iss >> x >> y >> z;
      x *= scale; y *= scale; z *= scale;
      // reconstruct: x y z + everything else
      std::string tail;
      std::getline(iss, tail);
      fout << x << " " << y << " " << z << tail << "\n";
    }
    // copy faces & any other ASCII data
    while (std::getline(fin, line)) {
      fout << line << "\n";
    }
  }
  else {
    // --- Binary branch ---
    bool needsSwap = (format==BIN_BE);
    std::vector<char> rec(recordSize);
    for (size_t i = 0; i < vertCount; ++i) {
      fin.read(rec.data(), recordSize);
      if (!fin) throw std::runtime_error("Unexpected EOF in binary body");
      // decode, swap if BE
      auto adjust = [&](size_t off){
        uint32_t iv;
        std::memcpy(&iv, rec.data()+off, sizeof(iv));
        if (needsSwap) iv = swap32(iv);
        float f;
        std::memcpy(&f, &iv, sizeof(f));
        f *= static_cast<float>(scale);
        std::memcpy(&iv, &f, sizeof(f));
        if (needsSwap) iv = swap32(iv);
        std::memcpy(rec.data()+off, &iv, sizeof(iv));
      };
      adjust(offX);
      adjust(offY);
      adjust(offZ);
      fout.write(rec.data(), recordSize);
    }
    // copy remainder (faces, other elements) verbatim
    constexpr size_t BUF = 1<<20;
    std::vector<char> buf(BUF);
    while (!fin.eof()) {
      fin.read(buf.data(), BUF);
      std::streamsize n = fin.gcount();
      if (n>0) fout.write(buf.data(), n);
    }
  }
}

void WriteOBJ(const fs::path &path, const pgs::Landmarks &ldms) {
  // Open the file
  std::ofstream file{path};
  if (not file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + path.string());
  }

  // Write vertices
  for (const auto &[_, ldm] : ldms) {
    if (ldm.X) {
      const auto &pt = ldm.X.value();
      file << "v " << pt.x() << " " << pt.y() << " " << pt.z() << "\n";
    }
  }

  // Close file
  file.flush();
  file.close();
  if (file.fail()) {
    throw std::runtime_error("Failed to write file: " + path.string());
  }
}

void WritePLY(const fs::path &path, const pgs::Landmarks &ldms) {
  // Iterate the vertices first
  std::size_t numVs{0};
  std::stringstream ss;
  for (const auto &[_, ldm] : ldms) {
    if (ldm.X) {
      ++numVs;
      const auto &pt = ldm.X.value();
      ss << pt.x() << " " << pt.y() << " " << pt.z() << " ";
      ss << 255 << " " << 255 << " " << 0 << "\n";
    }
  }

  // Open the file
  std::ofstream file{path};
  if (not file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + path.string());
  }
  // Write the header
  file << "ply\n";
  file << "format ascii 1.0\n";
  file << "element vertex " << numVs << "\n";
  file << "property float x\n";
  file << "property float y\n";
  file << "property float z\n";
  file << "property uchar red\n";
  file << "property uchar green\n";
  file << "property uchar blue\n";
  file << "end_header\n";

  // Write vertices
  file << ss.rdbuf();

  // Close file
  file.flush();
  file.close();
  if (file.fail()) {
    throw std::runtime_error("Failed to write file: " + path.string());
  }
}

void WriteMesh(const fs::path &path, const pgs::Landmarks &ldms) {
  if (el::is_file_type(path, "obj")) {
    WriteOBJ(path, ldms);
  } else if (el::is_file_type(path, "ply")) {
    WritePLY(path, ldms);
  } else {
    throw std::runtime_error("ERROR: Unrecognized mesh type: " +
                             path.extension().string());
  }
}

struct ScaleStats {
  std::vector<double> scales;  // all per-marker estimates
  double              summary; // median or mean, depending on method
};

/**
 * Compute per‐marker scale estimates via Umeyama + weighted median.
 */
auto ComputeUmeyamaScaleStats(
    const pgs::Landmarks &landmarks,
    const std::set<int>  &markerIDs,
    double                markerSize) -> ScaleStats
{
  struct SW { double scale, weight; };
  std::vector<SW> sws;
  sws.reserve(markerIDs.size());

  // Reference corner positions in marker‐local frame:
  const std::array<Vec3,4> refCorners = {{
    {0.0,          0.0,         0.0},
    {markerSize,   0.0,         0.0},
    {markerSize,   markerSize,  0.0},
    {0.0,          markerSize,  0.0}
  }};

  // Build one similarity per marker
  for (int mID : markerIDs) {
    std::vector<Vec3> P_ref, P_obs;
    P_ref.reserve(4); P_obs.reserve(4);

    for (int c = 0; c < 4; ++c) {
      auto it = landmarks.find(GetLandmarkID(mID, c));
      if (it != landmarks.end() && it->second.X) {
        P_ref.push_back(refCorners[c]);
        P_obs.push_back(it->second.X.value());
      }
    }
    if (P_obs.size() < 3) continue;  // need ≥3 to solve

    // Pack into 3×N mats
    const size_t N = P_obs.size();
    Mat3X M_ref(3, int(N)), M_obs(3, int(N));
    for (size_t i = 0; i < N; ++i) {
      M_ref.col(int(i)) = P_ref[i];
      M_obs.col(int(i)) = P_obs[i];
    }

    // Umeyama obs→ref
    Eigen::Matrix4d T = Eigen::umeyama(M_obs, M_ref, /*withScaling=*/true);

    // Extract scale = norm of first column of R*s
    double s = T.block<3,3>(0,0).col(0).norm();
    sws.push_back({s, static_cast<double>(N)});
  }

  ScaleStats stats;
  // collect just the scales for the histogram
  stats.scales.reserve(sws.size());
  for (auto &p : sws) stats.scales.push_back(p.scale);

  if (sws.empty()) {
    // no valid markers
    std::cerr << "WARNING: no markers yielded ≥3 corners; defaulting scale=1\n";
    stats.summary = 1.0;
  } else {
    // sort by scale
    std::sort(sws.begin(), sws.end(),
              [](auto &a, auto &b){ return a.scale < b.scale; });

    // total weight
    double W = 0.0;
    for (auto &p : sws) W += p.weight;
    double half = 0.5 * W;

    // precompute weighted mean (fallback)
    double mean_s = 0.0;
    for (auto &p : sws) mean_s += p.scale * p.weight;
    mean_s /= W;

    // default to mean
    stats.summary = mean_s;

    // try weighted median
    double cum = 0.0;
    bool found = false;
    for (auto &p : sws) {
      cum += p.weight;
      if (cum >= half) {
        stats.summary = p.scale;
        found = true;
        break;
      }
    }

    if (!found) {
      std::cerr << "WARNING: weighted median failed; falling back to weighted mean = "
                << stats.summary << "\n";
    }
  }

  return stats;
}

/**
 * Compute scale by measuring each marker’s edge length vs. the known markerSize.
 *
 * landmarks   : map from “markerID.cornerID” → triangulated Vec3  
 * markerIDs   : set of all detected marker IDs  
 * markerSize  : the *true* length (in your desired world units) of each marker edge  
 *
 * Returns a list of (expected/observed) scale factors for every adjacent‐corner pair,
 * plus their mean.
 */
auto ComputeEdgeScaleStats(
    const pgs::Landmarks &landmarks,
    const std::set<int>  &markerIDs,
    double                markerSize) -> ScaleStats
{
  ScaleStats stats;
  for (int mID : markerIDs) {
    for (int c = 0; c < 4; ++c) {
      int next = (c+1)%4;
      auto X0 = landmarks.at(GetLandmarkID(mID, c)).X;
      auto X1 = landmarks.at(GetLandmarkID(mID, next)).X;
      if (!X0 || !X1) continue;
      double observed = (X1.value() - X0.value()).norm();
      if (observed > 0.0)
        stats.scales.push_back(markerSize / observed);
    }
  }
  if (stats.scales.empty()) {
    stats.summary = 1.0;
  } else {
    double sum = std::accumulate(stats.scales.begin(),
                                 stats.scales.end(), 0.0);
    stats.summary = sum / stats.scales.size();
  }
  return stats;
}

/**
 * Write an SVG histogram of the given scales, with a red line at `median`.
 */
void WriteScaleHistogram(
    const std::string& path,
    const std::vector<double>& scales,
    double centerValue,
    const std::string& centerLabel)
{
    if (scales.empty()) {
        std::cerr << "Warning: no scale samples to histogram\n";
        return;
    }

    // Histogram parameters
    constexpr int nbins = 50;
    double minv = *std::min_element(scales.begin(), scales.end());
    double maxv = *std::max_element(scales.begin(), scales.end());
    double binw = (maxv - minv) / nbins;

    // Bin counts
    std::vector<int> hist(nbins, 0);
    for (double v : scales) {
        int b = std::min(int((v - minv) / binw), nbins - 1);
        hist[b]++;
    }
    int maxc = *std::max_element(hist.begin(), hist.end());

    // SVG canvas size
    const int SVGW = 800, SVGH = 600;
    const int M = 50;             // margin
    const int PW = SVGW - 2 * M;  // plot width
    const int PH = SVGH - 2 * M;  // plot height

    std::ofstream svg(path);
    if (!svg) {
        throw std::runtime_error("Cannot open histogram SVG for writing");
    }

    // SVG header
    svg << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
           "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n"
           "  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
           "<svg width=\"" << SVGW << "\" height=\"" << SVGH << "\"\n"
           "     xmlns=\"http://www.w3.org/2000/svg\"\n"
           "     style=\"background-color:white\">\n";

    // Axes
    svg << "<g stroke=\"black\" stroke-width=\"1\">\n"
           "  <line x1=\"" << M << "\" y1=\"" << (M + PH)
        << "\" x2=\"" << (M + PW) << "\" y2=\"" << (M + PH)
        << "\" />\n"
           "  <line x1=\"" << M << "\" y1=\"" << M
        << "\" x2=\"" << M << "\" y2=\"" << (M + PH)
        << "\" />\n"
           "</g>\n";

    // Ticks & labels
    constexpr int numXTicks = 10, numYTicks = 5, tickLen = 6;
    svg << "<g fill=\"black\" font-size=\"12\" font-family=\"sans-serif\">\n";
    // X ticks
    for (int i = 0; i <= numXTicks; ++i) {
        double x = M + i * double(PW) / numXTicks;
        double val = minv + i * (maxv - minv) / numXTicks;
        svg << "<line x1=\"" << x << "\" y1=\"" << (M + PH)
            << "\" x2=\"" << x << "\" y2=\"" << (M + PH + tickLen)
            << "\" stroke=\"black\" />\n";
        svg << "<text x=\"" << x << "\" y=\"" << (M + PH + tickLen + 15)
            << "\" text-anchor=\"middle\">"
            << std::fixed << std::setprecision(2) << val
            << "</text>\n";
    }
    // Y ticks
    for (int i = 0; i <= numYTicks; ++i) {
        double y = M + PH - i * double(PH) / numYTicks;
        int cnt = int(i * double(maxc) / numYTicks);
        svg << "<line x1=\"" << M << "\" y1=\"" << y
            << "\" x2=\"" << (M - tickLen) << "\" y2=\"" << y
            << "\" stroke=\"black\" />\n";
        svg << "<text x=\"" << (M - tickLen - 5) << "\" y=\"" << (y + 4)
            << "\" text-anchor=\"end\">" << cnt << "</text>\n";
    }
    svg << "</g>\n";

    // Bars
    double barW = double(PW) / nbins;
    svg << "<g fill=\"steelblue\">\n";
    for (int i = 0; i < nbins; ++i) {
        double x = M + i * barW;
        double h = double(hist[i]) / maxc * PH;
        double y = M + (PH - h);
        svg << "<rect x=\"" << x << "\" y=\"" << y
            << "\" width=\"" << (barW - 1) << "\" height=\"" << h
            << "\" />\n";
    }
    svg << "</g>\n";

    // Mean/Median line + label
    double lineX = M + (centerValue - minv) / (maxv - minv) * PW;
    svg << "<line x1=\"" << lineX << "\" y1=\"" << M
        << "\" x2=\"" << lineX << "\" y2=\"" << (M + PH)
        << "\" stroke=\"red\" stroke-width=\"2\" />\n";
    svg << "<text x=\"" << (lineX + 5) << "\" y=\"" << (M + 20)
        << "\" fill=\"red\" font-size=\"14\">"
        << centerLabel << " = "
        << std::fixed << std::setprecision(2) << centerValue
        << "</text>\n";

    // SVG footer
    svg << "</svg>\n";
    svg.close();
}

enum EXIT_CODE {
  SUCCESS = 0,
  HELP = 1,
  BAD_ARG = 2,
  NO_VIEWS = 3,
  NO_LDMS = 4,
  NO_SCALES = 5
};
}

auto LoadFilterFile(const fs::path &path) {
  // Open the file
  std::ifstream file{path};
  if (not file.is_open()) {
    throw std::runtime_error("Cannot open file for reading: " + path.string());
  }

  // Get all lines, excluding duplicates
  std::string line;
  std::unordered_set<std::string> lines;
  while (std::getline(file, line)) {
    lines.insert(line);
  }

  return lines;
}

auto main(int argc, char* argv[]) -> int
{
  // clang-format off
  po::options_description parser("options");
  parser.add_options()
    ("help,h", "print help message")
    ("input-scene,i", po::value<std::string>()->required(), "input sfm scene file")
    ("output-scene,o", po::value<std::string>(), "output sfm scene file")
    ("scale-method", po::value<std::string>()->default_value("umeyama"), "scale method: \"umeyama\" (weighted-median) or \"edge\" (mean of edge lengths)")
    ("input-mesh",  po::value<std::string>(), "input mesh file (ascii .ply only)")
    ("output-mesh", po::value<std::string>(), "output (scaled) mesh file (.ply)")
    ("histogram-out",         po::value<std::string>(), "where to write the scale‐histogram SVG")
    ("marker-size,s", po::value<double>()->required(), "ArUco marker size in desired world units")
    ("detection-method,m", po::value<std::string>()->default_value("markers"), "detection method: markers, sample-square")
    ("sfm-root", po::value<std::string>(), "use the given directory as the sfm root when loading image files")
    ("include-from", po::value<std::string>(), "only consider image files listed by name in the provided txt file")
    ("exclude-from", po::value<std::string>(), "do not consider any of the image files listed by name in the provided txt file")
    ("undistort-images", po::bool_switch(), "undistort images before running marker detection")
    ("min-marker-pix", po::value<int>()->default_value(32), "minimum marker size in pixels")
    ("detect-inverted", po::bool_switch(), "attempt to detect inverted markers")
    ("no-ransac", po::bool_switch(), "use RANSAC to make marker triangulation more resilient to false positive matches")
    ("save-debug-images", po::value<std::string>(), "save debug images to the given directory")
    ("save-landmarks", po::value<std::string>(), "save unscaled, triangulated landmarks to the given mesh file (obj, ply)")
    ("save-scaled-landmarks", po::value<std::string>(), "save scaled, triangulated landmarks to the given mesh file (obj, ply)")
    ("progress,p", po::bool_switch(), "Show progress bar")
  ;
  // clang-format on

  po::variables_map args;
  po::store(po::parse_command_line(argc, argv, parser), args);
  if (argc == 1 or args.count("help") > 0) {
    std::cout << parser << "\n";
    return HELP;
  }
  po::notify(args);

  std::string scaleMethod = el::to_lower_copy(
    args["scale-method"].as<std::string>()
  );
  if (scaleMethod != "umeyama" && scaleMethod != "edge") {
    std::cerr << "ERROR: --scale-method must be \"umeyama\" or \"edge\"\n";
    return EXIT_CODE::BAD_ARG;
  }
  // Get the input and output files
  fs::path sfmPath = args["input-scene"].as<std::string>();

  // Marker size (0.47 cm for the sample square)
  auto markerSize = args["marker-size"].as<double>();

  // Write a histogram if requested
  bool doHistogram = args.count("histogram-out") > 0;
  std::string histPath;
  if (doHistogram) histPath = args["histogram-out"].as<std::string>();

  // Detection method
  auto method = el::to_lower_copy(args["detection-method"].as<std::string>());
  std::function detect = DetectMarkers;
  if (method == "markers") {
    detect = DetectMarkers;
  } else if (method == "sample-square") {
    detect = DetectSampleSquare;
  } else {
    std::cout << "ERROR: Unrecognized detection method: \'" << method << "\'\n";
    return BAD_ARG;
  }
  ar::DetectorParameters params;
  params.useAruco3Detection = true;
  params.detectInvertedMarker = args["detect-inverted"].as<bool>();
  params.cornerRefinementMethod = ar::CORNER_REFINE_SUBPIX;
  auto minMarkerSize = static_cast<double>(args["min-marker-pix"].as<int>());

  // Boolean options
  auto undistortImages = args["undistort-images"].as<bool>();
  auto saveDebugImages = args.count("save-debug-images") > 0;

  // Load SfM file
  sfm::SfM_Data sfmData;
  sfm::Load(sfmData, sfmPath.string(), sfm::ALL);
  std::cout << "Loaded SfM scene: ";
  std::cout << sfmData.GetViews().size() << " views, ";
  std::cout << sfmData.GetPoses().size() << " poses, ";
  std::cout << sfmData.GetIntrinsics().size() << " intrinsics\n";

  fs::path sfmRoot = sfmData.s_root_path;
  if (args.count("sfm-root") > 0) {
    sfmRoot = args["sfm-root"].as<std::string>();
    std::cout << "Using custom SfM root: " << sfmRoot.string() << "\n";
  }

  // Set up debug directory
  fs::path debugDir;
  if (saveDebugImages) {
    debugDir = args["save-debug-images"].as<std::string>();
    fs::create_directories(debugDir);
  }

  // View filters
  using FilterFunction =
      std::function<bool(const std::shared_ptr<sfm::View> &)>;
  std::vector<FilterFunction> filters{
      [&sfmData](const std::shared_ptr<sfm::View> &view) {
        return sfmData.IsPoseAndIntrinsicDefined(view.get());
      }};
  if (args.count("include-from") > 0) {
    auto includes = LoadFilterFile(args["include-from"].as<std::string>());
    filters.emplace_back([includes](const std::shared_ptr<sfm::View> &view) {
      return includes.count(view->s_Img_path) > 0;
    });
  }
  if (args.count("exclude-from") > 0) {
    auto excludes = LoadFilterFile(args["exclude-from"].as<std::string>());
    filters.emplace_back([excludes](const std::shared_ptr<sfm::View> &view) {
      return excludes.count(view->s_Img_path) == 0;
    });
  }
  std::function filter =
      [&filters](const std::shared_ptr<sfm::View> &view) -> bool {
    return std::all_of(filters.begin(), filters.end(),
                       [&view](const auto &f) { return f(view); });
  };
  sfm::Views views;
  std::copy_if(sfmData.GetViews().begin(), sfmData.GetViews().end(),
               std::inserter(views, views.end()),
               [&filter](const auto &pair) { return filter(pair.second); });
  if (views.empty()) {
    std::cout << "ERROR: No views selected!\n";
    return NO_VIEWS;
  }

  // All observed markers
  std::set<int> markerIDs;
  // All observed marker corners
  pgs::Landmarks landmarks;

  // Detect landmarks
  std::string imgType = undistortImages ? "corrected" : "original";
  std::cout << "Detecting landmarks in " + imgType + " images (using "
            << views.size() << " views)\n";
  std::unique_ptr<ProgressBar> bar;
  std::size_t iter{0};
  auto numIters = views.size();
  auto pad = std::to_string(numIters).size();
  std::size_t viewsWithLandmarks{0};
  if (args["progress"].as<bool>()) {
    bar = std::make_unique<ProgressBar>(
        option::BarWidth{50}, option::Start{" ["},
        option::ForegroundColor{Color::unspecified},
        option::MaxProgress{numIters});
  }
  for (const auto &[viewID, view] : views) {
    // Ignore if it doesn't pass the filter
    if (not filter(view)) {
      if (bar) {
        bar->tick();
      }
      continue;
    }

    // Load the image
    auto path = view->s_Img_path;
    auto image = cv::imread(sfmRoot / path);

    // Undistort the images
    if (undistortImages) {
      auto cam = sfmData.intrinsics.at(view->id_intrinsic);
      image = UndistortImage(image, cam.get());
    }

    // Detect markers
    params.minMarkerLengthRatioOriginalImg =
        minMarkerSize / static_cast<double>(std::max(image.rows, image.cols));
    auto res = detect(image, params);

    if (not res.markerIds.empty()) {
      viewsWithLandmarks += 1;
    }

    // Draw markers
    if (saveDebugImages and not res.markerIds.empty()) {
      ar::drawDetectedMarkers(image, res.markerCorners, res.markerIds);
      if (not res.charucoIDs.empty()) {
        ar::drawDetectedCornersCharuco(image, res.charucoCorners, res.charucoIDs);
      }
      // Write marker image
      auto outFile = debugDir / fs::path(path).replace_extension("jpg");
      cv::imwrite(outFile, image);
    }

    // For each aruco ID found
    for (std::size_t idx = 0; idx < res.markerIds.size(); ++idx) {
      // Keep a list of all discovered markers
      const auto markerID = res.markerIds[idx];
      markerIDs.insert(markerID);

      // Track the marker corner observations
      const auto &corners = res.markerCorners[idx];
      for (int i = 0; i < corners.size(); i++) {
        // Get the stored landmark
        pgs::Landmark *ldm;
        const auto cornerID = GetLandmarkID(markerID, i);
        if (landmarks.count(cornerID) == 0) {
          landmarks.insert({cornerID, {}});
        }
        ldm = &landmarks.at(cornerID);

        auto corner = corners[i];
        ldm->obs.emplace_back(viewID, Vec2{corner.x, corner.y});
      }
    }
    ++iter;
    if (bar) {
      bar->set_option(option::PostfixText{el::to_padded_string(iter, pad, ' ') +
                                          "/" + std::to_string(numIters)});
      bar->tick();
    }
  }
  std::cout << "Detected landmarks in " << viewsWithLandmarks << " of "
            << views.size() << " views\n";

  // Triangulate the control points
  std::size_t numTriangulated{0};
  bool useRansac = not args["no-ransac"].as<bool>();
  auto postTxt = useRansac ? " w/RANSAC\n" : "\n";
  std::cout << "Triangulating landmarks" << postTxt;
  for (auto &[ldmID, ldm] : landmarks) {
    const auto &obs = ldm.obs;
    if (obs.size() < 3) {
      std::cout << "WARNING: Not enough observations to triangulate landmark ";
      std::cout << ldmID << "\n";
      continue;
    }

    // Collect observations and poses
    std::vector<RansacObservation> x;
    x.reserve(obs.size());
    for (const auto &[viewID, o] : obs) {
      auto view = sfmData.views.at(viewID);
      auto cam = sfmData.intrinsics.at(view->id_intrinsic);
      Mat3X pt;
      if (undistortImages) {
        pt = (*cam)(o);
      } else {
        pt = (*cam)(cam->get_ud_pixel(o));
      }

      auto pose = sfmData.GetPoseOrDie(view.get());
      x.emplace_back(o, pt, cam, pose);
    }

    // Triangulate
    bool success{false};
    Vec3 X;
    if (not args["no-ransac"].as<bool>()) {
      std::tie(success, X) = TriangulateRansac(x);
    } else {
      std::tie(success, X) = Triangulate(x);
    }

    if (not success) {
      std::cout << "WARNING: Could not triangulate landmark: " << ldmID << "\n";
      continue;
    }
    ldm.X = X;
    numTriangulated += 1;
  }

  // Need at least 2 triangulated points to even try measuring landmarks
  std::cout << "Triangulated " << numTriangulated << " of " << landmarks.size()
            << " landmarks\n";
  if (numTriangulated < 2) {
    std::cout << "ERROR: Not enough landmarks to estimate scale!\n";
    return NO_LDMS;
  }

  // Decide which stats to compute
  ScaleStats stats;
  if (scaleMethod == "edge") {
    stats = ComputeEdgeScaleStats(landmarks, markerIDs, markerSize);
    std::cout << "Edge-length mean scale: " << stats.summary << "\n";
  } else {
    stats = ComputeUmeyamaScaleStats(landmarks, markerIDs, markerSize);
    std::cout << "Umeyama median scale:   " << stats.summary << "\n";
  }

  if (doHistogram) {
      if (scaleMethod == "edge") {
          // For the edge-based method, label the line "mean"
          WriteScaleHistogram(histPath, stats.scales, stats.summary, "mean");
      } else {
          // For Umeyama, label the line "median"
          WriteScaleHistogram(histPath, stats.scales, stats.summary, "median");
      }
      std::cout << "Saved histogram → " << histPath << "\n";
  }

  // Scale and save the scene
  if (args.count("output-scene") > 0) {
    fs::path outPath = args["output-scene"].as<std::string>();
    std::cout << "Saving scaled SfM data\n";
    sfm::ApplySimilarity({{}, stats.summary}, sfmData);
    sfm::Save(sfmData, outPath.string(), sfm::ALL);
  }

  if (args.count("input-mesh") && args.count("output-mesh")) {
    ScalePLYMesh(
      fs::path(args["input-mesh"].as<std::string>()),
      fs::path(args["output-mesh"].as<std::string>()),
      stats.summary
    );
    std::cout << "Rescaled mesh → " << args["output-mesh"].as<std::string>() << "\n";
  }

  // Write the landmarks mesh
  if (args.count("save-landmarks") > 0 and not markerIDs.empty()) {
    std::cout << "Saving unscaled landmark mesh\n";
    fs::path ldmMesh = args["save-landmarks"].as<std::string>();
    fs::create_directories(fs::weakly_canonical(ldmMesh).parent_path());
    WriteMesh(ldmMesh, landmarks);
  }

  // Write the landmaFrks mesh
  if (args.count("save-scaled-landmarks") > 0 and not markerIDs.empty()) {
    std::cout << "Saving scaled landmark mesh\n";
    fs::path ldmMesh = args["save-scaled-landmarks"].as<std::string>();
    ScaleLandmarks(landmarks, stats.summary);
    fs::create_directories(fs::weakly_canonical(ldmMesh).parent_path());
    WriteMesh(ldmMesh, landmarks);
  }
  std::cout << "Done.\n";
}
