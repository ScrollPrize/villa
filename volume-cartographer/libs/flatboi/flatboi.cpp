// flatboi.cpp
#include <igl/slim.h>
#include <igl/MappingEnergyType.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/read_triangle_mesh.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <limits>
#include <array>
#include <chrono>
#include <iomanip>
#include <csignal>
#include <sstream>

// OpenCV for compressed image writing
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

// ------------------------------
// Minimal W&B pipe logger via temp Python file (no-op if python/wandb missing)
// ------------------------------
class WBLogger {
public:
  WBLogger(const std::string& obj_path, int n_iters) {
    // Allow explicit opt-out. If WANDB_DISABLED is set and truthy, skip logger entirely.
    if (const char* d = std::getenv("WANDB_DISABLED")) {
      std::string v(d);
      std::transform(v.begin(), v.end(), v.begin(), ::tolower);
      if (v=="1" || v=="true" || v=="yes") { enabled_ = false; return; }
    }
    // Derive run name: <volume or parent>_<stem>_<UTC-YYYYMMDD-HHMMSS>
    std::string vol_or_parent = "flatboi";
    try {
      fs::path p = fs::absolute(obj_path);
      for (auto parent = p.parent_path(); !parent.empty(); parent = parent.parent_path()) {
        auto name = parent.filename().string();
        if (name.size() >= 7 && name.substr(name.size()-7) == ".volpkg") {
          vol_or_parent = parent.stem().string();
          break;
        }
        if (parent.parent_path().empty()) vol_or_parent = p.parent_path().filename().string();
      }
    } catch (...) {}
    fs::path pp(obj_path);
    std::string stem = pp.stem().string();

    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &tt);
#else
    gmtime_r(&tt, &tm_utc);
#endif
    std::ostringstream ts; ts << std::put_time(&tm_utc, "%Y%m%d-%H%M%S");
    run_name_ = vol_or_parent + "_" + stem + "_" + ts.str();

    setenv_if_absent("WANDB_PROJECT", "flattening");
    setenv_var("WB_RUN_NAME", run_name_.c_str());

    // Write a temp Python script (only double quotes inside!)
    try {
      script_path_ = fs::temp_directory_path() / ("wblogger_" + run_name_ + ".py");
    } catch (...) {
      enabled_ = false; return;
    }

    const char* py = R"PY(
import os, sys, json

def consume_noop():
    # Consume stdin to avoid broken pipe; exit when told to finish
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if msg.get("type") == "finish":
            break
    sys.exit(0)

try:
    import wandb
except Exception:
    # wandb not installed -> silently no-op
    consume_noop()

entity  = os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_ORG")
project = os.environ.get("WANDB_PROJECT", "flattening")
name    = os.environ.get("WB_RUN_NAME", "flatboi")

# Try to login from env/cached config; if it fails, become a no-op
try:
    key = os.environ.get("WANDB_API_KEY")
    if key:
        try:
            wandb.login(key=key, relogin=True)
        except Exception:
            pass
    else:
        # Use cached credentials if present; don't prompt in this non-tty
        try:
            wandb.login(relogin=False)
        except Exception:
            pass

    run = wandb.init(project=project, entity=entity, name=name)
    wandb.define_metric("iter")
    wandb.define_metric("symmetric_dirichlet", step_metric="iter")
    wandb.define_metric("stretch/*", step_metric="iter")

except Exception:
    # Any failure (e.g., no API key, no cached login) -> no-op
    consume_noop()

# Normal logging path
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        continue

    t = msg.get("type", "log")
    if t == "config":
        try:
            run.config.update(msg.get("data", {}), allow_val_change=True)
        except Exception:
            pass
    elif t == "image":
        key  = msg.get("key")
        path = msg.get("path")
        if key and path:
            try:
                wandb.log({key: wandb.Image(path)})
            except Exception:
                pass
    elif t == "log":
        try:
            wandb.log(msg.get("data", {}))
        except Exception:
            pass
    elif t == "finish":
        break

wandb.finish()
)PY";

    std::ofstream pyf(script_path_);
    if(!pyf){ enabled_ = false; return; }
    pyf << py;
    pyf.close();

    // Build command (quote the path)
#if defined(_WIN32)
    std::string cmd = "python -u \"" + script_path_.string() + "\"";
    pipe_ = _popen(cmd.c_str(), "w");
#else
    // Robust launcher:
    //  - Prefer python3 if present
    //  - Fallback to python (Micromamba/Conda envs commonly expose only 'python')
    //  - Final fallback to 'cat >/dev/null' so writes never SIGPIPE even if Python is missing
    // Also, ignore SIGPIPE so a dying child cannot terminate flatboi.
    std::string quoted = "\"" + script_path_.string() + "\"";
    std::string cmd =
      "(command -v python3 >/dev/null 2>&1 && exec python3 -u " + quoted +
      " || command -v python  >/dev/null 2>&1 && exec python  -u " + quoted +
      " || exec cat >/dev/null)";
    // Hardening: ensure writes to a closed pipe don't kill the process.
    // (Linux lacks O_NOSIGPIPE for pipes; MSG_NOSIGNAL doesn't apply to stdio.)
    std::signal(SIGPIPE, SIG_IGN);

    pipe_ = popen(cmd.c_str(), "w");
#endif
    if (!pipe_) { enabled_ = false; cleanup_script(); return; }

    enabled_ = send_config(obj_path, n_iters);
    if (!enabled_) { close_pipe(); cleanup_script(); }
  }

  ~WBLogger() { finish(); }

  bool enabled() const { return enabled_; }

  void log_energy(double val, int step) {
    if (!enabled_) return;
    std::ostringstream js;
    js << "{\"type\":\"log\",\"data\":{\"symmetric_dirichlet\":" << val << ",\"iter\":" << step << "}}\n";
    write_line(js.str());
  }

  void log_stretch(double l2_mean, double l2_median, double linf, double area_err, int step) {
    if (!enabled_) return;
    std::ostringstream js;
    js << "{\"type\":\"log\",\"data\":{\"iter\":" << step
       << ",\"stretch/l2_mean\":"   << l2_mean
       << ",\"stretch/l2_median\":" << l2_median
       << ",\"stretch/linf\":"      << linf
       << ",\"stretch/area_error\":"<< area_err
       << "}}\n";
    write_line(js.str());
  }

  void log_image(const std::string& key, const std::string& path) {
    if (!enabled_) return;
    std::ostringstream js;
    js << "{\"type\":\"image\",\"key\":\"" << json_escape(key)
       << "\",\"path\":\"" << json_escape(path) << "\"}\n";
    write_line(js.str());
  }

  void finish() {
    if (finished_) return;
    finished_ = true;
    if (enabled_) {
      write_line(std::string("{\"type\":\"finish\"}\n"));
    }
    close_pipe();
    cleanup_script();
    enabled_ = false;
  }

private:
  bool enabled_ = false;
  bool finished_ = false;
  std::string run_name_;
  fs::path script_path_;
#if defined(_WIN32)
  FILE* pipe_ = nullptr;
  static void setenv_var(const char* k, const char* v) { _putenv_s(k, v); }
  static void setenv_if_absent(const char* k, const char* v) {
    size_t len=0; getenv_s(&len,nullptr,0,k);
    if (len==0) _putenv_s(k,v);
  }
  void close_pipe(){ if(pipe_){ _pclose(pipe_); pipe_=nullptr; } }
#else
  FILE* pipe_ = nullptr;
  static void setenv_var(const char* k, const char* v) { setenv(k, v, 1); }
  static void setenv_if_absent(const char* k, const char* v) {
    if (!std::getenv(k)) setenv(k, v, 1);
  }
  void close_pipe(){ if(pipe_){ pclose(pipe_); pipe_=nullptr; } }
#endif

  void cleanup_script(){
    std::error_code ec; if(!script_path_.empty()) fs::remove(script_path_, ec);
  }

  static std::string json_escape(const std::string& s){
    std::string out; out.reserve(s.size()+8);
    for(char c: s){
      switch(c){
        case '\"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b";  break;
        case '\f': out += "\\f";  break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default: out += (unsigned char)c < 0x20 ? '?' : c;
      }
    }
    return out;
  }

  bool write_line(const std::string& s){
    if(!pipe_) return false;
    // Best-effort guard: if the consumer vanished despite our fallbacks,
    // avoid UB from fwrite size mismatch + SIGPIPE already ignored.
    if (std::fwrite(s.data(), 1, s.size(), pipe_) != s.size()) return false;
    std::fflush(pipe_);
    return !std::ferror(pipe_);
  }

  bool send_config(const std::string& obj_path, int n_iters){
    std::ostringstream js;
    js << "{\"type\":\"config\",\"data\":{\"obj_path\":\"" << json_escape(obj_path)
       << "\",\"iterations\":" << n_iters << "}}\n";
    return write_line(js.str());
  }
};

// ------------------------------
// Flatboi
// ------------------------------
struct Flatboi {
  std::string input_obj;
  int max_iter = 50;

  Eigen::MatrixXd V;          // #V x 3
  Eigen::MatrixXi F;          // #F x 3
  Eigen::MatrixXd TC;         // #TC x 2
  Eigen::MatrixXi FTC;        // #F x 3
  bool have_per_corner_uv = false;

  Flatboi(const std::string& obj_path, int max_iter_) : input_obj(obj_path), max_iter(max_iter_) {
    read_mesh();
  }

  void read_mesh() {
    Eigen::MatrixXd N; Eigen::MatrixXi FN;
    if(!igl::readOBJ(input_obj, V, TC, N, F, FTC, FN)) {
      throw std::runtime_error("Failed to read OBJ: " + input_obj);
    }
    have_per_corner_uv = (TC.size() > 0 && FTC.size() > 0);
    if(F.cols() != 3) throw std::runtime_error("Expecting a triangle mesh (F.cols()==3).");
    if(V.rows() == 0 || F.rows() == 0) throw std::runtime_error("Empty mesh.");
  }

  Eigen::VectorXi generate_boundary() const {
    Eigen::VectorXi bnd;
    igl::boundary_loop(F, bnd);
    if (bnd.size() == 0) throw std::runtime_error("Mesh has no boundary (closed surface).");
    return bnd;
  }

  void harmonic_ic(Eigen::VectorXi& bnd, Eigen::MatrixXd& bnd_uv, Eigen::MatrixXd& uv) const {
    bnd = generate_boundary();
    igl::map_vertices_to_circle(V, bnd, bnd_uv);
    if(!igl::harmonic(V, F, bnd, bnd_uv, 1, uv)) {
      throw std::runtime_error("harmonic_ic: igl::harmonic failed.");
    }
  }

  void original_ic(Eigen::VectorXi& bnd, Eigen::MatrixXd& bnd_uv, Eigen::MatrixXd& uv) const {
    uv.setZero(V.rows(), 2);
    if(have_per_corner_uv) {
      for (int t = 0; t < F.rows(); ++t)
        for (int v = 0; v < F.cols(); ++v) {
          int vi  = F(t,v);
          int tci = FTC(t,v);
          uv.row(vi) = TC.row(tci);
        }
    }
    bnd = generate_boundary();
    bnd_uv.resize(bnd.size(), 2);
    for (int i = 0; i < bnd.size(); ++i) bnd_uv.row(i) = uv.row(bnd(i));
  }

  static Eigen::MatrixXd shift_uv(const Eigen::MatrixXd& uv) {
    Eigen::RowVector2d mn = uv.colwise().minCoeff();
    return uv.rowwise() - mn;
  }

  static void triangle_stretch(
      const Eigen::Matrix<double,3,3>& t3d,
      const Eigen::Matrix<double,3,2>& t2d,
      double& L2, double& G, double& area3d, double& area2d_abs)
  {
    using std::sqrt;
    const Eigen::RowVector3d q1 = t3d.row(0), q2 = t3d.row(1), q3 = t3d.row(2);
    const double s1=t2d(0,0), t1=t2d(0,1);
    const double s2=t2d(1,0), t2v=t2d(1,1);
    const double s3=t2d(2,0), t3v=t2d(2,1);

    const double A = ((s2 - s1) * (t3v - t1) - (s3 - s1) * (t2v - t1)) / 2.0;
    if (std::abs(A) < 1e-30) { L2=G=area3d=area2d_abs=0.0; return; }

    const Eigen::RowVector3d Ss = ( q1*(t2v - t3v) + q2*(t3v - t1) + q3*(t1 - t2v) ) / (2.0*A);
    const Eigen::RowVector3d St = ( q1*(s3 - s2)   + q2*(s1 - s3)   + q3*(s2 - s1)   ) / (2.0*A);

    const double a = Ss.dot(Ss), b = Ss.dot(St), c = St.dot(St);
    G  = sqrt( ((a + c) + sqrt((a - c)*(a - c) + 4.0*b*b)) / 2.0 );
    L2 = sqrt( (a + c) / 2.0 );

    const double ab = (q2 - q1).norm(), bc = (q3 - q2).norm(), ca = (q1 - q3).norm();
    const double s  = (ab + bc + ca) / 2.0;
    area3d = sqrt(std::max(0.0, s*(s - ab)*(s - bc)*(s - ca)));
    area2d_abs = std::abs(A);
  }

  static double weighted_median(const std::vector<double>& data, const std::vector<double>& weights) {
    std::vector<size_t> idx(data.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j){ return data[i] < data[j]; });
    double wsum = 0.0; for (double w: weights) wsum += w;
    double acc = 0.0;
    for (size_t k = 0; k < idx.size(); ++k) {
      acc += weights[idx[k]];
      if (acc >= 0.5 * wsum) return data[idx[k]];
    }
    return data[idx.back()];
  }

  // returns (L2_mean, L2_median, Linf, area_error)
  std::tuple<double,double,double,double> stretch_metrics(const Eigen::MatrixXd& uv_in) const {
    bool per_vertex = (uv_in.rows() == V.rows() && uv_in.cols() == 2);
    const int T = F.rows();
    std::vector<double> l2_all(T,0.0), linf_all(T,0.0), area3d_all(T,0.0), area2d_all(T,0.0), per_tri_area(T,0.0);
    double nom=0.0, sum_area3d=0.0;

    for (int t = 0; t < T; ++t) {
      Eigen::Matrix<double,3,3> tri3d;
      tri3d.row(0)=V.row(F(t,0)); tri3d.row(1)=V.row(F(t,1)); tri3d.row(2)=V.row(F(t,2));
      Eigen::Matrix<double,3,2> tri2d;
      if (per_vertex) {
        tri2d.row(0)=uv_in.row(F(t,0)); tri2d.row(1)=uv_in.row(F(t,1)); tri2d.row(2)=uv_in.row(F(t,2));
      } else if (have_per_corner_uv) {
        tri2d.row(0)=TC.row(FTC(t,0));  tri2d.row(1)=TC.row(FTC(t,1));  tri2d.row(2)=TC.row(FTC(t,2));
      } else tri2d.setZero();

      double L2,G,A3,A2; triangle_stretch(tri3d,tri2d,L2,G,A3,A2);
      linf_all[t]=G; area3d_all[t]=A3; area2d_all[t]=A2; l2_all[t]=L2*L2;
      nom += l2_all[t]*area3d_all[t]; sum_area3d += area3d_all[t];
    }

    const double l2_mean   = (sum_area3d>0.0)? std::sqrt(nom/sum_area3d) : 0.0;
    const double l2_median = weighted_median(l2_all, area3d_all);
    const double linf      = *std::max_element(linf_all.begin(), linf_all.end());

    double sumA3=0.0,sumA2=0.0; for (int t=0;t<T;++t){ sumA3+=area3d_all[t]; sumA2+=area2d_all[t]; }
    for (int t=0;t<T;++t){
      const double alpha=(sumA3>0)? area3d_all[t]/sumA3 : 0.0;
      const double beta =(sumA2>0)? area2d_all[t]/sumA2 : 0.0;
      per_tri_area[t] = (alpha>beta) ? (1.0 - (beta/(alpha + 1e-30))) : (1.0 - (alpha/(beta + 1e-30)));
    }
    double area_err=0.0; for (double v: per_tri_area) area_err += v; area_err /= std::max(1, T);
    return {l2_mean, l2_median, linf, area_err};
  }

  // Run SLIM; initial_condition = "original" or "harmonic"
  std::pair<Eigen::MatrixXd, std::vector<double>>
  slim_run(const std::string& initial_condition, WBLogger* wblog=nullptr) {
    Eigen::VectorXi bnd; Eigen::MatrixXd bnd_uv, uv;

    if (initial_condition == "original") {
      original_ic(bnd, bnd_uv, uv);
      auto [l2m, l2med, linf, area] = stretch_metrics(uv);
      std::cout << "Starting metrics from VC3D flattening -- "
                << "L2(mean): " << l2m << ", L2(median): " << l2med
                << ", Linf: " << linf << ", Area Error: " << area << "\n";
      if (wblog) wblog->log_stretch(l2m, l2med, linf, area, /*step=*/0);
    } else if (initial_condition == "harmonic") {
      harmonic_ic(bnd, bnd_uv, uv);
    } else {
      throw std::runtime_error("Unknown initial_condition: " + initial_condition);
    }

    igl::SLIMData data;
    igl::slim_precompute(
      V, F, uv, data,
      igl::MappingEnergyType::SYMMETRIC_DIRICHLET,
      bnd, bnd_uv,
      /*soft_penalty=*/0.0
    );

    std::vector<double> energies;
    energies.reserve(max_iter + 1);
    energies.push_back(data.energy); // iter=0
    if (wblog) wblog->log_energy(data.energy, /*step=*/0);

    std::cout << "PROGRESS 0/" << max_iter << std::endl;

    for (int it = 0; it < max_iter; ++it) {
      igl::slim_solve(data, 1);
      energies.push_back(data.energy);
      std::cout << "[it " << (it+1) << "] E = " << data.energy << std::endl; // flush
      std::cout << "PROGRESS " << (it+1) << "/" << max_iter << std::endl;
      if (wblog) wblog->log_energy(data.energy, /*step=*/it+1);

      if (((it+1) % 20) == 0) {
        auto [l2m, l2med, linf, area] = stretch_metrics(data.V_o.leftCols<2>());
        std::cout << "  stretch: L2(mean)=" << l2m
                  << ", L2(median)=" << l2med
                  << ", Linf=" << linf
                  << ", AreaErr=" << area << "\n";
        if (wblog) wblog->log_stretch(l2m,l2med,linf,area, /*step=*/it+1);
      }
    }

    auto [l2m_f, l2med_f, linf_f, area_f] = stretch_metrics(data.V_o.leftCols<2>());
    std::cout << "Final stretch -- L2(mean): " << l2m_f
              << ", L2(median): " << l2med_f
              << ", Linf: " << linf_f
              << ", Area Error: " << area_f << "\n";
    if (wblog && (max_iter % 20) != 0) {
      wblog->log_stretch(l2m_f, l2med_f, linf_f, area_f, /*step=*/max_iter);
    }

    return { data.V_o.leftCols<2>(), energies };
  }

  void save_obj(const Eigen::MatrixXd& uv) const {
    const fs::path in = fs::path(input_obj);
    const fs::path out = in.parent_path() / (in.stem().string() + "_flatboi.obj");

    Eigen::MatrixXd shifted = shift_uv(uv);
    Eigen::MatrixXd UVc(F.rows()*3, 2);
    Eigen::MatrixXi FUV(F.rows(), 3);
    for (int t = 0; t < F.rows(); ++t) {
      for (int v = 0; v < 3; ++v) {
        const int corner_idx = t*3 + v;
        const int vi = F(t,v);
        UVc.row(corner_idx) = shifted.row(vi);
        FUV(t,v) = corner_idx;
      }
    }

    Eigen::MatrixXd CN; CN.resize(0,3);
    Eigen::MatrixXi FN; FN.resize(0,3);
    if (!igl::writeOBJ(out.string(), V, F, CN, FN, UVc, FUV)) {
        throw std::runtime_error("Failed to write OBJ with UVs");
    }
    std::cout << "Wrote: " << out << "\n";
  }

  // === UV heatmaps (OpenCV PNG write; no clipping; V increases downward) ===
  struct HeatCfg {
    int  target_max_dim;
    int  margin_px;
    int  supersample;
    int  png_compression; // 0..9
    HeatCfg(): target_max_dim(2048), margin_px(8), supersample(1), png_compression(9) {}
  };

  static inline double clamp01(double x){ return std::max(0.0, std::min(1.0,x)); }
  template <typename T> static inline T clamp(T v, T lo, T hi){ return std::max(lo, std::min(hi, v)); }
  static inline double lerp(double a,double b,double t){ return a + (b-a)*t; }

  static inline Eigen::RowVector3i hsv2rgb(double h_deg, double s=1.0, double v=1.0){
    h_deg = fmod(fmod(h_deg,360.0)+360.0,360.0);
    double c = v*s;
    double x = c*(1.0 - std::fabs(fmod(h_deg/60.0,2.0)-1.0));
    double m = v - c;
    double r=0,g=0,b=0;
    if(h_deg < 60)       { r=c; g=x; b=0; }
    else if(h_deg < 120) { r=x; g=c; b=0; }
    else if(h_deg < 180) { r=0; g=c; b=x; }
    else if(h_deg < 240) { r=0; g=x; b=c; }
    else if(h_deg < 300) { r=x; g=0; b=c; }
    else                 { r=c; g=0; b=x; }
    return Eigen::RowVector3i(
      int(std::clamp((r+m)*255.0, 0.0, 255.0)),
      int(std::clamp((g+m)*255.0, 0.0, 255.0)),
      int(std::clamp((b+m)*255.0, 0.0, 255.0)));
  }
  static inline Eigen::RowVector3i cm_rainbow(double t){ return hsv2rgb(lerp(240.0, 0.0, clamp01(t)), 1.0, 1.0); }
  static inline Eigen::RowVector3i cm_diverging(double x){
    x = std::clamp(x,-1.0,1.0);
    if(x < 0.0){
      double t = -x;
      Eigen::RowVector3d c0(49,54,149), c1(255,255,255);
      Eigen::RowVector3d c = (1.0-t)*c1 + t*c0;
      return Eigen::RowVector3i( (int)c[0], (int)c[1], (int)c[2] );
    }else{
      double t = x;
      Eigen::RowVector3d c0(255,255,255), c1(165,0,38);
      Eigen::RowVector3d c = (1.0-t)*c0 + t*c1;
      return Eigen::RowVector3i( (int)c[0], (int)c[1], (int)c[2] );
    }
  }

  struct Raster {
    int W=0, H=0;
    std::vector<unsigned char> R,G,B,A;
    std::vector<double> Z;
    Raster(){}
    Raster(int w,int h): W(w),H(h),R(w*h,0),G(w*h,0),B(w*h,0),A(w*h,255),Z(w*h,-std::numeric_limits<double>::infinity()){}
    inline int idx(int x,int y) const { return y*W + x; }
    inline void plot(int x,int y, const Eigen::RowVector3i& c, double z){
      const int i = idx(x,y);
      if(z >= Z[i]){ R[i]= (unsigned char)c[0]; G[i]=(unsigned char)c[1]; B[i]=(unsigned char)c[2]; A[i]=255; Z[i]=z; }
    }
  };

  static inline bool barycentric(const Eigen::Vector2d& p,
                                 const Eigen::Vector2d& a,
                                 const Eigen::Vector2d& b,
                                 const Eigen::Vector2d& c,
                                 double& w0,double& w1,double& w2){
    const Eigen::Vector2d v0 = b-a, v1 = c-a, v2 = p-a;
    const double den = v0.x()*v1.y() - v1.x()*v0.y();
    if(std::abs(den) < 1e-30) return false;
    w1 = (v2.x()*v1.y() - v1.x()*v2.y())/den;
    w2 = (v0.x()*v2.y() - v2.x()*v0.y())/den;
    w0 = 1.0 - w1 - w2;
    const double eps = -1e-9;
    return (w0>=eps && w1>=eps && w2>=eps);
  }

  // PPM fallback if OpenCV write fails
  static bool write_ppm(const fs::path& out_ppm, const Raster& img){
    std::ofstream os(out_ppm, std::ios::binary);
    if(!os) return false;
    os << "P6\n" << img.W << " " << img.H << "\n255\n";
    for(int y=0;y<img.H;++y) for(int x=0;x<img.W;++x){
      const int i = img.idx(x,y);
      os.put((char)img.R[i]); os.put((char)img.G[i]); os.put((char)img.B[i]);
    }
    os.close();
    return (bool)os;
  }

  // OpenCV writer (PNG). Returns true on success.
  static bool write_image_cv_png(const fs::path& out_png, const Raster& img, int compression_level){
    cv::Mat mat(img.H, img.W, CV_8UC3);
    for(int y=0;y<img.H;++y){
      cv::Vec3b* row = mat.ptr<cv::Vec3b>(y);
      for(int x=0;x<img.W;++x){
        const int i = img.idx(x,y);
        // OpenCV uses BGR
        row[x] = cv::Vec3b(img.B[i], img.G[i], img.R[i]);
      }
    }
    std::vector<int> params = { cv::IMWRITE_PNG_COMPRESSION, clamp(compression_level,0,9) };
    try {
      return cv::imwrite(out_png.string(), mat, params);
    } catch (...) {
      return false;
    }
  }

  // Returns the actual paths written for logging
  struct HeatPaths { fs::path L2, Linf, kappa, logarea; };

  // Declarations (avoid nested default-arg pitfalls)
  HeatPaths save_uv_heatmaps(const Eigen::MatrixXd& uv_in) const;
  HeatPaths save_uv_heatmaps(const Eigen::MatrixXd& uv_in, const HeatCfg& cfg) const;
};

// No-arg convenience: default cfg
Flatboi::HeatPaths Flatboi::save_uv_heatmaps(const Eigen::MatrixXd& uv_in) const {
  HeatCfg cfg;
  return save_uv_heatmaps(uv_in, cfg);
}

// Full implementation (OpenCV writer)
Flatboi::HeatPaths Flatboi::save_uv_heatmaps(const Eigen::MatrixXd& uv_in, const HeatCfg& cfg) const {
  HeatPaths paths{};
  if(uv_in.rows() != V.rows() || uv_in.cols() != 2){
    std::cerr << "save_uv_heatmaps: expecting per-vertex UVs (#V x 2)\n"; return paths;
  }

  const fs::path in = fs::path(input_obj);
  const fs::path base = in.parent_path() / in.stem();

  // Shift & pack UVs
  Eigen::MatrixXd UV = shift_uv(uv_in);
  Eigen::RowVector2d mn = UV.colwise().minCoeff();
  Eigen::RowVector2d mx = UV.colwise().maxCoeff();
  const double w = std::max(1e-16, mx[0]-mn[0]);
  const double h = std::max(1e-16, mx[1]-mn[1]);

  const int W = (w >= h) ? cfg.target_max_dim : std::max(1, int(std::round(cfg.target_max_dim * (w/h))));
  const int H = (w >= h) ? std::max(1, int(std::round(cfg.target_max_dim * (h/w)))) : cfg.target_max_dim;
  const int M = cfg.margin_px;
  const double sx = (W - 2.0*M) / w;
  const double sy = (H - 2.0*M) / h;
  const double s  = std::min(sx, sy);

  // No vertical flip; V -> y increasing downward (matches tifxyz)
  auto uv_to_px = [&](const Eigen::RowVector2d& q)->Eigen::Vector2d{
    const double X = M + (q[0]-mn[0])*s;
    const double Y = M + (q[1]-mn[1])*s;
    return Eigen::Vector2d( X, Y );
  };

  const int T = F.rows();

  // Per-triangle metrics (true min/max; no clipping)
  std::vector<double> m_L2(T), m_Linf(T), m_kappa(T), m_logA(T);
  double L2_lo= std::numeric_limits<double>::infinity(), L2_hi=-L2_lo;
  double Li_lo= std::numeric_limits<double>::infinity(), Li_hi=-Li_lo;
  double K_lo = std::numeric_limits<double>::infinity(), K_hi =-K_lo;
  double LA_lo= std::numeric_limits<double>::infinity(), LA_hi=-LA_lo;

  for(int t=0;t<T;++t){
    const Eigen::RowVector3d q1 = V.row(F(t,0));
    const Eigen::RowVector3d q2 = V.row(F(t,1));
    const Eigen::RowVector3d q3 = V.row(F(t,2));
    const Eigen::RowVector2d p1 = uv_in.row(F(t,0));
    const Eigen::RowVector2d p2 = uv_in.row(F(t,1));
    const Eigen::RowVector2d p3 = uv_in.row(F(t,2));

    const double A2 = ((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])) * 0.5;
    const double A2abs = std::abs(A2);
    if(A2abs < 1e-30){
      m_L2[t]=m_Linf[t]=m_kappa[t]=m_logA[t]=0.0;
      L2_lo=std::min(L2_lo,0.0); L2_hi=std::max(L2_hi,0.0);
      Li_lo=std::min(Li_lo,0.0); Li_hi=std::max(Li_hi,0.0);
      K_lo =std::min(K_lo ,0.0); K_hi =std::max(K_hi ,0.0);
      LA_lo=std::min(LA_lo,0.0); LA_hi=std::max(LA_hi,0.0);
      continue;
    }

    const double s1=p1[0], t1=p1[1];
    const double s2=p2[0], t2=p2[1];
    const double s3=p3[0], t3=p3[1];

    const Eigen::RowVector3d Ss = ( q1*(t2 - t3) + q2*(t3 - t1) + q3*(t1 - t2) ) / (2.0*A2);
    const Eigen::RowVector3d St = ( q1*(s3 - s2) + q2*(s1 - s3) + q3*(s2 - s1) ) / (2.0*A2);

    const double a = Ss.dot(Ss);
    const double b = Ss.dot(St);
    const double c = St.dot(St);

    const double tr = a + c;
    const double disc = std::sqrt(std::max(0.0, (a - c)*(a - c) + 4.0*b*b));
    const double lam_max = 0.5*(tr + disc);
    const double lam_min = 0.5*(tr - disc);
    const double sigma_max = std::sqrt(std::max(0.0, lam_max));
    const double sigma_min = std::sqrt(std::max(0.0, lam_min));
    const double L2_rms    = std::sqrt(std::max(0.0, 0.5*(a + c)));

    const double ab = (q2 - q1).norm();
    const double bc = (q3 - q2).norm();
    const double ca = (q1 - q3).norm();
    const double sP = 0.5*(ab + bc + ca);
    const double A3 = std::sqrt(std::max(0.0, sP*(sP-ab)*(sP-bc)*(sP-ca)));
    const double area_factor = (A2abs>0.0) ? (A3 / A2abs) : 0.0;
    const double log_area    = std::log(std::max(1e-30, area_factor));

    m_L2[t]=L2_rms; m_Linf[t]=sigma_max; m_kappa[t]=(sigma_min>1e-30)? (sigma_max/sigma_min):0.0; m_logA[t]=log_area;

    L2_lo = std::min(L2_lo, m_L2[t]); L2_hi = std::max(L2_hi, m_L2[t]);
    Li_lo = std::min(Li_lo, m_Linf[t]); Li_hi = std::max(Li_hi, m_Linf[t]);
    K_lo  = std::min(K_lo , m_kappa[t]); K_hi = std::max(K_hi , m_kappa[t]);
    LA_lo = std::min(LA_lo, m_logA[t]); LA_hi = std::max(LA_hi, m_logA[t]);
  }

  const double logA_abs_max = std::max(std::abs(LA_lo), std::abs(LA_hi));
  auto fix_range = [](double lo, double hi){ if(!(hi>lo)) hi = lo+1.0; return std::pair<double,double>(lo,hi); };
  auto L2_rng=fix_range(L2_lo,L2_hi), Li_rng=fix_range(Li_lo,Li_hi), K_rng=fix_range(K_lo,K_hi);
  const double logA_den = (logA_abs_max>0.0)? logA_abs_max : 1.0;

  auto rasterize_metric = [&](const std::vector<double>& metric,
                              bool signed_diverging, double m_lo, double m_hi,
                              fs::path out_path)->fs::path{
    Raster img(W,H);
    const int SS = std::max(1, cfg.supersample);

    for(int t=0;t<T;++t){
      Eigen::Vector2d a = uv_to_px(UV.row(F(t,0)));
      Eigen::Vector2d b = uv_to_px(UV.row(F(t,1)));
      Eigen::Vector2d c = uv_to_px(UV.row(F(t,2)));

      const double minx = std::floor(std::min({a.x(), b.x(), c.x()}));
      const double maxx = std::ceil (std::max({a.x(), b.x(), c.x()}));
      const double miny = std::floor(std::min({a.y(), b.y(), c.y()}));
      const double maxy = std::ceil (std::max({a.y(), b.y(), c.y()}));

      const int x0 = clamp((int)minx, 0, W-1), x1 = clamp((int)maxx, 0, W-1);
      const int y0 = clamp((int)miny, 0, H-1), y1 = clamp((int)maxy, 0, H-1);

      for(int y=y0; y<=y1; ++y){
        for(int x=x0; x<=x1; ++x){
          double cov = 0.0;
          for(int sy=0; sy<SS; ++sy){
            for(int sx=0; sx<SS; ++sx){
              const double px = x + (SS==1 ? 0.5 : ((sx+0.5)/SS));
              const double py = y + (SS==1 ? 0.5 : ((sy+0.5)/SS));
              double w0,w1,w2;
              if(barycentric(Eigen::Vector2d(px,py), a,b,c, w0,w1,w2)) cov += 1.0;
            }
          }
          if(cov <= 0.0) continue;

          Eigen::RowVector3i rgb;
          const double denom = (m_hi - m_lo);
          if(signed_diverging){
            const double xnorm = std::max(-1.0, std::min(1.0, metric[t] / logA_den));
            rgb = cm_diverging(xnorm);
          }else{
            const double tnorm = clamp01( (metric[t] - m_lo) / ((denom!=0.0)?denom:1.0) );
            rgb = cm_rainbow(tnorm);
          }
          img.plot(x,y,rgb, signed_diverging ? std::abs(metric[t]) : metric[t]);
        }
      }
    }

    // Try OpenCV PNG first
    if(!write_image_cv_png(out_path, img, cfg.png_compression)){
      // Fallback to PPM with same basename
      fs::path ppm = out_path; ppm.replace_extension(".ppm");
      if(write_ppm(ppm, img)){
        std::cout << "Wrote heatmap (PPM fallback): " << ppm << "\n";
        return ppm;
      }else{
        std::cerr << "Failed to write heatmap: " << out_path << " (and fallback PPM)\n";
      }
    }else{
      std::cout << "Wrote heatmap (PNG): " << out_path << "\n";
    }
    return out_path;
  };

  fs::path pL2  = base; pL2  += "_heat_L2.png";
  fs::path pLi  = base; pLi  += "_heat_Linf.png";
  fs::path pKap = base; pKap += "_heat_kappa.png";
  fs::path pLA  = base; pLA  += "_heat_logarea.png";

  pL2  = rasterize_metric(m_L2,   false, L2_rng.first, L2_rng.second, pL2);
  pLi  = rasterize_metric(m_Linf, false, Li_rng.first, Li_rng.second, pLi);
  pKap = rasterize_metric(m_kappa,false, K_rng.first , K_rng.second , pKap);
  pLA  = rasterize_metric(m_logA, true , 0.0        , 0.0          , pLA);

  {
    std::ofstream os(base.string() + "_heat_ranges.txt");
    if(os){
      os << std::scientific;
      os << "L2_rms  : [" << L2_rng.first   << ", " << L2_rng.second   << "] (rainbow, full range)\n";
      os << "Linf    : [" << Li_rng.first   << ", " << Li_rng.second   << "] (rainbow, full range)\n";
      os << "kappa   : [" << K_rng.first    << ", " << K_rng.second    << "] (rainbow, full range)\n";
      os << "logArea : [-" << logA_den      << ", " << logA_den       << "] (diverging, symmetric)\n";
      std::cout << "Wrote: " << base.string() + "_heat_ranges.txt" << "\n";
    }
  }

  Flatboi::HeatPaths ret;
  ret.L2      = pL2;
  ret.Linf    = pLi;
  ret.kappa   = pKap;
  ret.logarea = pLA;
  return ret;
}

// ------------------------------
// Persist energies
// ------------------------------
static void save_energies(const fs::path& input, const std::vector<double>& E) {
  fs::path out = input.parent_path() / (input.stem().string() + "_energies_flatboi.txt");
  std::ofstream os(out);
  if(!os) { std::cerr << "Failed to open " << out << " for writing\n"; return; }
  os.setf(std::ios::fixed); os.precision(10);
  for (double e : E) os << e << "\n";
  std::cout << "Wrote: " << out << "\n";
}

// ------------------------------
// main
// ------------------------------
int main(int argc, char** argv) {
  if(argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <mesh.obj> <iters>\n";
    return 1;
  }
  const std::string obj_path = argv[1];
  const int iters = std::atoi(argv[2]);
  if(iters <= 0) { std::cerr << "iters must be > 0\n"; return 1; }
  if(!fs::exists(obj_path)) { std::cerr << "File not found: " << obj_path << "\n"; return 1; }
  if(fs::path(obj_path).extension() != ".obj") { std::cerr << "Input must be .obj\n"; return 1; }

  std::ios::sync_with_stdio(false);
  std::cout.setf(std::ios::unitbuf); // line-buffered: flush on '\n'
  
  try {
    Flatboi fb(obj_path, iters);

    WBLogger wblog(obj_path, iters); // optional; becomes no-op if unavailable

    // Run SLIM from original UVs (matches Python default), with W&B logging
    auto [uv_out, energies] = fb.slim_run("original", wblog.enabled()? &wblog : nullptr);

    // Persist
    save_energies(obj_path, energies);
    fb.save_obj(uv_out);

    // Heatmaps (post-SLIM) via OpenCV PNG + log them if W&B is available
    auto paths = fb.save_uv_heatmaps(uv_out);
    if (wblog.enabled()) {
      wblog.log_image("heatmap/L2",       paths.L2.string());
      wblog.log_image("heatmap/Linf",     paths.Linf.string());
      wblog.log_image("heatmap/kappa",    paths.kappa.string());
      wblog.log_image("heatmap/log_area", paths.logarea.string());
      wblog.finish();
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
