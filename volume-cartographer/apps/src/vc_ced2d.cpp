#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <limits>
#include <mutex>
#include <filesystem>
#include <atomic>
#include <iomanip>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

// Zarr / xtensor
#include <nlohmann/json.hpp>
#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/filesystem/dataset.hxx"
#include "z5/attributes.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)

namespace po = boost::program_options;
using json = nlohmann::json;

// Forward declaration for skeletonization utility used by coherence helpers
static cv::Mat skeletonize_mask(const cv::Mat& bin);

struct Config {
    float lambda_ = 1.0f;     // Edge threshold parameter
    float sigma_ = 3.0f;      // Gaussian smoothing for gradients
    float rho_ = 5.0f;        // Gaussian smoothing for structure tensor
    float step_size_ = 0.24f; // Diffusion time step (<= 0.25)
    float m_ = 1.0f;          // Exponent for diffusivity
    int   num_steps_ = 100;   // Iterations
    int   downsample_ = 1;    // Downsample factor (>=1)
    int   dilate_ = 0;        // Dilation radius in pixels (>=0)
    bool  apply_threshold_ = false; // Binarize output
    bool  no_threshold_ = false;    // If true, disable all thresholding; diffuse full image and write uint8
    bool  inverse_ = false;         // If true, invert final output to highlight incoherent regions
    bool  use_otsu_ = false;        // If true, use Otsu; else use threshold_value_
    double threshold_value_ = 0.0;  // Threshold in [0,255]
    int   remove_small_objects_ = 250; // Minimum area (pixels). 0 disables.
    int   jobs_ = 1;                  // Parallel files processed concurrently in folder mode
    bool  show_progress_ = true;      // Per-iteration inline progress
    bool  coherence_field_ = false;   // If true, write coherence field instead of diffused image
    bool  direction_field_ = false;   // If true with coherence-field, write RGB direction-of-coherence visualization
    double min_val_ = std::numeric_limits<double>::quiet_NaN(); // Optional clamp min for processing
    double max_val_ = std::numeric_limits<double>::quiet_NaN(); // Optional clamp max for processing
    bool  skeletonize_input_ = false; // If true (with coherence outputs), skeletonize input mask before coherence
};

// Constants (match Python)
static constexpr float EPS   = static_cast<float>(std::ldexp(1.0, -52)); // 2^-52
static constexpr float GAMMA = 0.01f;                                     // minimum diffusivity
static constexpr float CM    = 7.2848f;                                   // exponential constant

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static std::vector<float> gaussian_kernel_1d(float sigma) {
    if (sigma <= 0.f) return {1.f};
    int radius = static_cast<int>(std::ceil(3.0f * sigma));
    int size = 2 * radius + 1;
    std::vector<float> k(size);
    float denom = 2.0f * sigma * sigma;
    float sum = 0.f;
    for (int i = 0; i < size; ++i) {
        int x = i - radius;
        float v = std::exp(-(x * x) / denom);
        k[i] = v;
        sum += v;
    }
    for (int i = 0; i < size; ++i) k[i] /= sum;
    return k;
}

static void gaussian_blur(const std::vector<float>& src, int H, int W, float sigma, std::vector<float>& dst) {
    if (sigma <= 0.f) {
        if (&dst != &src) { dst = src; }
        return;
    }
    cv::Mat srcM(H, W, CV_32F, const_cast<float*>(src.data()));
    dst.resize(H * W);
    cv::Mat dstM(H, W, CV_32F, dst.data());
    cv::GaussianBlur(srcM, dstM, cv::Size(0, 0), sigma, sigma, cv::BORDER_REPLICATE);
}

static void compute_gradients(const std::vector<float>& img, int H, int W,
                              std::vector<float>& gx, std::vector<float>& gy) {
    gx.assign(H * W, 0.f);
    gy.assign(H * W, 0.f);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int xm = clampi(x - 1, 0, W - 1);
            int xp = clampi(x + 1, 0, W - 1);
            int ym = clampi(y - 1, 0, H - 1);
            int yp = clampi(y + 1, 0, H - 1);
            gx[y * W + x] = 0.5f * (img[y * W + xp] - img[y * W + xm]);
            gy[y * W + x] = 0.5f * (img[yp * W + x] - img[ym * W + x]);
        }
    }
}

static void compute_structure_tensor(const std::vector<float>& gx, const std::vector<float>& gy, int H, int W, float rho,
                                     std::vector<float>& s11, std::vector<float>& s12, std::vector<float>& s22) {
    std::vector<float> gx2(H * W), gy2(H * W), gxy(H * W);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < H * W; ++i) {
        float gxv = gx[i];
        float gyv = gy[i];
        gx2[i] = gxv * gxv;
        gy2[i] = gyv * gyv;
        gxy[i] = gxv * gyv;
    }
    gaussian_blur(gx2, H, W, rho, s11);
    gaussian_blur(gxy, H, W, rho, s12);
    gaussian_blur(gy2, H, W, rho, s22);
}

static void compute_alpha(const std::vector<float>& s11, const std::vector<float>& s12, const std::vector<float>& s22,
                          int HW, std::vector<float>& alpha) {
    alpha.resize(HW);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < HW; ++i) {
        float a = s11[i] - s22[i];
        float b = s12[i];
        alpha[i] = std::sqrt(a * a + 4.0f * b * b);
    }
}

static void compute_c2(const std::vector<float>& alpha, float lambda_, float m, int HW, std::vector<float>& c2) {
    c2.resize(HW);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < HW; ++i) {
        float h1 = (alpha[i] + EPS) / lambda_;
        float h2 = (std::abs(m - 1.0f) < 1e-10f) ? h1 : std::pow(h1, m);
        float h3 = std::exp(-CM / h2);
        c2[i] = GAMMA + (1.0f - GAMMA) * h3;
    }
}

static inline void invert_c2_mechanics(std::vector<float>& c2) {
    // Map c2 in [GAMMA,1] to inverted preference: high where original was low, and vice versa.
    // c2_inv = 1 - c2 + GAMMA, clamped to [GAMMA,1].
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(c2.size()); ++i) {
        float v = 1.0f - c2[i] + GAMMA;
        if (v < GAMMA) v = GAMMA;
        if (v > 1.0f) v = 1.0f;
        c2[i] = v;
    }
}

static void compute_diffusion_tensor(const std::vector<float>& s11, const std::vector<float>& s12, const std::vector<float>& s22,
                                     const std::vector<float>& alpha, const std::vector<float>& c2,
                                     int HW,
                                     std::vector<float>& d11, std::vector<float>& d12, std::vector<float>& d22) {
    d11.resize(HW); d12.resize(HW); d22.resize(HW);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < HW; ++i) {
        float dd = (c2[i] - GAMMA) * (s11[i] - s22[i]) / (alpha[i] + EPS);
        d11[i] = 0.5f * (GAMMA + c2[i] + dd);
        d12[i] = (GAMMA - c2[i]) * s12[i] / (alpha[i] + EPS);
        d22[i] = 0.5f * (GAMMA + c2[i] - dd);
    }
}

static void diffusion_step(const std::vector<float>& img, const std::vector<float>& d11, const std::vector<float>& d12, const std::vector<float>& d22,
                           int H, int W, float step_size, std::vector<float>& img_out) {
    img_out.assign(H * W, 0.f);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int i = y * W + x;
            int yN = clampi(y - 1, 0, H - 1);
            int yS = clampi(y + 1, 0, H - 1);
            int xW = clampi(x - 1, 0, W - 1);
            int xE = clampi(x + 1, 0, W - 1);

            int idxC  = i;
            int idxN  = yN * W + x;
            int idxS  = yS * W + x;
            int idxW  = y * W + xW;
            int idxE  = y * W + xE;
            int idxNW = yN * W + xW;
            int idxNE = yN * W + xE;
            int idxSW = yS * W + xW;
            int idxSE = yS * W + xE;

            float img_c = img[idxC];
            float img_n = img[idxN];
            float img_s = img[idxS];
            float img_w = img[idxW];
            float img_e = img[idxE];
            float img_nw = img[idxNW];
            float img_ne = img[idxNE];
            float img_sw = img[idxSW];
            float img_se = img[idxSE];

            float d11_c = d11[idxC];
            float d11_n = d11[idxN];
            float d11_s = d11[idxS];

            float d22_c = d22[idxC];
            float d22_w = d22[idxW];
            float d22_e = d22[idxE];

            float d12_c = d12[idxC];
            float d12_n = d12[idxN];
            float d12_s = d12[idxS];
            float d12_w = d12[idxW];
            float d12_e = d12[idxE];

            float c_cop = d22_c + d22_w; // (i,j) + (i,j-1)
            float a_amo = d11_s + d11_c; // (i+1,j) + (i,j)
            float a_apo = d11_n + d11_c; // (i-1,j) + (i,j)
            float c_com = d22_c + d22_e; // (i,j) + (i,j+1)

            float first_deriv = (
                c_cop * img_w +
                a_amo * img_s -
                (a_amo + a_apo + c_com + c_cop) * img_c +
                a_apo * img_n +
                c_com * img_e
            );

            float bmo = d12_s;
            float bop = d12_w;
            float bpo = d12_n;
            float bom = d12_e;

            float second_deriv = (
                -1.0f * ((bmo + bop) * img_sw + (bpo + bom) * img_ne) +
                (bpo + bop) * img_nw +
                (bmo + bom) * img_se
            );

            img_out[idxC] = img_c + step_size * (0.5f * first_deriv + 0.25f * second_deriv);
        }
    }
}

static std::mutex g_print_mtx;

static void ced_run(const cv::Mat& input, cv::Mat& output, const Config& cfg,
                    const char* progress_label = nullptr, int progress_mod = 1) {

    const int H = input.rows;
    const int W = input.cols;
    std::vector<float> img(H * W);
    const bool has_min = !std::isnan(cfg.min_val_);
    const bool has_max = !std::isnan(cfg.max_val_);
    if (input.type() == CV_8UC1) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = input.ptr<uint8_t>(y);
            for (int x = 0; x < W; ++x) {
                float v = static_cast<float>(row[x]);
                if (has_min && v < cfg.min_val_) v = static_cast<float>(cfg.min_val_);
                if (has_max && v > cfg.max_val_) v = static_cast<float>(cfg.max_val_);
                img[y * W + x] = v;
            }
        }
    } else if (input.type() == CV_16UC1) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            const uint16_t* row = input.ptr<uint16_t>(y);
            for (int x = 0; x < W; ++x) {
                float v = static_cast<float>(row[x]);
                if (has_min && v < cfg.min_val_) v = static_cast<float>(cfg.min_val_);
                if (has_max && v > cfg.max_val_) v = static_cast<float>(cfg.max_val_);
                img[y * W + x] = v;
            }
        }
    } else if (input.type() == CV_32FC1) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            const float* row = input.ptr<float>(y);
            for (int x = 0; x < W; ++x) {
                float v = row[x];
                if (has_min && v < cfg.min_val_) v = static_cast<float>(cfg.min_val_);
                if (has_max && v > cfg.max_val_) v = static_cast<float>(cfg.max_val_);
                img[y * W + x] = v;
            }
        }
    } else {
        throw std::runtime_error("Unsupported input image type; use 8U, 16U, or 32F single-channel TIFF");
    }


    std::vector<float> img_smooth(H * W), gx(H * W), gy(H * W), s11(H * W), s12(H * W), s22(H * W),
                       alpha(H * W), c2(H * W), d11(H * W), d12(H * W), d22(H * W), img_new(H * W);
    for (int step = 0; step < cfg.num_steps_; ++step) {
        if (cfg.show_progress_) {
            if (progress_label == nullptr) {
                std::cout << "\rStep " << (step + 1) << "/" << cfg.num_steps_ << std::flush;
            } else {
                bool do_print = (progress_mod <= 1) || ((step % progress_mod) == 0) || (step + 1 == cfg.num_steps_);
                if (do_print) {
                    std::lock_guard<std::mutex> lock(g_print_mtx);
                    std::cout << "[" << progress_label << "] Step " << (step + 1)
                              << "/" << cfg.num_steps_ << std::endl;
                }
            }
        }

        gaussian_blur(img, H, W, cfg.sigma_, img_smooth);
        compute_gradients(img_smooth, H, W, gx, gy);
        compute_structure_tensor(gx, gy, H, W, cfg.rho_, s11, s12, s22);
        compute_alpha(s11, s12, s22, H * W, alpha);
        compute_c2(alpha, cfg.lambda_, cfg.m_, H * W, c2);
        if (cfg.inverse_) {
            invert_c2_mechanics(c2);
        }
        compute_diffusion_tensor(s11, s12, s22, alpha, c2, H * W, d11, d12, d22);
        diffusion_step(img, d11, d12, d22, H, W, cfg.step_size_, img_new);
        img.swap(img_new);
    }
    if (cfg.show_progress_) {
        if (progress_label == nullptr) {
            std::cout << "\nDiffusion complete!" << std::endl;
        } else {
            std::lock_guard<std::mutex> lock(g_print_mtx);
            std::cout << "[" << progress_label << "] Complete" << std::endl;
        }
    }

    output.create(H, W, input.type());
    if (input.type() == CV_8UC1) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            uint8_t* row = output.ptr<uint8_t>(y);
            for (int x = 0; x < W; ++x) {
                float v = img[y * W + x];
                if (!cfg.no_threshold_ && v < 1.f) v = 0.f; // suppress tiny noise unless no-threshold is requested
                v = std::min(std::max(v, 0.0f), 255.0f);
                row[x] = static_cast<uint8_t>(std::lround(v));
            }
        }
    } else if (input.type() == CV_16UC1) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            uint16_t* row = output.ptr<uint16_t>(y);
            for (int x = 0; x < W; ++x) {
                float v = img[y * W + x];
                v = std::min(std::max(v, 0.0f), 65535.0f);
                row[x] = static_cast<uint16_t>(std::lround(v));
            }
        }
    } else { // CV_32FC1
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            float* row = output.ptr<float>(y);
            for (int x = 0; x < W; ++x) row[x] = img[y * W + x];
        }
    }
}

static cv::Mat compute_coherence_field_full(const cv::Mat& input, const Config& cfg) {
    // Convert to float buffer
    const int H = input.rows;
    const int W = input.cols;
    std::vector<float> img(H * W);
    if (input.type() == CV_8UC1) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = input.ptr<uint8_t>(y);
            for (int x = 0; x < W; ++x) img[y * W + x] = static_cast<float>(row[x]);
        }
    } else if (input.type() == CV_16UC1) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            const uint16_t* row = input.ptr<uint16_t>(y);
            for (int x = 0; x < W; ++x) img[y * W + x] = static_cast<float>(row[x]);
        }
    } else if (input.type() == CV_32FC1) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            const float* row = input.ptr<float>(y);
            for (int x = 0; x < W; ++x) img[y * W + x] = row[x];
        }
    } else {
        throw std::runtime_error("Unsupported input image type; use 8U, 16U, or 32F single-channel TIFF");
    }

    // Optionally downsample for speed
    int dH = H, dW = W;
    float sigma = cfg.sigma_;
    float rho   = cfg.rho_;
    std::vector<float> img_work;
    if (cfg.downsample_ > 1) {
        int f = cfg.downsample_;
        dW = std::max(1, (W + f - 1) / f);
        dH = std::max(1, (H + f - 1) / f);
        cv::Mat srcM(H, W, CV_32F, img.data());
        cv::Mat ds; cv::resize(srcM, ds, cv::Size(dW, dH), 0, 0, cv::INTER_AREA);
        img_work.assign(dH * dW, 0.f);
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < dH; ++y) {
            const float* row = ds.ptr<float>(y);
            const bool has_min = !std::isnan(cfg.min_val_);
            const bool has_max = !std::isnan(cfg.max_val_);
            for (int x = 0; x < dW; ++x) {
                float v = row[x];
                if (has_min && v < cfg.min_val_) v = static_cast<float>(cfg.min_val_);
                if (has_max && v > cfg.max_val_) v = static_cast<float>(cfg.max_val_);
                img_work[y * dW + x] = v;
            }
        }
        sigma = std::max(0.f, cfg.sigma_ / f);
        rho   = std::max(0.f, cfg.rho_   / f);
    } else {
        // Clamp if requested
        const bool has_min = !std::isnan(cfg.min_val_);
        const bool has_max = !std::isnan(cfg.max_val_);
        if (has_min || has_max) {
            img_work.resize(H * W);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < H * W; ++i) {
                float v = img[i];
                if (has_min && v < cfg.min_val_) v = static_cast<float>(cfg.min_val_);
                if (has_max && v > cfg.max_val_) v = static_cast<float>(cfg.max_val_);
                img_work[i] = v;
            }
        } else {
            img_work.swap(img);
        }
    }

    // Optional skeletonization: build a binary mask from img_work, skeletonize, and keep only skeleton with unit value
    if (cfg.skeletonize_input_) {
        cv::Mat workM(dH, dW, CV_32F, img_work.data());
        cv::Mat mask; cv::compare(workM, 0.0f, mask, cv::CMP_GT);
        cv::Mat skel = skeletonize_mask(mask);
        // Replace img_work with skeletonized binary field (1.0 on skeleton, 0.0 elsewhere)
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < dH; ++y) {
            const uint8_t* srow = skel.ptr<uint8_t>(y);
            for (int x = 0; x < dW; ++x) img_work[y * dW + x] = (srow[x] ? 1.0f : 0.0f);
        }
    }

    // Compute coherence measures on working resolution
    std::vector<float> img_smooth(dH * dW), gx(dH * dW), gy(dH * dW), s11(dH * dW), s12(dH * dW), s22(dH * dW), alpha(dH * dW), c2(dH * dW);
    gaussian_blur(img_work, dH, dW, sigma, img_smooth);
    compute_gradients(img_smooth, dH, dW, gx, gy);
    compute_structure_tensor(gx, gy, dH, dW, rho, s11, s12, s22);
    compute_alpha(s11, s12, s22, dH * dW, alpha);
    compute_c2(alpha, cfg.lambda_, cfg.m_, dH * dW, c2);

    // Normalize c2 from [GAMMA,1] -> [0,1]
    float denom = (1.0f - GAMMA);
    cv::Mat field_ds(dH, dW, CV_32FC1);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < dH; ++y) {
        float* row = field_ds.ptr<float>(y);
        for (int x = 0; x < dW; ++x) {
            float v = (c2[y * dW + x] - GAMMA) / denom;
            if (v < 0.f) v = 0.f; if (v > 1.f) v = 1.f;
            row[x] = v;
        }
    }

    if (cfg.downsample_ > 1) {
        cv::Mat field_full; cv::resize(field_ds, field_full, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
        return field_full;
    }
    return field_ds;
}

static cv::Mat compute_direction_field_rgb(const cv::Mat& input, const Config& cfg) {
    // Compute orientation (coherence direction) and coherence magnitude, output as BGR 8UC3
    const int H = input.rows;
    const int W = input.cols;
    // Prepare float image
    cv::Mat f;
    if (input.type() == CV_8UC1) {
        input.convertTo(f, CV_32F, 1.0);
    } else if (input.type() == CV_16UC1) {
        input.convertTo(f, CV_32F, 1.0);
    } else if (input.type() == CV_32FC1) {
        f = input;
    } else {
        throw std::runtime_error("Unsupported input image type for direction field");
    }

    int dH = H, dW = W;
    float sigma = cfg.sigma_;
    float rho   = cfg.rho_;
    cv::Mat fds;
    if (cfg.downsample_ > 1) {
        int fct = cfg.downsample_;
        dW = std::max(1, (W + fct - 1) / fct);
        dH = std::max(1, (H + fct - 1) / fct);
        cv::resize(f, fds, cv::Size(dW, dH), 0, 0, cv::INTER_AREA);
        sigma = std::max(0.f, cfg.sigma_ / fct);
        rho   = std::max(0.f, cfg.rho_   / fct);
    } else {
        fds = f;
    }

    // Work buffers
    std::vector<float> img(dH * dW), img_smooth(dH * dW), gx(dH * dW), gy(dH * dW), s11(dH * dW), s12(dH * dW), s22(dH * dW), alpha(dH * dW), c2(dH * dW);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < dH; ++y) {
        const float* row = fds.ptr<float>(y);
        const bool has_min = !std::isnan(cfg.min_val_);
        const bool has_max = !std::isnan(cfg.max_val_);
        for (int x = 0; x < dW; ++x) {
            float v = row[x];
            if (has_min && v < cfg.min_val_) v = static_cast<float>(cfg.min_val_);
            if (has_max && v > cfg.max_val_) v = static_cast<float>(cfg.max_val_);
            img[y * dW + x] = v;
        }
    }

    // Optional skeletonization on the downsampled float image
    if (cfg.skeletonize_input_) {
        cv::Mat fimg(dH, dW, CV_32F);
        for (int y = 0; y < dH; ++y) {
            float* row = fimg.ptr<float>(y);
            for (int x = 0; x < dW; ++x) row[x] = img[y * dW + x];
        }
        cv::Mat mask; cv::compare(fimg, 0.0f, mask, cv::CMP_GT);
        cv::Mat skel = skeletonize_mask(mask);
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < dH; ++y) {
            const uint8_t* srow = skel.ptr<uint8_t>(y);
            for (int x = 0; x < dW; ++x) img[y * dW + x] = (srow[x] ? 1.0f : 0.0f);
        }
    }
    gaussian_blur(img, dH, dW, sigma, img_smooth);
    compute_gradients(img_smooth, dH, dW, gx, gy);
    compute_structure_tensor(gx, gy, dH, dW, rho, s11, s12, s22);
    compute_alpha(s11, s12, s22, dH * dW, alpha);
    compute_c2(alpha, cfg.lambda_, cfg.m_, dH * dW, c2);

    // HSV image (OpenCV hue: 0..180). Use hue for direction, value for coherence magnitude.
    cv::Mat hsv(dH, dW, CV_8UC3);
    const float invDenom = 1.0f / std::max(1e-6f, (1.0f - GAMMA));
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < dH; ++y) {
        cv::Vec3b* row = hsv.ptr<cv::Vec3b>(y);
        for (int x = 0; x < dW; ++x) {
            int i = y * dW + x;
            float a = s11[i];
            float b = s12[i];
            float c = s22[i];
            // Principal orientation angle for largest eigenvalue
            float theta = 0.5f * std::atan2(2.0f * b, a - c);
            // Coherent direction is perpendicular to gradient-dominated eigenvector
            float phi = theta + static_cast<float>(CV_PI) * 0.5f;
            float hue = std::fmod((phi < 0 ? phi + 2.0f * static_cast<float>(CV_PI) : phi), 2.0f * static_cast<float>(CV_PI));
            uint8_t Hh = static_cast<uint8_t>(std::lround((hue / (2.0f * static_cast<float>(CV_PI))) * 180.0f));

            float coh = (c2[i] - GAMMA) * invDenom; // 0..1
            if (coh < 0.f) coh = 0.f; if (coh > 1.f) coh = 1.f;
            uint8_t Vv = static_cast<uint8_t>(std::lround(coh * 255.0f));
            // Full saturation for clarity
            row[x] = cv::Vec3b(Hh, 255, Vv);
        }
    }
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    if (cfg.downsample_ > 1) {
        cv::Mat up; cv::resize(bgr, up, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
        return up;
    }
    return bgr;
}

static bool find_nonzero_bbox(const cv::Mat& img, cv::Rect& bbox) {
    const int H = img.rows, W = img.cols;
    int minx = W, miny = H, maxx = -1, maxy = -1;
    if (img.type() == CV_8UC1) {
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = img.ptr<uint8_t>(y);
            for (int x = 0; x < W; ++x) if (row[x] != 0) { if (x < minx) minx = x; if (y < miny) miny = y; if (x > maxx) maxx = x; if (y > maxy) maxy = y; }
        }
    } else if (img.type() == CV_16UC1) {
        for (int y = 0; y < H; ++y) {
            const uint16_t* row = img.ptr<uint16_t>(y);
            for (int x = 0; x < W; ++x) if (row[x] != 0) { if (x < minx) minx = x; if (y < miny) miny = y; if (x > maxx) maxx = x; if (y > maxy) maxy = y; }
        }
    } else if (img.type() == CV_32FC1) {
        for (int y = 0; y < H; ++y) {
            const float* row = img.ptr<float>(y);
            for (int x = 0; x < W; ++x) if (row[x] != 0.0f) { if (x < minx) minx = x; if (y < miny) miny = y; if (x > maxx) maxx = x; if (y > maxy) maxy = y; }
        }
    } else {
        return false;
    }
    if (maxx < 0) return false;
    bbox = cv::Rect(minx, miny, maxx - minx + 1, maxy - miny + 1);
    return true;
}

static cv::Mat binarize_and_dilate(const cv::Mat& img, int dilate_radius) {
    // Binary mask: any nonzero pixel -> 255
    cv::Mat mask;
    cv::compare(img, 0, mask, cv::CMP_GT); // mask is 8U, values 0 or 255
    if (dilate_radius > 0) {
        int k = 2 * dilate_radius + 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        cv::dilate(mask, mask, kernel, cv::Point(-1, -1), 1);
    }
    return mask;
}

static cv::Mat process_one_image(const cv::Mat& img, const Config& cfg, const char* progress_label = nullptr) {
    // If no-threshold is requested, still find the nonzero ROI (optionally dilated) but
    // diffuse the original intensity values inside that ROI (no binarization of the target or output).
    if (cfg.no_threshold_) {
        cv::Mat mask = binarize_and_dilate(img, cfg.dilate_);
        cv::Rect bbox;
        if (!find_nonzero_bbox(mask, bbox)) {
            return img.clone();
        }
        int H = img.rows, W = img.cols;
        int margin = static_cast<int>(std::ceil(3.f * std::max(cfg.sigma_, cfg.rho_))) + 2;
        int x0 = std::max(0, bbox.x - margin);
        int y0 = std::max(0, bbox.y - margin);
        int x1 = std::min(W, bbox.x + bbox.width + margin);
        int y1 = std::min(H, bbox.y + bbox.height + margin);
        cv::Rect ext(x0, y0, x1 - x0, y1 - y0);

        cv::Mat region = img(ext).clone();

        cv::Mat processed;
        if (cfg.downsample_ > 1) {
            int f = cfg.downsample_;
            int dW = std::max(1, (ext.width  + f - 1) / f);
            int dH = std::max(1, (ext.height + f - 1) / f);
            cv::Mat region_ds; cv::resize(region, region_ds, cv::Size(dW, dH), 0, 0, cv::INTER_AREA);
            Config cfg_ds = cfg; cfg_ds.sigma_ = cfg.sigma_ / f; cfg_ds.rho_ = cfg.rho_ / f; if (cfg_ds.sigma_ < 0.f) cfg_ds.sigma_ = 0.f; if (cfg_ds.rho_ < 0.f) cfg_ds.rho_ = 0.f;
            cv::Mat out_ds; ced_run(region_ds, out_ds, cfg_ds, progress_label, std::max(1, cfg_ds.num_steps_/10));
            cv::resize(out_ds, processed, region.size(), 0, 0, cv::INTER_LINEAR);
        } else {
            ced_run(region, processed, cfg, progress_label, std::max(1, cfg.num_steps_/10));
        }

        cv::Mat out = img.clone();
        int ox = bbox.x - ext.x; int oy = bbox.y - ext.y;
        cv::Mat inROI = processed(cv::Rect(ox, oy, bbox.width, bbox.height));
        inROI.copyTo(out(bbox));
        return out;
    }

    // 1) Binarize and optionally dilate to build a mask
    cv::Mat mask = binarize_and_dilate(img, cfg.dilate_);

    // 2) Compute bbox from mask; if empty, return original
    cv::Rect bbox;
    if (!find_nonzero_bbox(mask, bbox)) {
        return img.clone();
    }
    int H = img.rows, W = img.cols;
    int margin = static_cast<int>(std::ceil(3.f * std::max(cfg.sigma_, cfg.rho_))) + 2;
    int x0 = std::max(0, bbox.x - margin);
    int y0 = std::max(0, bbox.y - margin);
    int x1 = std::min(W, bbox.x + bbox.width + margin);
    int y1 = std::min(H, bbox.y + bbox.height + margin);
    cv::Rect ext(x0, y0, x1 - x0, y1 - y0);

    cv::Mat region = img(ext).clone();
    cv::Mat mask_ext = mask(ext);
    // Build a binarized input image: inside mask -> max, else 0
    cv::Mat region_bin = cv::Mat::zeros(region.size(), region.type());
    if (region.type() == CV_8UC1) {
        region_bin.setTo(255, mask_ext);
    } else if (region.type() == CV_16UC1) {
        region_bin.setTo(65535, mask_ext);
    } else { // CV_32FC1
        region_bin.setTo(1.0f, mask_ext);
    }

    cv::Mat processed;
    if (cfg.downsample_ > 1) {
        int f = cfg.downsample_;
        int dW = std::max(1, (ext.width  + f - 1) / f);
        int dH = std::max(1, (ext.height + f - 1) / f);
        cv::Mat region_ds; cv::resize(region_bin, region_ds, cv::Size(dW, dH), 0, 0, cv::INTER_AREA);
        Config cfg_ds = cfg; cfg_ds.sigma_ = cfg.sigma_ / f; cfg_ds.rho_ = cfg.rho_ / f; if (cfg_ds.sigma_ < 0.f) cfg_ds.sigma_ = 0.f; if (cfg_ds.rho_ < 0.f) cfg_ds.rho_ = 0.f;
        cv::Mat out_ds; ced_run(region_ds, out_ds, cfg_ds, progress_label, std::max(1, cfg_ds.num_steps_/10));
        cv::resize(out_ds, processed, region.size(), 0, 0, cv::INTER_LINEAR);
    } else {
        ced_run(region_bin, processed, cfg, progress_label, std::max(1, cfg.num_steps_/10));
    }

    cv::Mat out = img.clone();
    int ox = bbox.x - ext.x; int oy = bbox.y - ext.y;
    cv::Mat inROI = processed(cv::Rect(ox, oy, bbox.width, bbox.height));
    inROI.copyTo(out(bbox));
    return out;
}

static bool is_directory(const std::string& p) {
    namespace fs = std::filesystem;
    std::error_code ec;
    return fs::exists(p, ec) && fs::is_directory(p, ec);
}

static bool ensure_dir(const std::string& p) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(p, ec)) return fs::is_directory(p, ec);
    return fs::create_directories(p, ec);
}

static bool is_all_zero(const cv::Mat& img) {
    if (img.empty()) return true;
    if (img.type() == CV_8UC1) {
        return cv::countNonZero(img) == 0;
    } else if (img.type() == CV_16UC1) {
        cv::Mat nz;
        cv::compare(img, 0, nz, cv::CMP_NE);
        return cv::countNonZero(nz) == 0;
    } else if (img.type() == CV_32FC1) {
        cv::Mat nz;
        cv::compare(img, 0.0f, nz, cv::CMP_NE);
        return cv::countNonZero(nz) == 0;
    }
    return false;
}


// Zhang-Suen thinning (skeletonization) for binary masks (CV_8UC1, 0/255)
static cv::Mat skeletonize_mask(const cv::Mat& bin) {
    CV_Assert(bin.type() == CV_8UC1);
    cv::Mat img;
    // Ensure strictly 0/1 values
    cv::threshold(bin, img, 0, 1, cv::THRESH_BINARY);
    img.convertTo(img, CV_8U);
    cv::Mat prev = cv::Mat::zeros(img.size(), CV_8U);
    cv::Mat diff;
    auto neighbors = [&](int y, int x, uint8_t p[9]) {
        // p2 p3 p4
        // p1 p  p5
        // p8 p7 p6
        p[0] = img.at<uint8_t>(y-1, x    ); // p2
        p[1] = img.at<uint8_t>(y-1, x+1  ); // p3
        p[2] = img.at<uint8_t>(y,   x+1  ); // p4
        p[3] = img.at<uint8_t>(y+1, x+1  ); // p5
        p[4] = img.at<uint8_t>(y+1, x    ); // p6
        p[5] = img.at<uint8_t>(y+1, x-1  ); // p7
        p[6] = img.at<uint8_t>(y,   x-1  ); // p8
        p[7] = img.at<uint8_t>(y-1, x-1  ); // p1
        p[8] = img.at<uint8_t>(y,   x    ); // center (not used in sums)
    };
    auto A = [&](const uint8_t p[8]) {
        int count = 0;
        for (int i = 0; i < 7; ++i) if (p[i] == 0 && p[i+1] == 1) ++count;
        if (p[7] == 0 && p[0] == 1) ++count;
        return count;
    };
    auto B = [&](const uint8_t p[8]) {
        int s = 0; for (int i = 0; i < 8; ++i) s += p[i]; return s;
    };
    do {
        img.copyTo(prev);
        std::vector<cv::Point> to_zero;
        // Step 1
        for (int y = 1; y < img.rows - 1; ++y) {
            const uint8_t* row = img.ptr<uint8_t>(y);
            for (int x = 1; x < img.cols - 1; ++x) if (row[x]) {
                uint8_t p9[9]; neighbors(y, x, p9);
                uint8_t p[8] = {p9[0],p9[1],p9[2],p9[3],p9[4],p9[5],p9[6],p9[7]};
                int bp = B(p);
                if (bp < 2 || bp > 6) continue;
                if (A(p) != 1) continue;
                if (p[0] & p[2] & p[4]) continue; // p2*p4*p6==1
                if (p[2] & p[4] & p[6]) continue; // p4*p6*p8==1
                to_zero.emplace_back(x, y);
            }
        }
        for (const auto& pt : to_zero) img.at<uint8_t>(pt.y, pt.x) = 0;
        to_zero.clear();
        // Step 2
        for (int y = 1; y < img.rows - 1; ++y) {
            const uint8_t* row = img.ptr<uint8_t>(y);
            for (int x = 1; x < img.cols - 1; ++x) if (row[x]) {
                uint8_t p9[9]; neighbors(y, x, p9);
                uint8_t p[8] = {p9[0],p9[1],p9[2],p9[3],p9[4],p9[5],p9[6],p9[7]};
                int bp = B(p);
                if (bp < 2 || bp > 6) continue;
                if (A(p) != 1) continue;
                if (p[0] & p[2] & p[6]) continue; // p2*p4*p8==1
                if (p[0] & p[4] & p[6]) continue; // p2*p6*p8==1
                to_zero.emplace_back(x, y);
            }
        }
        for (const auto& pt : to_zero) img.at<uint8_t>(pt.y, pt.x) = 0;
        cv::absdiff(img, prev, diff);
    } while (cv::countNonZero(diff) > 0);
    // Back to 0/255
    img *= 255;
    return img;
}

static void remove_small_objects(cv::Mat& bin, int min_area) {
    if (min_area <= 0) return;
    CV_Assert(bin.type() == CV_8UC1);
    // Connected components on binary 0/255 image (treat >0 as 1)
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
    if (n <= 1) return; // only background
    // Build keep mask
    std::vector<uint8_t> keep(n, 0);
    keep[0] = 0; // background
    for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= min_area) keep[i] = 255;
    }
    // Write back
    for (int y = 0; y < bin.rows; ++y) {
        const int* lrow = labels.ptr<int>(y);
        uint8_t* brow = bin.ptr<uint8_t>(y);
        for (int x = 0; x < bin.cols; ++x) brow[x] = keep[lrow[x]];
    }
}

static cv::Mat to_uint8_scaled_and_threshold(const cv::Mat& in, const Config& cfg) {
    cv::Mat u8;
    if (in.type() == CV_8UC1) {
        u8 = in.clone();
    } else if (in.type() == CV_16UC1) {
        // Map [0..65535] -> [0..255]
        in.convertTo(u8, CV_8U, 1.0 / 257.0, 0.0);
    } else if (in.type() == CV_32FC1) {
        // Map [0..1] -> [0..255]
        in.convertTo(u8, CV_8U, 255.0, 0.0);
    } else {
        throw std::runtime_error("Unsupported image type for output conversion");
    }

    if (cfg.apply_threshold_) {
        cv::Mat bin;
        if (cfg.use_otsu_) {
            cv::threshold(u8, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        } else {
            double t = std::max(0.0, std::min(255.0, cfg.threshold_value_));
            cv::threshold(u8, bin, t, 255, cv::THRESH_BINARY);
        }
        // Remove small connected components (default 250 px). 0 disables.
        remove_small_objects(bin, cfg.remove_small_objects_);
        return bin;
    }
    return u8;
}

int main(int argc, char** argv) {
    std::string in_path, out_path;
    Config cfg;
    int num_threads = -1;
    int group_idx = 0; // OME-Zarr group index
    int min_z = 0;     // Z range start (inclusive) for Zarr input
    int max_z = -1;    // Z range end (inclusive); -1 means last

    try {
        po::options_description desc("CED2D options");
        desc.add_options()
            ("help,h", "Show help")
            ("input,i", po::value<std::string>(&in_path)->required(), "Input path: TIFF file, directory of TIFFs, or OME-Zarr (.zarr)")
            ("output,o", po::value<std::string>(&out_path)->required(), "Output: TIFF file/dir; if input is OME-Zarr and -o ends with .zarr => write OME-Zarr, else write per-Z TIFFs into the directory")
            ("group-idx,g", po::value<int>(&group_idx)->default_value(0), "OME-Zarr group index for zarr input (default 0)")
            ("lambda", po::value<float>(&cfg.lambda_)->default_value(1.0f), "Edge threshold parameter")
            ("sigma", po::value<float>(&cfg.sigma_)->default_value(3.0f), "Gaussian sigma for gradients")
            ("rho", po::value<float>(&cfg.rho_)->default_value(5.0f), "Gaussian sigma for structure tensor")
            ("step-size", po::value<float>(&cfg.step_size_)->default_value(0.24f), "Diffusion step size (<=0.25)")
            ("m", po::value<float>(&cfg.m_)->default_value(1.0f), "Exponent m for diffusivity")
            ("num-steps", po::value<int>(&cfg.num_steps_)->default_value(100), "Number of diffusion steps")
            ("downsample", po::value<int>(&cfg.downsample_)->default_value(1), "Downsample factor (>=1)")
            ("dilate", po::value<int>(&cfg.dilate_)->default_value(0), "Dilate radius (pixels, >=0) after binarization")
            ("min-z", po::value<int>(&min_z)->default_value(0), "Minimum input Z slice (inclusive) for Zarr input")
            ("max-z", po::value<int>(&max_z)->default_value(-1), "Maximum input Z slice (inclusive) for Zarr input; -1 = last slice")
            ("coherence-field", po::bool_switch(&cfg.coherence_field_)->default_value(false), "Write coherence field instead of diffused image (higher=more coherent)")
            ("direction-field", po::bool_switch(&cfg.direction_field_)->default_value(false), "With --coherence-field: write RGB direction-of-coherence field (TIFF outputs only)")
            ("min-val", po::value<double>(&cfg.min_val_)->default_value(std::numeric_limits<double>::quiet_NaN()), "Minimum intensity clamp for processing (units of input)")
            ("max-val", po::value<double>(&cfg.max_val_)->default_value(std::numeric_limits<double>::quiet_NaN()), "Maximum intensity clamp for processing (units of input)")
            ("skeletonize", po::bool_switch(&cfg.skeletonize_input_)->default_value(false), "With --coherence-field: skeletonize input mask before computing coherence/direction")
        ;

        // threshold option: optional value, if no value -> Otsu
        double threshold_opt = std::numeric_limits<double>::quiet_NaN();
        desc.add_options()("threshold", po::value<double>(&threshold_opt)->implicit_value(std::numeric_limits<double>::quiet_NaN()),
                           "Binarize output as uint8; if value omitted => Otsu, else numeric threshold [0..255]");
        // no-threshold: disable any thresholding on target or output
        desc.add_options()("no-threshold", po::bool_switch(&cfg.no_threshold_)->default_value(false),
                           "Do not threshold target or output; still crop to nonzero region and diffuse original intensities; write uint8-scaled output");
        // inverse: invert diffusion mechanics to enhance non-coherent regions
        desc.add_options()("inverse", po::bool_switch(&cfg.inverse_)->default_value(false),
                           "Invert diffusion mechanics to enhance non-coherent regions instead of coherent ones");

        // remove-small-objects: optional value; default and implicit both 250. Set to 0 to disable.
        desc.add_options()("remove-small-objects", po::value<int>(&cfg.remove_small_objects_)->default_value(250)->implicit_value(250),
                           "Remove connected components smaller than N pixels from binary output (0 disables)");

        desc.add_options()
            ("threads", po::value<int>(&num_threads)->default_value(-1), "OpenMP threads for per-image compute (-1 auto)")
            ("jobs", po::value<int>(&cfg.jobs_)->default_value(1), "Parallel file jobs in folder mode (>=1)")
        ;

        if (argc == 1) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        po::notify(vm);

        if (cfg.step_size_ > 0.25f) {
            std::cerr << "Warning: step-size > 0.25 may be unstable; clamping to 0.25\n";
            cfg.step_size_ = 0.25f;
        }
        if (cfg.downsample_ < 1) cfg.downsample_ = 1;
        if (cfg.dilate_ < 0) cfg.dilate_ = 0;
        if (vm.count("threshold") && vm.count("no-threshold")) {
            throw std::runtime_error("Options --threshold and --no-threshold are mutually exclusive");
        }
        if (vm.count("threshold")) {
            cfg.apply_threshold_ = true;
            cfg.use_otsu_ = std::isnan(threshold_opt);
            if (!cfg.use_otsu_) cfg.threshold_value_ = threshold_opt;
        }
        if (cfg.direction_field_ && !cfg.coherence_field_) {
            throw std::runtime_error("--direction-field requires --coherence-field");
        }
        if (!std::isnan(cfg.min_val_) && !std::isnan(cfg.max_val_) && cfg.min_val_ > cfg.max_val_) {
            throw std::runtime_error("--min-val cannot be greater than --max-val");
        }

        int jobs = std::max(1, cfg.jobs_);
        int compute_threads = 1;
        #ifdef _OPENMP
        if (jobs > 1) {
            compute_threads = 1; // parallelize over files; keep per-image compute single-threaded
        } else {
            compute_threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
        }
        omp_set_num_threads(compute_threads);
        #else
        (void)num_threads;
        compute_threads = 1;
        #endif

        const bool dir_mode = is_directory(in_path);

        auto is_zarr_root = [&](const std::string& p) -> bool {
            namespace fs = std::filesystem;
            std::error_code ec;
            fs::path pp(p);
            if (!fs::exists(pp, ec) || !fs::is_directory(pp, ec)) return false;
            if (pp.extension() == ".zarr") return true;
            return fs::exists(pp / ".zgroup", ec) || fs::exists(pp / ".zarray", ec);
        };

        if (dir_mode && is_zarr_root(in_path)) {
            // Zarr volume mode: read OME-Zarr, run CED per-slice; write OME-Zarr or TIFFs
            namespace fs = std::filesystem;
            fs::path in_root(in_path);
            fs::path out_root(out_path);
            const bool out_is_zarr = (out_root.extension() == ".zarr");

            // Open source dataset at group_idx
            // Two cases:
            // 1) in_path points to zarr root (has .zgroup) -> use group_idx subfolder (e.g. 0)
            // 2) in_path points directly to a dataset (has .zarray) -> open that dataset (ignore group_idx)
            const bool path_is_dataset = fs::exists(in_root / ".zarray");
            z5::filesystem::handle::Dataset inHandle = [&]() {
                std::string dimsep = ".";
                if (path_is_dataset) {
                    fs::path ds_path = in_root;
                    try {
                        json j = json::parse(std::ifstream((ds_path / ".zarray").string()));
                        if (j.contains("dimension_separator")) dimsep = j["dimension_separator"].get<std::string>();
                    } catch (...) {}
                    z5::filesystem::handle::Group parent(ds_path.parent_path(), z5::FileMode::FileMode::r);
                    return z5::filesystem::handle::Dataset(parent, ds_path.filename().string(), dimsep);
                } else {
                    z5::filesystem::handle::Group root(in_root, z5::FileMode::FileMode::r);
                    try {
                        json j = json::parse(std::ifstream((in_root / std::to_string(group_idx) / ".zarray").string()));
                        if (j.contains("dimension_separator")) dimsep = j["dimension_separator"].get<std::string>();
                    } catch (...) {}
                    return z5::filesystem::handle::Dataset(root, std::to_string(group_idx), dimsep);
                }
            }();
            std::unique_ptr<z5::Dataset> dsIn = z5::filesystem::openDataset(inHandle);

            const auto& shape = dsIn->shape(); // [Z, Y, X]
            if (shape.size() != 3) {
                std::cerr << "Expected 3D OME-Zarr (Z,Y,X); got dims=" << shape.size() << std::endl;
                return 1;
            }
            const size_t Z = shape[0];
            const size_t Y = shape[1];
            const size_t X = shape[2];

            std::cout << "Input Zarr shape ZYX = [" << Z << ", " << Y << ", " << X << "]\n";

            // Compute Z range
            size_t z0 = 0, z1 = (Z > 0 ? Z - 1 : 0);
            if (Z == 0) {
                std::cout << "Empty Zarr dataset (Z=0). Nothing to do." << std::endl;
                return 0;
            }
            z0 = static_cast<size_t>(std::max(0, std::min<int>(min_z, static_cast<int>(Z) - 1)));
            if (max_z < 0) {
                z1 = Z - 1;
            } else {
                z1 = static_cast<size_t>(std::max(0, std::min<int>(max_z, static_cast<int>(Z) - 1)));
            }
            if (z1 < z0) {
                std::cout << "Z range empty after clamping: [" << min_z << ", " << max_z << "] -> [" << z0 << ", " << z1 << "]\n";
                return 0;
            }

            // Common progress setup
            int jobs = std::max(1, cfg.jobs_);
            std::atomic<size_t> done{0};
            const size_t total = (z1 - z0 + 1);
            {
                std::lock_guard<std::mutex> lock(g_print_mtx);
                std::cout << std::fixed << std::setprecision(1)
                          << "\rProgress: 0/" << total << " (0.0%)" << std::flush;
            }

            if (!out_is_zarr) {
                // Write per-Z TIFFs into output directory
                if (!ensure_dir(out_root.string())) {
                    std::cerr << "Failed to create output directory: " << out_root << std::endl;
                    return 1;
                }

                // Determine zero-padding width: at least 5 digits, or more if needed (based on max index in range)
                size_t pad_width = std::max<size_t>(5, std::to_string(z1).size());

                #ifdef _OPENMP
                #pragma omp parallel for num_threads(jobs) schedule(dynamic)
                #endif
                for (long long zi = static_cast<long long>(z0); zi <= static_cast<long long>(z1); ++zi) {
                    const size_t z = static_cast<size_t>(zi);
                    cv::Mat src;
                    // Read one Z slab
                    if (dsIn->getDtype() == z5::types::Datatype::uint8) {
                        xt::xarray<uint8_t> slab = xt::empty<uint8_t>({1ul, Y, X});
                        z5::types::ShapeType off = {z, 0ul, 0ul};
                        z5::multiarray::readSubarray<uint8_t>(dsIn, slab, off.begin());
                        src.create(static_cast<int>(Y), static_cast<int>(X), CV_8UC1);
                        for (size_t y = 0; y < Y; ++y) {
                            uint8_t* row = src.ptr<uint8_t>(static_cast<int>(y));
                            for (size_t x = 0; x < X; ++x) row[x] = slab(0, y, x);
                        }
                    } else if (dsIn->getDtype() == z5::types::Datatype::uint16) {
                        xt::xarray<uint16_t> slab = xt::empty<uint16_t>({1ul, Y, X});
                        z5::types::ShapeType off = {z, 0ul, 0ul};
                        z5::multiarray::readSubarray<uint16_t>(dsIn, slab, off.begin());
                        src.create(static_cast<int>(Y), static_cast<int>(X), CV_16UC1);
                        for (size_t y = 0; y < Y; ++y) {
                            uint16_t* row = src.ptr<uint16_t>(static_cast<int>(y));
                            for (size_t x = 0; x < X; ++x) row[x] = slab(0, y, x);
                        }
                    } else if (dsIn->getDtype() == z5::types::Datatype::float32) {
                        xt::xarray<float> slab = xt::empty<float>({1ul, Y, X});
                        z5::types::ShapeType off = {z, 0ul, 0ul};
                        z5::multiarray::readSubarray<float>(dsIn, slab, off.begin());
                        src.create(static_cast<int>(Y), static_cast<int>(X), CV_32FC1);
                        for (size_t y = 0; y < Y; ++y) {
                            float* row = src.ptr<float>(static_cast<int>(y));
                            for (size_t x = 0; x < X; ++x) row[x] = slab(0, y, x);
                        }
                    } else {
                        #ifdef _OPENMP
                        if (omp_get_thread_num() == 0)
                        #endif
                        std::cerr << "Unsupported Zarr dtype; only uint8/uint16/float32 supported." << std::endl;
                        continue;
                    }

                    // Skip empty slices
                    if (is_all_zero(src)) {
                        size_t d = ++done;
                        if ((d % 1) == 0) {
                            double pct = 100.0 * double(d) / double(total);
                            std::lock_guard<std::mutex> lock(g_print_mtx);
                            std::cout << std::fixed << std::setprecision(1)
                                      << "\rProgress: " << d << "/" << total << " (" << pct << "%)" << std::flush;
                        }
                        continue;
                    }

                    Config local_cfg = cfg; local_cfg.show_progress_ = false;
                    // Write TIFF named by z-index
                    std::ostringstream name;
                    name << std::setw(static_cast<int>(pad_width)) << std::setfill('0') << z << ".tif";
                    fs::path out_file = out_root / name.str();
                    std::vector<int> params = { cv::IMWRITE_TIFF_COMPRESSION, 32773 };
                    if (local_cfg.coherence_field_ && local_cfg.direction_field_) {
                        cv::Mat dirrgb = compute_direction_field_rgb(src, local_cfg);
                        if (!cv::imwrite(out_file.string(), dirrgb)) {
                            #ifdef _OPENMP
                            if (omp_get_thread_num() == 0)
                            #endif
                            std::cerr << "Failed to write RGB TIFF: " << out_file << std::endl;
                        }
                    } else {
                        cv::Mat out_u8;
                        if (local_cfg.coherence_field_) {
                            cv::Mat coh = compute_coherence_field_full(src, local_cfg);
                            out_u8 = to_uint8_scaled_and_threshold(coh, local_cfg);
                        } else {
                            cv::Mat out = process_one_image(src, local_cfg);
                            out_u8 = to_uint8_scaled_and_threshold(out, local_cfg);
                        }
                        if (!cv::imwrite(out_file.string(), out_u8, params)) {
                            #ifdef _OPENMP
                            if (omp_get_thread_num() == 0)
                            #endif
                            std::cerr << "Failed to write TIFF: " << out_file << std::endl;
                        }
                    }
                    
                    if (false) {
                        #ifdef _OPENMP
                        if (omp_get_thread_num() == 0)
                        #endif
                        ;
                    }

                    size_t d = ++done;
                    if ((d % 1) == 0) {
                        double pct = 100.0 * double(d) / double(total);
                        std::lock_guard<std::mutex> lock(g_print_mtx);
                        std::cout << std::fixed << std::setprecision(1)
                                  << "\rProgress: " << d << "/" << total << " (" << pct << "%)" << std::flush;
                    }
                }
                std::cout << std::endl;
                std::cout << "Saved per-Z TIFFs to: " << out_root << std::endl;
                return 0;
            }

            // Direction field RGB currently not supported for OME-Zarr output
            if (cfg.coherence_field_ && cfg.direction_field_) {
                std::cerr << "--direction-field is only supported when writing TIFFs."
                          << " For OME-Zarr output, omit --direction-field or write TIFFs with -o <dir>." << std::endl;
                return 1;
            }

            // Else: Prepare output zarr root and level-0 dataset
            z5::filesystem::handle::File outFile(out_root);
            z5::createFile(outFile, true);

            const size_t CH = 128, CW = 128; // chunking in Y,X; Z chunk = 1 for per-slice IO
            const size_t CZ = 1;
            auto make_shape = [](size_t z, size_t y, size_t x){ return std::vector<size_t>{z, y, x}; };
            std::vector<size_t> shape0{Z, Y, X};
            std::vector<size_t> chunks0{CZ, std::min(CH, Y), std::min(CW, X)};
            nlohmann::json compOpts0 = {{"cname","zstd"},{"clevel",1},{"shuffle",0}};
            auto dsOut0 = z5::createDataset(outFile, "0", "uint8", shape0, chunks0, std::string("blosc"), compOpts0);

            // Process slices, parallelized over Z
            // jobs/done/total already declared above

            #ifdef _OPENMP
            #pragma omp parallel for num_threads(jobs) schedule(dynamic)
            #endif
            for (long long zi = static_cast<long long>(z0); zi <= static_cast<long long>(z1); ++zi) {
                // Read one slice from input
                const size_t z = static_cast<size_t>(zi);
                cv::Mat src;
                // Support uint8 / uint16; other types -> convert later
                if (dsIn->getDtype() == z5::types::Datatype::uint8) {
                    xt::xarray<uint8_t> slab = xt::empty<uint8_t>({1ul, Y, X});
                    z5::types::ShapeType off = {z, 0ul, 0ul};
                    z5::multiarray::readSubarray<uint8_t>(dsIn, slab, off.begin());
                    src.create(static_cast<int>(Y), static_cast<int>(X), CV_8UC1);
                    for (size_t y = 0; y < Y; ++y) {
                        uint8_t* row = src.ptr<uint8_t>(static_cast<int>(y));
                        for (size_t x = 0; x < X; ++x) row[x] = slab(0, y, x);
                    }
                } else if (dsIn->getDtype() == z5::types::Datatype::uint16) {
                    xt::xarray<uint16_t> slab = xt::empty<uint16_t>({1ul, Y, X});
                    z5::types::ShapeType off = {z, 0ul, 0ul};
                    z5::multiarray::readSubarray<uint16_t>(dsIn, slab, off.begin());
                    src.create(static_cast<int>(Y), static_cast<int>(X), CV_16UC1);
                    for (size_t y = 0; y < Y; ++y) {
                        uint16_t* row = src.ptr<uint16_t>(static_cast<int>(y));
                        for (size_t x = 0; x < X; ++x) row[x] = slab(0, y, x);
                    }
                } else if (dsIn->getDtype() == z5::types::Datatype::float32) {
                    xt::xarray<float> slab = xt::empty<float>({1ul, Y, X});
                    z5::types::ShapeType off = {z, 0ul, 0ul};
                    z5::multiarray::readSubarray<float>(dsIn, slab, off.begin());
                    src.create(static_cast<int>(Y), static_cast<int>(X), CV_32FC1);
                    for (size_t y = 0; y < Y; ++y) {
                        float* row = src.ptr<float>(static_cast<int>(y));
                        for (size_t x = 0; x < X; ++x) row[x] = slab(0, y, x);
                    }
                } else {
                    #ifdef _OPENMP
                    if (omp_get_thread_num() == 0)
                    #endif
                    std::cerr << "Unsupported Zarr dtype; only uint8/uint16/float32 supported." << std::endl;
                    continue;
                }

                // Process slice
                Config local_cfg = cfg; local_cfg.show_progress_ = false;
                cv::Mat out_u8;
                if (local_cfg.coherence_field_) {
                    cv::Mat coh = compute_coherence_field_full(src, local_cfg);
                    out_u8 = to_uint8_scaled_and_threshold(coh, local_cfg);
                } else {
                    cv::Mat out = process_one_image(src, local_cfg);
                    out_u8 = to_uint8_scaled_and_threshold(out, local_cfg);
                }

                // Write to output level-0 at [z, 0, 0]
                xt::xarray<uint8_t> slabOut = xt::empty<uint8_t>({1ul, Y, X});
                for (size_t y = 0; y < Y; ++y) {
                    const uint8_t* row = out_u8.ptr<uint8_t>(static_cast<int>(y));
                    for (size_t x = 0; x < X; ++x) slabOut(0, y, x) = row[x];
                }
                z5::types::ShapeType offOut = {z, 0ul, 0ul};
                z5::multiarray::writeSubarray<uint8_t>(dsOut0, slabOut, offOut.begin());

                size_t d = ++done;
                if ((d % 1) == 0) {
                    double pct = 100.0 * double(d) / double(total);
                    std::lock_guard<std::mutex> lock(g_print_mtx);
                    std::cout << std::fixed << std::setprecision(1)
                              << "\rProgress: " << d << "/" << total << " (" << pct << "%)" << std::flush;
                }
            }
            std::cout << std::endl;

            // Write attributes and OME-NGFF multiscales metadata
            nlohmann::json attrs;
            attrs["axes"] = nlohmann::json::array({
                nlohmann::json{{"name","z"},{"type","space"},{"unit","pixel"}},
                nlohmann::json{{"name","y"},{"type","space"},{"unit","pixel"}},
                nlohmann::json{{"name","x"},{"type","space"},{"unit","pixel"}}
            });
            attrs["canvas_size"] = {static_cast<int>(X), static_cast<int>(Y)};
            attrs["note_axes_order"] = "ZYX (slice, row, col)";
            attrs["source_zarr"] = in_root.string();

            nlohmann::json multiscale;
            multiscale["version"] = "0.4";
            multiscale["name"] = "ced2d";
            multiscale["axes"] = attrs["axes"];
            multiscale["datasets"] = nlohmann::json::array();
            for (int level = 0; level <= 5; ++level) {
                nlohmann::json dset;
                dset["path"] = std::to_string(level);
                double s = std::pow(2.0, level);
                dset["coordinateTransformations"] = nlohmann::json::array({
                    nlohmann::json{{"type","scale"},{"scale", nlohmann::json::array({s, s, s})}},
                    nlohmann::json{{"type","translation"},{"translation", nlohmann::json::array({0.0, 0.0, 0.0})}}
                });
                multiscale["datasets"].push_back(dset);
            }
            multiscale["metadata"] = nlohmann::json{{"downsampling_method","mean"}};
            attrs["multiscales"] = nlohmann::json::array({multiscale});
            z5::filesystem::writeAttributes(outFile, attrs);

            // Build additional pyramid levels by mean pooling 2x2x2 from previous level (as uint8)
            for (int targetLevel = 1; targetLevel <= 5; ++targetLevel) {
                auto src = z5::openDataset(outFile, std::to_string(targetLevel - 1));
                const auto& sShape = src->shape();
                size_t sz = std::max<size_t>(1, sShape[0] / 2);
                size_t sy = std::max<size_t>(1, sShape[1] / 2);
                size_t sx = std::max<size_t>(1, sShape[2] / 2);
                std::vector<size_t> dShape{sz, sy, sx};
                std::vector<size_t> dChunks{1ul, std::min(CH, sy), std::min(CW, sx)};
                nlohmann::json compOpts = {{"cname","zstd"},{"clevel",1},{"shuffle",0}};
                auto dst = z5::createDataset(outFile, std::to_string(targetLevel), "uint8", dShape, dChunks, std::string("blosc"), compOpts);

                const size_t tileZ = 1, tileY = CH, tileX = CW;
                const size_t tilesY = (sy + tileY - 1) / tileY;
                const size_t tilesX = (sx + tileX - 1) / tileX;
                std::atomic<size_t> tilesDone{0};
                const size_t totalTiles = tilesY * tilesX;
                #ifdef _OPENMP
                #pragma omp parallel for schedule(dynamic) collapse(2)
                #endif
                for (long long ty = 0; ty < static_cast<long long>(tilesY); ++ty) {
                    for (long long tx = 0; tx < static_cast<long long>(tilesX); ++tx) {
                        size_t y0 = static_cast<size_t>(ty) * tileY;
                        size_t x0 = static_cast<size_t>(tx) * tileX;
                        size_t ly = std::min(tileY, sy - y0);
                        size_t lx = std::min(tileX, sx - x0);
                        size_t lz = sz;

                        xt::xarray<uint8_t> srcChunk = xt::empty<uint8_t>({lz, ly*2ul, lx*2ul});
                        {
                            z5::types::ShapeType off = {0ul, y0*2ul, x0*2ul};
                            z5::multiarray::readSubarray<uint8_t>(src, srcChunk, off.begin());
                        }
                        xt::xarray<uint8_t> dstChunk = xt::empty<uint8_t>({lz, ly, lx});
                        for (size_t zz = 0; zz < lz; ++zz) {
                            for (size_t yy = 0; yy < ly; ++yy) {
                                for (size_t xx = 0; xx < lx; ++xx) {
                                    uint32_t a = 0;
                                    a += srcChunk(zz, 2*yy + 0, 2*xx + 0);
                                    if (2*xx + 1 < srcChunk.shape()[2]) a += srcChunk(zz, 2*yy + 0, 2*xx + 1);
                                    if (2*yy + 1 < srcChunk.shape()[1]) a += srcChunk(zz, 2*yy + 1, 2*xx + 0);
                                    if (2*yy + 1 < srcChunk.shape()[1] && 2*xx + 1 < srcChunk.shape()[2]) a += srcChunk(zz, 2*yy + 1, 2*xx + 1);
                                    dstChunk(zz, yy, xx) = static_cast<uint8_t>(a / 4u);
                                }
                            }
                        }
                        z5::types::ShapeType offD = {0ul, y0, x0};
                        z5::multiarray::writeSubarray<uint8_t>(dst, dstChunk, offD.begin());
                        size_t d = ++tilesDone;
                        #ifdef _OPENMP
                        if (omp_get_thread_num() == 0)
                        #endif
                        if ((d % 1) == 0) {
                            int pct = static_cast<int>(100.0 * double(d) / double(totalTiles));
                            std::lock_guard<std::mutex> lock(g_print_mtx);
                            std::cout << "\r[downsample L" << targetLevel << "] tiles " << d << "/" << totalTiles
                                      << " (" << pct << "%)" << std::flush;
                        }
                    }
                }
                std::cout << std::endl;
            }

            std::cout << "Saved OME-Zarr: " << out_root.string() << std::endl;
            return 0;
        }

        if (dir_mode) {
            // Folder mode: process all .tif/.tiff files
            if (!ensure_dir(out_path)) {
                std::cerr << "Cannot create/open output directory: " << out_path << std::endl;
                return 1;
            }
            std::vector<cv::String> files;
            cv::glob(in_path + "/*.tif", files, false);
            std::vector<cv::String> files_tiff;
            cv::glob(in_path + "/*.tiff", files_tiff, false);
            files.insert(files.end(), files_tiff.begin(), files_tiff.end());
            if (files.empty()) {
                std::cerr << "No TIFF files found in directory: " << in_path << std::endl;
                return 1;
            }
            std::cout << "Found " << files.size() << " TIFF files in " << in_path << "\n";
            std::vector<int> params = { cv::IMWRITE_TIFF_COMPRESSION, 32773 }; // packbits
            jobs = std::max(1, cfg.jobs_);
            bool multi = jobs > 1;
            std::cout << "Folder jobs: " << jobs << " (per-image compute threads: " << compute_threads << ")\n";

            const int total = static_cast<int>(files.size());
            std::atomic<int> completed{0};
            {
                std::lock_guard<std::mutex> lock(g_print_mtx);
                std::cout << std::fixed << std::setprecision(1)
                          << "\rProgress: 0/" << total << " (0.0%), remaining " << total << std::flush;
            }

            #ifdef _OPENMP
            if (multi) {
                #pragma omp parallel for num_threads(jobs) schedule(dynamic)
                for (int i = 0; i < static_cast<int>(files.size()); ++i) {
                    const auto f = files[i];
                    cv::Mat img = cv::imread(f, cv::IMREAD_UNCHANGED);
                    namespace fs = std::filesystem;
                    std::string base = fs::path(f).filename().string();
                    if (img.empty() || img.channels() != 1 || img.dims != 2) {
                        int done = ++completed;
                        int rem = total - done;
                        double pct = 100.0 * (double)done / (double)total;
                        std::lock_guard<std::mutex> lock(g_print_mtx);
                        std::cout << std::fixed << std::setprecision(1)
                                  << "\rProgress: " << done << "/" << total << " (" << pct << "%), remaining " << rem << std::flush;
                        continue;
                    }
                    Config local_cfg = cfg; local_cfg.show_progress_ = false;
                    std::string out_file = (fs::path(out_path) / base).string();
                    if (local_cfg.coherence_field_ && local_cfg.direction_field_) {
                        cv::Mat dirrgb = compute_direction_field_rgb(img, local_cfg);
                        (void)cv::imwrite(out_file, dirrgb);
                    } else {
                        cv::Mat out_u8;
                        if (local_cfg.coherence_field_) {
                            cv::Mat coh = compute_coherence_field_full(img, local_cfg);
                            out_u8 = to_uint8_scaled_and_threshold(coh, local_cfg);
                        } else {
                            cv::Mat out = process_one_image(img, local_cfg);
                            out_u8 = to_uint8_scaled_and_threshold(out, local_cfg);
                        }
                        (void)cv::imwrite(out_file, out_u8, params);
                    }
                    int done = ++completed;
                    int rem = total - done;
                    double pct = 100.0 * (double)done / (double)total;
                    std::lock_guard<std::mutex> lock(g_print_mtx);
                    std::cout << std::fixed << std::setprecision(1)
                              << "\rProgress: " << done << "/" << total << " (" << pct << "%), remaining " << rem << std::flush;
                }
            } else
            #endif
            {
                const int total = static_cast<int>(files.size());
                int completed = 0;
                for (size_t idx = 0; idx < files.size(); ++idx) {
                    const auto& f = files[idx];
                    namespace fs = std::filesystem;
                    std::string base = fs::path(f).filename().string();
                    cv::Mat img = cv::imread(f, cv::IMREAD_UNCHANGED);
                    if (!(img.empty() || img.channels() != 1 || img.dims != 2)) {
                        std::string out_file = (fs::path(out_path) / base).string();
                        if (cfg.coherence_field_ && cfg.direction_field_) {
                            cv::Mat dirrgb = compute_direction_field_rgb(img, cfg);
                            (void)cv::imwrite(out_file, dirrgb);
                        } else {
                            cv::Mat out_u8;
                            if (cfg.coherence_field_) {
                                cv::Mat coh = compute_coherence_field_full(img, cfg);
                                out_u8 = to_uint8_scaled_and_threshold(coh, cfg);
                            } else {
                                cv::Mat out = process_one_image(img, cfg);
                                out_u8 = to_uint8_scaled_and_threshold(out, cfg);
                            }
                            (void)cv::imwrite(out_file, out_u8, params);
                        }
                    }
                    int done = ++completed;
                    int rem = total - done;
                    double pct = 100.0 * (double)done / (double)total;
                    std::cout << std::fixed << std::setprecision(1)
                              << "\rProgress: " << done << "/" << total << " (" << pct << "%), remaining " << rem << std::flush;
                }
            }
            std::cout << std::endl << "Done folder processing." << std::endl;
        } else {
            cv::Mat img = cv::imread(in_path, cv::IMREAD_UNCHANGED);
            if (img.empty()) {
                std::cerr << "Failed to read input TIFF: " << in_path << std::endl;
                return 1;
            }
            if (img.channels() != 1) {
                std::cerr << "Only single-channel (grayscale) 2D TIFFs are supported" << std::endl;
                return 1;
            }
            if (img.dims != 2) {
                std::cerr << "Only 2D TIFFs are supported" << std::endl;
                return 1;
            }

            std::vector<int> params = { cv::IMWRITE_TIFF_COMPRESSION, 32773 };
            if (cfg.coherence_field_ && cfg.direction_field_) {
                cv::Mat dirrgb = compute_direction_field_rgb(img, cfg);
                if (!cv::imwrite(out_path, dirrgb)) {
                    std::cerr << "Failed to write output TIFF: " << out_path << std::endl;
                    return 1;
                }
            } else {
                cv::Mat out_u8;
                if (cfg.coherence_field_) {
                    cv::Mat coh = compute_coherence_field_full(img, cfg);
                    out_u8 = to_uint8_scaled_and_threshold(coh, cfg);
                } else {
                    cv::Mat out = process_one_image(img, cfg);
                    out_u8 = to_uint8_scaled_and_threshold(out, cfg);
                }
                if (!cv::imwrite(out_path, out_u8, params)) {
                    std::cerr << "Failed to write output TIFF: " << out_path << std::endl;
                    return 1;
                }
            }
            std::cout << "Saved: " << out_path << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
