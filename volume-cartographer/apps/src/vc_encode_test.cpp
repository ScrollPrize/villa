// vc_encode_test: A/B matrix comparing per-codec quality/size points.
//
// --codec h265 (default): sweeps air_clamp values at fixed QP (128³ chunks).
// --codec c3d           : sweeps target_ratio values (256³ chunks).
//
// Reports avg encoded size, ratio, MAE, RMSE, PSNR vs original raw.
//
// Usage: vc_encode_test <chunks-dir> [--codec {h265|c3d}] [--qp N] [--max K]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "utils/video_codec.hpp"
#include "utils/c3d_codec.hpp"

namespace fs = std::filesystem;

enum class CodecKind { H265, C3d };

struct Config {
    const char* name;
    int air_clamp;       // h265 only; 0 = off
    float target_ratio;  // c3d only; > 1.0
};

struct Agg {
    size_t bytes = 0;
    double mse = 0;       // vs intended ground truth (zeroed below air-clamp)
    double mae = 0;
    int max_err = 0;
    double mat_mse = 0;   // material-only (voxels with orig > air_clamp)
    size_t mat_n_total = 0;
    size_t n = 0;
};

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: vc_encode_test <chunks-dir> [--codec {h265|c3d}] [--qp N] [--max K]\n");
        return 1;
    }
    std::string dir = argv[1];
    int qp = 36;
    int max_chunks = 100000;
    CodecKind codec = CodecKind::H265;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--qp" && i + 1 < argc) qp = std::atoi(argv[++i]);
        else if (a == "--max" && i + 1 < argc) max_chunks = std::atoi(argv[++i]);
        else if (a == "--codec" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "c3d")       codec = CodecKind::C3d;
            else if (v == "h265") codec = CodecKind::H265;
            else { fprintf(stderr, "--codec must be h265 or c3d\n"); return 1; }
        }
    }

    const int CHUNK_DIM = (codec == CodecKind::C3d) ? 256 : 128;
    const size_t CHUNK_VOXELS = size_t(CHUNK_DIM) * CHUNK_DIM * CHUNK_DIM;

    std::vector<std::string> files;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        if (fs::file_size(e.path()) != CHUNK_VOXELS) continue;
        files.push_back(e.path().string());
        if ((int)files.size() >= max_chunks) break;
    }
    if (files.empty()) {
        fprintf(stderr, "No chunks matching %d³ in %s\n", CHUNK_DIM, dir.c_str());
        return 1;
    }
    if (codec == CodecKind::C3d) {
        printf("Chunks: %zu (256³ c3d)\n\n", files.size());
    } else {
        printf("Chunks: %zu (128³ h265, qp %d)\n\n", files.size(), qp);
    }

    std::vector<Config> configs;
    if (codec == CodecKind::C3d) {
        configs = {
            {"ratio=5",    0,   5.0f},
            {"ratio=10",   0,  10.0f},
            {"ratio=25",   0,  25.0f},
            {"ratio=50",   0,  50.0f},
            {"ratio=100",  0, 100.0f},
        };
    } else {
        configs = {
            {"baseline",     0,  0.0f},
            {"air=32",      32,  0.0f},
            {"air=48",      48,  0.0f},
            {"air=64",      64,  0.0f},
            {"air=72",      72,  0.0f},
        };
    }
    std::vector<Agg> agg(configs.size());

    for (const auto& f : files) {
        std::vector<std::byte> orig(CHUNK_VOXELS);
        std::ifstream fin(f, std::ios::binary);
        fin.read(reinterpret_cast<char*>(orig.data()), CHUNK_VOXELS);
        if (!fin) continue;

        for (size_t c = 0; c < configs.size(); ++c) {
            std::vector<std::byte> enc;
            std::vector<std::byte> dec;
            if (codec == CodecKind::C3d) {
                utils::C3dCodecParams p;
                p.target_ratio = configs[c].target_ratio;
                p.depth = CHUNK_DIM; p.height = CHUNK_DIM; p.width = CHUNK_DIM;
                enc = utils::c3d_encode(std::span<const std::byte>(orig), p);
                dec = utils::c3d_decode(std::span<const std::byte>(enc), CHUNK_VOXELS, p);
            } else {
                utils::VideoCodecParams p;
                p.qp = qp;
                p.depth = CHUNK_DIM; p.height = CHUNK_DIM; p.width = CHUNK_DIM;
                p.air_clamp = configs[c].air_clamp;
                enc = utils::video_encode(std::span<const std::byte>(orig), p);
                dec = utils::video_decode(std::span<const std::byte>(enc), CHUNK_VOXELS, p);
            }

            // Intended ground truth: original with v<=air_clamp zeroed
            // (c3d: air_clamp is always 0, so gt==a).
            // Material defined as orig > MAT_THRESH (fixed across configs for
            // apples-to-apples PSNR comparison).
            const int t = configs[c].air_clamp;
            constexpr int MAT_THRESH = 80;
            double se = 0, ae = 0, mat_se = 0;
            size_t mat_n = 0;
            int me = 0;
            for (size_t i = 0; i < CHUNK_VOXELS; ++i) {
                int a = (int)(uint8_t)orig[i];
                int gt = (t > 0 && a <= t) ? 0 : a;
                int b = (int)(uint8_t)dec[i];
                int d = std::abs(gt - b);
                if (d > me) me = d;
                ae += d; se += double(d) * d;
                if (a > MAT_THRESH) { mat_se += double(d) * d; ++mat_n; }
            }
            agg[c].bytes += enc.size();
            agg[c].mse += se / CHUNK_VOXELS;
            agg[c].mae += ae / CHUNK_VOXELS;
            agg[c].max_err = std::max(agg[c].max_err, me);
            if (mat_n > 0) agg[c].mat_mse += mat_se / mat_n;
            agg[c].mat_n_total += mat_n;
            agg[c].n++;
        }
    }

    const double raw_bytes = double(CHUNK_VOXELS) * files.size();
    printf("%-10s  ratio   mae    rmse   psnr    mat_psnr  max_err\n", "config");
    for (size_t c = 0; c < configs.size(); ++c) {
        if (!agg[c].n) continue;
        double ratio = raw_bytes / double(agg[c].bytes);
        double mae = agg[c].mae / agg[c].n;
        double mse = agg[c].mse / agg[c].n;
        double rmse = std::sqrt(mse);
        double psnr = mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse)
                              : std::numeric_limits<double>::infinity();
        double mat_mse = agg[c].mat_mse / agg[c].n;
        double mat_psnr = mat_mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mat_mse)
                                       : std::numeric_limits<double>::infinity();
        printf("%-10s  %5.1fx  %5.3f  %5.3f  %6.2f   %6.2f    %d\n",
               configs[c].name, ratio, mae, rmse, psnr, mat_psnr, agg[c].max_err);
    }
    return 0;
}
