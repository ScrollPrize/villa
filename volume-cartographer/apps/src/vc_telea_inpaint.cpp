// vc_inpaint_telea.cpp
// Inpaint black "holes" in a folder of PNGs using OpenCV Telea method.
// - A "hole" = connected black component (all RGB <= threshold) that DOES NOT touch the image border.
// - Scratches/holes are both handled by Telea; radius is configurable.
// - Alpha channel is preserved if present.
//
// Usage:
//   ./vc_inpaint_telea <input_dir> <output_dir> [radius=3] [black_threshold=8] [min_area=1] [recursive=1]
//
// Example:
//   ./vc_inpaint_telea ./in ./out 4 10 3 1
//
// Notes:
// - Only .png (case-insensitive) are processed.
// - Output keeps the same relative path under <output_dir>.
// - If no holes are found, the image is copied as-is.

#include <iostream>
#include <filesystem>
#include <string>
#include <algorithm>
#include <cctype>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

namespace fs = std::filesystem;

static inline bool iequals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::tolower(a[i]) != std::tolower(b[i])) return false;
    return true;
}

static inline bool is_png(const fs::path& p) {
    return iequals(p.extension().string(), ".png");
}

// Build a 1-channel 8-bit mask (255 where to inpaint) for "black holes"
// - Pixels with B,G,R <= thr are candidates
// - Keep only connected components that DO NOT touch image borders
// - Drop components with area < min_area
static cv::Mat build_hole_mask(const cv::Mat& bgr_or_bgra, int thr, int min_area) {
    CV_Assert(bgr_or_bgra.channels() == 1 || bgr_or_bgra.channels() == 3 || bgr_or_bgra.channels() == 4);

    cv::Mat bgr;
    if (bgr_or_bgra.channels() == 4) {
        std::vector<cv::Mat> ch;
        cv::split(bgr_or_bgra, ch); // B,G,R,A
        cv::merge(std::vector<cv::Mat>{ch[0], ch[1], ch[2]}, bgr);
    } else if (bgr_or_bgra.channels() == 1) {
        // promote gray->BGR for uniformity
        cv::cvtColor(bgr_or_bgra, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = bgr_or_bgra;
    }

    const int H = bgr.rows, W = bgr.cols;

    // Candidate black mask
    std::vector<cv::Mat> ch(3);
    cv::split(bgr, ch);
    cv::Mat mask0 = (ch[0] <= thr) & (ch[1] <= thr) & (ch[2] <= thr);
    mask0.convertTo(mask0, CV_8U, 255); // 0/255

    if (cv::countNonZero(mask0) == 0) return mask0; // nothing to inpaint

    // Connected components with stats
    cv::Mat labels, stats, centroids;
    int ncc = cv::connectedComponentsWithStats(mask0, labels, stats, centroids, 8, CV_32S);

    cv::Mat keep = cv::Mat::zeros(mask0.size(), CV_8U);

    for (int lbl = 1; lbl < ncc; ++lbl) { // skip background 0
        int left   = stats.at<int>(lbl, cv::CC_STAT_LEFT);
        int top    = stats.at<int>(lbl, cv::CC_STAT_TOP);
        int width  = stats.at<int>(lbl, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(lbl, cv::CC_STAT_HEIGHT);
        int area   = stats.at<int>(lbl, cv::CC_STAT_AREA);

        if (area < std::max(1, min_area)) continue;

        bool touches = (left == 0) || (top == 0) || (left + width == W) || (top + height == H);
        if (touches) continue; // not a "hole"

        // add this component to keep
        keep.setTo(255, labels == lbl);
    }

    return keep;
}

static bool inpaint_one(const fs::path& in_png, const fs::path& out_png,
                        float radius, int black_thr, int min_area) {
    cv::Mat img = cv::imread(in_png.string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Error: could not read " << in_png << "\n";
        return false;
    }

    const bool has_alpha = (img.channels() == 4);
    cv::Mat alpha;
    cv::Mat bgr;
    if (has_alpha) {
        std::vector<cv::Mat> ch;
        cv::split(img, ch); // B,G,R,A
        alpha = ch[3].clone();
        cv::merge(std::vector<cv::Mat>{ch[0], ch[1], ch[2]}, bgr);
    } else if (img.channels() == 1) {
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    } else if (img.channels() == 3) {
        bgr = img;
    } else {
        std::cerr << "Error: unsupported channel count (" << img.channels() << ") in " << in_png << "\n";
        return false;
    }

    cv::Mat mask = build_hole_mask(has_alpha ? img : bgr, black_thr, min_area);
    if (mask.empty() || cv::countNonZero(mask) == 0) {
        // Nothing to inpaint: just copy/write input
        fs::create_directories(out_png.parent_path());
        if (!cv::imwrite(out_png.string(), img)) {
            std::cerr << "Error: write failed (no-inpaint path) to " << out_png << "\n";
            return false;
        }
        return true;
    }

    cv::Mat out_bgr;
    cv::inpaint(bgr, mask, out_bgr, radius, cv::INPAINT_TELEA);

    cv::Mat out;
    if (has_alpha) {
        std::vector<cv::Mat> ch{ out_bgr.channels()==3 ? std::vector<cv::Mat>{} : std::vector<cv::Mat>{} };
        std::vector<cv::Mat> bgr_ch;
        cv::split(out_bgr, bgr_ch);
        std::vector<cv::Mat> merged{ bgr_ch[0], bgr_ch[1], bgr_ch[2], alpha };
        cv::merge(merged, out);
    } else if (img.channels() == 1) {
        // If original was gray, convert result back to gray
        cv::cvtColor(out_bgr, out, cv::COLOR_BGR2GRAY);
    } else {
        out = out_bgr;
    }

    fs::create_directories(out_png.parent_path());
    if (!cv::imwrite(out_png.string(), out)) {
        std::cerr << "Error: could not write " << out_png << "\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_dir> <output_dir> [radius=3] [black_threshold=8] [min_area=1] [recursive=1]\n";
        return 1;
    }

    const fs::path in_dir  = argv[1];
    const fs::path out_dir = argv[2];

    if (!fs::exists(in_dir) || !fs::is_directory(in_dir)) {
        std::cerr << "Error: input_dir is not a directory: " << in_dir << "\n";
        return 2;
    }
    if (!fs::exists(out_dir)) {
        fs::create_directories(out_dir);
    }

    float radius = (argc >= 4) ? std::max(0.1f, std::stof(argv[3])) : 3.0f;
    int black_thr = (argc >= 5) ? std::max(0, std::stoi(argv[4])) : 8;
    int min_area  = (argc >= 6) ? std::max(1, std::stoi(argv[5])) : 1;
    bool recursive= (argc >= 7) ? (std::stoi(argv[6]) != 0) : true;

    size_t ok = 0, fail = 0, skipped = 0;

    auto process_entry = [&](const fs::directory_entry& e) {
        const fs::path p = e.path();
        if (!e.is_regular_file() || !is_png(p)) { ++skipped; return; }

        fs::path rel = fs::relative(p, in_dir);
        fs::path out_p = out_dir / rel;
        // Make sure parent exists; inpaint_one also creates it, but we can precreate for skipped copies
        fs::create_directories(out_p.parent_path());

        if (inpaint_one(p, out_p, radius, black_thr, min_area)) {
            std::cout << "OK: " << p << "  ->  " << out_p << "\n";
            ++ok;
        } else {
            std::cerr << "FAIL: " << p << "\n";
            ++fail;
        }
    };

    if (recursive) {
        for (const auto& e : fs::recursive_directory_iterator(in_dir)) process_entry(e);
    } else {
        for (const auto& e : fs::directory_iterator(in_dir)) process_entry(e);
    }

    std::cout << "\nDone. Success: " << ok << ", Failed: " << fail << ", Skipped (non-PNG): " << skipped << "\n"
              << "Params — radius: " << radius << ", black_threshold: " << black_thr
              << ", min_area: " << min_area << ", recursive: " << (recursive?1:0) << "\n";

    return (fail == 0) ? 0 : 3;
}
