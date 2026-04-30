#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Args {
    fs::path input;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " -i <image>\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if ((key == "--input" || key == "-i") && i + 1 < argc) {
            args.input = argv[++i];
        } else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + key);
        }
    }

    if (args.input.empty()) {
        throw std::runtime_error("--input/-i is required");
    }
    return args;
}

cv::Mat load_grayscale(const fs::path& path) {
    cv::Mat src = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (src.empty()) {
        throw std::runtime_error("failed to read input image: " + path.string());
    }

    if (src.channels() > 1) {
        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    return src;
}

cv::Mat to_u8_for_threshold(const cv::Mat& src) {
    if (src.depth() == CV_8U) {
        return src;
    }

    cv::Mat src_float;
    src.convertTo(src_float, CV_32F);

    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(src_float, &min_value, &max_value);
    if (max_value <= min_value) {
        return cv::Mat::zeros(src.size(), CV_8U);
    }

    cv::Mat normalized;
    src_float.convertTo(normalized, CV_8U, 255.0 / (max_value - min_value),
                        -min_value * 255.0 / (max_value - min_value));
    return normalized;
}

cv::Mat binarize_otsu(const cv::Mat& gray) {
    cv::Mat u8 = to_u8_for_threshold(gray);
    cv::Mat binary;
    cv::threshold(u8, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return binary;
}

cv::Mat normalized_dt_u16(const cv::Mat& dt) {
    double max_value = 0.0;
    cv::minMaxLoc(dt, nullptr, &max_value);
    cv::Mat out;
    if (max_value <= 0.0) {
        out = cv::Mat::zeros(dt.size(), CV_16U);
    } else {
        dt.convertTo(out, CV_16U, 65535.0 / max_value);
    }
    return out;
}

cv::Mat trace_distance_ridges(const cv::Mat& dt, float min_distance) {
    CV_Assert(dt.type() == CV_32F);

    cv::Mat ridges = cv::Mat::zeros(dt.size(), CV_8U);
    for (int y = 1; y < dt.rows - 1; ++y) {
        for (int x = 1; x < dt.cols - 1; ++x) {
            const float center = dt.at<float>(y, x);
            if (center < min_distance) {
                continue;
            }

            bool is_local_max = true;
            for (int dy = -1; dy <= 1 && is_local_max; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    if (dt.at<float>(y + dy, x + dx) > center) {
                        is_local_max = false;
                        break;
                    }
                }
            }

            if (is_local_max) {
                ridges.at<std::uint8_t>(y, x) = 255;
            }
        }
    }
    return ridges;
}

int transition_count(const std::uint8_t p[8]) {
    int count = 0;
    for (int i = 0; i < 8; ++i) {
        if (p[i] == 0 && p[(i + 1) % 8] != 0) {
            ++count;
        }
    }
    return count;
}

int nonzero_count(const std::uint8_t p[8]) {
    return static_cast<int>(std::count_if(p, p + 8, [](std::uint8_t v) {
        return v != 0;
    }));
}

void thinning_iteration(cv::Mat& img, int iter) {
    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8U);

    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            if (img.at<std::uint8_t>(y, x) == 0) {
                continue;
            }

            const std::uint8_t p[8] = {
                img.at<std::uint8_t>(y - 1, x),
                img.at<std::uint8_t>(y - 1, x + 1),
                img.at<std::uint8_t>(y, x + 1),
                img.at<std::uint8_t>(y + 1, x + 1),
                img.at<std::uint8_t>(y + 1, x),
                img.at<std::uint8_t>(y + 1, x - 1),
                img.at<std::uint8_t>(y, x - 1),
                img.at<std::uint8_t>(y - 1, x - 1),
            };

            const int nz = nonzero_count(p);
            const int transitions = transition_count(p);
            if (nz < 2 || nz > 6 || transitions != 1) {
                continue;
            }

            const bool remove =
                iter == 0
                    ? (p[0] * p[2] * p[4] == 0 && p[2] * p[4] * p[6] == 0)
                    : (p[0] * p[2] * p[6] == 0 && p[0] * p[4] * p[6] == 0);
            if (remove) {
                marker.at<std::uint8_t>(y, x) = 255;
            }
        }
    }

    img.setTo(0, marker);
}

cv::Mat zhang_suen_thinning(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8U);

    cv::Mat img;
    cv::threshold(src, img, 0, 1, cv::THRESH_BINARY);

    cv::Mat prev = cv::Mat::zeros(img.size(), CV_8U);
    cv::Mat diff;
    do {
        thinning_iteration(img, 0);
        thinning_iteration(img, 1);
        cv::absdiff(img, prev, diff);
        img.copyTo(prev);
    } while (cv::countNonZero(diff) > 0);

    img *= 255;
    return img;
}

void write_image(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("failed to write image: " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const fs::path workdir = fs::current_path();

        const cv::Mat gray = load_grayscale(args.input);
        const cv::Mat binary = binarize_otsu(gray);

        cv::Mat dt;
        cv::distanceTransform(binary, dt, cv::DIST_L2, cv::DIST_MASK_PRECISE);

        const cv::Mat ridges = trace_distance_ridges(dt, 1.0f);
        const cv::Mat skeleton = zhang_suen_thinning(ridges);
        const cv::Mat dt_u16 = normalized_dt_u16(dt);

        const std::string stem = args.input.stem().string();
        write_image(workdir / (stem + "_dt.tif"), dt_u16);
        write_image(workdir / (stem + "_ridges.tif"), ridges);
        write_image(workdir / (stem + "_skeleton.tif"), skeleton);

        std::cout << "Wrote:\n"
                  << "  " << (workdir / (stem + "_dt.tif")) << "\n"
                  << "  " << (workdir / (stem + "_ridges.tif")) << "\n"
                  << "  " << (workdir / (stem + "_skeleton.tif")) << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
