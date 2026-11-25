#include <iostream>
#include <chrono>
#include <fstream>
#include "vc/core/util/ImageMat.hpp"
#include <opencv2/imgproc.hpp>
#include <vc_reader/thinning.hpp>

// Helper to get a modified output path
std::string getOutputPath(const std::string& originalPath, const std::string& suffix) {
    size_t dotPos = originalPath.rfind('.');
    if (dotPos == std::string::npos) {
        return originalPath + "_" + suffix;
    }
    return originalPath.substr(0, dotPos) + "_" + suffix + originalPath.substr(dotPos);
}

int main(int argc, char** argv) {
    cv::setNumThreads(0);

    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image_base> [cpu|cuda]" << std::endl;
        return 1;
    }

    const std::string backend = argc == 4 ? argv[3] : "cpu";
    if (backend != "cpu" && backend != "cuda") {
        std::cerr << "Backend must be cpu or cuda\n"; return 1;
    }
#ifndef VC_READER_CUDA
    if (backend == "cuda") { std::cerr << "CUDA support is not built\n"; return 1; }
#endif
    try {
    std::string inputPath = argv[1];
    std::string basePath = argv[2];

    cv::Mat inputImage = vc::core::util::imageMat(vcr::sources::readImagePage(inputPath, 0));
    if (inputImage.empty()) {
        std::cerr << "Error: Could not read input image at " << inputPath << std::endl;
        return 1;
    }

    if (inputImage.channels() == 3) cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2GRAY);
    else if (inputImage.channels() == 4) cv::cvtColor(inputImage, inputImage, cv::COLOR_BGRA2GRAY);
    if (inputImage.depth() == CV_16U) inputImage.convertTo(inputImage, CV_8U, 1.0 / 256);

    if (inputImage.type() != CV_8UC1) throw std::invalid_argument("expected an 8-bit or 16-bit grayscale/RGB image");
    if (!inputImage.isContinuous()) inputImage = inputImage.clone();
    const auto thin = [&](cv::Mat& output, vcr::thinning::Algorithm algorithm) {
        output.create(inputImage.size(), CV_8UC1);
        // Preserve the app's rounded 0/255 normalization for skeleton methods.
        cv::Mat binary;
        inputImage.convertTo(binary, CV_8U, 1.0 / 255.0);
#ifdef VC_READER_CUDA
        if (backend == "cuda") {
            vcr::thinning::cuda::skeleton(binary.data, output.data, inputImage.rows, inputImage.cols, 1, algorithm);
            return;
        }
#endif
        vcr::thinning::skeleton(binary.data, output.data, inputImage.rows, inputImage.cols, 1, algorithm);
    };

    // --- Custom Thinning ---
    cv::Mat customOutput;
    auto startCustom = std::chrono::high_resolution_clock::now();
    customOutput.create(inputImage.size(), CV_8UC1);
    vcr::thinning::Scratch scratch;
    vcr::thinning::trace(inputImage.data, inputImage.rows, inputImage.cols, customOutput.data,
        nullptr, scratch, nullptr, {}, backend == "cuda" ? vcr::thinning::Backend::Cuda : vcr::thinning::Backend::Cpu);
    auto endCustom = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> customTime = endCustom - startCustom;
    std::cout << "Custom thinning took: " << customTime.count() << " ms" << std::endl;
    std::string customOutputPath = getOutputPath(basePath, "custom");
    vcr::sources::writeImage(customOutputPath, vc::core::util::imageBuffer(customOutput).view());
    std::cout << "Custom thinning output saved to " << customOutputPath << std::endl;

    // --- Skeletonization (Zhang-Suen) ---
    cv::Mat zhangSuenOutput;
    auto startZhang = std::chrono::high_resolution_clock::now();
    thin(zhangSuenOutput, vcr::thinning::Algorithm::ZhangSuen);
    auto endZhang = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> zhangTime = endZhang - startZhang;
    std::cout << "Zhang-Suen thinning took: " << zhangTime.count() << " ms" << std::endl;
    std::string zhangOutputPath = getOutputPath(basePath, "zhangsuen");
    vcr::sources::writeImage(zhangOutputPath, vc::core::util::imageBuffer(zhangSuenOutput).view());
    std::cout << "Zhang-Suen output saved to " << zhangOutputPath << std::endl;

    // --- Skeletonization (Guo-Hall) ---
    cv::Mat guoHallOutput;
    auto startGuo = std::chrono::high_resolution_clock::now();
    thin(guoHallOutput, vcr::thinning::Algorithm::GuoHall);
    auto endGuo = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> guoTime = endGuo - startGuo;
    std::cout << "Guo-Hall thinning took: " << guoTime.count() << " ms" << std::endl;
    std::string guoOutputPath = getOutputPath(basePath, "guohall");
    vcr::sources::writeImage(guoOutputPath, vc::core::util::imageBuffer(guoHallOutput).view());
    std::cout << "Guo-Hall output saved to " << guoOutputPath << std::endl;

    // --- Skeletonization (Zhang-Suen then Guo-Hall) ---
    cv::Mat combinedOutput;
    auto startCombined = std::chrono::high_resolution_clock::now();
    thin(combinedOutput, vcr::thinning::Algorithm::ZhangSuenGuoHall);
    auto endCombined = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> combinedTime = endCombined - startCombined;
    std::cout << "Zhang-Suen then Guo-Hall thinning took: " << combinedTime.count() << " ms" << std::endl;
    std::string combinedOutputPath = getOutputPath(basePath, "zhangsuen_guohall");
    vcr::sources::writeImage(combinedOutputPath, vc::core::util::imageBuffer(combinedOutput).view());
    std::cout << "Combined output saved to " << combinedOutputPath << std::endl;

    } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
    return 0;
}
