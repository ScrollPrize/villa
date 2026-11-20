#include "inklabels/InkLabelIO.hpp"

#include "inklabels/InkLabelLayer.hpp"

#include "vc/core/util/ImageMat.hpp"

#include <vc_reader/image_write.hpp>
#include <vc_reader/source_reader.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <stdexcept>

namespace fs = std::filesystem;

namespace vc3d::inklabels {
namespace {

void writeLabelImage(const fs::path& path, const cv::Mat_<uint8_t>& image)
{
    fs::create_directories(path.parent_path());
    // vc_reader writes beside the final name (<name>.tif.tmp.*), so label
    // discovery never picks up a partial file, and renames it into place.
    vcr::sources::TiffOptions options;
    options.compression = "lzw";
    options.tile = std::array<int, 2>{1024, 1024};
    vcr::sources::writeTiff(path, vc::core::util::imageBuffer(image).view(), options);
}

cv::Mat_<uint8_t> readNonzero(const fs::path& path)
{
    if (!fs::exists(path)) {
        throw std::runtime_error("missing label file " + path.string());
    }
    cv::Mat image = vc::core::util::imageMat(vcr::sources::readImagePage(path.string(), 0));
    if (image.empty()) {
        throw std::runtime_error("cannot read label file " + path.string());
    }
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2GRAY);
    } else if (image.channels() != 1) {
        throw std::runtime_error("label file is not grayscale: " + path.string());
    }
    return image != 0;
}

} // namespace

TrainingLabelFiles trainingLabelFiles(const fs::path& directory, const std::string& prefix, int version)
{
    const std::string suffix = version > 0 ? "_v" + std::to_string(version) : std::string();
    return {directory / (prefix + "_inklabels" + suffix + ".tif"),
            directory / (prefix + "_supervision_mask" + suffix + ".tif")};
}

void writeTrainingLabels(const TrainingLabelFiles& files, const cv::Mat_<uint8_t>& labels)
{
    if (labels.empty()) {
        throw std::runtime_error("no labels to write");
    }
    const cv::Mat_<uint8_t> ink = labels == static_cast<uint8_t>(Label::Ink);
    const cv::Mat_<uint8_t> supervision = labels != static_cast<uint8_t>(Label::Unlabelled);
    writeLabelImage(files.inklabels, ink);
    writeLabelImage(files.supervisionMask, supervision);
}

LoadedTrainingLabels readTrainingLabels(const TrainingLabelFiles& files)
{
    const cv::Mat_<uint8_t> ink = readNonzero(files.inklabels);
    const cv::Mat_<uint8_t> supervision = readNonzero(files.supervisionMask);
    if (ink.size() != supervision.size()) {
        throw std::runtime_error("ink labels and supervision mask differ in size");
    }
    LoadedTrainingLabels loaded;
    loaded.labels = cv::Mat_<uint8_t>(ink.size(), static_cast<uint8_t>(Label::Unlabelled));
    loaded.labels.setTo(static_cast<uint8_t>(Label::NoInk), supervision);
    loaded.labels.setTo(static_cast<uint8_t>(Label::Ink), ink & supervision);
    loaded.inkOutsideSupervision = static_cast<std::size_t>(cv::countNonZero(ink & ~supervision));
    return loaded;
}

} // namespace vc3d::inklabels
