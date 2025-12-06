#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include <nlohmann/json.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



using json = nlohmann::json;

int main(int argc, char *argv[])
{
    if (argc != 4) {
        std::cout << "usage: " << argv[0] << " <tiffxyz-src> <mask> <tiffxyz-src>" << std::endl;
        return EXIT_SUCCESS;
    }
    
    std::filesystem::path seg_path = argv[1];
    std::filesystem::path mask_path = argv[2];
    std::filesystem::path tgt_path = argv[3];
    
    std::unique_ptr<QuadSurface> surf;
    try {
        surf = load_quad_from_tifxyz(seg_path);
    }
    catch (...) {
        std::cout << "error when loading: " << seg_path << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat_<cv::Vec3f> *points = surf->rawPointsPtr();
    cv::Mat_<uint8_t> mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    cv::Mat_<uint8_t> mask_points(mask.size(), 0);

    for(int j=0;j<points->rows;j++)
        for(int i=0;i<points->cols;i++)
            if ((*points)(j,i)[0] == -1)
                mask_points(j,i) = 1;

    //closing operation to skip 1-2 pixel holes
    cv::Mat m = cv::getStructuringElement(cv::MORPH_RECT, {3,3});
    cv::erode(mask_points, mask_points, m, {-1,-1}, 1);
    cv::dilate(mask_points, mask_points, m, {-1,-1}, 1);

    for(int r=0;r<12;r++)
        cv::dilate(mask_points, mask_points, m, {-1,-1}, 1);

    std::cout << "sizes " << points->size() << mask.size() << std::endl;
    if (mask.size() != points->size())
        throw std::runtime_error("mask must be same size as tiffxyz");

    cv::erode(mask, mask, m, {-1,-1}, 1);

    cv::Mat_<cv::Vec3f> points_orig = points->clone();

    for(int r=0;r<100;r++) {
        int dia = int(float(100-r)/100*11)*2+1;
        std::cout << dia << std::endl;
        cv::GaussianBlur((*points), (*points), {dia,dia}, 0);
        points_orig.copyTo((*points), mask);
    }
    points->setTo(cv::Vec3f(-1,-1,-1), mask_points);

    surf->save(tgt_path);

    return EXIT_SUCCESS;
}
