#include <nlohmann/json.hpp>

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include XTENSORINCLUDE(io, xio.hpp)
#include XTENSORINCLUDE(views, xview.hpp)

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <filesystem>
#include <omp.h>

#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/StreamOperators.hpp"

using shape = z5::types::ShapeType;
using namespace xt::placeholders;


using json = nlohmann::json;



class MeasureLife
{
public:
    MeasureLife(std::string msg)
    {
        std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    ~MeasureLife()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

std::string time_str()
{
    using namespace std::chrono;

    // get current time
    auto now = system_clock::now();

    // get number of milliseconds for the current second
    // (remainder after division into seconds)
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream oss;

    oss << std::put_time(&bt, "%Y%m%d%H%M%S"); // HH:MM:SS
    oss << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
}

template <typename T, typename I>
float get_val(I &interp, cv::Vec3d l) {
    T v;
    interp.Evaluate(l[2], l[1], l[0], &v);
    return v;
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        std::cout << "usage: " << argv[0] << " <zarr-volume> <video-file> segments..." << std::endl;
        return EXIT_SUCCESS;
    }

    std::filesystem::path vol_path = argv[1];
    std::filesystem::path tgt_fn = argv[2];
    std::vector<std::filesystem::path> seg_dirs;
    for(int i=3;i<argc;i++)
        seg_dirs.push_back(argv[i]);

    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, "1", json::parse(std::ifstream(vol_path/"1/.zarray")).value<std::string>("dimension_separator","."));
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group 1 " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;

    cv::Size tgt_size = {3840, 2160};

    ChunkCache<uint8_t> chunk_cache(10e9);

    cv::VideoWriter vid(tgt_fn, cv::VideoWriter::fourcc('H','F','Y','U'), 5, tgt_size);

    for(auto &path : seg_dirs) {
        std::unique_ptr<QuadSurface> surf;
        try {
            surf = load_quad_from_tifxyz(path);
        }
        catch (...) {
            std::cout << "error, skipping: " << path << std::endl;
            continue;
        }
        cv::Mat_<cv::Vec3f> points = surf->rawPoints();
        float f = std::min(float(tgt_size.height) / points.rows, float(tgt_size.width) / points.cols);
        cv::resize(points, points, {0,0}, f, f, cv::INTER_CUBIC);

        cv::Mat_<uint8_t> img;

        points = points*0.5;

        readInterpolated3D(img, ds.get(), points, &chunk_cache);

        cv::Mat_<uint8_t> frame(tgt_size, 0);
        int pad_x = (tgt_size.width - img.size().width)/2;
        int pad_y = (tgt_size.height - img.size().height)/2;
        cv::Rect roi = {pad_x, pad_y, img.size().width, img.size().height };
        std::cout << tgt_size << roi << img.size << vid.getBackendName() << std::endl;
        img.copyTo(frame(roi));

        cv::Mat col;
        cv::cvtColor(frame, col, cv::COLOR_GRAY2BGR);

        vid << col;
        cv::imwrite("col.tif", col);
    }

    return EXIT_SUCCESS;
}
