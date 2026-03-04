#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include "vc/core/types/VcDataset.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/cache/SimpleCacheFactory.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/StreamOperators.hpp"


using shape = std::vector<size_t>;



shape chunkId(const std::unique_ptr<vc::VcDataset> &ds, shape coord)
{
    shape div = ds->defaultChunkShape();
    shape id = coord;
    for(int i=0;i<id.size();i++)
        id[i] /= div[i];
    return id;
}

shape idCoord(const std::unique_ptr<vc::VcDataset> &ds, shape id)
{
    shape mul = ds->defaultChunkShape();
    shape coord = id;
    for(int i=0;i<coord.size();i++)
        coord[i] *= mul[i];
    return coord;
}

void timed_plane_slice(Surface &plane, vc::cache::TieredChunkCache *cache, int size, std::string msg, vc::Sampling method)
{
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
    cv::Mat_<uint8_t> img;

    auto start = std::chrono::high_resolution_clock::now();
    plane.gen(&coords, &normals, {size, size}, cv::Vec3f(0, 0, 0), 1.0, {0,0,0});
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end-start).count() << "s gen_coords() " << msg << std::endl;
    start = std::chrono::high_resolution_clock::now();
    readInterpolated3D(img, cache, 0, coords, method);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end-start).count() << "s slicing  " << size*size/1024.0/1024.0/std::chrono::duration<double>(end-start).count() << "MiB/s " << msg << std::endl;
}


int main(int argc, char *argv[])
{
  assert(argc == 2 || argc == 3);
  std::filesystem::path vol_path(argv[1]);
  auto ds = std::make_unique<vc::VcDataset>(vol_path / "1");

   vc::Sampling sampling = (argc == 3 && strncmp(argv[2],"nearest",7) == 0)
       ? vc::Sampling::Nearest : vc::Sampling::Trilinear;

  std::cout << "ds shape " << ds->shape() << std::endl;
  std::cout << "chunk shape " << ds->defaultChunkShape() << std::endl;
  if (sampling == vc::Sampling::Nearest) {
    std::cout << "doing nearest neighbor interpolation" << std::endl;
  }

  cv::Mat_<cv::Vec3f> coords;
  cv::Mat_<cv::Vec3f> normals;
  cv::Mat_<uint8_t> img;
  
//   std::vector<cv::Mat> chs;
//   cv::imreadmulti("../grid_slice_coords.tif", chs, cv::IMREAD_UNCHANGED);
//   cv::Mat_<cv::Vec3f> points;
//   cv::merge(chs, points);
//
//   QuadSurface gen_grid(points, {1,1});
  
  PlaneSurface gen_plane({2000,2000,2000},{0.5,0.5,0.5});
  // PlaneSurface gen_plane({2000,2000,2000},{0.0,0.0,1.0});
  // PlaneSurface gen_plane({0,0,0},{0.0,0.0,1.0});
  // PlaneSurface gen_plane({2000,2000,0},{0.0,0.0,1.0});
  
  PlaneSurface plane_x({2000,2000,2000},{1.0,0.0,0.0});
  PlaneSurface plane_y({2000,2000,2000},{0.0,1.0,0.0});
  PlaneSurface plane_z({2000,2000,2000},{0.0,0.0,1.0});
  
  // gen_plane.gen_coords(coords, 1000, 1000);
  // gen_grid.gen(&coords, &normals, {1000, 1000}, cv::Vec3f(0, 0, 0), 1.0, {0,0,0});

    auto chunk_cache = vc::cache::createSimpleTieredCache(ds.get(), size_t(10*10e9), ds->path());

  // auto start = std::chrono::high_resolution_clock::now();
  // readInterpolated3D(img,ds.get(),coords, &chunk_cache);
  // auto end = std::chrono::high_resolution_clock::now();
  // std::cout << std::chrono::duration<double>(end-start).count() << "s cold" << std::endl;
  
  // cv::imwrite("plane.tif", img);
  
//   for(int r=0;r<10;r++) {
//     start = std::chrono::high_resolution_clock::now();
//     readInterpolated3D(img,ds.get(),coords, &chunk_cache);
//     end = std::chrono::high_resolution_clock::now();
//     std::cout << std::chrono::duration<double>(end-start).count() << "s cached" << std::endl;
//   }
//

  const int size = 1024;

  std::cout << "testing different slice directions / caching" << std::endl;
  for(int r=0;r<3;r++) {
      timed_plane_slice(plane_x, chunk_cache.get(), size,"yz cold", sampling);
      timed_plane_slice(plane_x, chunk_cache.get(), size,"yz", sampling);
      timed_plane_slice(plane_y, chunk_cache.get(), size,"xz cold", sampling);
      timed_plane_slice(plane_y, chunk_cache.get(), size,"xz", sampling);
      timed_plane_slice(plane_z, chunk_cache.get(), size,"xy cold", sampling);
      timed_plane_slice(plane_z, chunk_cache.get(), size,"xy", sampling);
      timed_plane_slice(gen_plane, chunk_cache.get(), size,"diag cold", sampling);
      timed_plane_slice(gen_plane, chunk_cache.get(), size,"diag", sampling);
  }


  {
    auto start = std::chrono::high_resolution_clock::now();

    for(float shift = -50;shift<50;shift++) {
        PlaneSurface plane_s({2000,2000,2000+shift},{0.0,0.0,1.0});

        cv::Mat_<cv::Vec3f> coords;
        cv::Mat_<cv::Vec3f> normals;
        cv::Mat_<uint8_t> img;

        plane_s.gen(&coords, &normals, {size, size}, cv::Vec3f(0, 0, 0), 1.0, {0,0,0});

        readInterpolated3D(img, chunk_cache.get(), 0, coords);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end-start).count() << "s slicing / " << 100*size*size/1024.0/1024.0/std::chrono::duration<double>(end-start).count() << "MiB/s " << " shift (cold)"  << std::endl;
  }

  {
      auto start = std::chrono::high_resolution_clock::now();

      for(float shift = -50;shift<50;shift++) {
          PlaneSurface plane_s({2000,2000,2000+shift},{0.0,0.0,1.0});

          cv::Mat_<cv::Vec3f> coords;
          cv::Mat_<cv::Vec3f> normals;
          cv::Mat_<uint8_t> img;

          plane_s.gen(&coords, &normals, {size, size}, cv::Vec3f(0, 0, 0), 1.0, {0,0,0});

          readInterpolated3D(img, chunk_cache.get(), 0, coords);
      }

      auto end = std::chrono::high_resolution_clock::now();
      std::cout << std::chrono::duration<double>(end-start).count() << "s slicing / " << 100*size*size/1024.0/1024.0/std::chrono::duration<double>(end-start).count() << "MiB/s " << " shift (warm)"  << std::endl;
  }


  
  // readInterpolated3D(img,ds.get(),coords);
  // m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
  // cv::imwrite("plane.tif", m);
  
  
  // m = cv::Mat(coords.shape(0), coords.shape(1), CV_32FC3, coords.data());
  // std::vector<cv::Mat> chs;
  // cv::split(m, chs);
  // cv::imwrite("coords_x.tif", chs[2]);
  // cv::imwrite("coords_y.tif", chs[1]);
  // cv::imwrite("coords_z.tif", chs[0]);
    
  return 0;
}
