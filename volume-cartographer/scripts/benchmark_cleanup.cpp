#include <vc_reader/pathfinding.hpp>
#include <vc_reader/grid.hpp>
#include <vc_object_store/geometry/VertexIndex.hpp>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
int main(){
 for(int side:{32,128,512}) {
  std::mt19937 rng(42);std::vector<float> weights(side*side);for(auto& w:weights)w=1+rng()%20;
  for(bool gpu:{false,true}) {
   std::vector<double> times;std::vector<uint32_t> path;
   for(int i=0;i<6;++i){auto start=std::chrono::steady_clock::now();
    path=gpu?vcr::pathfinding::cuda::shortestPath(weights.data(),side,side,0,weights.size()-1):vcr::pathfinding::shortestPath(weights.data(),side,side,0,weights.size()-1);
    if(i)times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
   }
   std::sort(times.begin(),times.end());std::cout<<side<<" "<<(gpu?"cuda":"cpu")<<" min/median/max ms "<<times.front()<<" "<<times[2]<<" "<<times.back()<<" path "<<path.size()<<std::endl;
  }
 }
 const int side=256;
 cv::Mat_<cv::Vec3f> points(side,side);
 for(int r=0;r<side;++r)for(int c=0;c<side;++c)points(r,c)={float(c),float(r),100.f+float((r*c)%31)};
 auto start=std::chrono::steady_clock::now();
 vc::geometry::VertexIndex index;index.rebuild(points);
 std::cout<<"index build ms "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<std::endl;
 std::mt19937 rng(42);std::vector<cv::Vec3f> queries(10000);
 for(auto& q:queries)q={float(rng()%256),float(rng()%256),float(100+rng()%31)};
 start=std::chrono::steady_clock::now();std::uint64_t checksum=0;
 for(const auto& q:queries)checksum+=index.nearest(q)->index;
 std::cout<<"10000 vertex queries ms "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<" checksum "<<checksum<<std::endl;
 vcr::grid::Vertices vertices{points.ptr<float>(),side,side,points.step1()};
 vcr::grid::Surface view(vertices);auto snapshot=vcr::grid::Surface::snapshot(vertices);
 std::vector<float> output(512*512*3);
 for(bool owned:{false,true}) {
  std::vector<double> times;
  for(int i=0;i<11;++i){start=std::chrono::steady_clock::now();
   (owned?*snapshot:view).resample({0,0,.49,.49,512,512},output.data(),nullptr);
   if(i)times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
  }
  std::sort(times.begin(),times.end());std::cout<<"512x512 resample "<<(owned?"snapshot":"view")<<" min/median/max ms "<<times.front()<<" "<<times[5]<<" "<<times.back()<<std::endl;
 }
#if defined(__linux__) || defined(__APPLE__)
 rusage usage{};getrusage(RUSAGE_SELF,&usage);
 std::cout<<"process peak RSS (platform units) "<<usage.ru_maxrss<<std::endl;
#endif

}
