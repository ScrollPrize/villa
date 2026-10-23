#include <vc_object_store/objects/ObjectStore.hpp>
#include <chrono>
#include <iostream>
#include <set>
using namespace vc::objects;
int main(int argc,char** argv) try {
 if (argc!=2) { std::cerr << "Usage: bench_fiber_open PARIS4.vcobj\n"; return 2; }
 auto now=std::chrono::steady_clock::now();
 auto report=[&](const char* stage){auto next=std::chrono::steady_clock::now();std::cout<<stage<<"="<<std::chrono::duration<double>(next-now).count()<<"s\n"<<std::flush;now=next;};
 OpenOptions options;options.readOnly=true;
 auto db=ObjectStore::open(argv[1],options);auto txn=db.read();report("open");
 auto groups=db.objectCoordinateGroups(txn,"fiber");report("groups");
 if(groups.empty()) { std::cerr<<"No active fibers in store\n"; return 1; }
 ObjectSelection page;page.kind="fiber";page.orderBy="length_desc";page.limit=200;page.coordinates=groups.front().coordinates;
 auto rows=db.selectObjects(txn,page,std::vector<std::string>{"name","fiber"});
 for (const auto& row:rows) db.currentObjectRevision(txn,row.uuid);
 report("page_200_with_pins");
 // Fixed source-voxel ROI in Paris 4, near the first imported fiber.
 RegionSlice slice;slice.origin={17000,15500,55000};slice.localBounds={-1024,-1024,-3,1024,1024,3};
 RegionOptions region;region.maximumObjects=5000;region.coordinates=groups.front().coordinates;
 auto result=db.geometryRegion(txn,slice,region);report("region_5000");
 std::set<std::string> ids;for(const auto& run:result.runs)ids.insert(run.objectUuid);
 std::cout<<"groups="<<groups.size()<<" count="<<groups.front().count<<" fibers="<<ids.size()<<" runs="<<result.runs.size()<<" stop="<<regionStopName(result.stop)<<" sql_steps="<<result.stats.sqlSteps<<" processing_bytes="<<result.stats.processingBytes<<" output_bytes="<<result.stats.outputBytes<<'\n';
 if(ids.size()>5000) return 1;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
