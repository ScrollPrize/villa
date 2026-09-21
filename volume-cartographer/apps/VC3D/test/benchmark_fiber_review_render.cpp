#include "ConstraintGraphWorkspace.hpp"
#include "CState.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Logging.hpp"
#include <QApplication>
#include <QElapsedTimer>
#include <QThread>
#include <QCryptographicHash>
#include <iostream>
#include <fstream>
#include <numeric>
namespace cg=vc::constraints;
int main(int argc,char**argv){
 if(argc!=4){std::cerr<<"Usage: benchmark_fiber_review_render REVIEW.json VOLUME.zarr OUTPUT.json\n";return 2;}
 QApplication app(argc,argv);app.setOrganizationName("VCReviewRenderBench");app.setApplicationName("VCReviewRenderBench");
 try{
  CState main;auto volume=Volume::New(argv[2]);main.setCurrentVolume(volume);
  ConstraintGraphWorkspace workspace(&main);workspace.resize(1600,1000);workspace.show();workspace.synchronizeVolume(volume);
  workspace.openSession(argv[1]);
  QElapsedTimer timer;timer.start();
  while(workspace.busy()||timer.elapsed()<1000){QCoreApplication::processEvents();QThread::msleep(2);if(timer.elapsed()>300000)throw std::runtime_error("open timeout");}
  CChunkedVolumeViewer* viewer=nullptr;for(auto*v:workspace.viewerManager()->baseViewers())if(v->surfName()=="constraint-graph")viewer=dynamic_cast<CChunkedVolumeViewer*>(v);
  if(!viewer)throw std::runtime_error("no viewer");
  auto review=cg::loadReview(argv[1]);auto& fiber=review.fibers.at(review.selected);
  const cg::Graph* alignment=nullptr;for(auto&[id,d]:fiber.decisions)if(d.status==cg::ReviewStatus::Accepted){alignment=&d.alignment;break;}
  if(!alignment||review.settings.volumeScale!=1)throw std::runtime_error("requires accepted alignment at volume scale 1");
  std::vector<std::shared_ptr<Surface>> surfaces;for(int i=0;i<34;++i){auto f=cg::renderSourceLayer(*alignment,false);if(!f.valid)throw std::runtime_error(f.message);surfaces.push_back(f.surface);}
  auto base=workspace.committed().surface;
  auto camera=viewer->cameraState();camera.scale=.75;viewer->applyCameraState(camera);
  uint64_t submitted=0,completed=0;double worker=0;QElapsedTimer clock;clock.start();qint64 lastFrame=clock.elapsed();
  QObject::connect(viewer,&CChunkedVolumeViewer::renderFrameSubmitted,[&](uint64_t s){submitted=s;});
  QObject::connect(viewer,&CChunkedVolumeViewer::renderFrameCompleted,[&](uint64_t s,double ms){completed=s;worker+=ms;lastFrame=clock.elapsed();});
  auto settle=[&]{auto start=clock.elapsed();while(true){QCoreApplication::processEvents();QThread::msleep(2);if(completed>=submitted&&clock.elapsed()-lastFrame>250&&clock.elapsed()-start>250)return;if(clock.elapsed()-start>180000)throw std::runtime_error("render timeout");}};
  settle();SetProfileLoggingEnabled(true);
  nlohmann::json report;
  for(int count:{1,8,32}){
   std::vector<double> samples,ui,workers;
   for(int i=-3;i<12;++i){std::vector<std::shared_ptr<Surface>> layers(surfaces.begin(),surfaces.begin()+count);layers.push_back(surfaces[32+((i+4)%2)]);std::vector<float> opacity(count+1,.2f);opacity.back()=.5f;
    const auto start=clock.elapsed();worker=0;QElapsedTimer call;call.start();viewer->setSurfaceLayers(base,std::move(layers),std::move(opacity));double synchronous=call.nsecsElapsed()/1e6;settle();
    if(i>=0){samples.push_back(double(lastFrame-start));ui.push_back(synchronous);workers.push_back(worker);}
   }
   auto image=viewer->graphicsView()->viewport()->grab().toImage().convertToFormat(QImage::Format_RGB32);
   auto hash=QCryptographicHash::hash(QByteArrayView(reinterpret_cast<const char*>(image.constBits()),image.sizeInBytes()),QCryptographicHash::Sha256).toHex();
   auto sorted=samples;std::sort(sorted.begin(),sorted.end());
   report[std::to_string(count)]={{"mean_ms",std::accumulate(samples.begin(),samples.end(),0.)/samples.size()},{"p50_ms",sorted[sorted.size()/2]},{"p95_ms",sorted.back()},{"samples_ms",samples},{"worker_ms",workers},{"ui_ms",ui},{"sha256",hash.toStdString()},{"size",{image.width(),image.height()}}};
   std::cout<<count<<" layers "<<report[std::to_string(count)].dump()<<std::endl;
  }
  // Compare cached composition against a complete, explicitly invalidated
  // render after changes that must invalidate the appropriate cache entries.
  auto snapshot=[&]{return viewer->graphicsView()->viewport()->grab().toImage().convertToFormat(QImage::Format_RGB32);};
  auto verify=[&](const char* label){settle();const auto cached=snapshot();viewer->invalidateVis();viewer->renderVisible(true,"benchmark forced reference");settle();if(snapshot()!=cached)throw std::runtime_error(std::string("Cached image differs after ")+label);};
  std::vector<std::shared_ptr<Surface>> checkLayers(surfaces.begin(),surfaces.begin()+8);
  for(auto&[id,d]:fiber.decisions)if(d.status==cg::ReviewStatus::Accepted && &d.alignment!=alignment){auto other=cg::renderSourceLayer(d.alignment,false);if(!other.valid)throw std::runtime_error(other.message);checkLayers.push_back(other.surface);break;}
  std::vector<float> opacity(checkLayers.size(),.2f);opacity.back()=.7f;
  viewer->setSurfaceLayers(base,checkLayers,opacity);verify("layer replacement");
  std::swap(checkLayers.front(),checkLayers.back());viewer->setSurfaceLayers(base,checkLayers,opacity);verify("layer reorder");
  opacity[0]=.85f;viewer->setSurfaceLayers(base,checkLayers,opacity);verify("accepted opacity");
  opacity.back()=0;viewer->setSurfaceLayers(base,checkLayers,opacity);verify("zero proposal opacity");
  opacity.back()=.65f;viewer->setSurfaceLayers(base,checkLayers,opacity);verify("proposal opacity");
  std::vector<QColor> tints(checkLayers.size(),QColor(100,180,255,10));tints.back()=QColor{};
  viewer->setSurfaceLayers(base,checkLayers,opacity,tints);verify("committed tint");
  tints[0]=QColor(255,100,150,10);viewer->setSurfaceLayers(base,checkLayers,opacity,tints);verify("cached tint change");
  viewer->setSurfaceLayers(base,checkLayers,opacity);verify("tint removal");
  checkLayers.erase(checkLayers.begin()+2);opacity.erase(opacity.begin()+2);viewer->setSurfaceLayers(base,checkLayers,opacity);verify("layer removal");
  viewer->adjustZoomByFactor(1.1f);verify("zoom");
  camera=viewer->cameraState();viewer->centerOnSurfacePoint({camera.surfacePtrX+11,camera.surfacePtrY-7});verify("pan");
  viewer->setVolumeWindow(30,170);verify("windowing");
  report["cache_invalidation_checks"]="passed";
  std::ofstream(argv[3])<<report.dump(2);std::cout<<"Cached pixels match fresh renders for replacement/reordering/removal, opacity, tint changes/removal, zoom, pan, windowing\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
}
