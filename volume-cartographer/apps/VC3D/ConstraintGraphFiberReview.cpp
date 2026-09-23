#include "ConstraintGraphWorkspace.hpp"
#include "ConstraintGraphSelectionBar.hpp"
#include "CState.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include <QCheckBox>
#include <QComboBox>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QToolBar>
#include <QVBoxLayout>
#include <QWidgetAction>
#include <algorithm>
#include <future>
#include <mutex>
#include <thread>

namespace cg=vc::constraints;
namespace {
// volumeScale only scales the display copy of a layer; every other setting
// can change the rasterized geometry and therefore invalidates a cached layer.
bool sameLayerSettings(const cg::Settings& a,const cg::Settings& b) {
    return a.spacing==b.spacing&&a.radius==b.radius&&a.fiberScale==b.fiberScale&&a.patchScale==b.patchScale&&
        a.normalScale==b.normalScale&&a.stiffness==b.stiffness;
}
size_t layerBytes(const cg::FitResult& layer) {
    size_t bytes=sizeof(cg::FitResult);
    if(layer.surface)if(const auto* points=layer.surface->rawPointsPtr())bytes+=points->total()*sizeof(cv::Vec3f);
    for(const auto& [id,p]:layer.graph.nodes)bytes+=p.uv.size()*sizeof(cv::Vec2d);
    return bytes;
}
}
bool ConstraintGraphWorkspace::reviewing() const {return _mode&&_mode->currentIndex()==1;}
cg::FitResult ConstraintGraphWorkspace::reviewPatchLayer(const cg::FitResult& fit,const std::atomic_bool* cancel) {
    // The review fit policy validates and returns the rasterized patch-only
    // layer itself, marked by its local chart count. Rendering the same graph
    // again produced an identical surface at the cost of a second rasterization.
    if(!fit.valid||!fit.surface||!fit.localChartVertices)return cg::renderSourceLayer(fit.graph,false,cancel);
    cg::FitResult layer;layer.graph.settings=fit.graph.settings;layer.graph.windingLayout=fit.graph.windingLayout;
    for(const auto& [id,p]:fit.graph.nodes)if(!p.source->fiber)layer.graph.nodes.emplace(id,p);
    if(layer.graph.nodes.empty())return cg::renderSourceLayer(fit.graph,false,cancel);
    layer.graph.seed=layer.graph.nodes.begin()->first;
    layer.surface=fit.surface;layer.origin=fit.origin;layer.stretchP95=fit.stretchP95;layer.message=fit.message;layer.valid=true;
    return layer;
}
std::vector<std::pair<std::string,cg::FitResult>> ConstraintGraphWorkspace::renderReviewLayers(const std::string& fiber,
    const std::vector<std::pair<std::string,cg::ReviewDecision>>& accepted,const cg::Settings& s,const ReviewLayerCache& cache,
    const std::atomic_bool* cancel,const cg::Progress& progress) {
    std::vector<std::pair<std::string,cg::FitResult>> layers(accepted.size());
    std::vector<size_t> missing;
    for(size_t i=0;i<accepted.size();++i) {
        const auto& [patch,d]=accepted[i];layers[i].first=patch;
        const auto& placement=d.alignment.nodes.at(patch);
        const auto found=cache.find({fiber,patch});
        if(found!=cache.end()&&found->second.placement.source==placement.source&&found->second.placement.uv.sharesStorage(placement.uv)&&
           sameLayerSettings(found->second.layer.graph.settings,s)) {
            layers[i].second=found->second.layer;layers[i].second.graph.settings=s;continue;
        }
        missing.push_back(i);
    }
    if(missing.empty())return layers;
    if(progress)progress("Rendering "+std::to_string(missing.size())+" accepted patch layers");
    // Each layer rasterizes one immutable placement, so layers are independent
    // and deterministic. VC3D pins OpenCV's own parallelism to one thread;
    // fan out here and keep acceptance order.
    const size_t width=std::min<size_t>({missing.size(),std::max(1u,std::thread::hardware_concurrency()),8});
    for(size_t first=0;first<missing.size();first+=width) {
        if(cancel&&cancel->load())throw std::runtime_error("Operation cancelled");
        const size_t end=std::min(first+width,missing.size());
        std::vector<std::future<cg::FitResult>> futures;
        for(size_t j=first;j<end;++j) {
            const auto& alignment=accepted[missing[j]].second.alignment;
            futures.push_back(std::async(std::launch::async,[&alignment,cancel]{return cg::renderSourceLayer(alignment,false,cancel);}));
        }
        for(size_t j=first;j<end;++j) {
            auto layer=futures[j-first].get();
            if(!layer.valid)throw std::runtime_error(layer.message);
            layers[missing[j]].second=std::move(layer);
        }
    }
    return layers;
}
void ConstraintGraphWorkspace::rememberReviewLayers(const std::string& fiber,const std::vector<std::pair<std::string,cg::FitResult>>& layers) {
    if(!_review||!_review->fibers.contains(fiber))return;
    const auto& f=_review->fibers.at(fiber);
    for(const auto& [patch,layer]:layers) {
        const auto decision=f.decisions.find(patch);
        if(decision==f.decisions.end()||!cg::fitted(decision->second)||!layer.surface)continue;
        const auto& placement=decision->second.alignment.nodes.at(patch);
        auto& entry=_reviewLayerCache[{fiber,patch}];
        if(entry.layer.surface!=layer.surface) {
            _reviewLayerCacheBytes-=entry.bytes;entry.placement=placement;entry.layer=layer;
            entry.bytes=layerBytes(layer);_reviewLayerCacheBytes+=entry.bytes;
        }
        entry.used=++_reviewLayerClock;
    }
    // Evict other fibers' least recently shown layers beyond the byte budget.
    constexpr size_t limit=size_t(768)<<20;
    while(_reviewLayerCacheBytes>limit) {
        auto oldest=_reviewLayerCache.end();
        for(auto it=_reviewLayerCache.begin();it!=_reviewLayerCache.end();++it)
            if(it->first.first!=fiber&&(oldest==_reviewLayerCache.end()||it->second.used<oldest->second.used))oldest=it;
        if(oldest==_reviewLayerCache.end())break;
        _reviewLayerCacheBytes-=oldest->second.bytes;_reviewLayerCache.erase(oldest);
    }
}
const cg::ReviewFiber* ConstraintGraphWorkspace::displayedReviewFiber() const {
    if(!_review)return nullptr;
    const auto& id=_committed.valid?_committed.graph.seed:_review->selected;
    const auto found=_review->fibers.find(id);return found==_review->fibers.end()?nullptr:&found->second;
}

void ConstraintGraphWorkspace::setupReview(QToolBar* toolbar) {
    _mode=new QComboBox(toolbar);_mode->setObjectName("constraintGraphMode");
    _mode->addItems({tr("Full graph"),tr("Fiber review")});toolbar->insertWidget(toolbar->actions().front(),_mode);
    auto* inputs=findChild<QDockWidget*>("constraintGraphInputs");toolbar->addAction(inputs->toggleViewAction());
    std::vector<QAction*> fullActions;
    for(auto* action:toolbar->actions()) {
        auto* widget=qobject_cast<QWidgetAction*>(action);
        if(action->text()==tr("Export tifxyz…")||action->text()==tr("Refit graph")||(widget&&widget->defaultWidget()==_showCommitted))fullActions.push_back(action);
    }
    auto* dock=new QDockWidget(tr("Fiber review"),this);dock->setObjectName("constraintGraphFiberReview");
    addDockWidget(Qt::LeftDockWidgetArea,dock);_reviewPanel=dock;
    auto* panel=new QWidget(dock);auto* layout=new QVBoxLayout(panel);dock->setWidget(panel);
    layout->addWidget(new QLabel(tr("Fibers · accepted patches"),panel));
    _reviewFibers=new QListWidget(panel);_reviewFibers->setObjectName("constraintGraphReviewFibers");layout->addWidget(_reviewFibers,2);
    _reviewFibers->setTextElideMode(Qt::ElideMiddle);_reviewFibers->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    _reviewFibers->installEventFilter(this);
    _reverseReview=new QCheckBox(tr("Reverse order"),panel);layout->addWidget(_reverseReview);
    auto* lineStyle=new QFormLayout;layout->addLayout(lineStyle);
    const cg::FiberReview defaults;
    _committedTint=new QSpinBox(panel);_committedTint->setObjectName("constraintGraphCommittedTint");
    _committedTint->setRange(0,100);_committedTint->setSuffix(tr("%"));_committedTint->setValue(defaults.committedTint);
    _committedTint->setToolTip(tr("Boundary-color tint on committed context patches; 0% disables it. The active proposal stays untinted."));
    lineStyle->addRow(tr("Committed tint"),_committedTint);
    connect(_committedTint,qOverload<int>(&QSpinBox::valueChanged),this,[this](int value){
        if(_review){_review->committedTint=value;_dirty=true;}if(reviewing())displayReview();
    });
    _fiberLineOpacity=new QSpinBox(panel);_fiberLineOpacity->setObjectName("constraintGraphFiberLineOpacity");
    _fiberLineOpacity->setRange(0,100);_fiberLineOpacity->setSuffix(tr("%"));_fiberLineOpacity->setValue(defaults.fiberLineOpacity);
    _fiberLineWidth=new QDoubleSpinBox(panel);_fiberLineWidth->setObjectName("constraintGraphFiberLineWidth");
    _fiberLineWidth->setRange(.5,10);_fiberLineWidth->setSingleStep(.5);_fiberLineWidth->setDecimals(1);_fiberLineWidth->setSuffix(tr(" px"));_fiberLineWidth->setValue(defaults.fiberLineWidth);
    lineStyle->addRow(tr("Fiber line opacity"),_fiberLineOpacity);lineStyle->addRow(tr("Fiber line thickness"),_fiberLineWidth);
    _fiberLineOpacity->setToolTip(tr("Opacity of the fiber center line and contact highlights; 0% hides both."));
    _fiberLineWidth->setToolTip(tr("Thickness in screen pixels, independent of zoom."));
    connect(_fiberLineOpacity,qOverload<int>(&QSpinBox::valueChanged),this,[this](int value){
        if(_review){_review->fiberLineOpacity=value;_dirty=true;}if(reviewing())overlays();
    });
    connect(_fiberLineWidth,qOverload<double>(&QDoubleSpinBox::valueChanged),this,[this](double value){
        if(_review){_review->fiberLineWidth=value;_dirty=true;}if(reviewing())overlays();
    });
    layout->addWidget(new QLabel(tr("Decisions · double-click to reopen"),panel));
    _reviewDecisions=new QListWidget(panel);_reviewDecisions->setObjectName("constraintGraphReviewDecisions");layout->addWidget(_reviewDecisions);
    auto* reopen=new QPushButton(tr("Reopen selected pair"),panel);layout->addWidget(reopen);
    auto* autoForm=new QFormLayout;layout->addLayout(autoForm);
    _autoLinkContact=new QDoubleSpinBox(panel);_autoLinkContact->setObjectName("constraintGraphAutoLinkContact");
    _autoLinkContact->setRange(0,10000);_autoLinkContact->setDecimals(0);_autoLinkContact->setSingleStep(5);_autoLinkContact->setSuffix(tr(" vx"));
    _autoLinkContact->setValue(cg::AutoLinkSettings{}.anchorContact);
    _autoLinkContact->setToolTip(tr("Anchor patches need at least this much merged fiber contact and must agree with the fiber strip. Other patches link only through same-sheet overlap with linked patches."));
    autoForm->addRow(tr("Anchor contact"),_autoLinkContact);
    _autoLinkFiber=new QPushButton(tr("Auto link fiber"),panel);_autoLinkFiber->setObjectName("constraintGraphAutoLinkFiber");layout->addWidget(_autoLinkFiber);
    _autoLinkFiber->setToolTip(tr("Link this fiber's patches from geometry alone: fitted links are accepted, the rest rejected, all marked automatic. Human decisions are kept."));
    _autoLinkAll=new QPushButton(tr("Auto link all fibers"),panel);_autoLinkAll->setObjectName("constraintGraphAutoLinkAll");layout->addWidget(_autoLinkAll);
    _autoLinkAll->setToolTip(tr("Discover and auto link every fiber in the catalog. Cancel keeps the fibers finished so far. Not undoable."));
    _nextAutoRejected=new QPushButton(tr("Next auto-rejected"),panel);_nextAutoRejected->setObjectName("constraintGraphNextAutoRejected");layout->addWidget(_nextAutoRejected);
    _nextAutoRejected->setToolTip(tr("Reopen the next automatically rejected pair of this fiber for review."));
    _committedOpacity=new QSlider(Qt::Horizontal,toolbar);_committedOpacity->setRange(0,100);_committedOpacity->setValue(20);
    _committedOpacity->setObjectName("constraintGraphCommittedOpacity");_committedOpacity->setMaximumWidth(120);
    auto* opacityLabel=new QLabel(tr("Committed opacity 20%"),toolbar);
    auto* labelAction=toolbar->addWidget(opacityLabel);auto* opacityAction=toolbar->addWidget(_committedOpacity);
    labelAction->setVisible(false);opacityAction->setVisible(false);
    _rejectReview=new QPushButton(tr("Reject (Delete)"),_bar);_rejectReview->setObjectName("constraintGraphReject");_bar->layout()->addWidget(_rejectReview);
    _skipReview=new QPushButton(tr("Skip"),_bar);_skipReview->setObjectName("constraintGraphSkip");_bar->layout()->addWidget(_skipReview);
    _skipReview->setToolTip(tr("Skip this patch for the current fiber without rejecting it. Reopen it from the decisions list to review it again."));
    _alternateReview=new QPushButton(tr("Next placement"),_bar);_alternateReview->setObjectName("constraintGraphAlternatePlacement");_bar->layout()->addWidget(_alternateReview);
    connect(_mode,qOverload<int>(&QComboBox::activated),this,[this,labelAction,opacityAction](int index){
        if(_saving||(_dirty&&QMessageBox::question(this,tr("Change mode"),tr("Discard unsaved workspace changes?"),QMessageBox::Discard|QMessageBox::Cancel,QMessageBox::Cancel)!=QMessageBox::Discard)){
            _mode->setCurrentIndex(index==1?0:1);return;
        }
        stopDiscovery();cancelPreview();_committed={};_prepared.reset();_catalog.reset();_review.reset();
        _reviewLayers.clear();_reviewContextSurfaces.clear();_reviewSurfaceCache.clear();_reviewFiberSurface.reset();_reviewPatchSurface.reset();
        _reviewLayerCache.clear();_reviewLayerCacheBytes=0;
        _undo.clear();_redo.clear();_reviewUndo.clear();_reviewRedo.clear();_path.clear();_dirty=false;_selected.clear();
        {QSignalBlocker r(_radius),stride(_discoveryStride);_radius->setValue(index?1.5:4);_discoveryStride->setCurrentIndex(_discoveryStride->findData(index?1:4));}
        labelAction->setVisible(index==1);opacityAction->setVisible(index==1);refreshReviewList();display();updateControls();
    });
    // Visibility also follows sessions opened directly from disk.
    connect(_mode,qOverload<int>(&QComboBox::currentIndexChanged),this,[labelAction,opacityAction](int index){labelAction->setVisible(index==1);opacityAction->setVisible(index==1);});
    connect(_mode,qOverload<int>(&QComboBox::currentIndexChanged),this,[inputs,fullActions](int index){
        inputs->show();for(auto* action:fullActions)action->setVisible(index==0);
    });
    connect(_reviewFibers,&QListWidget::currentItemChanged,this,[this](QListWidgetItem* item){if(item)selectReviewFiber(item->data(Qt::UserRole).toString().toStdString());});
    connect(_reverseReview,&QCheckBox::toggled,this,[this](bool reverse){
        if(!_review||_review->selected.empty()||_saving)return;
        captureReviewEdit();auto& f=_review->fibers.at(_review->selected);f.reverse=reverse;f.pending.reset();cg::orderReviewProposals(f);_dirty=true;searchReview();
    });
    connect(_committedOpacity,&QSlider::valueChanged,this,[this,opacityLabel](int value){
        opacityLabel->setText(tr("Committed opacity %1%").arg(value));
        if(_review){_review->committedOpacity=value;_dirty=true;}if(reviewing())displayReview();
    });
    connect(_showAllBoundaries,&QCheckBox::toggled,this,[this](bool checked){if(reviewing()&&_review){_review->boundaries=checked;_dirty=true;}});
    connect(_rejectReview,&QPushButton::clicked,this,[this]{decideReview(cg::ReviewStatus::Rejected);});
    connect(_skipReview,&QPushButton::clicked,this,[this]{decideReview(cg::ReviewStatus::Skipped);});
    connect(_alternateReview,&QPushButton::clicked,this,[this]{
        if(!_review||_review->selected.empty())return;auto& f=_review->fibers.at(_review->selected);if(f.current<0)return;
        const auto id=f.proposals[f.current].id;
        for(size_t step=1;step<f.proposals.size();++step){int i=(f.current+int(step))%int(f.proposals.size());if(f.proposals[i].id==id){f.pending.reset();previewReview(i);return;}}
    });
    connect(reopen,&QPushButton::clicked,this,[this]{if(auto* item=_reviewDecisions->currentItem())reopenReview(item->data(Qt::UserRole).toString().toStdString());});
    connect(_autoLinkFiber,&QPushButton::clicked,this,[this]{autoLinkReview(false);});
    connect(_autoLinkAll,&QPushButton::clicked,this,[this]{autoLinkReview(true);});
    connect(_nextAutoRejected,&QPushButton::clicked,this,[this]{reopenNextAutoRejected();});
    connect(_reviewDecisions,&QListWidget::itemDoubleClicked,this,[this](QListWidgetItem* item){reopenReview(item->data(Qt::UserRole).toString().toStdString());});
}

void ConstraintGraphWorkspace::startReview(const cg::Settings& s) {
    if(_saving)return;
    if(_dirty&&QMessageBox::question(this,tr("New fiber review"),tr("Discard unsaved workspace changes?"),QMessageBox::Discard|QMessageBox::Cancel,QMessageBox::Cancel)!=QMessageBox::Discard)return;
    stopDiscovery();cancelPreview();
    run(tr("Loading fiber review inputs"),[s](auto* cancel,auto progress){
        Job job;vc::lasagna::LasagnaDatasetOpenOptions options;options.workingToBaseScale=s.normalScale;
        job.dataset=std::make_shared<vc::lasagna::LasagnaDataset>(vc::lasagna::LasagnaDataset::openLocation(s.normalManifest,options));
        job.normals=std::make_shared<vc::lasagna::LasagnaNormalSampler>(*job.dataset);
        job.catalog=std::make_shared<cg::Catalog>(cg::loadCatalog(s,cancel,progress));
        if(job.catalog->fibers.empty())throw std::runtime_error("No eligible fibers in this folder");
        return job;
    },[this,s](Job job){
        cg::FiberReview review;review.settings=s;review.stride=1;
        review.committedTint=_committedTint->value();
        review.fiberLineOpacity=_fiberLineOpacity->value();review.fiberLineWidth=_fiberLineWidth->value();
        for(const auto& [id,f]:job.catalog->fibers){cg::ReviewFiber state;state.path=f.path;state.direction=f.direction;review.fibers.emplace(id,std::move(state));}
        review.selected=review.fibers.begin()->first;installReview(std::move(review),{},std::move(job));_dirty=true;
    });
}

void ConstraintGraphWorkspace::installReview(cg::FiberReview review,const QString& path,Job job) {
    _mode->setCurrentIndex(1);_review=std::move(review);_catalog=std::move(job.catalog);_dataset=std::move(job.dataset);_normals=std::move(job.normals);
    _reviewAvailablePatches.clear();size_t removed=0;
    if(_catalog) {
        for(const auto& patch:_catalog->patches)_reviewAvailablePatches.insert(patch.id);
        for(auto& [id,f]:_review->fibers)removed+=cg::pruneUnavailableReviewProposals(f,_reviewAvailablePatches);
    }
    _reviewUndo.clear();_reviewRedo.clear();_committed={};_draft.reset();_preview={};_proposalView={};_prepared.reset();
    _reviewLayerCache.clear();_reviewLayerCacheBytes=0;_reviewLayers.clear();_reviewContextSurfaces.clear();_reviewSurfaceCache.clear();
    _reviewFiberSurface.reset();_reviewPatchSurface.reset();
    _path=path;_dirty=false;_selected.clear();setSettings(_review->settings);
    {QSignalBlocker a(_discoveryStride),d(_showAllBoundaries),lineOpacity(_fiberLineOpacity),lineWidth(_fiberLineWidth),tint(_committedTint);
        _committedTint->setValue(_review->committedTint);
        _fiberLineOpacity->setValue(_review->fiberLineOpacity);_fiberLineWidth->setValue(_review->fiberLineWidth);
        _review->stride=1;_discoveryStride->setCurrentIndex(_discoveryStride->findData(1));_patchOpacity->setValue(_review->proposalOpacity);
        _committedOpacity->setValue(_review->committedOpacity);_showAllBoundaries->setChecked(_review->boundaries);}
    _dirty=removed!=0;
    refreshReviewList();selectReviewFiber(_review->selected);if(!job.notice.empty())report(QString::fromStdString(job.notice));
    if(removed)report(tr("Removed %1 unavailable patch proposals from the saved lists. Decisions and embedded alignments were retained.").arg(removed));
    findChild<QDockWidget*>("constraintGraphInputs")->hide();
}

void ConstraintGraphWorkspace::captureReviewEdit() {
    if(!_review||_review->selected.empty()||!_draft)return;
    auto& f=_review->fibers.at(_review->selected);
    if(f.current>=0&&_draft->nodes.contains(f.proposals[f.current].id))f.pending=*_draft;
}

void ConstraintGraphWorkspace::selectReviewFiber(const std::string& id) {
    if(_saving||!_review||!_review->fibers.contains(id))return;
    const bool newFiber=!_committed.valid||_committed.graph.seed!=id;
    captureReviewEdit();stopDiscovery();cancelPreview();_prepared.reset();
    // The previous fiber and its accepted layers stay on screen until the new
    // layout is ready. Publishing a null surface here cleared the viewer's
    // scene and dropped every sampled tile, so each switch flashed blank and
    // re-sampled the CT from scratch. searchReview/previewReview only act once
    // the committed fiber matches the selection again.
    _dirty|=_review->selected!=id;_review->selected=id;_selected=id;auto snapshot=_review->fibers.at(id);const auto s=_review->settings;
    auto catalog=_catalog;auto normals=_normals;
    ReviewLayerCache cache;
    for(auto it=_reviewLayerCache.lower_bound({id,std::string{}});it!=_reviewLayerCache.end()&&it->first.first==id;++it)cache.insert(*it);
    {QSignalBlocker b(_reverseReview);_reverseReview->setChecked(snapshot.reverse);}
    refreshReviewList();displayReview();
    std::set<std::string> excluded;for(const auto& [patch,p]:_review->quarantined)excluded.insert(patch);
    run(tr("Laying out fiber"),[id,s,snapshot,catalog,normals,excluded,cache=std::move(cache)](auto* cancel,auto progress) mutable {
        Job job;
        if(normals&&!snapshot.base.nodes.empty()) {
            progress("Updating fiber strip");
            try {if(cg::regenerateReviewFiber(snapshot,*normals,cancel)){job.regeneratedStrips=true;job.regeneratedReviewFiber=snapshot;}}
            catch(const std::exception& e){if(cancel&&cancel->load())throw;job.notice="Could not update fiber strip; saved geometry retained: "+std::string(e.what());}
        }
        auto base=snapshot.base;base.settings=s;
        if(base.nodes.empty()) {
            if(!catalog||!normals||!catalog->fibers.contains(id))throw std::runtime_error("Fiber sources or normals unavailable; saved decisions remain accessible");
            auto node=cg::makeFiberNode(catalog->fibers.at(id),s,*normals,cancel);base.seed=id;base.nodes.emplace(id,cg::Placement{node,node->initialUv});
        }
        job.fit=cg::renderPlacement(base,cancel,progress);
        if(!job.fit.valid)throw std::runtime_error(job.fit.message);
        job.prepared=cg::prepareGraph(job.fit,cancel);
        snapshot.base=job.fit.graph;
        // A compact workspace stores decisions only: discover proposals here and
        // fit every accepted patch that has no placement yet, in decision order,
        // each using the placements fitted before it as overlap targets.
        std::vector<std::pair<std::string,cg::ReviewDecision*>> unfitted;
        for(auto& [patch,d]:snapshot.decisions)if(d.status==cg::ReviewStatus::Accepted&&!cg::fitted(d))unfitted.emplace_back(patch,&d);
        if(!unfitted.empty()||!snapshot.discovered) {
            if(!catalog||!normals)throw std::runtime_error("Sources and normals are required to lay out this review");
            if(!snapshot.discovered) {
                progress("Finding patches along the fiber");
                snapshot.proposals=cg::discover(*catalog,snapshot.base,id,cancel,progress,{},1,excluded,cg::ContactPolicy::FiberReview);
                snapshot.discovered=true;snapshot.contactPolicyVersion=1;cg::orderReviewProposals(snapshot);
            }
            std::sort(unfitted.begin(),unfitted.end(),[](const auto& a,const auto& b){return std::tie(a.second->order,a.first)<std::tie(b.second->order,b.first);});
            size_t done=0;
            for(auto& [patch,d]:unfitted) {
                progress("Fitting accepted patch "+std::to_string(++done)+"/"+std::to_string(unfitted.size()));
                const auto candidate=std::find_if(snapshot.proposals.begin(),snapshot.proposals.end(),[&](const auto& c){return c.id==patch;});
                std::string failure;
                if(candidate==snapshot.proposals.end())failure="No proposal for this patch along the fiber in the current catalog";
                else try {
                    auto edit=cg::proposeEdit(*catalog,job.prepared,*candidate,false,*normals,cancel,cg::ContactPolicy::FiberReview);
                    for(const auto& k:d->keypoints)cg::addKeypoint(edit,k);
                    cg::addReviewOverlapTargets(edit,snapshot,cancel);
                    auto fit=cg::fitEdit(edit,cancel,{},cg::FitPolicy::FiberReview);
                    if(fit.valid&&fit.graph.nodes.at(id).uv==snapshot.base.nodes.at(id).uv)d->alignment=std::move(fit.graph);
                    else failure=fit.valid?"Fit moved the fixed fiber":fit.message;
                } catch(const std::exception& e){if(cancel&&cancel->load())throw;failure=e.what();}
                if(!failure.empty()){d->status=cg::ReviewStatus::Skipped;d->automatic=true;d->note="Accepted link could not be registered: "+failure.substr(0,120);job.decisionsChanged=true;}
            }
            job.regeneratedReviewFiber=snapshot;
        }
        std::vector<std::pair<std::string,cg::ReviewDecision>> accepted;
        for(const auto& item:snapshot.decisions)if(cg::fitted(item.second))accepted.push_back(item);
        std::sort(accepted.begin(),accepted.end(),[](const auto& a,const auto& b){return std::tie(a.second.order,a.first)<std::tie(b.second.order,b.first);});
        for(auto& [patch,d]:accepted)d.alignment.settings=s;
        job.reviewLayers=renderReviewLayers(id,accepted,s,cache,cancel,progress);
        return job;
    },[this,id,newFiber](Job job){
        if(job.regeneratedReviewFiber){_review->fibers.at(id)=std::move(*job.regeneratedReviewFiber);_dirty|=job.decisionsChanged;}
        if(job.decisionsChanged)report(tr("Some accepted links could not be registered and were marked Auto-skipped; see their notes."));
        _committed=std::move(job.fit);_prepared=std::move(job.prepared);_review->fibers.at(id).base=_committed.graph;
        _reviewLayers=std::move(job.reviewLayers);rememberReviewLayers(id,_reviewLayers);rebuildReviewLayers();
        _reviewFiberSurface=displaySurface(_committed);_reviewPatchSurface.reset();
        displayReview();if(newFiber){initialView();_centerFirstReviewProposal=true;}refreshReviewList();searchReview();
        if(!job.notice.empty())report(QString::fromStdString(job.notice));
    },true,[this]{
        // Nothing valid represents the selected fiber; drop the stale view.
        _committed={};_prepared.reset();_reviewLayers.clear();_reviewContextSurfaces.clear();_reviewSurfaceCache.clear();
        _reviewFiberSurface.reset();_reviewPatchSurface.reset();displayReview();
    });
}
void ConstraintGraphWorkspace::refreshReviewList() {
    QSignalBlocker block(_reviewFibers);_reviewDecisions->clear();
    if(!_review){_reviewFibers->clear();return;}
    // Retain rows for the same catalog: resetting the model during selection
    // or a decision update discards the user's scroll position.
    bool sameFibers=_reviewFibers->count()==int(_review->fibers.size());
    int row=0;
    if(sameFibers)for(const auto& [id,f]:_review->fibers) {
        if(_reviewFibers->item(row++)->data(Qt::UserRole).toString()!=QString::fromStdString(id)){sameFibers=false;break;}
    }
    if(!sameFibers)_reviewFibers->clear();
    row=0;
    for(const auto& [id,f]:_review->fibers) {
        auto* item=sameFibers?_reviewFibers->item(row++):new QListWidgetItem(_reviewFibers);
        item->setText(tr("%1 patches · %2").arg(cg::acceptedPatchCount(f)).arg(QString::fromStdString(f.path.filename().string())));
        item->setData(Qt::UserRole,QString::fromStdString(id));item->setToolTip(QString::fromStdString(f.path.string()));
        if(id==_review->selected){if(_reviewFibers->currentItem()!=item)_reviewFibers->setCurrentItem(item);for(const auto& [patch,d]:f.decisions){
            const auto label=d.status==cg::ReviewStatus::Accepted?(d.automatic?tr("Auto-accepted · "):tr("Accepted · ")):d.status==cg::ReviewStatus::Skipped?(d.automatic?tr("Auto-skipped · "):tr("Skipped · ")):(d.automatic?tr("Auto-rejected · "):tr("Rejected · "));
            auto* decision=new QListWidgetItem(label+QString::fromStdString(patch),_reviewDecisions);decision->setData(Qt::UserRole,QString::fromStdString(patch));
            if(!d.note.empty())decision->setToolTip(QString::fromStdString(d.note));}}
    }
}

void ConstraintGraphWorkspace::searchReview() {
    if(_saving||!_review||_review->selected.empty()||!_committed.valid)return;
    // Another fiber's layout is still on screen; its job was superseded or is
    // running. Proposals only make sense against the committed fiber.
    if(_committed.graph.seed!=_review->selected){if(!_busy)selectReviewFiber(_review->selected);return;}
    auto& f=_review->fibers.at(_review->selected);
    if(_catalog)_dirty|=cg::pruneUnavailableReviewProposals(f,_reviewAvailablePatches)!=0;
    if(f.discovered) {
        _candidates=f.proposals;
        int index=f.current;if(index<0||index>=int(f.proposals.size())||f.decisions.contains(f.proposals[index].id))index=cg::nextReviewProposal(f);
        previewReview(index);return;
    }
    if(!_catalog){report(tr("Source catalog unavailable. Saved decisions and accepted patches are retained."));return;}
    cancelPreview();auto catalog=_catalog;auto graph=_committed.graph;const int stride=_review->stride;const auto id=_review->selected;
    std::set<std::string> excluded;for(const auto& [patch,p]:_review->quarantined)excluded.insert(patch);
    const auto current=f.current>=0&&f.current<int(f.proposals.size())?f.proposals[f.current].id:std::string{};
    const auto manual=!current.empty()&&f.pending&&std::any_of(f.pending->links.begin(),f.pending->links.end(),[](const auto& l){return l.keypoint;})
        ?std::optional<cg::Candidate>{f.proposals[f.current]}:std::nullopt;
    run(tr("Finding patches along fiber"),[catalog,graph,id,stride,excluded](auto* cancel,auto progress){Job job;job.candidates=cg::discover(*catalog,graph,id,cancel,progress,{},stride,excluded,cg::ContactPolicy::FiberReview);return job;},
        [this,id,current,manual](Job job){
            auto& fiber=_review->fibers.at(id);auto pending=std::move(fiber.pending);fiber.pending.reset();
            fiber.proposals=std::move(job.candidates);fiber.discovered=true;fiber.contactPolicyVersion=1;
            // Explicit manual edits survive migration even if the old automatic
            // proposal no longer meets the new two-contact eligibility rule.
            if(manual&&std::none_of(fiber.proposals.begin(),fiber.proposals.end(),[&](const auto& c){return c.id==current;}))fiber.proposals.push_back(*manual);
            cg::orderReviewProposals(fiber);
            const auto found=std::find_if(fiber.proposals.begin(),fiber.proposals.end(),[&](const auto& c){return c.id==current;});
            if(found!=fiber.proposals.end()){fiber.current=int(found-fiber.proposals.begin());fiber.pending=std::move(pending);}
            _dirty=true;searchReview();
        });
}

void ConstraintGraphWorkspace::previewReview(int index) {
    if(_saving||!_review||_review->selected.empty())return;
    if(!_committed.valid||_committed.graph.seed!=_review->selected)return;
    auto& f=_review->fibers.at(_review->selected);
    auto pending=(index==f.current)?f.pending:std::optional<cg::Graph>{};
    cancelPreview();_reviewPatchSurface.reset();_candidates=f.proposals;_dirty|=f.current!=index;f.current=index;f.pending=pending;
    if(_catalog&&index>=0&&index<int(f.proposals.size())&&cg::pruneUnavailableReviewProposals(f,_reviewAvailablePatches)) {
        _dirty=true;index=f.current;pending=f.pending;_candidates=f.proposals;
    }
    if(index<0||index>=int(f.proposals.size())){f.current=-1;displayReview();updateControls();report(tr("Review complete for this fiber."));return;}
    const auto candidate=f.proposals[index];if(f.decisions.contains(candidate.id))return;
    _candidate=index;_selected=_review->selected;
    _selectionUv=cg::placedLocation(_committed.graph,candidate.contacts.front().a);
    // Only frame the first proposal after selecting a fiber. Advancing,
    // reopening and refitting must preserve the user's camera.
    if(_centerFirstReviewProposal){_view[0]->centerOnSurfacePoint(cv::Vec2f(_selectionUv),false);_centerFirstReviewProposal=false;}
    if(!_prepared||(!pending&&(!_catalog||!_normals))){displayReview();updateControls();report(tr("Sources and normals are required to fit this proposal."));return;}
    auto catalog=_catalog;auto normals=_normals;auto prepared=_prepared;
    cg::ReviewFiber references;references.decisions=f.decisions;
    std::vector<cg::Link> keypoints;if(!pending)if(const auto saved=f.savedKeypoints.find(candidate.id);saved!=f.savedKeypoints.end())keypoints=saved->second;
    run(tr("Fitting patch to fiber and accepted overlaps"),[catalog,normals,prepared,candidate,pending,references,keypoints](auto* cancel,auto progress){
        Job job;auto edit=pending?cg::graphEdit(prepared,*pending):cg::proposeEdit(*catalog,prepared,candidate,false,*normals,cancel,cg::ContactPolicy::FiberReview);
        for(const auto& k:keypoints)cg::addKeypoint(edit,k);
        progress("Matching accepted patch overlaps");cg::addReviewOverlapTargets(edit,references,cancel);
        job.edit=edit;job.proposalView=cg::renderProposalSources(edit,candidate.id,cancel);
        job.fit=cg::fitEdit(edit,cancel,progress,cg::FitPolicy::FiberReview);
        if(job.fit.valid)job.patchView=reviewPatchLayer(job.fit,cancel);
        return job;
    },[this](Job job){installPreview(std::move(job));});
}

void ConstraintGraphWorkspace::rememberReview() {
    // A decision changes one fiber and the order counter. Snapshotting the
    // whole review copied every fiber's proposals (hundreds of thousands of
    // contacts) per keypress and retained twenty such copies.
    if(!_review||_review->selected.empty())return;captureReviewEdit();
    _reviewUndo.push_back({_review->selected,_review->fibers.at(_review->selected),_review->nextOrder});
    if(_reviewUndo.size()>20)_reviewUndo.erase(_reviewUndo.begin());_reviewRedo.clear();
}
void ConstraintGraphWorkspace::decideReview(cg::ReviewStatus status) {
    const bool accepted=status==cg::ReviewStatus::Accepted;
    if(_saving||!_review||_review->selected.empty()||(accepted&&(_busy||!_preview.valid)))return;
    auto& f=_review->fibers.at(_review->selected);if(f.current<0)return;
    const auto patch=f.proposals[f.current].id;rememberReview();
    cg::ReviewDecision decision;decision.status=status;decision.order=_review->nextOrder++;
    if(accepted){decision.alignment=_preview.graph;_reviewLayers.emplace_back(patch,_patchView);_reviewSurfaceCache[patch]={_patchView.surface,_reviewPatchSurface};}
    f.decisions[patch]=std::move(decision);f.pending.reset();_dirty=true;
    if(accepted)rememberReviewLayers(_review->selected,{_reviewLayers.back()});
    rebuildReviewLayers();refreshReviewList();const auto next=cg::nextReviewProposal(f,f.current);previewReview(next);
}
std::optional<cg::PatchEntry> ConstraintGraphWorkspace::reviewPatch() const {
    if(!_review||_review->selected.empty())return {};
    const auto& f=_review->fibers.at(_review->selected);
    if(f.current<0||f.current>=int(f.proposals.size()))return {};
    const auto& id=f.proposals[f.current].id;
    if(_draft){auto found=_draft->nodes.find(id);if(found!=_draft->nodes.end()&&!found->second.source->fiber)
        return cg::PatchEntry{id,found->second.source->path,{},{}};}
    if(f.pending){auto found=f.pending->nodes.find(id);if(found!=f.pending->nodes.end()&&!found->second.source->fiber)
        return cg::PatchEntry{id,found->second.source->path,{},{}};}
    if(_catalog)for(const auto& p:_catalog->patches)if(p.id==id)return p;
    return {};
}
void ConstraintGraphWorkspace::quarantineReview() {
    if(_busy||_saving)return;
    const auto patch=reviewPatch();if(!patch)return;
    auto path=_path;const bool firstSave=path.isEmpty();
    if(firstSave)path=QFileDialog::getSaveFileName(this,tr("Save fiber links before quarantining"),{},tr("Fiber links workspace (*.json)"));
    if(path.isEmpty())return;
    captureReviewEdit();auto snapshot=*_review;
    cancelComparison();stopDiscovery();
    run(tr("Quarantining selected patch"),[snapshot,path,patch,firstSave](auto*,auto progress)mutable{
        if(firstSave)snapshot.sessionRevision=cg::saveLinks(snapshot,path.toStdString());
        Job job;job.review=cg::quarantineReviewPatch(snapshot,path.toStdString(),*patch,progress);return job;
    },[this,path](Job job){
        _review=std::move(job.review);_path=path;
        std::erase_if(_reviewLayers,[&](const auto& layer){return _review->quarantined.contains(layer.first);});
        _reviewLayerCache.clear();_reviewLayerCacheBytes=0;
        rebuildReviewLayers();_reviewUndo.clear();_reviewRedo.clear();cancelPreview();refreshReviewList();_dirty=false;
        report(tr("Patch moved to %1 and excluded from all fibers.").arg(QString::fromStdString(cg::quarantineDirectory(path.toStdString()).string())));
        if(_closing){_closing=false;emit closeReady();}else {searchReview();_dirty=false;}
    },false);_saving=true;updateControls();
}
void ConstraintGraphWorkspace::reopenReview(const std::string& patch) {
    if(_saving||!_review||_review->selected.empty())return;
    auto& f=_review->fibers.at(_review->selected);auto found=f.decisions.find(patch);if(found==f.decisions.end())return;
    if(found->second.status!=cg::ReviewStatus::Accepted&&_catalog&&!_reviewAvailablePatches.contains(patch)) {
        report(tr("This patch is no longer in the source folder. Its saved decision has been retained."));return;
    }
    rememberReview();std::optional<cg::Graph> pending;if(cg::fitted(found->second))pending=found->second.alignment;
    f.decisions.erase(found);
    if(_committed.valid&&_committed.graph.seed==_review->selected){std::erase_if(_reviewLayers,[&](const auto& layer){return layer.first==patch;});rebuildReviewLayers();}
    const auto proposal=std::find_if(f.proposals.begin(),f.proposals.end(),[&](const auto& c){return c.id==patch;});
    if(proposal==f.proposals.end()){f.discovered=false;f.pending.reset();searchReview();}
    else {f.current=int(proposal-f.proposals.begin());f.pending=std::move(pending);previewReview(f.current);}
    _dirty=true;refreshReviewList();
}
void ConstraintGraphWorkspace::reviewHistory(bool redo) {
    if(_saving||!_review)return;auto& from=redo?_reviewRedo:_reviewUndo;auto& to=redo?_reviewUndo:_reviewRedo;if(from.empty())return;
    auto entry=std::move(from.back());from.pop_back();
    if(!_review->fibers.contains(entry.fiber))return;
    captureReviewEdit();to.push_back({entry.fiber,_review->fibers.at(entry.fiber),_review->nextOrder});cancelPreview();
    _review->fibers.at(entry.fiber)=std::move(entry.state);_review->nextOrder=entry.nextOrder;_review->selected=entry.fiber;_dirty=true;
    selectReviewFiber(entry.fiber);
}
void ConstraintGraphWorkspace::rebuildReviewLayers() {
    // Preserve scaled surface identities when accepting another patch. A new
    // identity would invalidate the viewer's cached pixels for every layer.
    auto previous=std::move(_reviewSurfaceCache);_reviewSurfaceCache.clear();
    _reviewContextSurfaces.clear();
    for(const auto& [id,fit]:_reviewLayers){
        const auto found=previous.find(id);
        auto surface=found!=previous.end()&&found->second.first.lock()==fit.surface?found->second.second:displaySurface(fit);
        _reviewSurfaceCache[id]={fit.surface,surface};_reviewContextSurfaces.push_back(std::move(surface));
    }
}
cg::Graph ConstraintGraphWorkspace::reviewDisplayGraph() const {
    auto graph=_committed.graph;
    const auto* fiber=displayedReviewFiber();if(!fiber)return graph;
    for(const auto& [patch,d]:fiber->decisions)if(cg::fitted(d)){graph.nodes.emplace(patch,d.alignment.nodes.at(patch));graph.links.insert(graph.links.end(),d.alignment.links.begin(),d.alignment.links.end());}
    if(_preview.valid)for(const auto& [id,p]:_preview.graph.nodes)if(!p.source->fiber)graph.nodes[id]=p;
    return graph;
}
void ConstraintGraphWorkspace::displayReview() {
    if(!_view[0])return;
    _state->setSurface("constraint-graph",_committed.surface?_reviewFiberSurface:nullptr);
    _state->setSurface("constraint-proposal",_proposalView.valid?displaySurface(_proposalView):nullptr);
    std::vector<std::shared_ptr<Surface>> layers=_reviewContextSurfaces;
    std::vector<float> opacities(layers.size(),_committedOpacity?_committedOpacity->value()/100.f:.2f);
    std::vector<QColor> tints;
    for(const auto& [id,fit]:_reviewLayers)tints.push_back(_patchColors.contextTint(id,_committedTint->value()));
    if(_preview.valid&&_reviewPatchSurface){layers.push_back(_reviewPatchSurface);opacities.push_back(_patchOpacity->value()/100.f);tints.emplace_back();}
    _view[0]->setSurfaceLayers(_reviewFiberSurface,std::move(layers),std::move(opacities),std::move(tints));
    _splitter->widget(1)->setVisible(_matching&&_proposalView.valid);overlays();
}
void ConstraintGraphWorkspace::updateReviewControls() {
    const bool review=reviewing();_reviewPanel->setVisible(review);_rejectReview->setVisible(review);_skipReview->setVisible(review);_alternateReview->hide();
    _mode->setEnabled(!_saving);
    _component->setVisible(!review);
    if(auto* form=qobject_cast<QFormLayout*>(_component->parentWidget()->layout()))if(auto* label=form->labelForField(_component))label->setVisible(!review);
    if(!review){_commit->setText(tr("Commit"));return;}
    _discoveryStride->setEnabled(false);
    _compareSources->hide();_whole->hide();_quarantine->show();_committedOpacity->setEnabled(!_saving);_patchOpacity->setEnabled(!_saving);
    _reviewFibers->setEnabled(!_saving);_reviewDecisions->setEnabled(!_saving);_reverseReview->setEnabled(!_saving);
    _fiberLineOpacity->setEnabled(!_saving);_fiberLineWidth->setEnabled(!_saving);_committedTint->setEnabled(!_saving);
    _commit->setText(tr("Commit (Enter)"));_cancelButton->setText(!_busy&&!_draft?tr("Resume"):tr("Pause"));
    const bool proposal=_review&&!_review->selected.empty()&&_review->fibers.at(_review->selected).current>=0;
    _quarantine->setEnabled(proposal&&!_busy&&!_saving&&reviewPatch().has_value());
    _quarantine->setToolTip(_path.isEmpty()?tr("Save this review, then move the patch to quarantine and exclude it from all fibers."):
        tr("Move the source patch to %1, exclude it from all fibers, and save the review.").arg(QString::fromStdString(cg::quarantineDirectory(_path.toStdString()).string())));
    _rejectReview->setEnabled(proposal&&!_saving);_skipReview->setEnabled(proposal&&!_saving);_pair->setEnabled(proposal&&!_busy&&_proposalView.valid);
    const bool sources=_review&&_catalog&&_normals&&!_busy&&!_saving;
    _autoLinkFiber->setEnabled(sources&&!_review->selected.empty()&&_committed.valid&&_committed.graph.seed==_review->selected);
    _autoLinkAll->setEnabled(sources);_autoLinkContact->setEnabled(!_busy&&!_saving);
    const auto autoRejected=autoRejectedPatches();
    _nextAutoRejected->setEnabled(!autoRejected.empty()&&!_busy&&!_saving);
    _nextAutoRejected->setText(autoRejected.empty()?tr("Next auto-rejected"):tr("Next auto-rejected (%1)").arg(autoRejected.size()));
    _cancelButton->setEnabled(proposal&&!_saving);
    int alternatives=0;
    if(proposal){const auto& f=_review->fibers.at(_review->selected);const auto& id=f.proposals[f.current].id;for(const auto& c:f.proposals)alternatives+=c.id==id;}
    if(_review&&!_review->selected.empty()) {
        const auto& f=_review->fibers.at(_review->selected);
        std::set<std::string> remaining;for(const auto& c:f.proposals)if(!f.decisions.contains(c.id))remaining.insert(c.id);
        _previous->setEnabled(!_saving&&!remaining.empty());_next->setEnabled(!_saving&&!remaining.empty());
        if(proposal)_candidateLabel->setText(tr("%1 remaining · %2").arg(remaining.size()).arg(QString::fromStdString(f.proposals[f.current].id)));
        else _candidateLabel->setText(f.discovered?tr("Review complete"):tr("Finding proposals…"));
    }
    _alternateReview->setEnabled(alternatives>1&&!_saving);_alternateReview->setText(tr("Next placement (%1)").arg(alternatives));
}
std::vector<std::string> ConstraintGraphWorkspace::autoRejectedPatches() const {
    std::vector<std::string> out;
    if(!_review||_review->selected.empty())return out;
    const auto& f=_review->fibers.at(_review->selected);
    // Proposal order, starting after the current position, so review walks the fiber.
    const int size=int(f.proposals.size());
    for(int step=1;step<=size;++step) {
        const auto& id=f.proposals[((f.current<0?-1:f.current)+step)%size].id;
        const auto d=f.decisions.find(id);
        if(d!=f.decisions.end()&&d->second.automatic&&d->second.status==cg::ReviewStatus::Rejected&&std::find(out.begin(),out.end(),id)==out.end())out.push_back(id);
    }
    return out;
}
void ConstraintGraphWorkspace::reopenNextAutoRejected() {
    const auto pending=autoRejectedPatches();if(pending.empty())return;
    reopenReview(pending.front());
}
void ConstraintGraphWorkspace::autoLinkReview(bool allFibers) {
    if(_busy||_saving||!_review||!_catalog||!_normals)return;
    if(!allFibers&&(_review->selected.empty()||!_committed.valid||_committed.graph.seed!=_review->selected))return;
    captureReviewEdit();
    std::vector<std::string> ids;
    if(allFibers)for(const auto& [id,f]:_review->fibers)ids.push_back(id);else ids.push_back(_review->selected);
    if(allFibers){_reviewUndo.clear();_reviewRedo.clear();}else rememberReview();
    cancelPreview();
    std::vector<std::pair<std::string,cg::ReviewFiber>> snapshots;
    for(const auto& id:ids)snapshots.emplace_back(id,_review->fibers.at(id));
    const auto s=_review->settings;auto catalog=_catalog;auto normals=_normals;
    std::set<std::string> excluded;for(const auto& [patch,p]:_review->quarantined)excluded.insert(patch);
    cg::AutoLinkSettings autoLink;autoLink.anchorContact=_autoLinkContact->value();
    const uint64_t firstOrder=_review->nextOrder;
    run(allFibers?tr("Auto linking all fibers"):tr("Auto linking fiber"),[snapshots=std::move(snapshots),s,catalog,normals,excluded,autoLink,firstOrder](auto* cancel,auto progress) mutable {
        Job job;job.nextOrder=firstOrder;
        std::map<std::string,cg::PatchEntry> entries;for(const auto& p:catalog->patches)entries.emplace(p.id,p);
        std::mutex mutex;std::map<std::string,std::shared_ptr<const cg::Node>> cache;
        cg::PatchLoader load=[&](const std::string& id)->std::shared_ptr<const cg::Node>{
            {std::lock_guard lock(mutex);if(auto it=cache.find(id);it!=cache.end())return it->second;}
            const auto found=entries.find(id);std::shared_ptr<const cg::Node> node;
            if(found!=entries.end())try{node=cg::loadPatchNode(found->second,s);}catch(const std::exception&){node=nullptr;}
            std::lock_guard lock(mutex);cache[id]=node;return node;
        };
        size_t done=0,linked=0,rejected=0,unregistered=0,kept=0;
        for(auto& [id,f]:snapshots) {
            if(cancel&&cancel->load())break;
            const auto prefix="Fiber "+std::to_string(++done)+"/"+std::to_string(snapshots.size())+": ";
            progress(prefix+"laying out");
            // Saved strips of another generation regenerate first, as selecting the fiber would.
            if(!f.base.nodes.empty())cg::regenerateReviewFiber(f,*normals,cancel);
            auto base=f.base;base.settings=s;
            if(base.nodes.empty()) {
                if(!catalog->fibers.contains(id))continue;
                auto node=cg::makeFiberNode(catalog->fibers.at(id),s,*normals,cancel);base.seed=id;base.nodes.emplace(id,cg::Placement{node,node->initialUv});
            }
            const auto fit=cg::renderPlacement(base,cancel);
            if(!fit.valid)throw std::runtime_error(fit.message);
            f.base=fit.graph;const auto prepared=cg::prepareGraph(fit,cancel);
            if(!f.discovered) {
                progress(prefix+"finding patches along the fiber");
                f.proposals=cg::discover(*catalog,f.base,id,cancel,{},{},1,excluded,cg::ContactPolicy::FiberReview);
                f.discovered=true;f.contactPolicyVersion=1;f.pending.reset();cg::orderReviewProposals(f);
            }
            const auto& fiberNode=*f.base.nodes.at(id).source;
            const auto result=cg::autoLinkFiber(fiberNode,f.proposals,load,autoLink,cancel,[&](const std::string& m){progress(prefix+m);});
            std::map<std::string,const cg::AutoLinkCandidate*> byId;for(const auto& c:result.candidates)byId.emplace(c.id,&c);
            // Fit linked patches in proposal order; earlier automatic accepts
            // supply overlap targets exactly as an interactive review would.
            size_t fitted=0;
            for(size_t i=0;i<f.proposals.size();++i) {
                const auto& candidate=f.proposals[i];
                const auto found=byId.find(candidate.id);if(found==byId.end())continue;
                const auto* c=found->second;byId.erase(found);
                if(const auto d=f.decisions.find(c->id);d!=f.decisions.end()&&!d->second.automatic){++kept;continue;}
                cg::ReviewDecision decision;decision.automatic=true;decision.order=job.nextOrder++;
                char buffer[160];
                if(c->linked) {
                    progress(prefix+"fitting linked patch "+std::to_string(++fitted)+"/"+std::to_string(result.linked));
                    auto edit=cg::proposeEdit(*catalog,prepared,candidate,false,*normals,cancel,cg::ContactPolicy::FiberReview);
                    cg::addReviewOverlapTargets(edit,f,cancel);
                    const auto fitted_=cg::fitEdit(edit,cancel,{},cg::FitPolicy::FiberReview);
                    if(fitted_.valid&&fitted_.graph.nodes.at(id).uv==f.base.nodes.at(id).uv) {
                        decision.status=cg::ReviewStatus::Accepted;decision.alignment=fitted_.graph;++linked;
                        if(c->anchor)std::snprintf(buffer,sizeof buffer,"Auto-linked anchor: %.0f vx fiber contact, strip agreement %.2f vs %.2f adjacent",c->contact,c->stripSame,c->stripAdjacent);
                        else std::snprintf(buffer,sizeof buffer,"Auto-linked in round %d: %.2f same-sheet vs %.2f adjacent overlap with linked patches",c->round,c->linkedSame,c->linkedAdjacent);
                    } else {
                        decision.status=cg::ReviewStatus::Skipped;++unregistered;
                        std::snprintf(buffer,sizeof buffer,"Auto-linked but could not register: %s",fitted_.message.substr(0,100).c_str());
                    }
                } else {
                    decision.status=cg::ReviewStatus::Rejected;++rejected;
                    if(c->contact>=autoLink.anchorContact)std::snprintf(buffer,sizeof buffer,"Auto-rejected: %.0f vx contact but strip agreement %.2f does not exceed %.2f adjacent",c->contact,c->stripSame,c->stripAdjacent);
                    else std::snprintf(buffer,sizeof buffer,"Auto-rejected: %.0f vx contact, %.2f same-sheet vs %.2f adjacent overlap with linked patches",c->contact,c->linkedSame,c->linkedAdjacent);
                }
                decision.note=buffer;f.decisions[c->id]=std::move(decision);
            }
            f.pending.reset();f.current=cg::nextReviewProposal(f);
            job.autoLinked.emplace_back(id,std::move(f));
        }
        job.notice="Auto link: "+std::to_string(linked)+" linked, "+std::to_string(rejected)+" rejected, "+std::to_string(unregistered)+" could not be registered"+
            (kept?", "+std::to_string(kept)+" human decisions kept":"")+" across "+std::to_string(job.autoLinked.size())+" fiber(s).";
        return job;
    },[this](Job job){
        for(auto& [id,f]:job.autoLinked)_review->fibers.at(id)=std::move(f);
        _review->nextOrder=std::max(_review->nextOrder,job.nextOrder);_dirty=true;
        _reviewLayerCache.clear();_reviewLayerCacheBytes=0;
        refreshReviewList();report(QString::fromStdString(job.notice));
        if(!_review->selected.empty())selectReviewFiber(_review->selected);
    });
}
void ConstraintGraphWorkspace::saveReview(bool choose) {
    if(_saving||!_review)return;captureReviewEdit();
    auto path=_path;if(choose||path.isEmpty())path=QFileDialog::getSaveFileName(this,tr("Save fiber links"),path,tr("Fiber links workspace (*.json)"));
    if(path.isEmpty()){_closing=false;return;}const auto snapshot=*_review;
    cancelPreview();
    // The compact vc_fiber_patch_links document: decisions, notes and manual
    // keypoints only. Strips, proposals and placements are rebuilt on load.
    run(tr("Saving fiber links"),[snapshot,path](auto*,auto){Job job;job.notice=cg::saveLinks(snapshot,path.toStdString());return job;},[this,path](Job job){
        _review->sessionRevision=job.notice;_path=path;_dirty=false;report(tr("Saved fiber links workspace"));if(_closing){_closing=false;emit closeReady();}else {searchReview();_dirty=false;}
    },false);_saving=true;updateControls();
}
