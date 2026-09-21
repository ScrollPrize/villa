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

namespace cg=vc::constraints;
bool ConstraintGraphWorkspace::reviewing() const {return _mode&&_mode->currentIndex()==1;}

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
    captureReviewEdit();stopDiscovery();cancelPreview();_prepared.reset();_committed={};_reviewLayers.clear();_reviewContextSurfaces.clear();_reviewSurfaceCache.clear();
    _reviewFiberSurface.reset();_reviewPatchSurface.reset();
    _dirty|=_review->selected!=id;_review->selected=id;_selected=id;auto snapshot=_review->fibers.at(id);const auto s=_review->settings;
    auto catalog=_catalog;auto normals=_normals;
    {QSignalBlocker b(_reverseReview);_reverseReview->setChecked(snapshot.reverse);}
    refreshReviewList();displayReview();
    run(tr("Laying out fiber"),[id,s,snapshot,catalog,normals](auto* cancel,auto progress) mutable {
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
        std::vector<std::pair<std::string,cg::ReviewDecision>> accepted;
        for(const auto& item:snapshot.decisions)if(item.second.status==cg::ReviewStatus::Accepted)accepted.push_back(item);
        std::sort(accepted.begin(),accepted.end(),[](const auto& a,const auto& b){return std::tie(a.second.order,a.first)<std::tie(b.second.order,b.first);});
        for(auto& [patch,d]:accepted){d.alignment.settings=s;auto layer=cg::renderSourceLayer(d.alignment,false,cancel);if(!layer.valid)throw std::runtime_error(layer.message);job.reviewLayers.emplace_back(patch,std::move(layer));}
        return job;
    },[this,id,newFiber](Job job){
        if(job.regeneratedReviewFiber){_review->fibers.at(id)=std::move(*job.regeneratedReviewFiber);_dirty=true;}
        _committed=std::move(job.fit);_prepared=std::move(job.prepared);_review->fibers.at(id).base=_committed.graph;
        _reviewLayers=std::move(job.reviewLayers);rebuildReviewLayers();_reviewFiberSurface=displaySurface(_committed);
        displayReview();if(newFiber){initialView();_centerFirstReviewProposal=true;}refreshReviewList();searchReview();
        if(!job.notice.empty())report(QString::fromStdString(job.notice));
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
            auto* decision=new QListWidgetItem((d.status==cg::ReviewStatus::Accepted?tr("Accepted · "):d.status==cg::ReviewStatus::Skipped?tr("Skipped · "):tr("Rejected · "))+QString::fromStdString(patch),_reviewDecisions);decision->setData(Qt::UserRole,QString::fromStdString(patch));}}
    }
}

void ConstraintGraphWorkspace::searchReview() {
    if(_saving||!_review||_review->selected.empty()||!_committed.valid)return;
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
    run(tr("Fitting patch to fiber and accepted overlaps"),[catalog,normals,prepared,candidate,pending,references](auto* cancel,auto progress){
        Job job;auto edit=pending?cg::graphEdit(prepared,*pending):cg::proposeEdit(*catalog,prepared,candidate,false,*normals,cancel,cg::ContactPolicy::FiberReview);
        progress("Matching accepted patch overlaps");cg::addReviewOverlapTargets(edit,references,cancel);
        job.edit=edit;job.proposalView=cg::renderProposalSources(edit,candidate.id,cancel);
        job.fit=cg::fitEdit(edit,cancel,progress,cg::FitPolicy::FiberReview);
        if(job.fit.valid)job.patchView=cg::renderSourceLayer(job.fit.graph,false,cancel);
        return job;
    },[this](Job job){installPreview(std::move(job));});
}

void ConstraintGraphWorkspace::rememberReview() {
    if(!_review)return;captureReviewEdit();_reviewUndo.push_back(*_review);if(_reviewUndo.size()>20)_reviewUndo.erase(_reviewUndo.begin());_reviewRedo.clear();
}
void ConstraintGraphWorkspace::decideReview(cg::ReviewStatus status) {
    const bool accepted=status==cg::ReviewStatus::Accepted;
    if(_saving||!_review||_review->selected.empty()||(accepted&&(_busy||!_preview.valid)))return;
    auto& f=_review->fibers.at(_review->selected);if(f.current<0)return;
    const auto patch=f.proposals[f.current].id;rememberReview();
    cg::ReviewDecision decision;decision.status=status;decision.order=_review->nextOrder++;
    if(accepted){decision.alignment=_preview.graph;_reviewLayers.emplace_back(patch,_patchView);_reviewSurfaceCache[patch]={_patchView.surface,_reviewPatchSurface};}
    f.decisions[patch]=std::move(decision);f.pending.reset();_dirty=true;
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
    if(firstSave)path=QFileDialog::getSaveFileName(this,tr("Save fiber review before quarantining"),{},tr("Fiber review workspace (*.json)"));
    if(path.isEmpty())return;
    captureReviewEdit();auto snapshot=*_review;
    cancelComparison();stopDiscovery();
    run(tr("Quarantining selected patch"),[snapshot,path,patch,firstSave](auto*,auto progress)mutable{
        if(firstSave)snapshot.sessionRevision=cg::saveReview(snapshot,path.toStdString());
        Job job;job.review=cg::quarantineReviewPatch(snapshot,path.toStdString(),*patch,progress);return job;
    },[this,path](Job job){
        _review=std::move(job.review);_path=path;
        std::erase_if(_reviewLayers,[&](const auto& layer){return _review->quarantined.contains(layer.first);});
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
    rememberReview();std::optional<cg::Graph> pending;if(found->second.status==cg::ReviewStatus::Accepted)pending=found->second.alignment;
    f.decisions.erase(found);std::erase_if(_reviewLayers,[&](const auto& layer){return layer.first==patch;});rebuildReviewLayers();
    const auto proposal=std::find_if(f.proposals.begin(),f.proposals.end(),[&](const auto& c){return c.id==patch;});
    if(proposal==f.proposals.end()){f.discovered=false;f.pending.reset();searchReview();}
    else {f.current=int(proposal-f.proposals.begin());f.pending=std::move(pending);previewReview(f.current);}
    _dirty=true;refreshReviewList();
}
void ConstraintGraphWorkspace::reviewHistory(bool redo) {
    if(_saving||!_review)return;auto& from=redo?_reviewRedo:_reviewUndo;auto& to=redo?_reviewUndo:_reviewRedo;if(from.empty())return;
    captureReviewEdit();to.push_back(*_review);cancelPreview();const auto revision=_review->sessionRevision;
    _review=std::move(from.back());_review->sessionRevision=revision;from.pop_back();_dirty=true;setSettings(_review->settings);selectReviewFiber(_review->selected);
    {QSignalBlocker opacity(_fiberLineOpacity),width(_fiberLineWidth),tint(_committedTint);_fiberLineOpacity->setValue(_review->fiberLineOpacity);_fiberLineWidth->setValue(_review->fiberLineWidth);_committedTint->setValue(_review->committedTint);}
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
    if(!_review||_review->selected.empty())return graph;
    for(const auto& [patch,d]:_review->fibers.at(_review->selected).decisions)if(d.status==cg::ReviewStatus::Accepted){graph.nodes.emplace(patch,d.alignment.nodes.at(patch));graph.links.insert(graph.links.end(),d.alignment.links.begin(),d.alignment.links.end());}
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
void ConstraintGraphWorkspace::saveReview(bool choose) {
    if(_saving||!_review)return;captureReviewEdit();
    auto path=_path;if(choose||path.isEmpty())path=QFileDialog::getSaveFileName(this,tr("Save fiber review"),path,tr("Fiber review workspace (*.json)"));
    if(path.isEmpty()){_closing=false;return;}const auto snapshot=*_review;
    cancelPreview();
    run(tr("Saving fiber review"),[snapshot,path](auto*,auto){Job job;job.notice=cg::saveReview(snapshot,path.toStdString());return job;},[this,path](Job job){
        _review->sessionRevision=job.notice;_path=path;_dirty=false;report(tr("Saved fiber review workspace"));if(_closing){_closing=false;emit closeReady();}else {searchReview();_dirty=false;}
    },false);_saving=true;updateControls();
}
