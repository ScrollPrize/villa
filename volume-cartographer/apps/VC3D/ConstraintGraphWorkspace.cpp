#include "ConstraintGraphWorkspace.hpp"
#include <QSpinBox>
#include "ConstraintGraphActivity.hpp"
#include "ConstraintGraphView.hpp"
#include "ConstraintGraphSelectionBar.hpp"
#include "CState.hpp"
#include "ViewerManager.hpp"
#include "OpenDataLasagna.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "volume_viewers/SurfaceSceneTransform.hpp"
#include "overlays/OverlayBatchItem.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include <QtConcurrent/QtConcurrent>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <QFutureWatcher>
#include <QCheckBox>
#include <QComboBox>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsPathItem>
#include <QGraphicsSimpleTextItem>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSplitter>
#include <QSlider>
#include <QStatusBar>
#include <QToolBar>
#include <QKeyEvent>
#include <QTimer>
#include <QElapsedTimer>
#include <QDebug>
#include <set>
#include <fstream>

namespace cg=vc::constraints;
namespace {
QDoubleSpinBox* number(QWidget* parent,double value) {
    auto* s=new QDoubleSpinBox(parent);s->setDecimals(4);s->setRange(.0001,10000);s->setValue(value);return s;
}
}
std::shared_ptr<QuadSurface> ConstraintGraphWorkspace::displaySurface(const cg::FitResult& result) {
    if(!result.surface)return {};
    if(result.graph.settings.volumeScale==1)return result.surface;
    auto points=result.surface->rawPointsPtr()->clone();double scale=result.graph.settings.volumeScale;
    if(scale!=1)for(auto& p:points)if(p[0]!=-1)p*=scale;
    auto surface=std::make_shared<QuadSurface>(points,result.surface->scale());surface->setStrictQuadRenderValidity(true);
    surface->shiftSurfaceOrigin(surface->gridToSurface({0,0})-result.origin);return surface;
}

ConstraintGraphWorkspace::ConstraintGraphWorkspace(CState* main,QWidget* parent):QMainWindow(parent),_main(main) {
    setObjectName("constraintGraphWorkspace");_pool.setMaxThreadCount(1);_discoveryPool.setMaxThreadCount(1);_comparisonPool.setMaxThreadCount(1);
    _state=new CState(this,main&&main->debugDownloadQueueEnabled());
    _manager=std::make_unique<ViewerManager>(_state,_state->pointCollection(),this);
    _manager->setPreferSurfaceTileFills(true);
    auto* toolbar=addToolBar(tr("Constraint Graph"));toolbar->setObjectName("constraintGraphToolbar");
    _volume=new QComboBox(toolbar);_volume->setObjectName("constraintGraphVolume");toolbar->addWidget(new QLabel(tr("Volume "),toolbar));toolbar->addWidget(_volume);
    auto action=[&](QString text,auto callback){auto* a=toolbar->addAction(text);connect(a,&QAction::triggered,this,callback);return a;};
    action(tr("Open…"),[this]{auto p=QFileDialog::getOpenFileName(this,tr("Open constraint graph"),_path,tr("Graph session (*.json)"));if(!p.isEmpty())openSession(p);});
    action(tr("Save"),[this]{save();});action(tr("Save as…"),[this]{save(true);});
    action(tr("Export tifxyz…"),[this]{if(reviewing()){report(tr("Save the fiber review workspace to preserve its associations."));return;}if(!_committed.valid)return;auto parent=QFileDialog::getExistingDirectory(this,tr("Export destination"));if(parent.isEmpty())return;
        auto path=std::filesystem::path(parent.toStdString())/"constraint-graph.tifxyz";auto snapshot=_committed;
        run(tr("Exporting tifxyz"),[snapshot,path](auto*,auto){cg::exportSurface(snapshot,path);return Job{};},[this,path](Job){report(tr("Exported %1").arg(QString::fromStdString(path.string())));},false);});
    action(tr("Undo"),[this]{navigateHistory(false);})->setShortcut(QKeySequence::Undo);
    action(tr("Redo"),[this]{navigateHistory(true);})->setShortcut(QKeySequence::Redo);
    action(tr("Fit view"),[this]{_view[0]->resetViewForCurrentContent(true);});
    action(tr("Refit graph"),[this]{
        if(reviewing()){rebuildPreview();return;}
        if(_saving||_busy||!_committed.valid)return;
        if(!_committed.graph.windingLayout.is_null()){report(tr("Imported winding layouts preserve their computed coordinates and cannot be refitted."));return;}
        auto draft=_draft.value_or(_committed.graph);auto selected=_selected;
        cancelPreview();_selected=selected;_candidates.clear();
        _bar->move(12,12);
        fitPreview(std::move(draft),std::move(selected),true,true);
    });
    _showCommitted=new QCheckBox(tr("Show committed"),toolbar);toolbar->addWidget(_showCommitted);connect(_showCommitted,&QCheckBox::toggled,this,[this]{display();});
    _showAllBoundaries=new QCheckBox(tr("Show all boundaries"),toolbar);
    _showAllBoundaries->setObjectName("constraintGraphShowAllBoundaries");
    _showAllBoundaries->setToolTip(tr("Show colored boundaries for committed fibers and patches too. When unchecked, proposals and the selected patch have boundaries."));
    toolbar->addWidget(_showAllBoundaries);
    connect(_showAllBoundaries,&QCheckBox::toggled,this,[this]{overlays();});
    auto* dock=new QDockWidget(tr("Graph inputs"),this);dock->setObjectName("constraintGraphInputs");addDockWidget(Qt::LeftDockWidgetArea,dock);
    auto* panel=new QWidget(dock);auto* form=new QFormLayout(panel);dock->setWidget(panel);
    dock->setMaximumWidth(460);
    auto folder=[&](const QString& label,QLineEdit*& edit){auto* row=new QWidget(panel);auto* layout=new QHBoxLayout(row);layout->setContentsMargins(0,0,0,0);
        edit=new QLineEdit(row);auto* browse=new QPushButton("…",row);browse->setMaximumWidth(30);layout->addWidget(edit);layout->addWidget(browse);form->addRow(label,row);
        connect(browse,&QPushButton::clicked,this,[this,edit,label]{auto p=QFileDialog::getExistingDirectory(this,label,edit->text());if(!p.isEmpty())edit->setText(p);});};
    folder(tr("Fibers"),_fibers);folder(tr("Patches"),_patches);
    _fibers->setObjectName("constraintGraphFiberFolder");_patches->setObjectName("constraintGraphPatchFolder");
    _normal=new QComboBox(panel);_normal->setEditable(true);_normal->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);_normal->setMinimumContentsLength(12);_normal->setObjectName("constraintGraphNormals");form->addRow(tr("Lasagna normals"),_normal);
    auto* browseNormal=new QPushButton(tr("Browse normals manifest…"),panel);form->addRow(browseNormal);connect(browseNormal,&QPushButton::clicked,this,[this]{auto p=QFileDialog::getOpenFileName(this,tr("Lasagna manifest"),{},tr("JSON (*.json)"));if(!p.isEmpty())_normal->setEditText(p);});
    _fiberScale=number(panel,.25);_patchScale=number(panel,1);_normalScale=number(panel,4);_volumeScale=number(panel,1);_radius=number(panel,4);_stiffness=number(panel,1);
    form->addRow(tr("Fiber → graph scale"),_fiberScale);form->addRow(tr("Patch → graph scale"),_patchScale);
    form->addRow(tr("Graph → normal base"),_normalScale);form->addRow(tr("Graph → volume scale"),_volumeScale);
    form->addRow(tr("Grid spacing"),new QLabel(tr("20 vx × 20 vx"),panel));form->addRow(tr("Proximity (vx)"),_radius);form->addRow(tr("Spacing stiffness"),_stiffness);
    _discoveryStride=new QComboBox(panel);_discoveryStride->setObjectName("constraintGraphDiscoveryStride");
    for(int stride:{1,4,8})_discoveryStride->addItem(stride==1?tr("1 — every point"):tr("%1 — every %1th point").arg(stride),stride);
    const int savedStride=QSettings{}.value("constraintGraph/discoveryStride",4).toInt();
    _discoveryStride->setCurrentIndex(std::max(0,_discoveryStride->findData(savedStride==1||savedStride==8?savedStride:4)));
    _discoveryStride->setToolTip(tr("Sample dense fiber points for proposals in both directions. Endpoints are always checked. Larger strides can miss short contacts; 1 checks every point. Fiber review always checks every point."));
    form->addRow(tr("Discovery stride"),_discoveryStride);
    auto* load=new QPushButton(tr("Load folders / New graph"),panel);load->setObjectName("constraintGraphLoad");form->addRow(load);connect(load,&QPushButton::clicked,this,[this]{try{start(settings());}catch(const std::exception& e){report(QString::fromUtf8(e.what()));}});
    _component=new QComboBox(panel);_component->setObjectName("constraintGraphComponent");form->addRow(tr("Seed component"),_component);connect(_component,qOverload<int>(&QComboBox::activated),this,&ConstraintGraphWorkspace::seed);
    _audit=new QLabel(tr("Fiber cycle check: load source folders to check."),panel);_audit->setObjectName("constraintGraphCycleAudit");_audit->setWordWrap(true);_audit->setTextInteractionFlags(Qt::TextSelectableByMouse);form->addRow(_audit);
    auto* auditDetails=new QPushButton(tr("Cycle check / input details…"),panel);auditDetails->setObjectName("constraintGraphAuditDetails");form->addRow(auditDetails);
    connect(auditDetails,&QPushButton::clicked,this,[this]{
        QMessageBox box(this);box.setWindowTitle(tr("Source fiber checks"));box.setText(_audit->text());
        if(_catalog){
            QString details=tr("Current source folder: %1\n\n").arg(QString::fromStdString(_catalog->fibers.empty()?std::string{}:_catalog->fibers.begin()->second.path.parent_path().string()));
            for(const auto& d:_catalog->cycleAudit.details)details+=QString::fromStdString(d)+"\n\n";
            for(const auto& w:_catalog->warnings)details+=QString::fromStdString(w)+"\n";
            box.setDetailedText(details);
        }
        box.exec();
    });
    _pairs=new QListWidget(panel);_pairs->setObjectName("constraintGraphKeypoints");form->addRow(tr("Keypoint pairs"),_pairs);
    auto* remove=new QPushButton(tr("Remove selected pair"),panel);form->addRow(remove);connect(remove,&QPushButton::clicked,this,&ConstraintGraphWorkspace::removePair);
    auto* edit=new QPushButton(tr("Replace selected pair"),panel);form->addRow(edit);connect(edit,&QPushButton::clicked,this,[this]{removePair();beginPair();});
    auto* activityDock=new QDockWidget(tr("Activity"),this);activityDock->setObjectName("constraintGraphActivityDock");
    activityDock->setFeatures(QDockWidget::DockWidgetMovable|QDockWidget::DockWidgetFloatable);
    _activity=new ConstraintGraphActivity(activityDock);activityDock->setWidget(_activity);
    addDockWidget(Qt::BottomDockWidgetArea,activityDock);
    connect(_activity->stopButton(),&QPushButton::clicked,this,&ConstraintGraphWorkspace::cancelPreview);
    _splitter=new QSplitter(Qt::Horizontal,this);setCentralWidget(_splitter);
    for(int i=0;i<2;++i) {
        _view[i]=_manager->createViewerInWidget(i==0?"constraint-graph":"constraint-proposal",_splitter);
        auto* widget=qobject_cast<QWidget*>(_view[i]->asQObject());_splitter->addWidget(widget);
        _view[i]->setResetViewOnSurfaceChange(false);
        auto* view=_view[i]->graphicsView();view->installEventFilter(this);
        connect(view,&CVolumeViewerView::sendMouseRelease,this,[this,i](QPointF p,Qt::MouseButton b,Qt::KeyboardModifiers m){if(b==Qt::LeftButton&&m==Qt::NoModifier)clicked(i,p);});
        _view[i]->connectOverlaysUpdated(this,[this,i]{updateOverlayTransform(i);positionControlBar();});
    }
    _splitter->widget(1)->hide();
    _bar=new ConstraintGraphSelectionBar(this);_bar->setObjectName("constraintGraphSelectionBar");_bar->setAutoFillBackground(true);
    auto* barLayout=new QVBoxLayout(_bar);barLayout->setContentsMargins(5,3,5,3);
    barLayout->addWidget(_bar->makeGrip());
    auto* candidateRow=new QHBoxLayout;auto* actionRow=new QHBoxLayout;
    barLayout->addLayout(candidateRow);barLayout->addLayout(actionRow);
    _previous=new QPushButton("‹",_bar);_next=new QPushButton("›",_bar);_previous->setFixedWidth(28);_next->setFixedWidth(28);_candidateLabel=new QLabel(_bar);_candidateLabel->setMaximumWidth(240);
    _whole=new QCheckBox(tr("Include linked component"),_bar);_whole->setChecked(true);
    _pair=new QPushButton(tr("Add keypoint pair"),_bar);_commit=new QPushButton(tr("Commit"),_bar);_commit->setObjectName("constraintGraphCommit");_cancelButton=new QPushButton(tr("Cancel"),_bar);
    for(auto* w:std::vector<QWidget*>{_previous,_candidateLabel,_next})candidateRow->addWidget(w);
    _quarantine=new QPushButton(tr("Quarantine patch"),_bar);_quarantine->setObjectName("constraintGraphQuarantine");
    connect(_quarantine,&QPushButton::clicked,this,&ConstraintGraphWorkspace::quarantineSelected);
    for(auto* w:std::vector<QWidget*>{_pair,_commit,_quarantine,_cancelButton})actionRow->addWidget(w);
    barLayout->addWidget(_whole);
    _compareSources=new QCheckBox(tr("Compare fiber / patch CT"),_bar);
    _compareSources->setObjectName("constraintGraphCompareSources");barLayout->addWidget(_compareSources);
    _opacityControls=new QWidget(_bar);auto* opacityRow=new QHBoxLayout(_opacityControls);
    opacityRow->setContentsMargins(0,0,0,0);barLayout->addWidget(_opacityControls);
    opacityRow->addWidget(new QLabel(tr("Patch opacity"),_bar));
    _patchOpacity=new QSlider(Qt::Horizontal,_bar);_patchOpacity->setObjectName("constraintGraphPatchOpacity");
    _patchOpacity->setRange(0,100);_patchOpacity->setValue(50);opacityRow->addWidget(_patchOpacity);
    auto* opacityValue=new QLabel(tr("50%"),_bar);opacityRow->addWidget(opacityValue);
    _patchOpacity->setToolTip(tr("Blend independent patch CT over fiber CT at the fitted positions. 0% shows fibers; 100% shows patches wherever they have coverage."));
    connect(_compareSources,&QCheckBox::toggled,this,[this]{updateComparison();});
    connect(_patchOpacity,&QSlider::valueChanged,this,[this,opacityValue](int value){
        opacityValue->setText(tr("%1%").arg(value));
        if(reviewing()){if(_review){_review->proposalOpacity=value;_dirty=true;}displayReview();}
        else _view[0]->setSurfaceComparisonOpacity(value/100.0f);
    });
    _bar->hide();
    connect(_previous,&QPushButton::clicked,this,[this]{selectCandidate(-1);});connect(_next,&QPushButton::clicked,this,[this]{selectCandidate(1);});
    connect(_pair,&QPushButton::clicked,this,&ConstraintGraphWorkspace::beginPair);connect(_commit,&QPushButton::clicked,this,&ConstraintGraphWorkspace::commit);
    connect(_cancelButton,&QPushButton::clicked,this,[this]{
        if(reviewing()&&_review&&!_review->selected.empty()) {
            auto& f=_review->fibers.at(_review->selected);
            if(!_busy&&!_draft&&f.current>=0){previewReview(f.current);return;}
            captureReviewEdit();
        }
        if(!_committed.graph.windingLayout.is_null())_selected.clear();
        cancelPreview();
    });
    connect(_whole,&QCheckBox::toggled,this,[this]{if(_candidate>=0)selectCandidate(0);});
    connect(_discoveryStride,qOverload<int>(&QComboBox::currentIndexChanged),this,[this]{
        if(reviewing()&&_review){_review->stride=_discoveryStride->currentData().toInt();for(auto& [id,f]:_review->fibers)f.discovered=false;_dirty=true;searchReview();return;}
        QSettings{}.setValue("constraintGraph/discoveryStride",_discoveryStride->currentData().toInt());
        if(!_committed.valid)return;
        cancelPreview();search();
    });
    connect(_radius,qOverload<double>(&QDoubleSpinBox::valueChanged),this,[this](double v){
        if(reviewing()&&_review){cancelPreview();_review->settings.radius=v;for(auto& [id,f]:_review->fibers){f.discovered=false;f.pending.reset();} _dirty=true;selectReviewFiber(_review->selected);return;}
        if(!_committed.valid)return;cancelPreview();_committed.graph.settings.radius=v;_dirty=true;search();});
    connect(_volumeScale,qOverload<double>(&QDoubleSpinBox::valueChanged),this,[this](double v){
        if(reviewing()&&_review){_review->settings.volumeScale=v;_dirty=true;selectReviewFiber(_review->selected);return;}
        if(!_committed.valid)return;
        _previewCache.clear();_committed.graph.settings.volumeScale=v;_preview.graph.settings.volumeScale=v;_proposalView.graph.settings.volumeScale=v;
        if(_draft)_draft->settings.volumeScale=v;_dirty=_committed.graph.windingLayout.is_null();display();
    });
    connect(_stiffness,qOverload<double>(&QDoubleSpinBox::valueChanged),this,[this](double v){
        if(reviewing()&&_review){captureReviewEdit();_review->settings.stiffness=v;_dirty=true;selectReviewFiber(_review->selected);return;}
        if(!_committed.valid)return;if(!_draft)_draft=_committed.graph;_draft->settings.stiffness=v;rebuildPreview();});
    if(_main){connect(_main,&CState::vpkgChanged,this,[this]{refreshNormals();synchronizeVolume(_main->currentVolume());});}
    refreshNormals();QSettings persisted;_fibers->setText(persisted.value("constraintGraph/fibers").toString());_patches->setText(persisted.value("constraintGraph/patches").toString());
    setupReview(toolbar);
    updateControls();
}
ConstraintGraphWorkspace::~ConstraintGraphWorkspace(){
    cancelComparison();_comparisonPool.waitForDone();
    stopDiscovery();_discoveryPool.waitForDone();
    ++_generation;if(_cancel)_cancel->store(true);_pool.waitForDone();
    for(int pane=0;pane<2;++pane){if(_overlayRoot[pane])for(auto* item:_items[pane])delete item;_items[pane].clear();_markers[pane].clear();delete _overlayRoot[pane].data();}
    delete _highlights.data();_manager.reset();
}
void ConstraintGraphWorkspace::report(const QString& message){_activity->note(message);}
void ConstraintGraphWorkspace::refreshAudit(){
    if(!_committed.graph.windingLayout.is_null()){
        _audit->setText(tr("Imported winding layout. Click a patch to see its boundaries and source name. Quarantine moves its source folder and excludes all its components from this workspace."));return;
    }
    if(!_catalog){_audit->setText(tr("Fiber cycle check unavailable: source catalog not loaded."));return;}
    const auto& audit=_catalog->cycleAudit;
    QString text=QString::fromStdString(audit.summary);
    if(!_catalog->warnings.empty())text+=tr("\n%1 input warnings; see details.").arg(_catalog->warnings.size());
    const bool warning=!audit.issues.empty()||!_catalog->warnings.empty();
    _audit->setText((warning?tr("Warning — "):QString{})+text);
    _audit->setToolTip(tr("Checks current source annotations across all components, independently of the fitted grid. See details for scope and affected links."));
}
void ConstraintGraphWorkspace::run(const QString& title,Worker worker,std::function<void(Job)> finish,bool stoppable,std::function<void()> failed) {
    if(_saving)return;
    if(_cancel)_cancel->store(true);auto cancel=std::make_shared<std::atomic_bool>(false);_cancel=cancel;auto generation=++_generation;
    _activity->finish(ConstraintGraphActivity::Operation,tr("Superseded"));
    _activity->begin(ConstraintGraphActivity::Operation,title,stoppable);
    if(_busy)_activity->update(ConstraintGraphActivity::Operation,tr("Waiting for the previous operation to stop…"));
    _busy=true;updateControls();QPointer<ConstraintGraphWorkspace> guard(this);
    // VC3D_CONSTRAINT_FIT_LOG=1 echoes every progress line with its elapsed
    // time to stderr as well as the Activity panel.
    static const bool logProgress=std::getenv("VC3D_CONSTRAINT_FIT_LOG")!=nullptr;
    const auto started=std::chrono::steady_clock::now();
    auto progress=[guard,generation,started](const std::string& message){
        if(logProgress)std::clog<<"constraint graph +"<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<" ms: "<<message<<std::endl;
        if(guard)QMetaObject::invokeMethod(guard,[guard,generation,message]{if(guard&&guard->_generation==generation)guard->_activity->update(ConstraintGraphActivity::Operation,QString::fromStdString(message));},Qt::QueuedConnection);};
    auto* watcher=new QFutureWatcher<Job>(this);
    connect(watcher,&QFutureWatcher<Job>::finished,this,[this,watcher,generation,cancel,finish=std::move(finish),failed=std::move(failed)]()mutable{
        auto job=watcher->result();watcher->deleteLater();
        if(generation!=_generation){
            if(_cancel==cancel&&!_busy)_activity->finish(ConstraintGraphActivity::Operation,tr("Cancelled"));
            return;
        }
        _busy=false;
        _saving=false;
        const auto outcome=!job.error.empty()?tr("Failed"):
            !job.fit.graph.nodes.empty()&&!job.fit.valid?tr("Needs attention"):tr("Completed");
        _activity->finish(ConstraintGraphActivity::Operation,outcome);
        if(!job.error.empty()){report(QString::fromStdString(job.error));_closing=false;if(failed)failed();updateControls();emit operationFinished(false);return;}
        try{finish(std::move(job));}catch(const std::exception& e){report(QString::fromUtf8(e.what()));emit operationFinished(false);updateControls();return;}
        updateControls();emit operationFinished(true);
    });
    watcher->setFuture(QtConcurrent::run(&_pool,[worker=std::move(worker),cancel,progress]{try{progress("Starting operation");return worker(cancel.get(),progress);}catch(const std::exception& e){Job job;job.error=e.what();return job;}catch(...){Job job;job.error="Unknown graph operation failure";return job;}}));
}
cg::Settings ConstraintGraphWorkspace::settings()const {
    cg::Settings s;s.fiberFolder=_fibers->text().toStdString();s.patchFolder=_patches->text().toStdString();
    s.fiberScale=_fiberScale->value();s.patchScale=_patchScale->value();s.normalScale=_normalScale->value();s.volumeScale=_volumeScale->value();s.radius=_radius->value();s.stiffness=_stiffness->value();
    s.normalManifest=_normal->currentData().isValid()&&_normal->currentText()==_normal->itemText(_normal->currentIndex())?_normal->currentData().toString().toStdString():_normal->currentText().toStdString();
    s.validate();if(s.normalManifest.empty())throw std::runtime_error("Select a Lasagna normals manifest");return s;
}
void ConstraintGraphWorkspace::setSettings(const cg::Settings& s){
    QSignalBlocker radius(_radius),stiff(_stiffness),volume(_volumeScale);_fibers->setText(QString::fromStdString(s.fiberFolder.string()));_patches->setText(QString::fromStdString(s.patchFolder.string()));
    _normal->setEditText(QString::fromStdString(s.normalManifest));_fiberScale->setValue(s.fiberScale);_patchScale->setValue(s.patchScale);_normalScale->setValue(s.normalScale);_volumeScale->setValue(s.volumeScale);_radius->setValue(s.radius);_stiffness->setValue(s.stiffness);
}
void ConstraintGraphWorkspace::refreshNormals(){
    auto previous=_normal->currentText();QSignalBlocker blocker(_normal);_normal->clear();
    if(_main&&_main->vpkg()){
        auto pkg=_main->vpkg();for(const auto& e:pkg->lasagnaDatasetEntries()) {
            auto location=vc3d::opendata::lasagnaSourceManifestLocation(e);
            if(!vc::lasagna::isRemoteLasagnaLocation(location)&&std::filesystem::path(location).is_relative())location=(pkg->path().parent_path()/location).string();
            _normal->addItem(QString::fromStdString(location),QString::fromStdString(location));
            if(e.location==pkg->selectedLasagnaDataset())_normal->setCurrentIndex(_normal->count()-1);
        }
    }
    if(!previous.isEmpty())_normal->setEditText(previous);
}
void ConstraintGraphWorkspace::synchronizeVolume(const std::shared_ptr<Volume>& volume){
    if(_main&&_state->vpkg()!=_main->vpkg())_state->setVpkg(_main->vpkg());
    if(_state->currentVolume()!=volume)_manager->switchVolume(volume);
    display();
}
void ConstraintGraphWorkspace::start(const cg::Settings& s){
    if(reviewing()){startReview(s);return;}
    if(_saving)return;
    if(_dirty&&QMessageBox::question(this,tr("New graph"),tr("Discard unsaved graph changes?"))!=QMessageBox::Yes)return;
    stopDiscovery();setSettings(s);report(tr("Loading fibers and patch bounds…"));
    run(tr("Loading inputs and building graph"),[s](auto* cancel,auto progress){Job j;progress("Opening normals dataset");vc::lasagna::LasagnaDatasetOpenOptions options;options.workingToBaseScale=s.normalScale;
        j.dataset=std::make_shared<vc::lasagna::LasagnaDataset>(vc::lasagna::LasagnaDataset::openLocation(s.normalManifest,options));j.normals=std::make_shared<vc::lasagna::LasagnaNormalSampler>(*j.dataset);
        j.catalog=std::make_shared<cg::Catalog>(cg::loadCatalog(s,cancel,progress));if(j.catalog->components.empty())throw std::runtime_error("No eligible fiber components; check source scales and fiber validation");
        try{j.fit=cg::fit(cg::seedGraph(*j.catalog,0,s,*j.normals,cancel),cancel,progress);}
        catch(const std::exception& e){if(cancel&&cancel->load())throw;j.fit.message=e.what();}
        return j;
    },[this,s](Job j){
        _catalog=j.catalog;_dataset=j.dataset;_normals=j.normals;_path.clear();_undo.clear();_redo.clear();_draft.reset();_preview={};_proposalView={};_candidate=-1;
        refreshAudit();
        _committed={};_dirty=false;_selected.clear();_candidates.clear();
        {QSignalBlocker blocker(_component);_component->clear();for(size_t i=0;i<_catalog->components.size();++i)_component->addItem(tr("%1 — %2 fibers").arg(i+1).arg(_catalog->components[i].size()));}
        if(!j.fit.valid){display();refreshPairs();report(QString::fromStdString(j.fit.message));return;}
        _committed=std::move(j.fit);_dirty=true;_selected.clear();
        QSettings persisted;persisted.setValue("constraintGraph/fibers",QString::fromStdString(s.fiberFolder.string()));persisted.setValue("constraintGraph/patches",QString::fromStdString(s.patchFolder.string()));
        display();initialView();search();
    });
}
void ConstraintGraphWorkspace::seed(int component){
    if(_saving)return;
    if(!_catalog||!_normals||component<0)return;
    if(_dirty&&QMessageBox::question(this,tr("Change seed"),tr("Start a new graph from this component and discard unsaved changes?"))!=QMessageBox::Yes)return;
    stopDiscovery();auto c=_catalog;auto n=_normals;auto s=_committed.valid?_committed.graph.settings:settings();
    run(tr("Building selected component"),[c,n,s,component](auto* cancel,auto progress){Job j;j.fit=cg::fit(cg::seedGraph(*c,component,s,*n,cancel),cancel,progress);return j;},[this](Job j){
        if(!j.fit.valid){report(QString::fromStdString(j.fit.message));return;}_committed=std::move(j.fit);_draft.reset();_preview={};_proposalView={};_undo.clear();_redo.clear();_dirty=true;_path.clear();_selected.clear();display();initialView();search();});
}
void ConstraintGraphWorkspace::stopDiscovery(){
    if(_activity)_activity->finish(ConstraintGraphActivity::Discovery,tr("Stopped"));
    ++_discoveryGeneration;if(_discoveryCancel)_discoveryCancel->store(true);
    _discovering=false;_available.clear();_previewCache.clear();
}
bool ConstraintGraphWorkspace::candidateNearSelection(const cg::Candidate& candidate)const{
    if(_selected.empty()||!_committed.graph.nodes.contains(_selected))return false;
    for(const auto& link:candidate.contacts) {
        if(link.a.node!=_selected)continue;
        if(!_committed.graph.nodes.at(_selected).source->fiber||
            cv::norm(cg::placedLocation(_committed.graph,link.a)-_selectionUv)<=_committed.graph.settings.spacing)return true;
    }
    return false;
}
void ConstraintGraphWorkspace::selectAvailable(bool preview){
    _candidates.clear();_candidate=-1;_awaitingProposals=false;
    if(_selected.empty())return;
    for(const auto& candidate:_available)if(candidateNearSelection(candidate))_candidates.push_back(candidate);
    std::stable_sort(_candidates.begin(),_candidates.end(),[](const auto& a,const auto& b){return a.distance!=b.distance?a.distance<b.distance:a.id<b.id;});
    _awaitingProposals=preview&&_candidates.empty()&&_discovering;
    updateControls();
    if(preview&&!_candidates.empty())selectCandidate(1);
    else if(_candidates.empty())report(_discovering?tr("Searching for proposals… Orange marks show verified attachments."):tr("No proposals at this location."));
}
void ConstraintGraphWorkspace::receiveDiscovery(std::vector<cg::Candidate> batch){
    appendHighlights(batch);updateOverlayTransform(0,true);
    // Append without reordering the active choice while discovery continues.
    if(_candidate>=0)for(const auto& candidate:batch)if(candidateNearSelection(candidate))_candidates.push_back(candidate);
    _available.insert(_available.end(),std::make_move_iterator(batch.begin()),std::make_move_iterator(batch.end()));
    if(_awaitingProposals&&!_busy&&_candidate<0&&!_draft)selectAvailable(true);
    else updateControls();
}
void ConstraintGraphWorkspace::search(){
    if(reviewing()){searchReview();return;}
    stopDiscovery();_candidates.clear();_candidate=-1;
    if(!_committed.valid||!_committed.graph.windingLayout.is_null())return;
    if(!_prepared||cg::preparedResult(*_prepared).surface!=_committed.surface||
       cg::preparedResult(*_prepared).graph.settings.radius!=_committed.graph.settings.radius){
        auto snapshot=_committed;auto edit=std::move(_prepareEdit);_prepareEdit.reset();_prepared.reset();
        run(tr("Preparing attachments"),[snapshot,edit](auto* cancel,auto progress){progress("Preparing committed graph for attachment");Job job;job.prepared=edit?cg::prepareGraph(snapshot,*edit,cancel):cg::prepareGraph(snapshot,cancel);return job;},
            [this](Job job){_prepared=std::move(job.prepared);search();});
        return;
    }
    if(!_catalog)return;
    _activity->begin(ConstraintGraphActivity::Discovery,tr("Attachment search"));
    _discovering=true;_awaitingProposals=!_selected.empty();auto cancel=std::make_shared<std::atomic_bool>(false);_discoveryCancel=cancel;
    const auto generation=_discoveryGeneration;auto catalog=_catalog;auto graph=_committed.graph;const int stride=_discoveryStride->currentData().toInt();
    QPointer<ConstraintGraphWorkspace> guard(this);
    overlays();updateControls();
    auto* watcher=new QFutureWatcher<QString>(this);
    connect(watcher,&QFutureWatcher<QString>::finished,this,[this,watcher,generation]{
        auto error=watcher->result();watcher->deleteLater();if(generation!=_discoveryGeneration)return;
        _discovering=false;
        _activity->finish(ConstraintGraphActivity::Discovery,error.isEmpty()?tr("%1 proposals found").arg(_available.size()):error);
        // Discovery owns its row; completing it must not erase a fit error.
        if(error.isEmpty()&&!_busy&&_candidate<0&&!_selected.empty()&&_awaitingProposals)selectAvailable(true);
        updateControls();
    });
    watcher->setFuture(QtConcurrent::run(&_discoveryPool,[guard,catalog,graph=std::move(graph),cancel,generation,stride]{
        std::vector<cg::Candidate> pending;QElapsedTimer timer;timer.start();
        auto publish=[&]{
            if(pending.empty()||cancel->load())return;
            if(guard)QMetaObject::invokeMethod(guard,[guard,generation,batch=std::move(pending)]()mutable{
                if(guard&&guard->_discoveryGeneration==generation)guard->receiveDiscovery(std::move(batch));
            },Qt::QueuedConnection);
            pending.clear();timer.restart();
        };
        try {
            cg::discover(*catalog,graph,{},cancel.get(),[guard,generation](const std::string& message){
                if(guard)QMetaObject::invokeMethod(guard,[guard,generation,message]{
                    if(guard&&guard->_discoveryGeneration==generation)
                        guard->_activity->update(ConstraintGraphActivity::Discovery,QString::fromStdString(message));
                },Qt::QueuedConnection);
            },[&](std::vector<cg::Candidate> batch){
                pending.insert(pending.end(),std::make_move_iterator(batch.begin()),std::make_move_iterator(batch.end()));
                if(timer.elapsed()>=500)publish();
            },stride);publish();return QString{};
        }catch(const std::exception& e){publish();return QString::fromUtf8(e.what());}
    }));
}
void ConstraintGraphWorkspace::selectCandidate(int delta){
    if(reviewing()){if(_review&&!_review->selected.empty()){auto& f=_review->fibers.at(_review->selected);previewReview(delta==0?f.current:cg::nextReviewProposal(f,f.current,delta));}return;}
    if(_saving)return;
    if(_candidates.empty()||!_catalog||!_normals||!_prepared)return;
    _candidate=(_candidate+delta+int(_candidates.size()))%int(_candidates.size());if(_candidate<0)_candidate=0;
    auto candidate=_candidates[_candidate];
    if(_selected.empty()&&!candidate.contacts.empty()) {_selected=candidate.contacts.front().a.node;_selectionUv=cg::placedLocation(_committed.graph,candidate.contacts.front().a);}
    auto c=_catalog;auto n=_normals;auto prepared=_prepared;bool whole=_whole->isChecked();
    auto same=[&](const CachedPreview& cached){
        if(cached.whole!=whole||cached.candidate.id!=candidate.id||cached.candidate.contacts.size()!=candidate.contacts.size())return false;
        auto location=[](const auto& a,const auto& b){return a.node==b.node&&a.lineIndex==b.lineIndex&&a.grid==b.grid;};
        for(size_t i=0;i<candidate.contacts.size();++i) {
            const auto& a=candidate.contacts[i];const auto& b=cached.candidate.contacts[i];
            if(!location(a.a,b.a)||!location(a.b,b.b))return false;
        }
        return true;
    };
    if(auto cached=std::find_if(_previewCache.begin(),_previewCache.end(),same);cached!=_previewCache.end()) {
        if(_cancel)_cancel->store(true);++_generation;_busy=false;_activity->finish(ConstraintGraphActivity::Operation,tr("Cancelled"));
        _firstPair.reset();_matching=false;_showCommitted->setChecked(false);
        installPreview(cached->job);return;
    }
    cancelComparison();_fiberView={};_patchView={};_placementOnly=true;
    _activeCandidate=candidate;
    _draft.reset();_preview={};_proposalView={};_firstPair.reset();_matching=false;_showCommitted->setChecked(false);display();
    QPointer<ConstraintGraphWorkspace> guard(this);
    const auto generation=_generation+1;
    run(tr("Fitting %1").arg(QString::fromStdString(candidate.id)),[guard,generation,c,n,prepared,candidate,whole](auto* cancel,auto progress){
        progress("Finding contacts for the selected attachment");
        auto edit=cg::proposeEdit(*c,prepared,candidate,whole,*n,cancel);
        progress("Building source preview");
        Job job;job.edit=edit;
        job.proposalView=cg::renderProposalSources(edit,candidate.id,cancel);
        if(guard&&!cancel->load())QMetaObject::invokeMethod(guard,[guard,generation,source=job.proposalView]{
            if(!guard||guard->_generation!=generation)return;
            guard->_proposalView=source;guard->display();guard->_view[1]->resetViewForCurrentContent(true);
            guard->_activity->update(ConstraintGraphActivity::Operation,guard->tr("Fitting attachment…"));
        },Qt::QueuedConnection);
        job.fit=cg::fitEdit(edit,cancel,progress);
        return job;
    },[this](Job job){finishPreview(std::move(job));});
}
void ConstraintGraphWorkspace::rebuildPreview(bool fitView){
    if(!_draft)return;
    _activeCandidate.reset();auto draft=*_draft;auto selected=_candidate>=0?_candidates[_candidate].id:_selected;
    if(reviewing()&&_review&&!_review->selected.empty()){_review->fibers.at(_review->selected).pending=draft;_dirty=true;}
    const bool settingsChanged=draft.settings.stiffness!=_committed.graph.settings.stiffness;
    fitPreview(std::move(draft),selected,fitView,settingsChanged||_fullPreview);
}
void ConstraintGraphWorkspace::fitPreview(cg::Graph draft,std::string selected,bool fitView,bool full){
    if(reviewing())full=false;
    if(!_prepared&&!full)return;
    cancelComparison();_placementOnly=true;_fiberView={};_patchView={};_preview={};updateComparison();
    auto prepared=_prepared;auto committed=_committed.graph;auto existingEdit=_previewEdit;const bool review=reviewing();
    std::optional<cg::ReviewFiber> references;
    if(review){references.emplace();references->decisions=_review->fibers.at(_review->selected).decisions;}
    run(full?tr("Refitting entire graph"):tr("Updating attachment"),[draft=std::move(draft),prepared,committed,selected,full,existingEdit,review,references](auto* cancel,auto progress)mutable{
        Job job;progress("Building source preview");
        if(full){
            job.fullFit=true;
            job.proposalView=cg::renderProposalSources(draft,committed,selected,cancel);
            job.fit=cg::fit(std::move(draft),cancel,progress);
        }else{
            auto edit=existingEdit?*existingEdit:cg::graphEdit(prepared,draft);
            if(references){progress("Matching accepted patch overlaps");cg::addReviewOverlapTargets(edit,*references,cancel);}
            job.edit=edit;
            job.proposalView=cg::renderProposalSources(edit,selected,cancel);
            job.fit=cg::fitEdit(edit,cancel,progress,review?cg::FitPolicy::FiberReview:cg::FitPolicy::Joint);
            if(review&&job.fit.valid)job.patchView=reviewPatchLayer(job.fit,cancel);
        }
        return job;
    },[this,fitView](Job job){finishPreview(std::move(job),fitView);});
}
void ConstraintGraphWorkspace::finishPreview(Job job,bool fitView){
    if(_activeCandidate){
        if(_previewCache.size()>=3)_previewCache.erase(_previewCache.begin());
        _previewCache.push_back({*_activeCandidate,_whole->isChecked(),job});
    }
    installPreview(std::move(job));if(fitView)_view[0]->resetViewForCurrentContent(true);
}
void ConstraintGraphWorkspace::installPreview(Job j){
    cancelComparison();
    _previewEdit=std::move(j.edit);
    _fullPreview=j.fullFit;
    _placementOnly=false;_activeCandidate.reset();
    j.fit.graph.settings.volumeScale=_committed.graph.settings.volumeScale;
    j.proposalView.graph.settings.volumeScale=_committed.graph.settings.volumeScale;
    _fiberView=std::move(j.fiberView);_patchView=std::move(j.patchView);
    _draft=j.fit.graph;_preview=std::move(j.fit);_proposalView=std::move(j.proposalView);
    auto message=_preview.valid?tr("Preview · correspondence RMS %1 vx · 95% spacing ratio %2").arg(_preview.correspondenceRms,0,'f',3).arg(_preview.stretchP95,0,'f',3):QString::fromStdString(_preview.message);
    if(reviewing()&&_preview.valid)message+=tr(" · %1 overlap samples · %2").arg(_previewEdit?_previewEdit->fixedTargets.size():0).arg(QString::fromStdString(_preview.message));
    report(message);
    if(reviewing()){_reviewPatchSurface=displaySurface(_patchView);captureReviewEdit();}
    display();refreshPairs();updateControls();
}
void ConstraintGraphWorkspace::commit(){
    if(reviewing()){decideReview(cg::ReviewStatus::Accepted);return;}
    if(_busy||_placementOnly||!_draft||!_preview.valid)return;
    cancelComparison();
    _prepareEdit=std::move(_previewEdit);_previewEdit.reset();
    _undo.push_back(_committed);if(_undo.size()>20)_undo.erase(_undo.begin());_redo.clear();_committed=_preview;_dirty=true;
    _draft.reset();_preview={};_proposalView={};_matching=false;_firstPair.reset();_candidate=-1;display();refreshPairs();search();
}
void ConstraintGraphWorkspace::cancelPreview(){
    if(_saving)return;
    cancelComparison();
    _previewEdit.reset();
    _fullPreview=false;
    if(_busy){_activity->update(ConstraintGraphActivity::Operation,tr("Stopping… waiting for the current step to return"));_activity->stopButton()->setEnabled(false);}
    if(_cancel)_cancel->store(true);++_generation;_busy=false;_awaitingProposals=false;_placementOnly=false;_activeCandidate.reset();_draft.reset();_preview={};_proposalView={};_matching=false;_firstPair.reset();_candidate=-1;_showCommitted->setChecked(false);display();refreshPairs();updateControls();
}
void ConstraintGraphWorkspace::navigateHistory(bool redo){
    if(reviewing()){reviewHistory(redo);return;}
    if(_saving)return;
    auto& from=redo?_redo:_undo;auto& to=redo?_undo:_redo;if(from.empty())return;cancelPreview();to.push_back(_committed);_committed=from.back();from.pop_back();_dirty=true;_selected.clear();setSettings(_committed.graph.settings);display();search();
}
void ConstraintGraphWorkspace::beginPair(){
    if(_saving)return;
    if(!_committed.valid)return;
    if(!_draft){if(_selected.empty()||!_prepared)return;_draft=_committed.graph;_previewEdit=cg::GraphEdit{_prepared};rebuildPreview();}
    _matching=true;_firstPair.reset();_splitter->widget(1)->show();overlays();report(tr("Click the feature in the proposal pane, then its match in the graph."));
}
void ConstraintGraphWorkspace::clicked(int pane,const QPointF& scene){
    if(_saving||(_busy&&(reviewing()||!_committed.graph.windingLayout.is_null())))return;
    const auto uvf=_view[pane]->sceneToSurfaceCoords(scene);cv::Vec2d uv{uvf[0],uvf[1]};
    if(_matching&&_draft){
        if(pane==1){auto p=cg::pick(_proposalView.graph,uv);if(p){_firstPair=*p;report(tr("Now click the same feature on the existing graph."));overlays();}return;}
        if(!_firstPair)return;
        const auto& graph=(!_showCommitted->isChecked()&&_preview.valid)?_preview.graph:_committed.graph;
        auto match=cg::pick(graph,uv,_selected);if(!match)match=cg::pick(graph,uv);if(!match||match->node==_firstPair->node)return;
        // User registration supersedes the proximity guesses for this pair.
        if(_fullPreview){
            const auto pair=std::minmax(match->node,_firstPair->node);
            _draft->links.eraseIf([&](const cg::Link& l){return !l.keypoint&&std::minmax(l.a.node,l.b.node)==pair;});
            _draft->links.push_back({*match,*_firstPair,true});
        }else{
            if(!_previewEdit)_previewEdit=cg::graphEdit(_prepared,*_draft);
            cg::addKeypoint(*_previewEdit,{*match,*_firstPair,true});_draft=cg::materialize(*_previewEdit);
        }
        _firstPair.reset();_matching=false;rebuildPreview();return;
    }
    if(reviewing()){
        if(pane==0)for(auto it=_reviewLayers.rbegin();it!=_reviewLayers.rend();++it)if(cg::pick(it->second.graph,uv,it->first)){reopenReview(it->first);break;}
        return;
    }
    if(pane!=0||!_committed.valid)return;
    const auto& showing=(!_showCommitted->isChecked()&&_preview.valid)?_preview:_committed;
    auto location=cg::pickRendered(showing,uv);
    if(!location||!_committed.graph.nodes.contains(location->node)) {
        if(!_committed.graph.windingLayout.is_null()){_selected.clear();overlays();updateControls();}
        return;
    }
    const auto committedUv=cg::placedLocation(_committed.graph,*location);
    _candidates.clear();cancelPreview();_selected=location->node;_selectionUv=committedUv;_bar->show();overlays();
    if(!_committed.graph.windingLayout.is_null()) {
        updateControls();report(tr("Selected %1").arg(QString::fromStdString(_committed.graph.nodes.at(_selected).source->path.string())));
    }else selectAvailable(true);
}
void ConstraintGraphWorkspace::quarantineSelected(){
    if(reviewing()){quarantineReview();return;}
    if(_busy||_saving||_path.isEmpty()||_committed.graph.windingLayout.is_null())return;
    const auto found=_committed.graph.nodes.find(_selected);
    if(found==_committed.graph.nodes.end()||found->second.source->fiber)return;
    const auto selected=_selected;const auto path=_path;auto snapshot=_committed;
    cancelComparison();stopDiscovery();
    run(tr("Quarantining selected patch"),[snapshot,path,selected](auto*,auto progress){
        Job job;job.fit=cg::quarantinePatches(snapshot,path.toStdString(),{selected},progress);return job;
    },[this](Job job){
        _committed=std::move(job.fit);_selected.clear();_draft.reset();_preview={};_proposalView={};
        _fiberView={};_patchView={};_previewCache.clear();_prepared.reset();_previewEdit.reset();_prepareEdit.reset();
        _undo.clear();_redo.clear();_available.clear();_candidates.clear();_candidate=-1;_dirty=false;
        display();refreshPairs();report(QString::fromStdString(_committed.message));
        if(_closing){_closing=false;emit closeReady();}
    },false);
    // File movement and publication must finish before closing/opening another
    // workspace. The ordinary activity panel shows progress without a modal.
    _saving=true;updateControls();
}
void ConstraintGraphWorkspace::refreshPairs(){
    _pairs->clear();const auto& graph=_draft?*_draft:_committed.graph;int index=0;
    for(size_t i=0;i<graph.links.size();++i)if(graph.links[i].keypoint){const auto& l=graph.links[i];auto* item=new QListWidgetItem(tr("%1: %2 ↔ %3").arg(++index).arg(QString::fromStdString(l.a.node),QString::fromStdString(l.b.node)),_pairs);item->setData(Qt::UserRole,qulonglong(i));}
}
void ConstraintGraphWorkspace::removePair(){
    if(_saving)return;
    auto* item=_pairs->currentItem();if(!item)return;auto index=item->data(Qt::UserRole).toULongLong();if(!_draft)_draft=_committed.graph;
    if(index<_draft->links.size()){
        if(_fullPreview)_draft->links.erase(_draft->links.cbegin()+index);
        else{
            if(!_previewEdit)_previewEdit=cg::graphEdit(_prepared,*_draft);
            cg::removeKeypoint(*_previewEdit,std::as_const(_draft->links)[index]);_draft=cg::materialize(*_previewEdit);
        }
    }
    rebuildPreview();
}
void ConstraintGraphWorkspace::display(){
    if(reviewing()){displayReview();return;}
    const auto& showing=(!_showCommitted->isChecked()&&_preview.valid)?_preview:_committed;
    _state->setSurface("constraint-graph",displaySurface(showing));_state->setSurface("constraint-proposal",displaySurface(_proposalView));
    _splitter->widget(1)->setVisible(_proposalView.valid);updateComparison();overlays();
}
void ConstraintGraphWorkspace::updateComparison(){
    if(reviewing()){displayReview();return;}
    bool fibers=false,patches=false;
    for(const auto& [id,p]:_preview.graph.nodes){fibers|=p.source->fiber;patches|=!p.source->fiber;}
    const bool available=_preview.valid&&!_placementOnly&&!_showCommitted->isChecked()&&fibers&&patches;
    _compareSources->setEnabled(available);_patchOpacity->setEnabled(available&&_compareSources->isChecked());
    if(available&&_compareSources->isChecked()&&_fiberView.valid&&_patchView.valid){
        _fiberView.graph.settings.volumeScale=_committed.graph.settings.volumeScale;
        _patchView.graph.settings.volumeScale=_committed.graph.settings.volumeScale;
        _view[0]->setSurfaceComparisonOpacity(_patchOpacity->value()/100.0f);
        _view[0]->setSurfaceComparison(displaySurface(_fiberView),displaySurface(_patchView));
    }else {
        _view[0]->setSurfaceComparison({},{});
        if(available&&_compareSources->isChecked()&&!_comparisonBusy)requestComparison();
    }
}
void ConstraintGraphWorkspace::cancelComparison(){
    if(_comparisonCancel)_comparisonCancel->store(true);
    ++_comparisonGeneration;_comparisonBusy=false;
    if(_activity)_activity->finish(ConstraintGraphActivity::Comparison,tr("Stopped"));
}
void ConstraintGraphWorkspace::requestComparison(){
    if(!_preview.valid||_comparisonBusy)return;
    _comparisonBusy=true;
    _activity->begin(ConstraintGraphActivity::Comparison,tr("Building fiber / patch comparison"));
    auto cancel=std::make_shared<std::atomic_bool>(false);_comparisonCancel=cancel;
    const auto generation=++_comparisonGeneration;auto graph=_preview.graph;
    auto* watcher=new QFutureWatcher<Job>(this);
    connect(watcher,&QFutureWatcher<Job>::finished,this,[this,watcher,generation]{
        auto job=watcher->result();watcher->deleteLater();
        if(generation!=_comparisonGeneration)return;
        _comparisonBusy=false;
        _activity->finish(ConstraintGraphActivity::Comparison,job.error.empty()&&job.fiberView.valid&&job.patchView.valid?tr("Ready"):tr("Unavailable"));
        if(!job.error.empty()||!job.fiberView.valid||!job.patchView.valid){
            QSignalBlocker blocker(_compareSources);_compareSources->setChecked(false);
            report(tr("Source comparison unavailable: %1").arg(QString::fromStdString(job.error.empty()?job.fiberView.message+" "+job.patchView.message:job.error)));
        }else{_fiberView=std::move(job.fiberView);_patchView=std::move(job.patchView);}
        updateComparison();
    });
    watcher->setFuture(QtConcurrent::run(&_comparisonPool,[graph=std::move(graph),cancel]{
        Job job;
        try{job.fiberView=cg::renderSourceLayer(graph,true,cancel.get());job.patchView=cg::renderSourceLayer(graph,false,cancel.get());}
        catch(const std::exception& error){job.error=error.what();}
        return job;
    }));
}
void ConstraintGraphWorkspace::initialView(){
    if(!_committed.valid)return;
    const auto view=vc3d::constraint_graph::initialView(_committed.graph);
    _view[0]->resetSurfaceOffsets();
    _view[0]->centerOnSurfacePoint(cv::Vec2f(view.center),false);
    _view[0]->adjustZoomByFactor(float(view.scale/_view[0]->getCurrentScale()));
}
void ConstraintGraphWorkspace::overlays(){
    const bool profile=qEnvironmentVariableIsSet("VC3D_PROFILE_CONSTRAINT_GRAPH");
    QElapsedTimer timer;if(profile)timer.start();
    qint64 pathNs=0,contactNs=0;
    for(int pane=0;pane<2;++pane){
        // A null root means the viewer cleared the scene and already deleted
        // all its children and markers. QPointer makes that reset observable.
        if(_overlayRoot[pane])for(auto* item:_items[pane])delete item;
        _items[pane].clear();_markers[pane].clear();
        if(pane==0){_highlightUv.clear();_highlightLocations.clear();if(_highlights)_highlights->setPointCommands({});}
        auto* scene=_view[pane]->graphicsView()->scene();if(!scene)continue;
        cg::FitResult reviewResult;cg::Graph reviewCommitted;
        if(reviewing()){reviewResult.graph=reviewDisplayGraph();reviewResult.valid=_committed.valid;reviewCommitted=reviewResult.graph;
            if(const auto* f=displayedReviewFiber();f&&f->current>=0)reviewCommitted.nodes.erase(f->proposals[f->current].id);}
        const auto& result=reviewing()&&pane==0?reviewResult:pane==1?_proposalView:(!_showCommitted->isChecked()&&_preview.valid?_preview:_committed);if(!result.valid)continue;
        if(!_overlayRoot[pane]){auto* root=new OverlayBatchItem;root->setFlag(QGraphicsItem::ItemHasNoContents);root->setZValue(20);scene->addItem(root);_overlayRoot[pane]=root;}
        if(pane==0&&!_highlights){_highlights=new OverlayBatchItem;_highlights->setZValue(25);scene->addItem(_highlights);}
        auto point=[](cv::Vec2d uv){return QPointF(float(uv[0]),float(uv[1]));};
        auto mark=[&](const cg::Location& l,QColor color,QString label={}){if(!result.graph.nodes.contains(l.node))return;auto uv=cg::placedLocation(result.graph,l);
            auto* dot=scene->addEllipse(-3,-3,6,6,QPen(color),QBrush(color));dot->setFlag(QGraphicsItem::ItemIgnoresTransformations);dot->setZValue(30);_items[pane].push_back(dot);_markers[pane].push_back({dot,uv,{}});
            if(!label.isEmpty()){auto* text=scene->addSimpleText(label);text->setBrush(color);text->setFlag(QGraphicsItem::ItemIgnoresTransformations);text->setZValue(31);_items[pane].push_back(text);_markers[pane].push_back({text,uv,{5,5}});}};
        const auto pathStart=profile?timer.nsecsElapsed():0;
        for(const auto& [id,p]:result.graph.nodes){
            if(!(result.graph.windingLayout.is_object()&&p.source->fiber)&&!vc3d::constraint_graph::showBoundary(id,reviewing()?reviewCommitted:_committed.graph,_showAllBoundaries->isChecked(),_selected))continue;
            QPainterPath path;const auto& n=*p.source;
            if(n.fiber){for(int i=0;i<n.cols;++i){auto q=point(p.uv[(n.rows/2)*n.cols+i]);if(i==0)path.moveTo(q);else path.lineTo(q);}}
            else{std::map<std::pair<int,int>,int> edges;for(const auto& t:n.triangles)for(int i=0;i<3;++i)++edges[std::minmax(t[i],t[(i+1)%3])];
                for(const auto& [edge,count]:edges)if(count==1){path.moveTo(point(p.uv[edge.first]));path.lineTo(point(p.uv[edge.second]));}}
            const auto selected=result.graph.nodes.find(_selected);
            const bool highlight=id==_selected||(selected!=result.graph.nodes.end()&&!n.fiber&&!n.path.empty()&&n.path==selected->second.source->path);
            QPen pen(highlight?QColor("#ffd166"):n.fiber?QColor("#50dfac"):_patchColors.color(id));pen.setCosmetic(true);pen.setWidthF(highlight?2.5:1.0);
            if(reviewing()&&n.fiber){auto color=pen.color();color.setAlphaF(_fiberLineOpacity->value()/100.0);pen.setColor(color);pen.setWidthF(_fiberLineWidth->value());}
            auto* item=new QGraphicsPathItem(path,_overlayRoot[pane].data());item->setPen(pen);item->setAcceptedMouseButtons(Qt::NoButton);_items[pane].push_back(item);
        }
        if(profile)pathNs+=timer.nsecsElapsed()-pathStart;
        const auto contactStart=profile?timer.nsecsElapsed():0;
        if(pane==0) {
            if(reviewing()){if(_candidate>=0&&_candidate<int(_candidates.size()))appendHighlights({_candidates[_candidate]});}
            else appendHighlights(_available);
        }
        if(profile)contactNs+=timer.nsecsElapsed()-contactStart;
        int pair=0;const auto& links=_draft?_draft->links:result.graph.links;
        for(const auto& link:links)if(link.keypoint){++pair;mark(link.a,QColor("#f97cff"),QString::number(pair));mark(link.b,QColor("#f97cff"),QString::number(pair));}
        if(_firstPair)mark(*_firstPair,QColor("#ffffff"),"+");
        updateOverlayTransform(pane,true);
    }
    positionControlBar();
    if(profile)qInfo().nospace()<<"constraint_graph_overlay rebuild_ms="<<timer.nsecsElapsed()/1e6<<" paths_ms="<<pathNs/1e6<<" contacts_ms="<<contactNs/1e6;
}
void ConstraintGraphWorkspace::appendHighlights(const std::vector<cg::Candidate>& candidates){
    const auto& graph=(!_showCommitted->isChecked()&&_preview.valid)?_preview.graph:_committed.graph;
    for(const auto& candidate:candidates)for(const auto& contact:candidate.contacts) {
        const auto& l=contact.a;
        if(!graph.nodes.contains(l.node)||!_highlightLocations.emplace(l.node,l.lineIndex,l.grid[0],l.grid[1]).second)continue;
        _highlightUv.push_back(cg::placedLocation(graph,l));
    }
}
void ConstraintGraphWorkspace::updateOverlayTransform(int pane,bool force){
    if(!_overlayRoot[pane])return;
    const auto transform=surfaceToSceneTransform(_view[pane]);
    if(!force&&_overlayRoot[pane]->transform()==transform)return;
    const bool profile=qEnvironmentVariableIsSet("VC3D_PROFILE_CONSTRAINT_GRAPH");
    QElapsedTimer timer;if(profile)timer.start();
    _overlayRoot[pane]->setTransform(transform);
    for(const auto& marker:_markers[pane])marker.item->setPos(transform.map(QPointF(float(marker.uv[0]),float(marker.uv[1])))+marker.offset);
    if(pane==0&&_highlights){
        const auto bounds=QRectF(_view[pane]->graphicsView()->viewport()->rect()).adjusted(-3,-3,3,3);
        std::vector<QPointF> points;points.reserve(_highlightUv.size());
        for(const auto& uv:_highlightUv){
            const auto p=transform.map(QPointF(float(uv[0]),float(uv[1])));
            if(!bounds.contains(p))continue;
            points.push_back(p);
        }
        QPen pen(QColor("#ffba45"));pen.setWidthF(4);pen.setCapStyle(Qt::RoundCap);
        if(reviewing()){auto color=pen.color();color.setAlphaF(_fiberLineOpacity->value()/100.0);pen.setColor(color);pen.setWidthF(_fiberLineWidth->value());}
        _highlights->setUniformPoints(std::move(points),pen);
    }
    if(profile)qInfo().nospace()<<"constraint_graph_overlay camera_ms="<<timer.nsecsElapsed()/1e6<<" cached_contacts="<<_highlightUv.size();
}
void ConstraintGraphWorkspace::positionControlBar(){
    if(!_bar)return;
    if(!_selected.empty()){
        auto* view=_view[0]->graphicsView();auto p=view->viewport()->mapTo(this,view->mapFromScene(_view[0]->surfaceCoordsToScene(_selectionUv[0],_selectionUv[1])));_bar->adjustSize();
        _bar->placeNear({p.x()-_bar->width()/2,p.y()+18});
    }
}
void ConstraintGraphWorkspace::updateControls(){
    const bool fixed=!_committed.graph.windingLayout.is_null();
    for(auto* control:std::vector<QWidget*>{_radius,_stiffness,_component,_discoveryStride})control->setEnabled(!_saving&&!fixed);
    _volumeScale->setEnabled(!_saving);
    const auto selected=_committed.graph.nodes.find(_selected);
    const bool patch=selected!=_committed.graph.nodes.end()&&!selected->second.source->fiber;
    _quarantine->setVisible(fixed);_quarantine->setEnabled(fixed&&patch&&!_busy&&!_saving&&!_path.isEmpty());
    _quarantine->setToolTip(_path.isEmpty()?tr("Open a saved winding workspace to quarantine a patch."):tr("Move the source patch to %1 and remove all its components from this workspace.").arg(QString::fromStdString(cg::quarantineDirectory(_path.toStdString()).string())));
    for(auto* control:std::vector<QWidget*>{_pair,_commit,_previous,_next,_compareSources,_opacityControls})control->setVisible(!fixed);
    _cancelButton->setText(fixed?tr("Clear selection"):tr("Cancel"));
    _commit->setEnabled(!_busy&&!_placementOnly&&_draft&&_preview.valid);_pair->setEnabled(!fixed&&!_busy&&(_proposalView.valid||!_selected.empty()));
    _previous->setEnabled(!_candidates.empty());_next->setEnabled(!_candidates.empty());_cancelButton->setEnabled(!_saving&&(fixed?!_selected.empty():_draft.has_value()||_busy));
    _whole->setVisible(_candidate>=0&&_candidate<int(_candidates.size())&&_candidates[_candidate].fiber);
    _candidateLabel->setWordWrap(fixed);
    if(fixed&&selected!=_committed.graph.nodes.end()) {
        _candidateLabel->setText(QString::fromStdString(selected->second.source->path.filename().string()));
        _candidateLabel->setToolTip(QString::fromStdString(selected->second.source->path.string()));
    }
    else if(_candidate>=0&&_candidate<int(_candidates.size()))_candidateLabel->setText(tr("%1/%2 · %3").arg(_candidate+1).arg(_candidates.size()).arg(QString::fromStdString(_candidates[_candidate].id)));
    else if(_draft&&_selected.empty())_candidateLabel->setText(tr("Graph fit"));
    else _candidateLabel->setText(_awaitingProposals?tr("Searching…"):tr("%1 proposals").arg(_candidates.size()));
    _bar->setVisible(!_selected.empty()||_draft.has_value());
    if(_mode)updateReviewControls();
    _bar->adjustSize();positionControlBar();
}
bool ConstraintGraphWorkspace::eventFilter(QObject* object,QEvent* event){
    if(event->type()==QEvent::KeyPress){auto* key=static_cast<QKeyEvent*>(event);
        if(reviewing()&&(key->key()==Qt::Key_Return||key->key()==Qt::Key_Enter)){decideReview(cg::ReviewStatus::Accepted);return true;}
        if(reviewing()&&key->key()==Qt::Key_Delete){decideReview(cg::ReviewStatus::Rejected);return true;}
        if(key->key()==Qt::Key_Tab||key->key()==Qt::Key_Backtab){selectCandidate(key->key()==Qt::Key_Backtab||key->modifiers().testFlag(Qt::ShiftModifier)?-1:1);return true;}
        if(key->key()==Qt::Key_Escape){if(!_saving&&!_committed.graph.windingLayout.is_null())_selected.clear();cancelPreview();return true;}
    }
    return QMainWindow::eventFilter(object,event);
}
void ConstraintGraphWorkspace::save(bool choose){
    if(reviewing()){saveReview(choose);return;}
    if(!_committed.valid||_busy)return;
    if(!_committed.graph.windingLayout.is_null()){report(tr("This layout is already saved in its workspace file. Use Export tifxyz to export the displayed surface."));return;}
    QString path=_path;if(choose||path.isEmpty())path=QFileDialog::getSaveFileName(this,tr("Save constraint graph"),path,tr("Graph session (*.json)"));
    if(path.isEmpty()){_closing=false;return;}auto snapshot=_committed;
    run(tr("Saving graph and tifxyz"),[snapshot,path](auto*,auto){cg::saveSession(snapshot,path.toStdString());return Job{};},[this,path](Job){_path=path;_dirty=false;report(tr("Saved graph and tifxyz"));if(_closing){_closing=false;_draft.reset();_preview={};_proposalView={};emit closeReady();}},false);
    _saving=true;updateControls();
}
void ConstraintGraphWorkspace::openSession(const QString& path){
    if(_saving)return;
    if(_dirty&&QMessageBox::question(this,tr("Open graph"),tr("Discard unsaved graph changes?"))!=QMessageBox::Yes)return;
    stopDiscovery();report(tr("Opening saved graph…"));
    run(tr("Opening saved graph"),[path](auto* cancel,auto progress){progress("Reading session and checking saved surface");Job j;
        {std::ifstream input(path.toStdString());auto document=nlohmann::json::parse(input);
            if(document.value("type",std::string{})=="vc3d_fiber_review"||cg::isLinksDocument(document))j.review=cg::reviewFromDocument(document);}
        if(j.review){
            const auto& s=j.review->settings;
            try{vc::lasagna::LasagnaDatasetOpenOptions options;options.workingToBaseScale=s.normalScale;
                j.dataset=std::make_shared<vc::lasagna::LasagnaDataset>(vc::lasagna::LasagnaDataset::openLocation(s.normalManifest,options));
                j.normals=std::make_shared<vc::lasagna::LasagnaNormalSampler>(*j.dataset);j.catalog=std::make_shared<cg::Catalog>(cg::loadCatalog(s,cancel,progress));
            }catch(const std::exception& e){if(cancel&&cancel->load())throw;j.notice="Saved review loaded; sources unavailable: "+std::string(e.what());}
            return j;
        }
        j.fit=cg::loadSession(path.toStdString());auto s=j.fit.graph.settings;
        if(!j.fit.graph.windingLayout.is_null()){j.notice=j.fit.message;return j;}
        try {
            vc::lasagna::LasagnaDatasetOpenOptions options;options.workingToBaseScale=s.normalScale;
            j.dataset=std::make_shared<vc::lasagna::LasagnaDataset>(vc::lasagna::LasagnaDataset::openLocation(s.normalManifest,options));j.normals=std::make_shared<vc::lasagna::LasagnaNormalSampler>(*j.dataset);j.catalog=std::make_shared<cg::Catalog>(cg::loadCatalog(s,cancel,progress));
        }catch(const std::exception& e){if(cancel&&cancel->load())throw;j.notice="Saved graph loaded; source catalog unavailable: "+std::string(e.what());}
        if(j.normals) {
            auto graph=j.fit.graph;
            try {
                progress("Updating saved fiber strips");
                if(cg::regenerateFiberStrips(graph,*j.normals,cancel)) {
                    auto rendered=cg::renderPlacement(graph,cancel,progress);
                    if(!rendered.valid)throw std::runtime_error(rendered.message);
                    j.fit=std::move(rendered);j.regeneratedStrips=true;
                }
            }catch(const std::exception& e){if(cancel&&cancel->load())throw;j.notice="Could not update fiber strips; saved geometry retained: "+std::string(e.what());}
        }
        return j;
    },[this,path](Job j){if(j.review){auto review=std::move(*j.review);installReview(std::move(review),path,std::move(j));return;}
        _mode->setCurrentIndex(0);_review.reset();_reviewLayers.clear();_reviewContextSurfaces.clear();
        if(!j.fit.valid){report(QString::fromStdString(j.fit.message));return;}_committed=std::move(j.fit);_catalog=j.catalog;_dataset=j.dataset;_normals=j.normals;_path=path;_dirty=j.regeneratedStrips;_draft.reset();_preview={};_proposalView={};_undo.clear();_redo.clear();_selected.clear();_candidates.clear();_candidate=-1;setSettings(_committed.graph.settings);
        {QSignalBlocker blocker(_component);_component->clear();if(_catalog)for(size_t i=0;i<_catalog->components.size();++i){_component->addItem(tr("%1 — %2 fibers").arg(i+1).arg(_catalog->components[i].size()));if(std::find(_catalog->components[i].begin(),_catalog->components[i].end(),_committed.graph.seed)!=_catalog->components[i].end())_component->setCurrentIndex(int(i));}}
        refreshAudit();display();refreshPairs();initialView();if(j.notice.empty())search();else report(QString::fromStdString(j.notice));});
}
bool ConstraintGraphWorkspace::prepareToClose(){
    if(_saving){_closing=true;return false;}
    if(!_dirty&&!_draft)return true;
    auto answer=QMessageBox::question(this,tr("Constraint Graph"),reviewing()?tr("Save fiber review decisions and pending edits before closing?"):tr("Save the committed graph before closing? Uncommitted proposals will be discarded."),QMessageBox::Save|QMessageBox::Discard|QMessageBox::Cancel);
    if(answer==QMessageBox::Discard)return true;if(answer==QMessageBox::Cancel)return false;
    if(reviewing())captureReviewEdit();
    if(_busy)cancelPreview();_closing=true;save();return false;
}
