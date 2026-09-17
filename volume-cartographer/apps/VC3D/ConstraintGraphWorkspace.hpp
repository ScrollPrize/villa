#pragma once
#include "vc/constraints/ConstraintGraph.hpp"
#include "ConstraintGraphPatchColors.hpp"
#include <QMainWindow>
#include <QThreadPool>
#include <QPointer>
#include <functional>
#include <memory>
#include <set>
#include <tuple>

class ConstraintGraphSelectionBar;
class ConstraintGraphActivity;
class CState;
class ViewerManager;
class VolumeViewerBase;
class QComboBox;
class QLineEdit;
class QDoubleSpinBox;
class QCheckBox;
class QPushButton;
class QLabel;
class QListWidget;
class QGraphicsItem;
class OverlayBatchItem;
class QSplitter;
class QSlider;
class Volume;
namespace vc::lasagna { class LasagnaDataset;class LasagnaNormalSampler; }

class ConstraintGraphWorkspace : public QMainWindow {
    Q_OBJECT
public:
    explicit ConstraintGraphWorkspace(CState* mainState,QWidget* parent=nullptr);
    ~ConstraintGraphWorkspace() override;
    ViewerManager* viewerManager() const {return _manager.get();}
    QComboBox* volumeSelectionControl() const {return _volume;}
    void synchronizeVolume(const std::shared_ptr<Volume>&);
    bool prepareToClose();
    // Used by the ordinary workspace UI and integration harnesses alike.
    void openSession(const QString&);
    void start(const vc::constraints::Settings&);
    bool busy() const {return _busy;}
    const vc::constraints::FitResult& committed() const {return _committed;}

signals:
    void closeReady();
    void operationFinished(bool success);

private:
    struct Job {
        std::shared_ptr<vc::constraints::Catalog> catalog;
        std::shared_ptr<vc::lasagna::LasagnaDataset> dataset;
        std::shared_ptr<vc::lasagna::LasagnaNormalSampler> normals;
        vc::constraints::FitResult fit,proposalView,fiberView,patchView;
        std::shared_ptr<const vc::constraints::PreparedGraph> prepared;
        std::optional<vc::constraints::GraphEdit> edit;
        bool fullFit=false;
        std::vector<vc::constraints::Candidate> candidates;
        std::string error;
        std::string notice;
    };
    using Worker=std::function<Job(const std::atomic_bool*,vc::constraints::Progress)>;
    void run(const QString& title,Worker,std::function<void(Job)>,bool stoppable=true);
    vc::constraints::Settings settings() const;
    void setSettings(const vc::constraints::Settings&);
    void seed(int component);
    void search();
    void stopDiscovery();
    void selectAvailable(bool preview);
    bool candidateNearSelection(const vc::constraints::Candidate&) const;
    void receiveDiscovery(std::vector<vc::constraints::Candidate>);
    void selectCandidate(int delta);
    void rebuildPreview(bool fitView=false);
    void installPreview(Job);
    void fitPreview(vc::constraints::Graph, std::string, bool fitView=false, bool full=false);
    void finishPreview(Job, bool fitView=false);
    void cancelComparison();
    void requestComparison();
    void commit();
    void cancelPreview();
    void navigateHistory(bool redo);
    void save(bool choosePath=false);
    void clicked(int pane,const QPointF& scene);
    void beginPair();
    void removePair();
    void refreshPairs();
    void display();
    void updateComparison();
    void overlays();
    void updateOverlayTransform(int pane, bool force=false);
    void positionControlBar();
    void initialView();
    void updateControls();
    void refreshNormals();
    void report(const QString&);
    void refreshAudit();
    bool eventFilter(QObject*,QEvent*) override;

    CState* _main;
    CState* _state;
    std::unique_ptr<ViewerManager> _manager;
    VolumeViewerBase* _view[2]{};
    QSplitter* _splitter{};
    ConstraintGraphSelectionBar* _bar{};
    ConstraintGraphActivity* _activity{};
    QComboBox* _volume{}, *_normal{}, *_component{}, *_discoveryStride{};
    QLineEdit* _fibers{}, *_patches{};
    QDoubleSpinBox* _radius{}, *_fiberScale{}, *_patchScale{}, *_normalScale{}, *_volumeScale{}, *_stiffness{};
    QCheckBox* _whole{}, *_showCommitted{}, *_showAllBoundaries{};
    QCheckBox* _compareSources{};
    QSlider* _patchOpacity{};
    QPushButton* _commit{}, *_pair{}, *_previous{}, *_next{}, *_cancelButton{};
    QLabel* _candidateLabel{}, *_audit{};
    QListWidget* _pairs{};
    std::vector<QGraphicsItem*> _items[2];
    QPointer<OverlayBatchItem> _overlayRoot[2];
    struct Marker { QGraphicsItem* item; cv::Vec2d uv; QPointF offset; };
    std::vector<Marker> _markers[2];
    QPointer<OverlayBatchItem> _highlights;
    std::vector<cv::Vec2d> _highlightUv;
    std::set<std::tuple<std::string,int64_t,double,double>> _highlightLocations;
    void appendHighlights(const std::vector<vc::constraints::Candidate>&);
    std::shared_ptr<vc::constraints::Catalog> _catalog;
    std::shared_ptr<vc::lasagna::LasagnaDataset> _dataset;
    std::shared_ptr<vc::lasagna::LasagnaNormalSampler> _normals;
    vc::constraints::FitResult _committed, _preview, _proposalView;
    vc::constraints::FitResult _fiberView, _patchView;
    std::shared_ptr<const vc::constraints::PreparedGraph> _prepared;
    std::optional<vc::constraints::GraphEdit> _previewEdit, _prepareEdit;
    std::optional<vc::constraints::Graph> _draft;
    std::vector<vc::constraints::FitResult> _undo,_redo;
    std::vector<vc::constraints::Candidate> _candidates, _available;
    QThreadPool _discoveryPool;
    uint64_t _discoveryGeneration=0;
    std::shared_ptr<std::atomic_bool> _discoveryCancel;
    bool _discovering=false,_awaitingProposals=false;
    int _candidate=-1;
    ConstraintGraphPatchColors _patchColors;
    std::string _selected;
    cv::Vec2d _selectionUv{0,0};
    std::optional<vc::constraints::Location> _firstPair;
    bool _matching=false,_busy=false,_dirty=false,_closing=false;
    bool _saving=false, _placementOnly=false;
    bool _fullPreview=false;
    struct CachedPreview { vc::constraints::Candidate candidate; bool whole; Job job; };
    std::vector<CachedPreview> _previewCache;
    std::optional<vc::constraints::Candidate> _activeCandidate;
    QString _path;
    QThreadPool _pool;
    QThreadPool _comparisonPool;
    std::shared_ptr<std::atomic_bool> _comparisonCancel;
    uint64_t _comparisonGeneration=0;
    bool _comparisonBusy=false;
    uint64_t _generation=0;
    std::shared_ptr<std::atomic_bool> _cancel;
};
