#pragma once
#include "vc/constraints/ConstraintGraph.hpp"
#include "vc/constraints/FiberReview.hpp"
#include "vc/constraints/AutoLink.hpp"
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
class QSpinBox;
class QCheckBox;
class QPushButton;
class QLabel;
class QListWidget;
class QGraphicsItem;
class OverlayBatchItem;
class QSplitter;
class QSlider;
class Volume;
class Surface;
class QToolBar;
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
        bool regeneratedStrips=false;
        std::optional<vc::constraints::ReviewFiber> regeneratedReviewFiber;
        std::optional<vc::constraints::FiberReview> review;
        std::vector<std::pair<std::string,vc::constraints::FitResult>> reviewLayers;
        // Auto-link results: replaced fibers plus the advanced order counter.
        std::vector<std::pair<std::string,vc::constraints::ReviewFiber>> autoLinked;
        uint64_t nextOrder=0;
        // Layout changed saved decisions (a stored link could not be fitted).
        bool decisionsChanged=false;
    };
    using Worker=std::function<Job(const std::atomic_bool*,vc::constraints::Progress)>;
    // failed runs on the UI thread when the worker throws; superseded jobs call neither.
    void run(const QString& title,Worker,std::function<void(Job)>,bool stoppable=true,std::function<void()> failed={});
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
    void quarantineSelected();
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
    bool reviewing() const;
    void setupReview(QToolBar*);
    void startReview(const vc::constraints::Settings&);
    void selectReviewFiber(const std::string&);
    void refreshReviewList();
    void searchReview();
    void previewReview(int);
    void decideReview(vc::constraints::ReviewStatus);
    std::optional<vc::constraints::PatchEntry> reviewPatch() const;
    void quarantineReview();
    void reopenReview(const std::string&);
    // Geometric auto-linking of the selected fiber (or every fiber): links are
    // fitted and accepted, the rest rejected, all marked automatic for review.
    void autoLinkReview(bool allFibers);
    void reopenNextAutoRejected();
    std::vector<std::string> autoRejectedPatches() const;
    void rememberReview();
    void reviewHistory(bool);
    void saveReview(bool);
    void displayReview();
    void updateReviewControls();
    void installReview(vc::constraints::FiberReview, const QString&, Job);
    void rebuildReviewLayers();
    void captureReviewEdit();
    // The fiber whose layout is on screen; differs from the selection while a
    // newly selected fiber is still being laid out.
    const vc::constraints::ReviewFiber* displayedReviewFiber() const;
    vc::constraints::Graph reviewDisplayGraph() const;
    static std::shared_ptr<QuadSurface> displaySurface(const vc::constraints::FitResult&);
    // The review fit already validated and returned the rendered patch layer.
    static vc::constraints::FitResult reviewPatchLayer(const vc::constraints::FitResult&,const std::atomic_bool*);
    // Rendered accepted layers, keyed by fiber and patch, valid while the
    // association keeps its placement identity and render settings.
    struct ReviewLayerEntry {
        vc::constraints::Placement placement;
        vc::constraints::FitResult layer;
        uint64_t used=0;
        size_t bytes=0;
    };
    using ReviewLayerCache=std::map<std::pair<std::string,std::string>,ReviewLayerEntry>;
    void rememberReviewLayers(const std::string& fiber,const std::vector<std::pair<std::string,vc::constraints::FitResult>>&);
    static std::vector<std::pair<std::string,vc::constraints::FitResult>> renderReviewLayers(const std::string& fiber,
        const std::vector<std::pair<std::string,vc::constraints::ReviewDecision>>& accepted,const vc::constraints::Settings&,
        const ReviewLayerCache&,const std::atomic_bool* cancel,const vc::constraints::Progress&);
    struct ReviewSnapshot { std::string fiber; vc::constraints::ReviewFiber state; uint64_t nextOrder=0; };

    QComboBox* _mode{};
    QListWidget* _reviewFibers{}, *_reviewDecisions{};
    QWidget* _reviewPanel{};
    QSlider* _committedOpacity{};
    QSpinBox* _committedTint{};
    std::set<std::string> _reviewAvailablePatches;
    QSpinBox* _fiberLineOpacity{};
    QDoubleSpinBox* _fiberLineWidth{};
    bool _centerFirstReviewProposal=false;
    QCheckBox* _reverseReview{};
    QPushButton* _rejectReview{}, *_skipReview{}, *_alternateReview{};
    QPushButton* _autoLinkFiber{}, *_autoLinkAll{}, *_nextAutoRejected{};
    QDoubleSpinBox* _autoLinkContact{};
    std::optional<vc::constraints::FiberReview> _review;
    std::vector<ReviewSnapshot> _reviewUndo,_reviewRedo;
    ReviewLayerCache _reviewLayerCache;
    uint64_t _reviewLayerClock=0;
    size_t _reviewLayerCacheBytes=0;
    std::vector<std::pair<std::string,vc::constraints::FitResult>> _reviewLayers;
    std::shared_ptr<Surface> _reviewFiberSurface, _reviewPatchSurface;
    std::vector<std::shared_ptr<Surface>> _reviewContextSurfaces;
    std::map<std::string,std::pair<std::weak_ptr<QuadSurface>,std::shared_ptr<Surface>>> _reviewSurfaceCache;

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
    QPushButton* _commit{}, *_pair{}, *_previous{}, *_next{}, *_cancelButton{}, *_quarantine{};
    QWidget* _opacityControls{};
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
