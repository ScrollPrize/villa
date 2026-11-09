#pragma once

// Ink label sessions on the segment view: which store is open, the paint
// tools, the keys, autosave, and the bridge-facing operations. The GUI (the
// label section of the Ink Detection dock, the keys and the mouse) and the
// agent bridge both drive this one controller.

#include "ink_labels/InkLabelAssist.hpp"
#include "ink_labels/InkLabelImport.hpp"
#include "ink_labels/InkLabelRaster.hpp"
#include "ink_labels/InkLabelStore.hpp"

#include "vc/core/util/RenderCanvas.hpp"

#include <QColor>
#include <QDateTime>
#include <QFuture>
#include <QHash>
#include <QJsonObject>
#include <QList>
#include <QObject>
#include <QPointer>
#include <QPointF>
#include <QRect>
#include <QSet>
#include <QString>
#include <QStringList>

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

class CState;
class InkLabelOverlayController;
class QAction;
class QLockFile;
class QShortcut;
class QTimer;
class QWidget;
class QuadSurface;
class SpiralBrushCursorWidget;
class ViewerManager;
class VolumeViewerBase;

class InkLabelController : public QObject
{
    Q_OBJECT

public:
    enum class Tool : uint8_t { Brush, Polygon, Fill };

    struct Deps {
        CState* state = nullptr;
        ViewerManager* viewers = nullptr;
        // The main window: the label keys are window shortcuts on it, so they
        // work wherever focus is in the window (a dock, the dock bar's popup,
        // the status bar), and dialogs are parented to it.
        QWidget* shortcutHost = nullptr;
        std::function<void(const QString&, int)> status;
        // Segmentation editing or approval painting is on.
        std::function<bool()> otherSurfaceToolActive;
        // Segmentation editing alone is on (label mode is refused then).
        std::function<bool()> segmentationEditing;
        std::function<void()> stopSurfacePainting;
        // The segment workspace is the current workspace. The keys are off
        // in other workspaces, which bind some of the same letters. Unset
        // means always.
        std::function<bool()> workspaceActive;
        // A permanent line in the status bar while label mode is on, for
        // example "Ink labels: ink, brush r 12"; an empty text hides it.
        std::function<void(const QString&)> modeLine;
    };

    // Why an open failed, for callers that map failures to error codes.
    enum class OpenError : uint8_t {
        None,
        NoProject,
        NoSegment,
        InvalidScale,
        LayerLocked,      // the store was saved for another canvas or segment
        AutosavePending,  // RecoverPolicy::Fail and a newer checkpoint exists
        SaveInFlight,
        ReadFailed,
        LockedByOther,    // readOnlyIfLocked is false and another VC3D holds it
        Busy,             // another open is waiting for an answer, or the segment changed meanwhile
        NoVersion,        // the requested label version does not exist
    };

    struct OpenOptions {
        vc3d::ink_labels::RecoverPolicy recover = vc3d::ink_labels::RecoverPolicy::Ask;
        bool readOnlyIfLocked = true;
        // Label version: 0 opens the newest one in the store (1 if none).
        int version = 0;
        // Allow a version newer than every existing one: a new, empty version.
        bool createVersion = false;
    };

    struct OpenOutcome {
        bool ok = false;
        bool readOnly = false;
        bool locked = false;
        OpenError error = OpenError::None;
        QString message;
        bool restoredAutosave = false;
        QStringList warnings;
        // LayerLocked: the canvas the labels were saved for, and the
        // segment's canvas now.
        QString expected, got;
        QString autosaveAtUtc;
    };

    struct SaveOptions {
        bool overwriteExternalEdits = false;
    };

    struct SessionInfo {
        vc::RenderCanvas canvas;
        vc3d::ink_labels::StorePaths paths;
        std::string segment;
        std::string segmentUuid;
        std::filesystem::path segmentDir;
        bool readOnly = false, locked = false, dirty = false;
        uint64_t stateSerial = 0;
        std::array<uint64_t, 8> counts{};
        // Read-only or locked: why, in a sentence for the banner.
        QString lockReason;
        // Locked: the canvas (or segment uuid) the labels were saved for, and
        // the segment's now.
        QString lockExpected, lockGot;
    };

    // Fill, suggestion and cleanup settings (the Assist section of the dock).
    struct AssistSettings {
        // Fill: what decides the region, and for the prediction guide which
        // side of the threshold passes.
        vc3d::ink_labels::GuideKind guide = vc3d::ink_labels::GuideKind::Prediction;
        uint8_t threshold = 128;
        bool above = true;
        int connectivity = 4;
        bool unlabeledOnly = false;
        // Suggestions: median kernel (1, 3 or 5) and smallest component.
        int median = 3;
        int minArea = 50;
        // Cleanup size n in label px.
        int cleanupSize = 3;
        bool operator==(const AssistSettings&) const = default;
    };

    // What ink_labels.read writes: the raw values (bits 1, 2, 4) or one bit
    // as 0/255.
    enum class ReadFormat : uint8_t { Values, Ink, Supervision, Validation };
    struct ReadOutcome {
        bool ok = false;
        QString error;
        cv::Rect rect;  // clipped to the canvas; empty when nothing was inside
        vc3d::ink_labels::LabelCounts counts;
    };

    struct LastSave {
        QDateTime atUtc;
        std::vector<std::pair<vc3d::ink_labels::LabelKind, std::filesystem::path>> files;
        std::map<vc3d::ink_labels::LabelKind, std::string> sha256;
    };

    InkLabelController(Deps deps, QObject* parent = nullptr);
    ~InkLabelController() override;

    [[nodiscard]] bool labelModeEnabled() const { return _labelMode; }
    // Retained until the next successful open/enable, including when no session exists.
    [[nodiscard]] const QString& labelModeRefusal() const { return _labelModeRefusal; }
    // Turning it on opens a session for the active segment if needed. Refused
    // (with a status message) without a project or segment, or while
    // segmentation editing is on.
    void setLabelModeEnabled(bool enabled);

    // (Overloads rather than "= {}": a nested struct's member initializers
    // are not usable in default arguments inside its own class.)
    OpenOutcome openForActiveSegment();
    OpenOutcome openForActiveSegment(const OpenOptions& options);
    // Commits any open stroke, checkpoints unsaved edits as an autosave, and
    // releases the store. The undo history ends with the session; reopening
    // restores the labels from the checkpoint with a fresh history.
    void closeSession();
    QFuture<vc3d::ink_labels::SaveOutcome> save();
    QFuture<vc3d::ink_labels::SaveOutcome> save(const SaveOptions& options);
    // Writes all three kinds as version max+1 (validation all zero if
    // unused), so the trainer's per-kind newest-version pick stays paired;
    // the session continues on the new version.
    QFuture<vc3d::ink_labels::SaveOutcome> saveAsNewVersion();
    // Label versions in the session's store.
    [[nodiscard]] std::vector<int> versions() const;
    // Opens another version of the active segment's labels.
    OpenOutcome openVersion(int version);
    // From a locked layer: start empty labels as a new version on the
    // segment's current canvas.
    OpenOutcome startNewVersion();
    // Reads a label image into one label bit on a worker and applies it as
    // one undo entry. A result that needs a decision (a legacy mask, a
    // resample) is reported through importFinished with needsDecision set and
    // nothing applied.
    bool importFile(const std::filesystem::path& path, const vc3d::ink_labels::ImportOptions& options);
    bool importFile(const std::filesystem::path& path, const vc3d::ink_labels::ImportOptions& options,
                    vc3d::ink_labels::Actor actor);
    // The trainer's dataset entry for the session's segment and version (see
    // vc3d::ink_labels::datasetEntry), and the same as indented JSON text.
    [[nodiscard]] QJsonObject datasetEntry() const;
    [[nodiscard]] QString datasetEntryText() const;
    // GUI save: asks before overwriting files changed outside VC3D.
    void saveInteractive();
    [[nodiscard]] bool saveInFlight() const;
    // A trainer save (not a checkpoint) of the session's store is running.
    [[nodiscard]] bool trainerSaveInFlight() const;
    // False while unsaved labels or writes are pending; then asks Save,
    // Discard or Cancel without blocking and calls retryClose once it is safe.
    // The answer lets exactly the next close through, so a close that another
    // guard cancels asks again next time.
    [[nodiscard]] bool readyToClose(std::function<void()> retryClose);
    // A checkable "Ink label mode" action for a menu, showing the I key
    // without binding it a second time.
    [[nodiscard]] QAction* labelModeAction() const { return _labelModeAction; }
    // Enables the keys for the current workspace, label mode and session;
    // call it when the current workspace changes.
    void updateShortcutStates();

    bool beginStroke(cv::Point2d labelPx, bool fromLastPoint, vc3d::ink_labels::Actor actor);
    void extendStroke(cv::Point2d labelPx);
    void endStroke();
    [[nodiscard]] bool strokeOpen() const;
    uint64_t applyShape(const vc3d::ink_labels::Shape& shape, vc3d::ink_labels::Actor actor);
    uint64_t applyMask(cv::Point origin, const cv::Mat_<uint8_t>& mask, vc3d::ink_labels::Actor actor);
    // With the explicit op instead of the panel's class and eraser.
    uint64_t applyShape(const vc3d::ink_labels::Shape& shape, vc3d::ink_labels::PaintOp op, vc3d::ink_labels::Actor actor,
                        const QString& label);
    uint64_t applyMask(cv::Point origin, const cv::Mat_<uint8_t>& mask, vc3d::ink_labels::PaintOp op, vc3d::ink_labels::Actor actor,
                       const QString& label);
    // Several shapes as one undo entry (an agent stroke through many points).
    uint64_t applyShapes(const std::vector<vc3d::ink_labels::Shape>& shapes, vc3d::ink_labels::PaintOp op, vc3d::ink_labels::Actor actor,
                         const QString& label);
    // Writes the current values of `rect` (clipped to the canvas, at most
    // 4096 x 4096) to a PNG or TIFF, without saving the labels.
    ReadOutcome readRect(const cv::Rect& rect, ReadFormat format, const std::filesystem::path& path) const;
    // Sets the exact label values (bits 1, 2, 4; at most kValueMask) of the
    // rectangle at `origin`, as ink_labels.read writes them, as one undo entry.
    uint64_t applyValues(cv::Point origin, const cv::Mat_<uint8_t>& values, vc3d::ink_labels::Actor actor, const QString& label);
    // Refuses to undo another actor's entry unless forced.
    bool undo(vc3d::ink_labels::Actor actor, bool force = false);
    bool redo(vc3d::ink_labels::Actor actor);

    void setPaintClass(vc3d::ink_labels::PaintClass cls);
    void setErase(bool erase);
    void setTool(Tool tool);
    void setRadius(double labelPx);
    void setBackgroundKeepsInk(bool keep);
    void setDisplay(bool showLabels, bool ink, bool supervision, bool validation, double opacity);
    void setAssistSettings(const AssistSettings& settings);
    [[nodiscard]] const AssistSettings& assistSettings() const { return _assist; }

    // Assisted labeling. Each runs on a worker from a raster snapshot and
    // applies its result on the GUI thread as one undo entry. `bound` is the
    // label rect to work in; by default the visible part of a viewer showing
    // the session, else the whole canvas. They return false (with a status
    // message) when refused: no writable session, another assist running, or
    // no single-channel prediction for the prediction guide.
    bool fillAt(cv::Point2d labelPx, std::optional<cv::Rect> bound = std::nullopt);
    // With an explicit paint operation instead of the panel's class and eraser.
    bool fillAt(cv::Point2d labelPx, std::optional<cv::Rect> bound, vc3d::ink_labels::PaintOp op);
    bool suggest(std::optional<cv::Rect> bound = std::nullopt);
    bool runCleanup(vc3d::ink_labels::Cleanup kind, std::optional<cv::Rect> bound = std::nullopt);
    // Picks or unpicks the suggestion under a label pixel.
    bool toggleSuggestionAt(cv::Point2d labelPx);
    void setSuggestionSelected(int id, bool selected);
    // Picks every suggestion (Shift+U).
    void pickAllSuggestions();
    // Applies the picked suggestions as ink (one entry), skipping pixels that
    // are already ink or background, and records them in the sidecar's
    // provenance. Returns the changed pixels. With nothing picked it keeps
    // the suggestions and says how to pick.
    uint64_t acceptSuggestions();
    uint64_t acceptSuggestions(vc3d::ink_labels::Actor actor);
    void discardSuggestions();
    // Esc: drops a running assist's result and any suggestions.
    void cancelAssist();
    [[nodiscard]] bool assistRunning() const { return _assistRunning; }

    // What the last fill, suggestion or cleanup did, or why it was refused.
    struct AssistReport {
        QString what;
        uint64_t changedPixels = 0;
        bool reachedBound = false;
        QString error;
    };
    // Starts an assist with `start` (fillAt, suggest or runCleanup) and runs
    // it to completion on the calling thread, for callers that need the
    // result now, such as the agent bridge. The work is the same as the
    // worker's and lands the same way.
    AssistReport runAssistNow(const std::function<bool()>& start);
    [[nodiscard]] const AssistReport& lastAssistReport() const { return _lastAssistReport; }
    [[nodiscard]] const vc3d::ink_labels::SuggestionSet* suggestions() const
    {
        return _suggestions ? &*_suggestions : nullptr;
    }
    [[nodiscard]] const std::set<int>& selectedSuggestions() const { return _selectedSuggestions; }
    // The Ink Detection dock's selected prediction, when it is single-channel.
    [[nodiscard]] std::optional<vc3d::ink_labels::PredictionGuide> predictionGuide() const;

    [[nodiscard]] vc3d::ink_labels::PaintClass paintClass() const { return _class; }
    [[nodiscard]] bool erase() const { return _erase; }
    [[nodiscard]] Tool tool() const { return _tool; }
    [[nodiscard]] double radius() const { return _radius; }
    [[nodiscard]] bool backgroundKeepsInk() const { return _backgroundKeepsInk; }
    [[nodiscard]] bool showLabels() const { return _showLabels; }
    [[nodiscard]] bool showInk() const { return _showInk; }
    [[nodiscard]] bool showSupervision() const { return _showSupervision; }
    [[nodiscard]] bool showValidation() const { return _showValidation; }
    [[nodiscard]] double opacity() const { return _opacity; }
    [[nodiscard]] vc3d::ink_labels::PaintOp currentOp() const;
    [[nodiscard]] QString toolName() const;
    // Ink red, background blue, validation yellow, eraser gray.
    [[nodiscard]] static QColor colorFor(vc3d::ink_labels::PaintClass cls, bool erase);
    // The color of the current class, or of the eraser.
    [[nodiscard]] QColor classColor() const;

    [[nodiscard]] const SessionInfo* session() const;
    [[nodiscard]] const vc3d::ink_labels::TileRaster* raster() const;
    [[nodiscard]] const vc3d::ink_labels::History* history() const;
    [[nodiscard]] const std::optional<LastSave>& lastSave() const { return _lastSave; }
    [[nodiscard]] const QDateTime& lastAutosave() const { return _lastAutosave; }
    // The QuadSurface the session is drawn on (null for test sessions).
    [[nodiscard]] QuadSurface* sessionSurface() const;
    [[nodiscard]] const std::vector<cv::Point2d>& openPolygon() const { return _polygon; }
    [[nodiscard]] InkLabelOverlayController* overlay() const { return _overlay.get(); }

    // Label pixel <-> viewer scene, through the session surface. nullopt when
    // the viewer does not show the session surface.
    [[nodiscard]] std::optional<QPointF> labelToScene(VolumeViewerBase* viewer, cv::Point2d labelPx) const;
    [[nodiscard]] std::optional<cv::Point2d> sceneToLabel(VolumeViewerBase* viewer, const QPointF& scenePos) const;
    // The viewer shows the session surface on the flattened (non-volumetric)
    // camera, so labels can be drawn and painted there.
    [[nodiscard]] bool viewerShowsSession(VolumeViewerBase* viewer) const;

    struct ViewportStrokeOutcome {
        bool ok = false;
        int delivered = 0;
        int handled = 0;
        QString error;
    };
    // App-local Qt events, delivered to the real viewport filter. Never OS input.
    // Validate the entire brush path first; edits retain Actor::Agent provenance.
    ViewportStrokeOutcome dispatchViewportStroke(VolumeViewerBase* viewer,
                                                 const std::vector<QPointF>& points);

signals:
    void sessionChanged();
    void rasterChanged(QRect labelRect);
    void historyChanged();
    void dirtyChanged(bool dirty);
    void toolStateChanged();
    void labelModeChanged(bool enabled);
    void labelModeRefusalChanged();
    void displayChanged();
    void saveFinished(const vc3d::ink_labels::SaveOutcome& outcome);
    void assistChanged();
    void assistFinished(const QString& what, quint64 changedPixels);
    void importFinished(const QString& path, const vc3d::ink_labels::ImportOutcome& outcome,
                        const vc3d::ink_labels::ImportOptions& options);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

#ifdef VC_TESTING
public:
    OpenOutcome openForTesting(const vc::RenderCanvas& canvas,
                               const std::filesystem::path& projectDir,
                               const std::string& segment);
    OpenOutcome openForTesting(const vc::RenderCanvas& canvas,
                               const std::filesystem::path& projectDir,
                               const std::string& segment,
                               const OpenOptions& options);
    // Runs the left-press path of the event filter at a label pixel, as if
    // the viewer showed the session surface. Returns whether it was consumed.
    bool pressForTesting(cv::Point2d labelPx, Qt::KeyboardModifiers modifiers = Qt::NoModifier);
    void releaseForTesting();
    [[nodiscard]] const QList<QShortcut*>& shortcutsForTesting() const { return _shortcuts; }
    void runAutosaveTickForTesting(bool ignoreTimers);
    [[nodiscard]] QFuture<vc3d::ink_labels::SaveOutcome> inFlightForTesting(const QString& storeDir) const
    {
        return _inFlight.value(storeDir).future;
    }
    void setPredictionGuideForTesting(std::optional<vc3d::ink_labels::PredictionGuide> guide)
    {
        _guideForTesting = std::move(guide);
    }
    [[nodiscard]] QFuture<void> assistFutureForTesting() const { return _assistFuture; }
    bool keyForTesting(int key);
#endif

private:
    struct Session;
    struct BoundViewer {
        VolumeViewerBase* viewer = nullptr;
        QPointer<QObject> view;
        QPointer<QObject> viewport;
        QPointer<SpiralBrushCursorWidget> cursor;
    };

    // A worker's answer: a mask to apply, or a suggestion set to show.
    struct AssistResult {
        QString what;
        QString editKind;
        std::optional<vc3d::ink_labels::MaskResult> mask;
        vc3d::ink_labels::PaintOp op;
        std::optional<vc3d::ink_labels::SuggestionSet> suggestions;
        QString error;
        // Imports: the outcome and the edit, made on a copy of the raster.
        std::optional<vc3d::ink_labels::ImportOutcome> import;
        std::optional<vc3d::ink_labels::HistoryEntry> entry;
        vc3d::ink_labels::ImportOptions importOptions;
        QString importPath;
        uint64_t baseSerial = 0;
    };

    struct OpenRequest {
        vc::RenderCanvas canvas;
        // The folder that holds ink_labels/: the project folder, or an
        // application data folder for projects kept in the remote cache.
        std::filesystem::path projectDir;
        std::string segment;
        std::string segmentUuid;
        std::filesystem::path segmentDir;
        std::weak_ptr<QuadSurface> surface;
        std::string geometrySha;
    };

    // Opening a store, step by step: take the store's lock (if its folder
    // exists), check the trainer sidecar against the segment, decide what to
    // do with a checkpoint, then load. The lock and folder of a store that
    // does not exist yet are taken at the first edit.
    OpenOutcome openStore(const OpenRequest& request, const OpenOptions& options);
    // Picks the label version to open: the newest, or the one asked for.
    bool resolveVersion(vc3d::ink_labels::StorePaths& paths, const OpenRequest& request, const OpenOptions& options,
                        OpenOutcome& out) const;
    bool acquireLock(Session& session, const OpenOptions& options, std::shared_ptr<QLockFile> heldLock, OpenOutcome& out);
    struct Sidecars {
        std::optional<vc3d::ink_labels::Sidecar> trainer;
        std::optional<vc3d::ink_labels::Sidecar> autosave;
    };
    // The reason the trainer sidecar locks the layer, or nothing.
    std::optional<OpenOutcome> checkSidecar(std::unique_ptr<Session>& session, const OpenRequest& request,
                                            const vc3d::ink_labels::CanvasKey& key, Sidecars& sidecars, OpenOutcome& out);
    // Trainer files or the checkpoint; nullopt when the open stops here.
    std::optional<vc3d::ink_labels::Target> chooseTarget(Session& session, const OpenRequest& request, const OpenOptions& options,
                                                         const vc3d::ink_labels::CanvasKey& key, Sidecars& sidecars,
                                                         OpenOutcome& out);
    bool loadInto(Session& session, vc3d::ink_labels::Target target, const vc3d::ink_labels::CanvasKey& key,
                  const Sidecars& sidecars, OpenOutcome& out);
    OpenOutcome lockLayer(std::unique_ptr<Session> session, OpenOutcome out, const QString& reason, const QString& expected,
                          const QString& got);
    std::optional<OpenRequest> activeSegmentRequest(OpenOutcome& failure) const;
    // canEdit, then the store's folder and lock if they are not taken yet.
    bool prepareEdit(QString* why);
    void installSession(std::unique_ptr<Session> session);
    void commitOpenStroke();
    void finishTransaction();
    uint64_t applyEdit(const std::function<uint64_t(vc3d::ink_labels::EditTxn&)>& edit, vc3d::ink_labels::Actor actor,
                       const QString& label, const QString& editKind);
    bool canStartAssist(QString* why) const;
    // A refused assist: the status message, and the report runAssistNow gives.
    void refuseAssist(const QString& what, const QString& why);
    void startAssist(std::function<AssistResult()> work);
    void finishAssist(const AssistResult& result);
    void finishImport(const AssistResult& result);
    [[nodiscard]] std::optional<cv::Rect> visibleLabelRect(VolumeViewerBase* viewer) const;
    [[nodiscard]] cv::Rect defaultBound() const;
    void refreshSuggestionPreview();
    bool handleAssistKey(int key);
    bool canEdit(QString* why) const;
    void notifyRasterChanged(const cv::Rect& labelRect);
    void notifyHistory();
    void createShortcuts();
    // Enter, Backspace and Esc while a polygon is open. Returns whether the
    // key was used.
    bool handleEditKey(int key);
    void statusMessage(const QString& text, int ms = 3000) const;
    void announceToolState();
    void updateModeLine();
    // Recomputes the geometry hash after the segment's grid changed.
    void refreshGeometrySha();
    vc3d::ink_labels::SaveRequest makeSaveRequest(vc3d::ink_labels::Target target, bool overwriteExternalEdits);
    QFuture<vc3d::ink_labels::SaveOutcome> startWrite(vc3d::ink_labels::SaveRequest request);
    void startCheckpoint();
    void onAutosaveTick();
    void onStateSurfaceChanged();
    void bindViewer(VolumeViewerBase* viewer);
    void unbindViewer(VolumeViewerBase* viewer);
    BoundViewer* boundFor(QObject* watched);
    void updateCursor(BoundViewer& bound, const QPointF& devicePos);
    void hideCursors();
    // Left-press decision shared by the event filter and tests.
    bool handlePress(cv::Point2d labelPx, Qt::KeyboardModifiers modifiers, double devicePxPerLabelPx,
                     vc3d::ink_labels::Actor actor = vc3d::ink_labels::Actor::User);
    void handleDrag(cv::Point2d labelPx, double devicePxPerLabelPx);
    void handleRelease();
    void closePolygon();
    void cancelPolygon();
    void updatePolygonPreview(std::optional<cv::Point2d> hover = std::nullopt);
    [[nodiscard]] double devicePxPerLabelPx(VolumeViewerBase* viewer, cv::Point2d labelPx) const;

    Deps _deps;
    bool _labelMode = false;
    QString _labelModeRefusal;
    void setLabelModeRefusal(const QString& reason);
    vc3d::ink_labels::PaintClass _class = vc3d::ink_labels::PaintClass::Ink;
    bool _erase = false;
    Tool _tool = Tool::Brush;
    double _radius = 12.0;
    bool _backgroundKeepsInk = false;
    bool _showLabels = true, _showInk = true, _showSupervision = true, _showValidation = true;
    double _opacity = 0.6;
    AssistSettings _assist;

    // Assist state: one worker at a time; a newer generation drops the result.
    QFuture<void> _assistFuture;
    uint64_t _assistGeneration = 0;
    bool _assistRunning = false;
    bool _assistNow = false;
    AssistReport _lastAssistReport;
    std::optional<vc3d::ink_labels::SuggestionSet> _suggestions;
    std::set<int> _selectedSuggestions;
#ifdef VC_TESTING
    std::optional<vc3d::ink_labels::PredictionGuide> _guideForTesting;
#endif

    std::unique_ptr<Session> _session;
    uint64_t _sessionGeneration = 0;
    std::unique_ptr<InkLabelOverlayController> _overlay;
    QList<QShortcut*> _shortcuts;
    QShortcut* _toggleShortcut = nullptr;
    QShortcut* _showShortcut = nullptr;
    QShortcut* _pickAllShortcut = nullptr;
    // Enter, Backspace and Esc: enabled only while they have work to do.
    QList<QShortcut*> _editKeyShortcuts;
    QAction* _labelModeAction = nullptr;
    bool _hintShown = false;
    // An open is waiting for the restore question; nested opens are refused.
    bool _openPending = false;
    std::vector<BoundViewer> _bound;
    QTimer* _autosaveTimer = nullptr;
    // One write per store at a time: a trainer save or a checkpoint.
    struct InFlight {
        uint64_t id = 0;
        vc3d::ink_labels::Target target = vc3d::ink_labels::Target::Trainer;
        QFuture<vc3d::ink_labels::SaveOutcome> future;
        // The store's lock, held until the write lands even if the session
        // closed; reopening the store takes it over.
        std::shared_ptr<QLockFile> lock;
    };
    QHash<QString, InFlight> _inFlight;
    uint64_t _nextWriteId = 1;
    QSet<QString> _checkpointedThisRun;
    std::optional<LastSave> _lastSave;
    QDateTime _lastAutosave;
    bool _closeBypass = false;

    // Polygon tool state.
    std::vector<cv::Point2d> _polygon;
    bool _lassoCandidate = false;
    bool _lasso = false;
    cv::Point2d _lastLassoPoint;
    double _lassoMinStep = 1.0;
    bool _leftDown = false;
    // The visible label rect of the viewer a fill click happened in.
    std::optional<cv::Rect> _pressBound;
};
