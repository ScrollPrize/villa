#pragma once

#include "SessionRecovery.hpp"

#include <QDialog>
#include <QPointer>

#include <functional>
#include <memory>

class QCheckBox;
class QLockFile;
class VCCollection;

// Offers what a crashed session left behind (see SessionRecovery): its
// points, restored into the project they were placed in, and its segment,
// selected again. The dialog is application modal, so nothing changes behind
// it before the user answers, and nothing is applied before Recover. The copy
// is deleted once its points are restored, or with Discard; Later keeps it for
// the next start. While the dialog is open it holds the session's lock, so a
// second VC3D started at the same time does not offer it too.
class SessionRecoveryDialog : public QDialog
{
    Q_OBJECT

public:
    // What the dialog needs from the main window.
    struct Actions
    {
        std::function<QString()> currentProjectPath;
        std::function<bool(const QString& path, QString* error)> openProject;
        std::function<bool(const QString& segmentId, QString* error)> activateSegment;
        std::function<void(const QString& text)> showStatus;
    };

    // Shows the dialog for the newest leftover session of recovery that no
    // other VC3D is offering, or returns nullptr when there is none. The
    // dialog deletes itself when it closes.
    static SessionRecoveryDialog* offer(QWidget* parent,
                                        const SessionRecovery& recovery,
                                        VCCollection* collection,
                                        Actions actions);

    // Lets go of the session's lock however the dialog closes.
    void done(int result) override;

private:
    SessionRecoveryDialog(QWidget* parent,
                          SessionRecovery::LeftoverSession session,
                          std::shared_ptr<QLockFile> claim,
                          VCCollection* collection,
                          Actions actions);

    void recover();
    void discardCopy();
    [[nodiscard]] bool projectIsOpen() const;
    void finish(const QString& status);

    SessionRecovery::LeftoverSession _session;
    std::shared_ptr<QLockFile> _claim;
    QPointer<VCCollection> _collection;
    Actions _actions;
    QCheckBox* _selectSegment{nullptr};
};
