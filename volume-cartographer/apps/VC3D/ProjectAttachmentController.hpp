#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

#include <functional>

class CWindow;

// Outcome of a project attachment change.
struct ProjectAttachmentOutcome {
    bool success{false};
    bool changed{false};          // false when the project already matched
    QString location;             // the location as stored in the project
    QString error;                // failure sentence when !success
    QStringList warnings;         // non-fatal notes
};

// Attaches and detaches project-level resources beside volumes and segments.
// Menu actions pick a location and present the outcome; other callers use the
// same operations with an explicit location. Every operation persists
// through the open VolumePkg and refreshes the project UI like the menus.
class ProjectAttachmentController final : public QObject
{
public:
    explicit ProjectAttachmentController(CWindow* window);

    // Loads a local or remote .lasagna.json manifest on a worker and attaches
    // its volumes. `onFinished` runs on the GUI thread once the attachment
    // completes, fails, or finds the open project changed. Returns false with
    // `error` set when the attachment cannot start.
    bool startLasagnaAttachment(const QString& location,
                                bool fiberInference,
                                std::function<void(const ProjectAttachmentOutcome&)> onFinished,
                                QString* error = nullptr);
    bool lasagnaAttachmentInFlight() const { return _lasagnaInFlight; }

private:
    bool hasProject(QString* error) const;

    CWindow* _window{nullptr};
    bool _lasagnaInFlight{false};
};
