#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

#include <functional>

class CWindow;

// Outcome of a project attachment change.
struct ProjectAttachmentOutcome {
    // Why an operation failed, for callers that report it by kind.
    enum class Error {
        None,
        NoProject,          // no project is open
        InvalidLocation,    // the location is missing, malformed or of the wrong kind
        NotFound,           // no such entry to detach
        NoVolume,           // the operation needs a current volume
        Unsupported,        // the current volume cannot take the attachment
        Busy,               // an attachment of this kind is already loading
        Failed,             // the project or a file could not be updated
    };

    bool success{false};
    bool changed{false};          // false when the project already matched
    QString location;             // the location as stored in the project
    QString previousLocation;     // a single-valued attachment that was replaced
    Error errorKind{Error::None};
    QString error;                // failure sentence when !success
    QStringList warnings;         // non-fatal notes, e.g. an unstamped umbilicus
};

// The entry lists of a project that hold attachments by location.
enum class ProjectEntryKind { Volume, Segments, NormalGrid, Lasagna };

// Attaches and detaches project-level resources beside volumes and segments:
// the umbilicus, normal grids and Lasagna manifests. Menu actions pick a
// location and present the outcome; other callers use the same operations
// with an explicit location. Every operation persists through the open
// VolumePkg and refreshes the project UI like the menus. Local locations are
// stored absolute and lexically normal (relative ones resolve against the
// project folder, as stored entries do), and an existing entry is found with
// vc::project::sameLocation, so another spelling of an attached location
// matches it instead of adding a second entry.
class ProjectAttachmentController final : public QObject
{
public:
    explicit ProjectAttachmentController(CWindow* window);

    // Validates a local umbilicus file, refusing malformed frame metadata,
    // then sets it on the project, replacing any other umbilicus
    // (`previousLocation`). A file that declares no frame is attached with a
    // warning. Attaching the attached file again changes nothing.
    ProjectAttachmentOutcome attachUmbilicus(const QString& file);
    // Clears the attached umbilicus so it is resolved automatically again.
    ProjectAttachmentOutcome detachUmbilicus();

    // Adds a local normal_grids root (xy/xz/yz subdirectories). Attaching an
    // attached root again changes nothing.
    ProjectAttachmentOutcome attachNormalGrid(const QString& location);

    // Removes the entry of `kind` stored for `location` (any spelling of it).
    // VolumePkg removes that stored location from every entry list it
    // appears in. Files are not touched.
    ProjectAttachmentOutcome detachEntry(ProjectEntryKind kind, const QString& location);

    // Loads a local or remote .lasagna.json manifest on a worker and attaches
    // its volumes. `onFinished` runs on the GUI thread once the attachment
    // completes, fails, or finds the open project changed.
    // Returns false with `refusal` describing why when the attachment cannot
    // start.
    bool startLasagnaAttachment(const QString& location,
                                bool fiberInference,
                                std::function<void(const ProjectAttachmentOutcome&)> onFinished,
                                ProjectAttachmentOutcome* refusal = nullptr);
    bool lasagnaAttachmentInFlight() const { return _lasagnaInFlight; }

private:
    bool hasProject(ProjectAttachmentOutcome* outcome) const;

    CWindow* _window{nullptr};
    bool _lasagnaInFlight{false};
};
