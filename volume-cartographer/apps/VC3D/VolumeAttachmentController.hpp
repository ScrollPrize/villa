#pragma once

#include <QObject>
#include <QString>

#include <functional>
#include <string>
#include <vector>

#include "vc/core/util/RemoteAuth.hpp"

class CWindow;
class Volume;

enum class VolumeAttachmentSelection {
    PreserveCurrent,
    SelectAttached,
};

enum class VolumeAttachmentPresentation {
    Silent,
    Interactive,
};

enum class VolumeAttachmentPreparationFailure {
    None,
    NoProject,
    InvalidLocation,
    RemoteConfiguration,
};

struct VolumeAttachmentRequest {
    QString location;
    std::vector<std::string> tags;
    vc::HttpAuth auth;
    QString remoteCacheRoot;
    VolumeAttachmentSelection selection{VolumeAttachmentSelection::PreserveCurrent};
};

struct VolumeAttachmentOutcome {
    bool success{false};
    bool alreadyAttached{false};
    QString error;
    QString volumeId;
    QString location;
    QString projectPath;
};

class VolumeAttachmentController final : public QObject
{
public:
    explicit VolumeAttachmentController(CWindow* window);

    bool prepare(
        const QString& location,
        std::vector<std::string> tags,
        VolumeAttachmentPresentation presentation,
        VolumeAttachmentRequest* request,
        QString* errorMessage = nullptr,
        VolumeAttachmentPreparationFailure* failure = nullptr);
    bool start(
        VolumeAttachmentRequest request,
        std::function<void(const VolumeAttachmentOutcome&)> onFinished,
        QString* errorMessage = nullptr);
    bool inFlight() const;

    bool resolveRemoteAuth(
        const QString& url,
        vc::HttpAuth* authOut,
        QString* errorMessage = nullptr) const;

private:
    struct TaskResult;

    QString remoteCacheDirectory(VolumeAttachmentPresentation presentation);
    QString configuredRemoteCacheDirectory() const;
    QString suggestedRemoteCacheDirectory() const;

    CWindow* _window{nullptr};
    bool _inFlight{false};
};
