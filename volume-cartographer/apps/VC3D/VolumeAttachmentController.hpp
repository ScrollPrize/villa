#pragma once

#include <QObject>
#include <QString>

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <vc_reader/native/reader_types.hpp>

class CWindow;
class Volume;

enum class VolumeAttachmentSelection {
    PreserveCurrent,
    SelectAttached,
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
        VolumeAttachmentRequest* request,
        QString* errorMessage = nullptr,
        VolumeAttachmentPreparationFailure* failure = nullptr);
    bool start(
        VolumeAttachmentRequest request,
        std::function<void(const VolumeAttachmentOutcome&)> onFinished,
        QString* errorMessage = nullptr);
    bool inFlight() const;
private:
    struct TaskResult;

    CWindow* _window{nullptr};
    bool _inFlight{false};
};
