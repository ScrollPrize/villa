#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QImage>
#include <QString>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

class CState;
class QuadSurface;
class VolumeViewerBase;

class InkDetectionOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    struct Option {
        QString label;
        std::string sampleId;
        std::string segmentId;
        std::string segmentLongId;
        std::filesystem::path localPath;
        bool singleChannel{false};
    };

    explicit InkDetectionOverlayController(CState* state, QObject* parent = nullptr);

    [[nodiscard]] const std::vector<Option>& options() const { return _options; }
    [[nodiscard]] const std::filesystem::path& selectedPath() const { return _selectedPath; }
    [[nodiscard]] int opacity() const { return _opacity; }
    [[nodiscard]] const std::string& colormapId() const { return _colormapId; }
    [[nodiscard]] bool selectedIsSingleChannel() const { return _selectedImage.singleChannel; }

public slots:
    void refreshAvailableDetections();
    void setSelectedPath(const std::filesystem::path& path);
    void clearSelection();
    void setOpacity(int opacity);
    void setColormapId(const std::string& id);

signals:
    void availableDetectionsChanged();
    void selectionChanged();

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer,
                           ViewerOverlayControllerBase::OverlayBuilder& builder) override;

private:
    struct LoadedImage {
        QImage image;
        cv::Mat_<uint8_t> scalar;
        bool singleChannel{false};
        std::filesystem::path path;
        std::string colormapId;
    };

    void loadSelectedImage();
    QImage renderSingleChannelImage(const cv::Mat_<uint8_t>& scalar,
                                    const std::string& colormapId) const;
    bool imageMatchesSurface(const QuadSurface& surface) const;

    CState* _state{nullptr};
    std::vector<Option> _options;
    std::filesystem::path _selectedPath;
    LoadedImage _selectedImage;
    int _opacity{70};
    std::string _colormapId;
};
