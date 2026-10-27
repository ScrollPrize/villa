#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QImage>
#include <QString>

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core/mat.hpp>

class CState;
class QuadSurface;
class SurfaceCatalog;
class VolumeViewerBase;

class InkDetectionOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    // A published ink detection image of the active surface: its file in the
    // surface's artifact directory (downloaded when first selected).
    struct Option {
        QString label;
        std::string name;  // the published file (vc::objects::CatalogFile::name)
        std::filesystem::path localPath;
    };

    explicit InkDetectionOverlayController(CState* state, QObject* parent = nullptr);
    // The catalog the active surface's ink detections are read from.
    void setSurfaceCatalog(SurfaceCatalog* catalog);

    [[nodiscard]] const std::vector<Option>& options() const { return _options; }
    [[nodiscard]] const std::filesystem::path& selectedPath() const { return _selectedPath; }
    [[nodiscard]] int opacity() const { return _opacity; }
    [[nodiscard]] const std::string& colormapId() const { return _colormapId; }
    [[nodiscard]] bool horizontalFlip() const { return _horizontalFlip; }
    [[nodiscard]] bool verticalFlip() const { return _verticalFlip; }
    [[nodiscard]] bool selectedIsSingleChannel() const { return _selectedImage.singleChannel; }
    [[nodiscard]] bool hasLoadedSelection() const;
#ifdef VC_TESTING
    [[nodiscard]] const QImage& selectedImageForTesting() const { return _selectedImage.image; }
#endif

public slots:
    void refreshAvailableDetections();
    void toggleVisibility();
    void setSelectedPath(const std::filesystem::path& path);
    void clearSelection();
    void setOpacity(int opacity);
    void setColormapId(const std::string& id);
    void setHorizontalFlip(bool enabled);
    void setVerticalFlip(bool enabled);

signals:
    void availableDetectionsChanged();
    void selectionChanged();
    void opacityChanged(int opacity);
    void flipChanged(bool horizontal, bool vertical);

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer,
                           ViewerOverlayControllerBase::OverlayBuilder& builder) override;

private:
    struct LoadedImage {
        QImage baseImage;
        QImage image;
        cv::Mat_<uint8_t> scalar;
        bool singleChannel{false};
        std::filesystem::path path;
        std::string colormapId;
    };

    void loadSelectedImage();
    QImage applyImageFlips(const QImage& image) const;
    QImage renderSingleChannelImage(const cv::Mat_<uint8_t>& scalar,
                                    const std::string& colormapId) const;
    void refreshOptionsForActiveSurface();
    std::string activeSegmentKey() const;
    bool imageMatchesSurface(const QuadSurface& surface) const;

    CState* _state{nullptr};
    SurfaceCatalog* _catalog{nullptr};
    std::vector<Option> _options;
    // The image being downloaded (its file), if any.
    std::filesystem::path _fetchingPath;
    std::filesystem::path _selectedPath;
    std::string _selectedSegmentKey;
    std::unordered_map<std::string, std::filesystem::path> _selectedPathBySegment;
    LoadedImage _selectedImage;
    int _opacity{70};
    int _opacityBeforeToggle{70};
    std::string _colormapId;
    bool _horizontalFlip{false};
    bool _verticalFlip{false};
};
