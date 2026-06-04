#pragma once

#include <QGraphicsView>

#include <filesystem>

#include "vc/atlas/Atlas.hpp"

class QGraphicsScene;

class AtlasCanvasWidget : public QGraphicsView
{
public:
    explicit AtlasCanvasWidget(QWidget* parent = nullptr);

    void setAtlas(const std::filesystem::path& atlasDir, const vc::atlas::Atlas& atlas);
    void clearAtlas();

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void rebuildScene();

    QGraphicsScene* _scene{nullptr};
    std::filesystem::path _atlasDir;
    vc::atlas::Atlas _atlas;
    bool _hasAtlas{false};
};
