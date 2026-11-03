#pragma once
#include "Fiber3DSlices.hpp"
#include "vc/core/util/Compositing.hpp"
#include <QMatrix4x4>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>

namespace vc3d::fibers3d
{
// Owns only bounded GPU display resources. Construct/destroy with the canvas
// context current. CT reads and resampling belong to the existing worker.
class Fiber3DVolumeRenderer : protected QOpenGLFunctions_3_3_Core
{
public:
    Fiber3DVolumeRenderer();
    ~Fiber3DVolumeRenderer();
    // Returns true while progressive upload needs another frame.
    bool draw(
        const slices::Images&,
        const std::array<CompositeParams, 3>&,
        const QMatrix4x4& mvp,
        const Point& origin,
        GLuint framebuffer,
        int width,
        int height,
        float low,
        float high,
        float opacity,
        bool interactive,
        int maxRaySamples = 512);
    const QString& error() const { return error_; }

private:
    struct Texture {
        GLuint id = 0;
        int layers = 0;
    };
    void clearImages();
    bool copyDepth(GLuint framebuffer, int width, int height);
    std::unique_ptr<QOpenGLShaderProgram> program_;
    slices::Images images_;
    std::vector<Texture> textures_;
    GLuint vao_ = 0, transfer_ = 0, depth_ = 0, depthFbo_ = 0;
    int depthWidth_ = 0, depthHeight_ = 0;
    std::array<CompositeParams, 3> params_;
    bool transferDirty_ = true;
    QString error_;
};
}  // namespace vc3d::fibers3d
