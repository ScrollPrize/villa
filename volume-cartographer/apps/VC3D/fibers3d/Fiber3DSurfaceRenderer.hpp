#pragma once
#include "Fiber3DSurfaceMesh.hpp"
#include <QMatrix4x4>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QQuaternion>

namespace vc3d::fibers3d
{
// Construct, draw and destroy with the owning canvas context current.
class Fiber3DSurfaceRenderer : protected QOpenGLFunctions_3_3_Core
{
public:
    Fiber3DSurfaceRenderer();
    ~Fiber3DSurfaceRenderer();
    // True while uploading. Subsequent frames reuse the resident buffers.
    bool draw(std::shared_ptr<const surface::Mesh>, const QMatrix4x4&, const Point& origin,
              const QQuaternion& rotation, float opacity);
    QString error() const { return error_; }
private:
    std::shared_ptr<const surface::Mesh> mesh_;
    QOpenGLShaderProgram program_;
    GLuint vao_ = 0, vertices_ = 0, indices_ = 0;
    size_t vertexBytes_ = 0, indexBytes_ = 0;
    QString error_;
};
}
