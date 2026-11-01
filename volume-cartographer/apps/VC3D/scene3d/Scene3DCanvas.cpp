#include "Scene3DCanvasState.hpp"
#include <QApplication>
#include <QContextMenuEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QPainter>
#include <QSurfaceFormat>
#include <QWheelEvent>
#include <QVector2D>
#include <limits>

namespace vc3d::scene3d
{
namespace
{
bool finite(const Point& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}
}
Scene3DCanvas::Scene3DCanvas(QWidget* parent) : QOpenGLWidget(parent), v(std::make_unique<State>())
{
    QSurfaceFormat fmt = format();
    fmt.setVersion(3, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setDepthBufferSize(24);
    fmt.setSamples(4);
    setFormat(fmt);
    setMinimumSize(180, 160);
    setFocusPolicy(Qt::StrongFocus);
    setToolTip(tr("Drag: orbit · Right-drag: pan · Wheel: zoom"));
    v->refine.setSingleShot(true);
    v->refine.setInterval(180);
    connect(&v->refine, &QTimer::timeout, this, [this] {
        v->interactive = false;
        update();
    });
}
Scene3DCanvas::~Scene3DCanvas()
{
    cleanupGL();
}
void Scene3DCanvas::cleanupGL()
{
    if (context())
        disconnect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &Scene3DCanvas::cleanupGL);
    if (!context() || !v->glReady)
        return;
    makeCurrent();
    v->surfaceRenderer.reset();
    v->surfaceUploading = bool(v->surfaceMesh);
    v->volumeRenderer.reset();
    v->volumeUploading = bool(v->ctVolume);
    v->ctProgram.reset();
    glDeleteTextures(GLsizei(v->sliceTextures.size()), v->sliceTextures.data());
    v->sliceTextures.clear();
    if (v->sliceBuffer)
        glDeleteBuffers(1, &v->sliceBuffer);
    v->sliceBuffer = 0;
    v->slicesDirty = true;
    if (v->ctVao)
        glDeleteVertexArrays(1, &v->ctVao);
    v->ctVao = 0;
    v->glReady = false;
    doneCurrent();
}
int Scene3DCanvas::maximumVolumeTextureEdge() const
{
    return v->maximumVolumeTextureEdge;
}
void Scene3DCanvas::initializeGL()
{
    if (!initializeOpenGLFunctions()) {
        v->ctError = tr("3D view requires OpenGL 3.3.");
        return;
    }
    v->glReady = true;
    GLint maxEdge = 256;
    glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &maxEdge);
    if (maxEdge > 0 && maxEdge != v->maximumVolumeTextureEdge) {
        v->maximumVolumeTextureEdge = maxEdge;
        emit volumeTextureLimitChanged(maxEdge);
    }
    v->ctError.clear();
    connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &Scene3DCanvas::cleanupGL, Qt::DirectConnection);
    glGenVertexArrays(1, &v->ctVao);
    v->ctProgram = std::make_unique<QOpenGLShaderProgram>();
    if (!v->ctProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, R"glsl(
        #version 330 core
        layout(location=0) in vec3 position;
        layout(location=1) in vec2 uv;
        uniform mat4 mvp;
        out vec2 texCoord;
        void main() { gl_Position = mvp * vec4(position,1); texCoord = uv; }
    )glsl") ||
        !v->ctProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, R"glsl(
        #version 330 core
        in vec2 texCoord;
        uniform sampler2D ct;
        uniform vec2 windowRange;
        uniform float opacity;
        out vec4 color;
        void main() {
            vec2 sampleValue = texture(ct, texCoord).rg;
            if (sampleValue.g < .999) discard;
            float gray = clamp((sampleValue.r * 255.0 - windowRange.x) / max(1.0,windowRange.y-windowRange.x),0.0,1.0);
            color = vec4(gray,gray,gray,opacity);
        }
    )glsl") ||
        !v->ctProgram->link())
        v->ctError = tr("Cannot initialize CT display: %1").arg(v->ctProgram->log());
}
void Scene3DCanvas::beginFrame()
{
    if (!v->glReady)
        return;
    const int pixelWidth = std::max(1, int(width()*devicePixelRatioF()));
    const int pixelHeight = std::max(1, int(height()*devicePixelRatioF()));
    // QPainter and the widget compositor also use this context. Restore
    // relevant state explicitly, including before clearing the framebuffer.
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glViewport(0, 0, pixelWidth, pixelHeight);
    glClearColor(0.045f, 0.06f, 0.075f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
void Scene3DCanvas::drawSurface()
{
    const auto matrix = v->matrix(width(), height());
    if (v->glReady && (v->surfaceMesh || v->surfaceRenderer)) {
        if (!v->surfaceRenderer)
            v->surfaceRenderer = std::make_unique<fibers3d::Fiber3DSurfaceRenderer>();
        v->surfaceUploading = v->surfaceRenderer->draw(v->surfaceVisible ? v->surfaceMesh : nullptr, matrix, v->origin, v->rotation, v->surfaceOpacity, v->surfaceTextured, v->windowLow, v->windowHigh);
        if (v->surfaceUploading)
            update();
    }
}
void Scene3DCanvas::paintGL()
{
    beginFrame();
    drawSurface();
    if (v->glReady)
        drawCTSlices();
    QPainter painter(this);
    painter.setPen(Qt::white);
    QString error = v->ctError;
    if (v->surfaceRenderer && !v->surfaceRenderer->error().isEmpty()) error += "\n" + v->surfaceRenderer->error();
    if (v->volumeRenderer && !v->volumeRenderer->error().isEmpty()) error += "\n" + v->volumeRenderer->error();
    painter.drawText(rect().adjusted(12,12,-12,-12), Qt::AlignTop | Qt::TextWordWrap, error);
}
void Scene3DCanvas::setPerspective(bool perspective)
{
    v->perspective = perspective;
    update();
}
void Scene3DCanvas::fitAll()
{
    const double inf = std::numeric_limits<double>::infinity();
    Bounds bounds{{inf, inf, inf}, {-inf, -inf, -inf}};
    auto expand = [&](const Point& p) {
        for (int k = 0; k < 3; ++k) {
            bounds.lower[k] = std::min(bounds.lower[k], p[k]);
            bounds.upper[k] = std::max(bounds.upper[k], p[k]);
        }
    };
    for (const auto& image : v->slices)
        for (const auto& p : slices::corners(*image))
            expand(p);
    if (v->surfaceVisible && v->surfaceMesh) {
        expand(v->surfaceMesh->bounds.lower);
        expand(v->surfaceMesh->bounds.upper);
    }
    if (v->ctVolume) {
        expand(v->ctVolume->bounds.lower);
        expand(v->ctVolume->bounds.upper);
    }
    fitBounds(bounds);
}
float Scene3DCanvas::pixelsPerVoxel() const
{
    return float(std::max(1, height()) * devicePixelRatioF() / (2.0 * v->scale));
}

void Scene3DCanvas::resizeGL(int, int)
{
    emit viewScaleChanged();
}

void Scene3DCanvas::clearContent()
{
    setCTSlices({});
    setSurfaceMesh({});
    setCTVolume({});
    v->fitted = false;
    v->origin = {};
    v->radius = 100.0;
    update();
}

void Scene3DCanvas::setCTWindow(float low, float high)
{
    v->windowLow = low;
    v->windowHigh = high;
    update();
}

GLuint Scene3DCanvas::uploadCTTexture(const ct::Tile& tile)
{
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, tile.width, tile.height, 0, GL_RG, GL_UNSIGNED_BYTE, tile.pixels.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    return texture;
}

void Scene3DCanvas::setCTSlices(slices::Images images)
{
    if (images == v->slices)
        return;
    v->slices = std::move(images);
    v->slicesDirty = true;
    if (!v->fitted && !v->slices.empty())
        fitAll();
    update();
}

const slices::Images& Scene3DCanvas::ctSlices() const
{
    return v->slices;
}

void Scene3DCanvas::setCTVolume(std::shared_ptr<const VolumeData> data)
{
    if (v->ctVolume == data) return;
    v->ctVolume = std::move(data);
    v->volumeUploading = bool(v->ctVolume);
    if (!v->fitted && v->ctVolume) fitAll();
    update();
}
std::shared_ptr<const VolumeData> Scene3DCanvas::ctVolume() const { return v->ctVolume; }
bool Scene3DCanvas::isCTVolumeReady() const
{
    return v->ctVolume && !v->volumeUploading && v->volumeRenderer && v->volumeRenderer->error().isEmpty();
}
void Scene3DCanvas::setCTVolumeDensity(float density)
{
    v->volumeDensity = std::clamp(density, 0.f, 100.f);
    update();
}

void Scene3DCanvas::setCTSliceOpacity(float opacity)
{
    v->sliceOpacity = std::clamp(opacity, 0.f, 1.f);
    update();
}

void Scene3DCanvas::setCTSliceTransfer(const std::array<CompositeParams, 3>& params)
{
    if (v->sliceTransfer == params)
        return;
    v->sliceTransfer = params;
    update();
}

void Scene3DCanvas::drawCTSlices()
{
    const bool volumetric = !v->slices.empty() && bool(v->slices.front()->slab);

    if (v->slicesDirty) {
        glDeleteTextures(GLsizei(v->sliceTextures.size()), v->sliceTextures.data());
        v->sliceTextures.clear();
        if (v->sliceBuffer)
            glDeleteBuffers(1, &v->sliceBuffer);
        v->sliceBuffer = 0;
        v->sliceRanges.clear();
        v->sliceTree = volumetric ? nullptr : slices::partition(v->slices);
        v->slicesDirty = false;
    }
    if ((volumetric || v->ctVolume) && !v->volumeRenderer)
        v->volumeRenderer = std::make_unique<fibers3d::Fiber3DVolumeRenderer>();
    if (v->volumeRenderer) {
        const auto ratio = devicePixelRatioF();
        auto transfer = v->sliceTransfer;
        int raySamples = 512;
        if (v->ctVolume) {
            const auto& slab = *v->ctVolume->images.front()->slab;
            raySamples = 2 * std::min(2048, std::max({slab.width, slab.height, slab.depth}));
            auto& p = transfer[0];
            p = CompositeParams{};
            const auto& bounds = v->ctVolume->bounds;
            double thickness = std::numeric_limits<double>::infinity();
            for (int k=0; k<3; ++k) thickness=std::min(thickness,std::max(1.,bounds.upper[k]-bounds.lower[k]));
            p.alphaOpacity = float(v->volumeDensity / thickness);
            p.alphaMin = std::clamp(v->windowLow/255.f,0.f,1.f);
            p.alphaMax = std::clamp(v->windowHigh/255.f,0.f,1.f);
            p.isoCutoff = 1;
        }
        v->volumeUploading = v->volumeRenderer
                ->draw(v->ctVolume ? v->ctVolume->images : (volumetric ? v->slices : slices::Images{}), transfer, v->matrix(width(), height()), v->origin, defaultFramebufferObject(), int(width() * ratio), int(height() * ratio), v->windowLow, v->windowHigh, v->ctVolume ? (v->surfaceVisible ? v->surfaceOpacity : 0.f) : v->sliceOpacity, v->interactive, raySamples);
        if (v->volumeUploading)
            update();
    }
    if (volumetric)
        return;
    if (v->slices.empty() || !v->ctProgram || !v->ctProgram->isLinked() || !v->ctError.isEmpty())
        return;
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(v->ctVao);
    if (v->sliceTextures.size() < v->slices.size()) {
        // One <= 2 MiB texture per repaint; camera motion never uploads CT.
        const auto texture = uploadCTTexture(v->slices[v->sliceTextures.size()]->tile);
        if (glGetError() != GL_NO_ERROR) {
            glDeleteTextures(1, &texture);
            v->ctError = tr("Cannot upload CT slices.");
        } else
            v->sliceTextures.push_back(texture);
        if (v->sliceTextures.size() < v->slices.size()) {
            glBindVertexArray(0);
            if (v->ctError.isEmpty())
                update();
            return;
        }
    }
    if (!v->sliceBuffer) {
        struct V {
            float x, y, z, u, v;
        };
        std::vector<V> vertices;
        v->sliceOrigin = v->slices.front()->plane.origin;
        for (const auto* polygon : slices::backToFront(v->sliceTree.get(), v->sliceOrigin)) {
            const auto& tile = v->slices[polygon->image]->tile;
            const size_t start = vertices.size();
            auto append = [&](const slices::Vertex& v) {
                const auto p = relative(v.position, this->v->sliceOrigin);
                vertices.push_back(
                    {p.x(), p.y(), p.z(), float((.5 + v.u * (tile.width - 1)) / tile.width), float((.5 + v.v * (tile.height - 1)) / tile.height)});
            };
            for (size_t i = 1; i + 1 < polygon->vertices.size(); ++i) {
                append(polygon->vertices[0]);
                append(polygon->vertices[i]);
                append(polygon->vertices[i + 1]);
            }
            v->sliceRanges[polygon] = {GLint(start), GLsizei(vertices.size() - start)};
        }
        glGenBuffers(1, &v->sliceBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, v->sliceBuffer);
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(vertices.size() * sizeof(V)), vertices.data(), GL_STATIC_DRAW);
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    glDisable(GL_SAMPLE_ALPHA_TO_ONE);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    auto& program = *v->ctProgram;
    program.bind();
    auto matrix = v->matrix(width(), height());
    matrix.translate(relative(v->sliceOrigin, v->origin));
    program.setUniformValue("mvp", matrix);
    program.setUniformValue("ct", 0);
    program.setUniformValue("windowRange", QVector2D(v->windowLow, v->windowHigh));
    program.setUniformValue("opacity", v->sliceOpacity);
    glBindBuffer(GL_ARRAY_BUFFER, v->sliceBuffer);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
    glVertexAttribDivisor(0, 0);
    glVertexAttribDivisor(1, 0);
    const double eyeDistance = v->perspective ? v->scale / std::tan(0.39269908169872415481) : 1.e12;
    const auto offset = v->rotation.conjugated().rotatedVector(QVector3D(0, 0, float(eyeDistance)));
    const Point eye{v->target[0] + offset.x(), v->target[1] + offset.y(), v->target[2] + offset.z()};
    for (const auto* polygon : slices::backToFront(v->sliceTree.get(), eye)) {
        glBindTexture(GL_TEXTURE_2D, v->sliceTextures[polygon->image]);
        const auto [start, count] = v->sliceRanges.at(polygon);
        glDrawArrays(GL_TRIANGLES, start, count);
    }
    program.release();
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
}

void Scene3DCanvas::fitBounds(const Bounds& bounds)
{
    if (!finite(bounds.lower) || !finite(bounds.upper))
        return;
    for (int k = 0; k < 3; ++k)
        v->target[k] = (bounds.lower[k] + bounds.upper[k]) * 0.5;
    const double aspect = double(std::max(1, width())) / std::max(1, height());
    v->scale = std::max(1.0, distance(bounds.lower, bounds.upper) * 0.6) / std::min(1.0, aspect);
    v->fitted = true;
    emit viewScaleChanged();
    update();
}

std::shared_ptr<const surface::Mesh> Scene3DCanvas::surfaceMesh() const
{
    return v->surfaceMesh;
}

bool Scene3DCanvas::isSurfaceMeshReady() const
{
    return v->surfaceMesh && (!v->surfaceVisible || (v->surfaceRenderer && !v->surfaceUploading && v->surfaceRenderer->error().isEmpty()));
}

void Scene3DCanvas::setSurfaceTextured(bool enabled)
{
    v->surfaceTextured = enabled;
    update();
}
void Scene3DCanvas::setSurfaceOpacity(float opacity)
{
    v->surfaceOpacity = std::clamp(opacity, 0.f, 1.f);
    update();
}

void Scene3DCanvas::setSurfaceVisible(bool visible)
{
    if (v->surfaceVisible == visible) return;
    v->surfaceVisible = visible;
    emit surfaceDisplayChanged();
    update();
}

bool Scene3DCanvas::isSurfaceVisible() const
{
    return v->surfaceVisible;
}

void Scene3DCanvas::fitSurface()
{
    if (v->surfaceMesh) {
        fitBounds(v->surfaceMesh->bounds);
    }
}

void Scene3DCanvas::focusPoint(Point point)
{
    if (!finite(point))
        return;
    v->target = point;
    update();
}

void Scene3DCanvas::mousePressEvent(QMouseEvent* event)
{
    if (event->button() != Qt::LeftButton && event->button() != Qt::RightButton && event->button() != Qt::MiddleButton) {
        QOpenGLWidget::mousePressEvent(event);
        return;
    }
    v->pressed = event->button();
    v->pressPosition = v->previousPosition = event->position();
    v->dragging = false;
    setFocus(Qt::MouseFocusReason);
    event->accept();
}

void Scene3DCanvas::mouseMoveEvent(QMouseEvent* event)
{
    if (v->pressed == Qt::NoButton)
        return;
    if (!v->dragging && (event->position() - v->pressPosition).manhattanLength() < QApplication::startDragDistance())
        return;
    v->dragging = true;
    v->interactive = true;
    v->refine.start();
    const QPointF delta = event->position() - v->previousPosition;
    v->previousPosition = event->position();
    if (v->pressed == Qt::LeftButton && !(event->modifiers() & Qt::ShiftModifier)) {
        v->rotation = QQuaternion::fromEulerAngles(float(delta.y() * .4), float(delta.x() * .4), 0) * v->rotation;
        v->rotation.normalize();
    } else {
        const double unitPerPixel = v->scale * 2.0 / std::max(1, height());
        const QVector3D shift =
            v->rotation.conjugated().rotatedVector(QVector3D(float(-delta.x() * unitPerPixel), float(delta.y() * unitPerPixel), 0));
        v->target[0] += shift.x();
        v->target[1] += shift.y();
        v->target[2] += shift.z();
    }
    update();
    event->accept();
}

void Scene3DCanvas::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() != v->pressed)
        return;
    // Context menus are handled here (release), never on a right-drag.
    if (!v->dragging && v->pressed != Qt::MiddleButton)
        emit positionClicked(event->position(), v->pressed == Qt::RightButton, event->globalPosition().toPoint());
    v->pressed = Qt::NoButton;
    event->accept();
}

void Scene3DCanvas::contextMenuEvent(QContextMenuEvent* event)
{
    if (event->reason() == QContextMenuEvent::Keyboard)
        emit emptyContextRequested(event->globalPos());
    event->accept();
}

void Scene3DCanvas::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Escape) {
        emit clearSelectionRequested();
        event->accept();
        return;
    }
    QOpenGLWidget::keyPressEvent(event);
}

void Scene3DCanvas::wheelEvent(QWheelEvent* event)
{
    v->interactive = true;
    v->refine.start();
    const double steps = !event->pixelDelta().isNull() ? event->pixelDelta().y() / 120.0 : event->angleDelta().y() / 120.0;
    v->scale = std::clamp(v->scale * std::exp(-steps * 0.15), 0.01, 1.0e9);
    emit viewScaleChanged();
    update();
    event->accept();
}

void Scene3DCanvas::setSurfaceMesh(std::shared_ptr<const surface::Mesh> mesh)
{
    if (mesh && mesh->indices.empty()) mesh.reset();
    if (v->surfaceMesh == mesh) return;
    v->surfaceMesh = std::move(mesh);
    v->surfaceUploading = bool(v->surfaceMesh);
    emit surfaceDisplayChanged();
    if (!v->fitted && v->surfaceVisible && v->surfaceMesh) fitSurface();
    update();
}

}
