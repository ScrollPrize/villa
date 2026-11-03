#include "Fiber3DCanvas.hpp"
#include "scene3d/Scene3DCanvasState.hpp"
#include "Fiber3DColor.hpp"
#include "Fiber3DRenderGeometry.hpp"
#include "Fiber3DRibbon.hpp"
#include "Fiber3DSlices.hpp"
#include "Fiber3DVolumeRenderer.hpp"
#include "Fiber3DSurfaceRenderer.hpp"
#include "Fiber3DSurfaceProximity.hpp"
#include <unordered_map>
#include <QFutureWatcher>
#include <QTimer>
#include <QtConcurrent/QtConcurrentRun>

#include <QApplication>
#include <QContextMenuEvent>
#include <QKeyEvent>
#include <QLinearGradient>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QPainter>
#include <QPainterPath>
#include <QQuaternion>
#include <QSurfaceFormat>
#include <QVector2D>
#include <QVector4D>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace vc3d::fibers3d
{
namespace
{
using render::distance;
using render::emptyBounds;
using render::expand;
using render::Segment;
using render::Vertex;
using scene3d::relative;
QVector3D rgb(const QColor& color)
{
    return {float(color.redF()), float(color.greenF()), float(color.blueF())};
}
constexpr auto vertexShader = R"glsl(
#version 330 core
layout(location=0) in vec3 a;
layout(location=1) in vec3 b;
layout(location=2) in vec4 info; // full length, family, span style, fiber index
layout(location=3) in float arc;
layout(location=4) in vec3 axialDirection;
uniform mat4 mvp;
uniform vec2 viewport;
uniform float width, selectedWidthBoost, selectedFiber, otherOpacity;
uniform bool showAnnotations, selectedAnnotationsOnly, showGaps, hideSelected, highlightSelection;
uniform int colorMode, directionMode;
uniform vec3 customColor, horizontalColor, verticalColor, selectedColor;
uniform sampler2D palette;
uniform vec2 lengthRange;
noperspective out vec2 positionPx;
flat out float segmentLengthPx, radiusPx, dashStartPx;
flat out int pattern, marker;
flat out vec4 color;

vec3 ramp(float t) {
    return texture(palette, vec2((clamp(t, 0.0, 1.0) * 255.0 + 0.5) / 256.0, 0.5)).rgb;
}
void main() {
    bool selected = abs(info.w - selectedFiber) < 0.25;
    bool annotated = showAnnotations && (!selectedAnnotationsOnly || selected);
    marker = info.z >= 10.0 ? int(info.z) : 0;
    pattern = 0;
    if (showGaps && (info.z == 1.0 || info.z == 2.0)) pattern = 1;
    if (annotated && info.z == 3.0) pattern = 2;
    vec3 c = customColor;
    if (colorMode == 1) c = info.y == 1.0 ? horizontalColor :
                           (info.y == 2.0 ? verticalColor : vec3(0.68));
    if (colorMode == 2) {
        vec3 t = axialDirection;
        float len = length(t);
        c = len == 0.0 ? vec3(0.8) : (directionMode == 0 ? t :
            ramp(acos(clamp(t[directionMode - 1], 0.0, 1.0)) / 1.57079632679));
    }
    if (colorMode == 3) c = info.x < 0.0 ? vec3(0.6) :
        ramp(lengthRange.y > lengthRange.x ?
             (info.x - lengthRange.x) / (lengthRange.y - lengthRange.x) : 0.5);
    if (selected && highlightSelection) c = selectedColor;
    color = vec4(c, selectedFiber < 0.0 || selected ? 1.0 : otherOpacity);
    vec4 p = mvp * vec4(a, 1.0), q = mvp * vec4(b, 1.0);
    float nearA = p.z + p.w, nearB = q.z + q.w;
    if ((nearA <= 0.0 && nearB <= 0.0) || (marker != 0 && !annotated) || (selected && hideSelected)) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        positionPx = vec2(0.0); segmentLengthPx = 0.0;
        radiusPx = 0.0; dashStartPx = 0.0;
        return;
    }
    // Clip against the near plane BEFORE dividing by w or extruding a quad.
    if (nearA < 0.0) p = mix(p, q, nearA / (nearA - nearB));
    if (nearB < 0.0) q = mix(q, p, nearB / (nearB - (p.z + p.w)));
    vec2 delta = (q.xy / q.w - p.xy / p.w) * viewport * 0.5;
    segmentLengthPx = length(delta);
    vec2 tangent = segmentLengthPx > 0.001 ? delta / segmentLengthPx : vec2(1.0, 0.0);
    vec2 normal = vec2(-tangent.y, tangent.x);
    radiusPx = marker != 0 ? 4.0 : (width + (selected ? selectedWidthBoost : 0.0)) * 0.5;
    // One antialias pixel is outside the opaque stroke.
    float extent = radiusPx + 1.0;
    const vec2 corners[6] = vec2[6](vec2(0,-1), vec2(1,-1), vec2(0,1),
                                  vec2(0,1), vec2(1,-1), vec2(1,1));
    vec2 corner = corners[gl_VertexID];
    vec4 clip = mix(p, q, corner.x);
    vec2 offset = normal * corner.y * extent + tangent * (corner.x * 2.0 - 1.0) * extent;
    clip.xy += offset * 2.0 / viewport * clip.w;
    gl_Position = clip;
    positionPx = vec2(corner.x * segmentLengthPx + (corner.x * 2.0 - 1.0) * extent,
                      corner.y * extent);
    dashStartPx = length(b - a) > 0.0 ? arc * segmentLengthPx / length(b - a) : 0.0;
}
)glsl";

constexpr auto fragmentShader = R"glsl(
#version 330 core
noperspective in vec2 positionPx;
flat in float segmentLengthPx, radiusPx, dashStartPx;
flat in int pattern, marker;
flat in vec4 color;
uniform int coveragePass; // 0: MSAA coverage; 1: opaque core; 2: antialias fringe
out vec4 fragColor;
void main() {
    float x = positionPx.x;
    float dx = max(max(-x, x - segmentLengthPx), 0.0);
    float edge = length(vec2(dx, positionPx.y));
    if (marker == 11) edge = max(abs(x), abs(positionPx.y));
    if (marker == 12) edge = (abs(x) + abs(positionPx.y)) * 0.70710678;
    float coverage = clamp(radiusPx + 0.5 - edge, 0.0, 1.0);
    if (coverage == 0.0 || color.a == 0.0) discard;
    if (coveragePass == 1 && coverage < 1.0) discard;
    if (coveragePass == 2 && coverage >= 1.0) discard;
    float phase = mod(max(x, 0.0) + dashStartPx, 12.0);
    if (pattern == 1 && phase > 7.0) discard;
    if (pattern == 2 && (phase > 5.0 && (phase < 8.0 || phase > 9.5))) discard;
    fragColor = vec4(color.rgb, color.a * coverage);
}
)glsl";
}  // namespace

struct Fiber3DCanvas::Impl {
    struct ProximityResult {
        render::Visibility visible;
        std::shared_ptr<const surface::Index> surfaceIndex;
    };
    std::shared_ptr<const surface::Index> surfaceIndex;
    bool nearSurfaceOnly = false;
    double surfaceDistance = 64;
    struct RibbonTile {
        GLuint texture = 0, buffer = 0;
        GLsizei vertices = 0;
    };
    std::shared_ptr<const ribbon::Image> ribbon;
    std::vector<RibbonTile> ribbonTiles;
    Point ribbonOrigin{};
    bool ribbonDirty = false;
    size_t ribbonTextureRefresh = SIZE_MAX;
    std::shared_ptr<const Scene> scene;
    DisplaySettings settings;
    QString selectedKey, error;
    int selectedIndex = -1;
    std::shared_ptr<render::Geometry> geometry;
    std::shared_ptr<const Scene> requested;
    QFutureWatcher<std::shared_ptr<render::Geometry>> preparation;
    std::shared_ptr<std::atomic_bool> cancelled;
    QFutureWatcher<ProximityResult> proximity;
    std::shared_ptr<std::atomic_bool> proximityCancelled;
    bool proximityPending = false, proximityBusy = false;
    render::Visibility visible;
    std::vector<render::Range> visibleMarkers;
    QString proximityError;
    bool preparing = false, pending = false, fitPending = false, selectedFitPending = false;
    std::array<GLuint, render::tolerances.size()> buffers{};
    std::array<size_t, render::tolerances.size()> uploaded{};
    std::array<bool, render::tolerances.size()> allocated{};
    std::vector<Vertex> selectedVertices;
    bool selectedDirty = true;
    GLuint selectedBuffer = 0, markerBuffer = 0;
    bool markersDirty = true;
    std::vector<render::Draw> commands;
    std::vector<render::Range> exactRanges;
    QMatrix4x4 commandMatrix;
    double commandTolerance = -1;
    int commandWidth = 0, commandHeight = 0;
    float commandLineWidth = -1;
    Bounds bounds = emptyBounds();
    std::optional<Point> cursorPosition, linePosition;
    bool dirty = false;
    GLuint vao = 0, paletteTexture = 0;
    int uploadedPalette = -1;
    std::unique_ptr<QOpenGLShaderProgram> program;

};

Fiber3DCanvas::Fiber3DCanvas(QWidget* parent) : Scene3DCanvas(parent), d(std::make_unique<Impl>())
{
    setToolTip(tr("Drag: orbit · Right-drag: pan · Wheel: zoom · Click: select · Right-click: actions"));
    connect(this, &Scene3DCanvas::positionClicked, this, &Fiber3DCanvas::pick);
    connect(&d->proximity, &QFutureWatcher<Impl::ProximityResult>::finished, this, [this] {
        d->proximityBusy = false;
        try {
            auto result = d->proximity.future().takeResult();
            if (!d->proximityCancelled->load() && !d->proximityPending) {
                d->visible = std::move(result.visible);
                d->surfaceIndex = std::move(result.surfaceIndex);
            }
        } catch (const std::exception& e) {
            if (!d->proximityCancelled->load() && !d->proximityPending)
                d->proximityError = tr("Cannot filter nearby fibers: %1").arg(QString::fromUtf8(e.what()));
        }
        if (d->proximityPending)
            startProximityBuild();
        visibilityUpdated();
    });
    connect(&d->preparation, &QFutureWatcher<std::shared_ptr<render::Geometry>>::finished, this, [this] {
        d->preparing = false;
        try {
            // Consume the future's result so clearing a scene also releases its
            // prepared arrays; the watcher must not keep another owner forever.
            auto geometry = d->preparation.future().takeResult();
            if (!d->pending && d->requested) {
                d->geometry = std::move(geometry);
                d->scene = d->geometry->scene;
                d->bounds = d->geometry->bounds;
                v->origin = d->geometry->origin;
                v->radius = d->geometry->chunks.empty() ? 1 : std::max(1., distance(d->bounds.lower, d->bounds.upper) * .5);
                d->dirty = true;
                d->commandTolerance = -1;
                setSelectedKey(d->selectedKey);
                if (!v->fitted || d->fitPending)
                    fit(d->selectedFitPending);
                d->fitPending = false;
                if (v->glReady && d->program && d->program->isLinked())
                    d->error.clear();
            }
        } catch (const std::exception& e) {
            if (!d->pending && d->requested)
                d->error = tr("Cannot prepare fibers: %1").arg(QString::fromUtf8(e.what()));
        }
        if (d->pending)
            startGeometryBuild();
        update();
    });
}

Fiber3DCanvas::~Fiber3DCanvas()
{
    if (d->cancelled)
        d->cancelled->store(true);
    if (d->proximityCancelled)
        d->proximityCancelled->store(true);
    // Workers own immutable snapshots and cancellation tokens. Closing the
    // view must not wait on their CPU work on the GUI thread.
    cleanupGL();
}

void Fiber3DCanvas::cleanupGL()
{
    // The base QOpenGLWidget destructor can destroy its context after our
    // private data has gone away. It must not call this slot a second time.
    if (context())
        disconnect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &Fiber3DCanvas::cleanupGL);
    if (!context() || !v->glReady)
        return;
    makeCurrent();
    for (auto& tile : d->ribbonTiles) {
        glDeleteTextures(1, &tile.texture);
        glDeleteBuffers(1, &tile.buffer);
    }
    d->ribbonTiles.clear();
    d->ribbonDirty = true;
    d->program.reset();
    glDeleteBuffers(GLsizei(d->buffers.size()), d->buffers.data());
    d->buffers.fill(0);
    if (d->selectedBuffer)
        glDeleteBuffers(1, &d->selectedBuffer);
    if (d->markerBuffer)
        glDeleteBuffers(1, &d->markerBuffer);
    d->selectedBuffer = d->markerBuffer = 0;
    if (d->vao)
        glDeleteVertexArrays(1, &d->vao);
    if (d->paletteTexture)
        glDeleteTextures(1, &d->paletteTexture);
    d->vao = d->paletteTexture = 0;
    d->uploadedPalette = -1;
    d->dirty = true;
    doneCurrent();
}

void Fiber3DCanvas::initializeGL()
{
    // Release fiber resources before the base cleanup marks the context unavailable.
    connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &Fiber3DCanvas::cleanupGL, Qt::DirectConnection);
    Scene3DCanvas::initializeGL();
    if (!v->glReady) return;
    d->program = std::make_unique<QOpenGLShaderProgram>();
    if (!d->program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShader) ||
        !d->program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShader) || !d->program->link()) {
        d->error = tr("Cannot initialize Fibers 3D: %1").arg(d->program->log());
        return;
    }
    d->error.clear();
    glGenVertexArrays(1, &d->vao);
    glGenTextures(1, &d->paletteTexture);
    glGenBuffers(GLsizei(d->buffers.size()), d->buffers.data());
    glGenBuffers(1, &d->selectedBuffer);
    glGenBuffers(1, &d->markerBuffer);
    d->ribbonDirty = true;
    d->dirty = true;
}

void Fiber3DCanvas::setCTRibbon(std::shared_ptr<const ribbon::Image> image)
{
    if (image == d->ribbon)
        return;
    bool sameMesh = image && d->ribbon && image->key == d->ribbon->key && image->tiles.size() == d->ribbon->tiles.size();
    if (sameMesh)
        for (size_t i = 0; i < image->tiles.size(); ++i) {
            const auto& a = image->tiles[i];
            const auto& b = d->ribbon->tiles[i];
            if (a.width != b.width || a.height != b.height || a.left != b.left || a.right != b.right) {
                sameMesh = false;
                break;
            }
        }
    d->ribbon = std::move(image);
    if (sameMesh && !d->ribbonDirty)
        d->ribbonTextureRefresh = 0;
    else
        d->ribbonDirty = true;
    update();
}
std::shared_ptr<const ribbon::Image> Fiber3DCanvas::ctRibbon() const
{
    return d->ribbon;
}

void Fiber3DCanvas::drawCTRibbon()
{
    if (d->ribbonDirty) {
        for (auto& tile : d->ribbonTiles) {
            glDeleteTextures(1, &tile.texture);
            glDeleteBuffers(1, &tile.buffer);
        }
        d->ribbonTiles.clear();
        d->ribbonDirty = false;
        d->ribbonTextureRefresh = SIZE_MAX;
        if (v->ctProgram && v->ctProgram->isLinked())
            v->ctError.clear();
    }
    if (!d->ribbon || !v->ctProgram || !v->ctError.isEmpty() || d->ribbon->key != d->selectedKey || !d->geometry ||
        (d->selectedIndex >= 0 && !d->visible.includes(size_t(d->selectedIndex))))
        return;
    const auto& image = *d->ribbon;
    if (image.tiles.empty())
        return;
    d->ribbonOrigin = image.tiles.front().left.front();
    glBindVertexArray(v->ctVao);
    glActiveTexture(GL_TEXTURE0);
    // Refine textures without discarding already visible geometry. Bound GPU
    // transfers per paint; a new chunk must not restart a multi-frame upload.
    size_t refreshed = 0;
    while (d->ribbonTextureRefresh < d->ribbonTiles.size() && refreshed < 1024 * 1024) {
        const size_t index = d->ribbonTextureRefresh++;
        const auto& tile = image.tiles[index];
        glBindTexture(GL_TEXTURE_2D, d->ribbonTiles[index].texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tile.width, tile.height, GL_RG, GL_UNSIGNED_BYTE, tile.pixels.data());
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        refreshed += tile.pixels.size();
    }
    if (d->ribbonTextureRefresh < d->ribbonTiles.size())
        update();
    // At most one <= 1 MiB new tile per frame. Camera changes reuse these objects.
    if (d->ribbonTiles.size() < image.tiles.size()) {
        const auto& tile = image.tiles[d->ribbonTiles.size()];
        Impl::RibbonTile gpu;
        struct V {
            float x, y, z, u, v;
        };
        std::vector<V> vertices;
        vertices.reserve(size_t(tile.height) * 2);
        for (int r = 0; r < tile.height; ++r)
            for (int side = 0; side < 2; ++side) {
                const auto p = relative(side ? tile.right[r] : tile.left[r], d->ribbonOrigin);
                vertices.push_back({p.x(), p.y(), p.z(), (side ? tile.width - .5f : .5f) / tile.width, (r + .5f) / tile.height});
            }
        gpu.vertices = GLsizei(vertices.size());
        glGenBuffers(1, &gpu.buffer);
        glBindBuffer(GL_ARRAY_BUFFER, gpu.buffer);
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(vertices.size() * sizeof(V)), vertices.data(), GL_STATIC_DRAW);
        gpu.texture = uploadCTTexture(tile);
        if (glGetError() != GL_NO_ERROR) {
            glDeleteTextures(1, &gpu.texture);
            glDeleteBuffers(1, &gpu.buffer);
            v->ctError = tr("Cannot upload CT ribbon; reduce its width.");
        } else
            d->ribbonTiles.push_back(gpu);
        if (d->ribbonTiles.size() < image.tiles.size() && v->ctError.isEmpty())
            update();
    }
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    glDisable(GL_SAMPLE_ALPHA_TO_ONE);
    glDisable(GL_SAMPLE_COVERAGE);
    glEnable(GL_MULTISAMPLE);
    // A small depth bias lets the centerline remain readable on its CT plane.
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1, 1);
    auto mvp = v->matrix(width(), height());
    mvp.translate(relative(d->ribbonOrigin, v->origin));
    auto& program = *v->ctProgram;
    program.bind();
    program.setUniformValue("mvp", mvp);
    program.setUniformValue("ct", 0);
    program.setUniformValue("windowRange", QVector2D(v->windowLow, v->windowHigh));
    program.setUniformValue("opacity", 1.f);
    for (const auto& tile : d->ribbonTiles) {
        glBindTexture(GL_TEXTURE_2D, tile.texture);
        glBindBuffer(GL_ARRAY_BUFFER, tile.buffer);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
        glVertexAttribDivisor(0, 0);
        glVertexAttribDivisor(1, 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, tile.vertices);
    }
    program.release();
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glDisable(GL_POLYGON_OFFSET_FILL);
    glDisable(GL_DEPTH_TEST);
}

void Fiber3DCanvas::setScene(std::shared_ptr<const Scene> scene)
{
    if (!scene) {
        clear();
        return;
    }
    d->requested = std::move(scene);
    d->pending = true;
    if (d->cancelled)
        d->cancelled->store(true);
    if (!d->preparing)
        startGeometryBuild();
    update();
}

void Fiber3DCanvas::startGeometryBuild()
{
    // A completed future can still have queued result/finished signals. Keep
    // its watcher intact until the finished handler has consumed the result.
    d->preparing = true;
    d->pending = false;
    d->cancelled = std::make_shared<std::atomic_bool>(false);
    const auto scene = d->requested;
    const auto token = d->cancelled;
    d->preparation.setFuture(QtConcurrent::run([scene, token] { return render::build(scene, token.get()); }));
}

std::shared_ptr<const Scene> Fiber3DCanvas::scene() const
{
    return d->scene;
}

bool Fiber3DCanvas::isPreparing() const
{
    return d->preparing || d->pending ||
           (d->geometry && !d->geometry->chunks.empty() && (d->dirty || d->uploaded[0] != d->geometry->levels[0].size()));
}

void Fiber3DCanvas::setDisplaySettings(const DisplaySettings& settings)
{
    const bool proximityChanged = settings.nearbyOnly != d->settings.nearbyOnly || settings.nearbyDistanceVoxels != d->settings.nearbyDistanceVoxels;
    d->settings = settings;
    setPerspective(settings.perspective);
    if (proximityChanged)
        updateProximity();
    update();  // Display changes reuse uploaded geometry.
}

void Fiber3DCanvas::setSelectedKey(const QString& key)
{
    d->selectedKey = key;
    d->selectedIndex = -1;
    if (d->scene && !key.isEmpty())
        for (size_t i = 0; i < d->scene->fibers.size(); ++i)
            if (d->scene->fibers[i].key == key) {
                d->selectedIndex = int(i);
                break;
            }
    updateSelectedGeometry();
    updateProximity();
    update();
}

void Fiber3DCanvas::clear()
{
    setCTRibbon({});
    if (d->cancelled)
        d->cancelled->store(true);
    d->pending = false;
    d->requested.reset();
    d->scene.reset();
    d->geometry.reset();
    d->cursorPosition.reset();
    d->linePosition.reset();
    d->selectedKey.clear();
    d->selectedIndex = -1;
    d->selectedVertices.clear();
    v->fitted = false;
    d->dirty = true;
    updateProximity();
    update();
}

bool Fiber3DCanvas::isFiltering() const
{
    return d->proximityPending || (d->proximityBusy && d->proximityCancelled && !d->proximityCancelled->load());
}

size_t Fiber3DCanvas::visibleFiberCount() const
{
    return d->visible.fibers.empty() ? (d->scene ? d->scene->fibers.size() : 0) : d->visible.count;
}

void Fiber3DCanvas::visibilityUpdated()
{
    d->commandTolerance = -1;
    d->visibleMarkers.clear();
    if (d->geometry)
        for (size_t i = 0; i < d->geometry->markers.size(); ++i) {
            if (!d->visible.includes(size_t(d->geometry->markers[i].fiber)))
                continue;
            if (!d->visibleMarkers.empty() && d->visibleMarkers.back().first + d->visibleMarkers.back().count == i)
                ++d->visibleMarkers.back().count;
            else
                d->visibleMarkers.push_back({i, 1});
        }
    emit visibilityChanged();
    update();
}

void Fiber3DCanvas::updateProximity()
{
    if (d->proximityCancelled)
        d->proximityCancelled->store(true);
    d->proximityPending = false;
    d->visible = {};
    d->proximityError.clear();
    if (d->geometry && ((d->settings.nearbyOnly && d->selectedIndex >= 0) || d->nearSurfaceOnly)) {
        // Never display stale matches. A surface filter starts empty, since
        // even the selected fibre may be outside its requested distance.
        std::vector<uint8_t> keep(d->scene->fibers.size(), 0);
        if (!d->nearSurfaceOnly && d->selectedIndex >= 0)
            keep[d->selectedIndex] = 1;
        d->visible = render::visibility(*d->geometry, std::move(keep));
        d->proximityPending = !d->nearSurfaceOnly || bool(v->surfaceMesh);
        if (d->proximityPending && !d->proximityBusy)
            startProximityBuild();
    }
    visibilityUpdated();
}

void Fiber3DCanvas::startProximityBuild()
{
    d->proximityPending = false;
    d->proximityBusy = true;
    d->proximityCancelled = std::make_shared<std::atomic_bool>(false);
    const auto token = d->proximityCancelled;
    const auto geometry = d->geometry;
    const auto selected = size_t(d->selectedIndex);
    const auto radius = d->settings.nearbyDistanceVoxels;
    const bool aroundSelected = d->settings.nearbyOnly && d->selectedIndex >= 0;
    const auto mesh = d->nearSurfaceOnly ? v->surfaceMesh : nullptr;
    const auto index = d->surfaceIndex;
    const double surfaceRadius = d->surfaceDistance;
    d->proximity.setFuture(QtConcurrent::run([geometry, selected, radius, token, aroundSelected, mesh, index, surfaceRadius] {
        Impl::ProximityResult result;
        result.surfaceIndex = index;
        if (aroundSelected)
            result.visible = render::nearbyFibers(*geometry, selected, radius, token.get());
        if (mesh) {
            if (!result.surfaceIndex || result.surfaceIndex->mesh != mesh)
                result.surfaceIndex = surface::index(mesh, *token);
            result.visible = surface::nearby(*geometry, *result.surfaceIndex, surfaceRadius, result.visible, *token);
        }
        return result;
    }));
}

void Fiber3DCanvas::updateSelectedGeometry()
{
    d->selectedVertices.clear();
    d->selectedDirty = true;
    if (!d->geometry || d->selectedIndex < 0)
        return;
    const auto& g = *d->geometry;
    size_t count = 0;
    for (auto ci : g.fiberChunks[d->selectedIndex])
        count += g.chunks[ci].levels[0].count;
    d->selectedVertices.reserve(count);
    for (auto ci : g.fiberChunks[d->selectedIndex]) {
        render::appendExact(g, g.chunks[ci], d->selectedVertices);
    }
}

void Fiber3DCanvas::fit(bool selectedOnly)
{
    if (d->preparing || d->pending) {
        d->fitPending = true;
        d->selectedFitPending = selectedOnly;
    }
    Bounds bounds = d->bounds;
    if (!d->geometry || d->geometry->chunks.empty()) {
        if (selectedOnly)
            return;
        bounds = emptyBounds();
        for (const auto& image : v->slices)
            for (const auto& p : slices::corners(*image))
                expand(bounds, p);
    } else if (selectedOnly) {
        if (d->selectedIndex < 0 || !d->visible.includes(size_t(d->selectedIndex)))
            return;
        bounds = d->geometry->fiberBounds[d->selectedIndex];
    } else if (!d->visible.fibers.empty()) {
        bounds = emptyBounds();
        for (size_t i = 0; i < d->visible.fibers.size(); ++i) {
            const auto& b = d->geometry->fiberBounds[i];
            if (!d->visible.includes(i) || !finite(b.lower))
                continue;
            expand(bounds, b.lower);
            expand(bounds, b.upper);
        }
    }
    if (!selectedOnly && v->surfaceVisible && v->surfaceMesh && !v->surfaceMesh->indices.empty()) {
        expand(bounds, v->surfaceMesh->bounds.lower);
        expand(bounds, v->surfaceMesh->bounds.upper);
    }
    fitBounds(bounds);
}

void Fiber3DCanvas::fitAll()
{
    fit(false);
}

void Fiber3DCanvas::setSurfaceFilter(bool enabled, double distanceVoxels)
{
    if (d->nearSurfaceOnly == enabled && d->surfaceDistance == distanceVoxels)
        return;
    d->nearSurfaceOnly = enabled;
    d->surfaceDistance = distanceVoxels;
    updateProximity();
}

void Fiber3DCanvas::fitSelected()
{
    fit(true);
}

void Fiber3DCanvas::setCursorPosition(std::optional<Point> position)
{
    if (position && !finite(*position))
        position.reset();
    if (d->cursorPosition == position)
        return;
    d->cursorPosition = position;
    update();
}

void Fiber3DCanvas::setLinePosition(std::optional<Point> position)
{
    if (d->linePosition == position)
        return;
    d->linePosition = position;
    update();
}

void Fiber3DCanvas::paintGL()
{
    beginFrame();
    const qreal ratio = devicePixelRatioF();
    const int pixelWidth = std::max(1, int(width() * ratio)), pixelHeight = std::max(1, int(height() * ratio));
    if (v->glReady) {
        if (d->dirty) {
            // Release the previous scene before allocating the new resident levels.
            glDeleteBuffers(GLsizei(d->buffers.size()), d->buffers.data());
            glGenBuffers(GLsizei(d->buffers.size()), d->buffers.data());
            glDeleteBuffers(1, &d->selectedBuffer);
            glDeleteBuffers(1, &d->markerBuffer);
            glGenBuffers(1, &d->selectedBuffer);
            glGenBuffers(1, &d->markerBuffer);
            d->uploaded.fill(0);
            d->allocated.fill(false);
            d->selectedDirty = d->markersDirty = true;
            d->dirty = false;
            d->commandTolerance = -1;
        }
    }
    const auto matrix = v->matrix(width(), height());
    drawSurface();
    if (v->glReady)
        drawCTRibbon();
    if (v->glReady && d->error.isEmpty() && d->program && d->geometry && !d->geometry->chunks.empty()) {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        // Test the actual widget framebuffer; the requested format may not be
        // available. Fractionally covered cap/edge pixels must not occlude the
        // opaque center of the next segment in a densely sampled curve.
        GLint samples = 0;
        glGetIntegerv(GL_SAMPLES, &samples);
        glEnable(GL_MULTISAMPLE);
        glDisable(GL_SAMPLE_COVERAGE);
        glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
        glDisable(GL_SAMPLE_ALPHA_TO_ONE);
        if (samples > 1) {
            glDisable(GL_BLEND);
            glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
            // Coverage already supplies alpha at each sample. Keep the resolved
            // widget opaque, avoiding another alpha multiplication by Qt.
            glEnable(GL_SAMPLE_ALPHA_TO_ONE);
        } else {
            glEnable(GL_BLEND);
        }
        glBlendEquation(GL_FUNC_ADD);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_CULL_FACE);
        glBindVertexArray(d->vao);
        const auto& geometry = *d->geometry;
        // Upload progressively, coarse levels first. Each repaint transfers at
        // most 8 MiB. Large scenes keep exact detail on demand; their resident
        // coarse levels and spatial index are reused during camera motion.
        size_t uploadBudget = 8 * 1024 * 1024 / sizeof(Vertex);
        bool uploading = false;
        for (size_t li = d->buffers.size(); li-- > 0;) {
            const auto& data = geometry.levels[li];
            if (d->uploaded[li] == data.size())
                continue;
            uploading = true;
            if (!uploadBudget)
                break;
            glBindBuffer(GL_ARRAY_BUFFER, d->buffers[li]);
            if (!d->allocated[li]) {
                glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(data.size() * sizeof(Vertex)), nullptr, GL_STATIC_DRAW);
                if (glGetError() != GL_NO_ERROR) {
                    d->error = tr("Not enough graphics memory. Reduce the displayed region or Memory budget.");
                    break;
                }
                d->allocated[li] = true;
            }
            const size_t count = std::min(uploadBudget, data.size() - d->uploaded[li]);
            glBufferSubData(GL_ARRAY_BUFFER, GLintptr(d->uploaded[li] * sizeof(Vertex)), GLsizeiptr(count * sizeof(Vertex)), data.data() + d->uploaded[li]);
            if (glGetError() != GL_NO_ERROR) {
                d->error = tr("Cannot upload fiber geometry. Reduce the displayed region or Memory budget.");
                break;
            }
            d->uploaded[li] += count;
            uploadBudget -= count;
        }
        if (uploading)
            update();
        if (d->selectedDirty) {
            glBindBuffer(GL_ARRAY_BUFFER, d->selectedBuffer);
            glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(d->selectedVertices.size() * sizeof(Vertex)), d->selectedVertices.data(), GL_STATIC_DRAW);
            d->selectedDirty = false;
        }
        if (d->settings.showAnnotations && d->markersDirty) {
            glBindBuffer(GL_ARRAY_BUFFER, d->markerBuffer);
            glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(geometry.markers.size() * sizeof(Vertex)), geometry.markers.data(), GL_STATIC_DRAW);
            d->markersDirty = false;
        }
        auto& program = *d->program;
        const auto& s = d->settings;
        program.bind();
        program.setUniformValue("mvp", matrix);
        program.setUniformValue("viewport", QVector2D(float(pixelWidth), float(pixelHeight)));
        program.setUniformValue("width", float(std::clamp(s.lineWidth, 0.5f, 12.f) * ratio));
        program.setUniformValue("selectedWidthBoost", s.highlightSelection ? float(2.0 * ratio) : 0.f);
        program.setUniformValue("highlightSelection", s.highlightSelection);
        const bool selectedVisible = d->selectedIndex >= 0 && d->visible.includes(size_t(d->selectedIndex));
        program.setUniformValue("selectedFiber", selectedVisible ? float(d->selectedIndex) : -1.f);
        program.setUniformValue("otherOpacity", selectedVisible ? std::clamp(s.unselectedOpacity, 0.f, 1.f) : 1.f);
        program.setUniformValue("showGaps", s.showGaps);
        program.setUniformValue("showAnnotations", s.showAnnotations);
        program.setUniformValue("selectedAnnotationsOnly", s.annotationsSelectedOnly);
        program.setUniformValue("colorMode", int(s.colorMode));
        program.setUniformValue("directionMode", int(s.directionMode));
        program.setUniformValue("customColor", rgb(s.custom));
        program.setUniformValue("horizontalColor", rgb(s.horizontal));
        program.setUniformValue("verticalColor", rgb(s.vertical));
        program.setUniformValue("selectedColor", rgb(vc3d::selectedFiberColor()));
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, d->paletteTexture);
        if (d->uploadedPalette != int(s.palette)) {
            std::array<uint8_t, 256 * 3> colors;
            const auto& stops = paletteStops(s.palette);
            for (int i = 0; i < 256; ++i) {
                colors[i * 3] = uint8_t(stops[i].red());
                colors[i * 3 + 1] = uint8_t(stops[i].green());
                colors[i * 3 + 2] = uint8_t(stops[i].blue());
            }
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 256, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, colors.data());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            d->uploadedPalette = int(s.palette);
        }
        program.setUniformValue("palette", 0);
        const auto range = s.fixedLengthRange ? lengthRange(*d->scene, s) : geometry.lengths;
        program.setUniformValue("lengthRange", QVector2D(float(range.first), float(range.second)));
        const double tolerance = v->interactive ? 1.25 : .5;  // physical screen pixels
        if (d->commandMatrix != matrix || d->commandTolerance != tolerance || d->commandWidth != pixelWidth ||
            d->commandHeight != pixelHeight || d->commandLineWidth != s.lineWidth) {
            d->commands =
                render::draws(geometry, render::View(matrix, v->origin, pixelWidth, pixelHeight, (s.lineWidth + 2) * ratio), tolerance, d->visible);
            d->exactRanges.assign(d->commands.size(), {});
            if (geometry.deferredExact) {
                std::vector<Vertex> exact;
                for (size_t i = 0; i < d->commands.size(); ++i) {
                    const auto& command = d->commands[i];
                    if (command.level != 0)
                        continue;
                    const auto first = exact.size();
                    for (size_t ci = command.begin; ci < command.end; ++ci)
                        render::appendExact(geometry, geometry.chunks[ci], exact);
                    d->exactRanges[i] = {first, exact.size() - first};
                }
                glBindBuffer(GL_ARRAY_BUFFER, d->buffers[0]);
                glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(exact.size() * sizeof(Vertex)), exact.data(), GL_DYNAMIC_DRAW);
                if (glGetError() != GL_NO_ERROR)
                    d->error = tr("Not enough graphics memory for exact fiber detail. Reduce the displayed region.");
            }
            d->commandMatrix = matrix;
            d->commandTolerance = tolerance;
            d->commandWidth = pixelWidth;
            d->commandHeight = pixelHeight;
            d->commandLineWidth = s.lineWidth;
        }
        auto drawRange = [&](GLuint buffer, size_t first, size_t count) {
            if (!count)
                return;
            glBindBuffer(GL_ARRAY_BUFFER, buffer);
            const int sizes[] = {3, 3, 4, 1, 3};
            const size_t offsets[] = {offsetof(Vertex, a), offsetof(Vertex, b), offsetof(Vertex, length), offsetof(Vertex, arc), offsetof(Vertex, direction)};
            for (int i = 0; i < 5; ++i) {
                glEnableVertexAttribArray(i);
                glVertexAttribPointer(i, sizes[i], GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(first * sizeof(Vertex) + offsets[i]));
                glVertexAttribDivisor(i, 1);
            }
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, GLsizei(count));
        };
        auto draw = [&] {
            program.setUniformValue("hideSelected", true);
            if (!selectedVisible || s.unselectedOpacity > 0)
                for (size_t i = 0; i < d->commands.size(); ++i) {
                    const auto& command = d->commands[i];
                    if (geometry.deferredExact && command.level == 0) {
                        const auto r = d->exactRanges[i];
                        drawRange(d->buffers[0], r.first, r.count);
                        continue;
                    }
                    size_t level = command.level;
                    while (level < d->buffers.size() && d->uploaded[level] != geometry.levels[level].size())
                        ++level;
                    if (level == d->buffers.size())
                        continue;
                    const auto first = geometry.chunks[command.begin].levels[level], last = geometry.chunks[command.end - 1].levels[level];
                    drawRange(d->buffers[level], first.first, last.first + last.count - first.first);
                }
            program.setUniformValue("hideSelected", false);
            if (selectedVisible)
                drawRange(d->selectedBuffer, 0, d->selectedVertices.size());
            if (s.showAnnotations)
                for (const auto& r : d->visibleMarkers)
                    drawRange(d->markerBuffer, r.first, r.count);
        };
        program.setUniformValue("coveragePass", samples > 1 ? 0 : 1);
        draw();
        if (samples <= 1) {
            glDepthMask(GL_FALSE);
            program.setUniformValue("coveragePass", 2);
            draw();
            glDepthMask(GL_TRUE);
        }
        program.release();
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindVertexArray(0);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
        glDisable(GL_SAMPLE_ALPHA_TO_ONE);
    }
    if (v->glReady)
        drawCTSlices();
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(QColor(210, 215, 222));
    if (v->surfaceVisible && v->surfaceMesh && v->surfaceRenderer) {
        if (!v->surfaceRenderer->error().isEmpty())
            painter.drawText(rect().adjusted(12, 66, -12, -12), Qt::AlignTop | Qt::TextWordWrap, v->surfaceRenderer->error());
        else if (v->surfaceUploading)
            painter.drawText(rect().adjusted(12, 66, -12, -12), Qt::AlignTop, tr("Uploading surface mesh…"));
    }
    if (v->volumeRenderer && !v->slices.empty() && v->slices.front()->slab && !v->volumeRenderer->error().isEmpty())
        painter.drawText(rect().adjusted(12, 40, -12, -12), Qt::AlignTop | Qt::TextWordWrap, v->volumeRenderer->error());
    if ((d->ribbon || !v->slices.empty()) && !v->ctError.isEmpty())
        painter.drawText(rect().adjusted(12, 12, -12, -12), Qt::AlignTop | Qt::TextWordWrap, v->ctError);
    if (!d->error.isEmpty()) {
        painter.drawText(rect().adjusted(16, 16, -16, -16), Qt::AlignCenter | Qt::TextWordWrap, d->error);
        return;
    }
    if (!d->geometry || d->geometry->chunks.empty()) {
        if (v->slices.empty() && !v->surfaceMesh)
            painter.drawText(
                rect().adjusted(16, 16, -16, -16),
                Qt::AlignCenter | Qt::TextWordWrap,
                d->preparation.isRunning()
                    ? tr("Preparing fibers…")
                    : (d->scene && !d->scene->message.isEmpty() ? d->scene->message : tr("No fibers in the displayed region")));
        return;
    }
    if (d->settings.showBounds) {
        const auto bounds = d->scene->clipBounds.value_or(d->bounds);
        painter.setPen(QPen(QColor(150, 170, 190, 65), 1));
        std::array<QVector4D, 8> corners;
        for (int i = 0; i < 8; ++i) {
            Point p;
            for (int k = 0; k < 3; ++k)
                p[k] = (i & (1 << k)) ? bounds.upper[k] : bounds.lower[k];
            corners[i] = matrix * QVector4D(relative(p, v->origin), 1);
        }
        auto screen = [&](const QVector4D& p) { return QPointF((p.x() / p.w() + 1) * width() * .5, (1 - p.y() / p.w()) * height() * .5); };
        for (int i = 0; i < 8; ++i)
            for (int k = 0; k < 3; ++k) {
                const int j = i ^ (1 << k);
                if (i < j && corners[i].w() > 0 && corners[j].w() > 0 && corners[i].z() >= -corners[i].w() && corners[j].z() >= -corners[j].w())
                    painter.drawLine(screen(corners[i]), screen(corners[j]));
            }
    }
    if (d->linePosition) {
        const QVector4D p = matrix * QVector4D(relative(*d->linePosition, v->origin), 1);
        if (p.w() > 0 && std::isfinite(p.w()) && std::abs(p.x()) <= p.w() && std::abs(p.y()) <= p.w() && std::abs(p.z()) <= p.w()) {
            const QPointF center((p.x() / p.w() + 1) * width() * .5, (1 - p.y() / p.w()) * height() * .5);
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(0, 245, 255));
            painter.drawEllipse(center, 3, 3);
            painter.setBrush(Qt::NoBrush);
        }
    }
    if (d->settings.showCursor && d->cursorPosition) {
        // Subtract the scene origin in double precision before projection, as
        // for the fiber geometry. Do not snap or clamp the CT focus to a fiber.
        const QVector4D p = matrix * QVector4D(relative(*d->cursorPosition, v->origin), 1);
        if (std::isfinite(p.w()) && p.w() > 0 && std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z()) &&
            std::abs(p.x()) <= p.w() && std::abs(p.y()) <= p.w() && std::abs(p.z()) <= p.w()) {
            const QPointF center((p.x() / p.w() + 1) * width() * .5, (1 - p.y() / p.w()) * height() * .5);
            QPainterPath reticle;
            reticle.addEllipse(center, 3, 3);
            for (const QPointF axis : {QPointF(1, 0), QPointF(-1, 0), QPointF(0, 1), QPointF(0, -1)}) {
                reticle.moveTo(center + axis * 6);
                reticle.lineTo(center + axis * 11);
            }
            // Logical pixels keep the navigation marker the same size on Retina.
            painter.setPen(QPen(QColor(10, 14, 18), 4, Qt::SolidLine, Qt::RoundCap));
            painter.drawPath(reticle);
            painter.setPen(QPen(Qt::white, 1.5, Qt::SolidLine, Qt::RoundCap));
            painter.drawPath(reticle);
        }
    }
    painter.setPen(QColor(210, 215, 222));
    const auto mode = d->settings.colorMode;
    QString legend;
    if (mode == ColorMode::Direction && d->settings.directionMode == DirectionMode::XYZ)
        legend = tr("Direction · X red · Y green · Z blue · no preferred sign");
    else if (mode == ColorMode::Family)
        legend = tr("H / V · unclassified gray");
    else if (mode == ColorMode::Custom)
        legend = tr("Uniform color");
    else {
        QLinearGradient gradient(12, 0, 172, 0);
        const auto& stops = paletteStops(d->settings.palette);
        for (int i = 0; i < 256; ++i)
            gradient.setColorAt(i / 255.0, stops[i]);
        painter.fillRect(QRect(12, 34, std::min(160, width() - 24), 8), gradient);
        if (mode == ColorMode::Length) {
            const auto range = d->settings.fixedLengthRange ? lengthRange(*d->scene, d->settings) : d->geometry->lengths;
            legend = tr("Full length · %1–%2 mm · unknown gray").arg(range.first, 0, 'g', 3).arg(range.second, 0, 'g', 3);
        } else {
            const QChar axis = QStringLiteral("XYZ")[int(d->settings.directionMode) - 1];
            legend = tr("Angle to %1 · 0°–90° · no preferred sign").arg(axis);
        }
    }
    painter.drawText(QRect(12, 8, width() - 24, 22), Qt::AlignLeft | Qt::AlignVCenter, legend);
    painter.drawText(QRect(12, height() - 26, width() - 24, 20), Qt::AlignLeft | Qt::AlignVCenter, tr("Drag: orbit · Right-drag: pan · Wheel: zoom"));
    if (d->preparing)
        painter.drawText(QRect(12, 52, width() - 24, 22), Qt::AlignLeft | Qt::AlignVCenter, tr("Preparing fibers…"));
    else if (d->geometry && d->uploaded[0] != d->geometry->levels[0].size())
        painter.drawText(QRect(12, 52, width() - 24, 22), Qt::AlignLeft | Qt::AlignVCenter, tr("Refining display…"));
    if (isFiltering() || !d->proximityError.isEmpty())
        painter.drawText(QRect(12, 78, width() - 24, 44), Qt::AlignLeft | Qt::TextWordWrap, isFiltering() ? tr("Finding nearby fibers…") : d->proximityError);
    if (d->nearSurfaceOnly && !v->surfaceMesh)
        painter.drawText(rect().adjusted(12, 66, -12, -12), Qt::AlignTop | Qt::TextWordWrap, tr("Surface filter: select a surface in Main and wait for its mesh."));
}

void Fiber3DCanvas::pick(const QPointF& position, bool context, const QPoint& globalPosition)
{
    const auto matrix = v->matrix(width(), height());
    if (!d->geometry)
        return;
    std::optional<Segment> best;
    const double tolerance = std::max(6.0, double(d->settings.lineWidth * .5 + 3.0));
    double bestDistance = tolerance;
    double bestDepth = 2.0, bestT = 0.0;
    const auto& g = *d->geometry;
    const auto candidates =
        render::pickChunks(g, render::View(matrix, v->origin, width(), height(), tolerance), position.x() * 2 / width() - 1, 1 - position.y() * 2 / height(), d->visible);
    for (auto ci : candidates) {
        const auto& chunk = g.chunks[ci];
        for (size_t si = chunk.first; si < chunk.end; ++si) {
            Segment segment;
            if (!render::segment(g, chunk, si, segment))
                continue;
            if (d->selectedIndex >= 0 && d->visible.includes(size_t(d->selectedIndex)) && int(segment.fiber) != d->selectedIndex &&
                d->settings.unselectedOpacity <= 0)
                continue;
            QVector4D a = matrix * QVector4D(relative(segment.a, v->origin), 1), b = matrix * QVector4D(relative(segment.b, v->origin), 1);
            const double nearA = a.z() + a.w(), nearB = b.z() + b.w();
            if (nearA <= 0 && nearB <= 0)
                continue;
            double clipA = 0, clipB = 1;
            if (nearA < 0) {
                clipA = nearA / (nearA - nearB);
                a = a * float(1 - clipA) + b * float(clipA);
            }
            if (nearB < 0) {
                clipB = nearA / (nearA - nearB);
                b = matrix * QVector4D(relative(interpolate(segment.a, segment.b, clipB), v->origin), 1);
            }
            if (a.w() <= 0 || b.w() <= 0)
                continue;
            const QPointF pa((a.x() / a.w() + 1) * width() * .5, (1 - a.y() / a.w()) * height() * .5);
            const QPointF pb((b.x() / b.w() + 1) * width() * .5, (1 - b.y() / b.w()) * height() * .5);
            const QPointF delta = pb - pa;
            const double squared = QPointF::dotProduct(delta, delta);
            const double t = squared > 0 ? std::clamp(QPointF::dotProduct(position - pa, delta) / squared, 0.0, 1.0) : 0;
            const QPointF offset = position - (pa + delta * t);
            const double dist = std::hypot(offset.x(), offset.y());
            const double depth = (1 - t) * a.z() / a.w() + t * b.z() / b.w();
            if (depth < -1 || depth > 1 || dist > tolerance)
                continue;
            // At a visual crossing choose the frontmost segment, not iteration order.
            if (best && (dist > bestDistance + 0.75 || (std::abs(dist - bestDistance) <= 0.75 && depth >= bestDepth)))
                continue;
            const double worldT = t * a.w() / ((1 - t) * b.w() + t * a.w());
            best = segment;
            bestDistance = dist;
            bestDepth = depth;
            bestT = clipA + (clipB - clipA) * worldT;
        }
    }
    if (!best || !d->scene) {
        if (context)
            emit emptyContextRequested(globalPosition);
        else
            emit clearSelectionRequested();
        return;
    }
    const auto point = interpolate(best->a, best->b, bestT);
    const double sourceT = best->begin + (best->end - best->begin) * bestT;
    const qint64 index = best->originalIndex + (sourceT >= .5 ? 1 : 0);
    const QString key = d->scene->fibers[best->fiber].key;
    if (context)
        emit fiberContextRequested(key, index, point[0], point[1], point[2], globalPosition);
    else {
        emit fiberPositionPicked(key, double(best->originalIndex) + sourceT);
        emit fiberPicked(key, index, point[0], point[1], point[2]);
    }
}

void Fiber3DCanvas::setSurfaceMesh(std::shared_ptr<const surface::Mesh> mesh)
{
    if (mesh && mesh->indices.empty())
        mesh.reset();
    if (v->surfaceMesh == mesh)
        return;
    Scene3DCanvas::setSurfaceMesh(std::move(mesh));
    d->surfaceIndex.reset();
    updateProximity();
}

void Fiber3DCanvas::fitSurface()
{
    if (v->surfaceMesh)
        d->fitPending = false;
    Scene3DCanvas::fitSurface();
}

}  // namespace vc3d::fibers3d
