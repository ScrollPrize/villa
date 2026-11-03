#include "Fiber3DVolumeRenderer.hpp"
#include "volume_viewers/VolumetricCompositor.hpp"
#include <QVector2D>
#include <QVector3D>
#include <QVector4D>
#include <algorithm>
#include <cmath>

namespace vc3d::fibers3d
{
namespace
{
const char* vertex = R"glsl(#version 330 core
out vec2 uv;
void main() {
    uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
)glsl";
const char* fragment = R"glsl(#version 330 core
in vec2 uv;
out vec4 color;
uniform sampler3D volume0, volume1, volume2;
uniform sampler2D depthImage, transfer;
uniform mat4 inverseMvp, toTexture[3];
uniform vec3 voxelSize[3];
uniform int count, axis[3];
uniform vec2 windowRange;
uniform float opacity, stepFactor, maxRaySamples;
vec3 world(float depth) {
    vec4 p = inverseMvp * vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    return p.xyz / p.w;
}
vec2 sampleVolume(int i, vec3 p) {
    if (i == 0) return texture(volume0, p).rg;
    if (i == 1) return texture(volume1, p).rg;
    return texture(volume2, p).rg;
}
bool interval(vec3 o, vec3 d, float limit, out float a, out float b) {
    a = 0.0; b = limit;
    for (int k = 0; k < 3; ++k) {
        if (abs(d[k]) < 1e-10) { if (o[k] < 0.0 || o[k] > 1.0) return false; }
        else { float x = -o[k] / d[k], y = (1.0-o[k]) / d[k];
               a = max(a, min(x,y)); b = min(b, max(x,y)); }
    }
    return b > a;
}
void main() {
    vec3 start = world(0.0), direction = normalize(world(1.0) - start);
    float limit = max(0.0, dot(world(texture(depthImage, uv).r) - start, direction));
    float enter[3], leave[3]; vec3 o[3], d[3];
    float t = limit;
    for (int i = 0; i < 3; ++i) {
        enter[i] = limit; leave[i] = -1.0;
        if (i >= count) continue;
        o[i] = (toTexture[i] * vec4(start, 1)).xyz;
        d[i] = (toTexture[i] * vec4(direction, 0)).xyz;
        float a, b;
        if (interval(o[i], d[i], limit, a, b)) { enter[i] = a; leave[i] = b; t = min(t, a); }
    }
    vec3 rgb = vec3(0); float transmittance = 1.0;
    // Slabs use <=512 samples per interval (at most six entry/exit events).
    // A standalone volume can use up to 4096 samples for higher-resolution data.
    // Empty gaps are skipped, and overlapping slabs are sampled once.
    for (int iteration = 0; iteration < 4096; ++iteration) {
        if (t >= limit || transmittance < 0.004) break;
        float boundary = limit, spacing = 1e30; bool insideSlab = false;
        for (int i = 0; i < 3; ++i) {
            if (i >= count || leave[i] <= t + 1e-4) continue;
            if (enter[i] > t + 1e-4) boundary = min(boundary, enter[i]);
            else { insideSlab = true; boundary = min(boundary, leave[i]);
                   spacing = min(spacing, max(min(voxelSize[i].x, min(voxelSize[i].y, voxelSize[i].z)) * stepFactor, (leave[i]-enter[i])/maxRaySamples)); }
        }
        if (!insideSlab) { if (boundary <= t) break; t = boundary; continue; }
        float dt = min(boundary - t, spacing);
        if (dt <= 1e-5) { t = boundary; continue; }
        float middle = t + 0.5 * dt, best = 1e30, value = 0.0; int source = -1;
        for (int i = 0; i < 3; ++i) {
            if (i >= count || middle < enter[i] || middle > leave[i]) continue;
            vec2 v = sampleVolume(i, o[i] + middle * d[i]);
            float quality = max(voxelSize[i].x, max(voxelSize[i].y, voxelSize[i].z));
            if (v.g > 0.001 && quality < best) { source = i; best = quality; value = v.r / v.g * 255.0; }
        }
        if (source >= 0) {
            float a = texture(transfer, vec2((floor(value + 0.5) + 0.5)/256.0, (float(axis[source])+0.5)/3.0)).r;
            // Unit-voxel alpha comes from VC3D's shared transfer function.
            // Optical-length correction keeps opacity stable across step sizes.
            a = 1.0 - pow(max(0.0, 1.0-a), dt);
            float gray = clamp((value - windowRange.x) / max(1e-3, windowRange.y-windowRange.x), 0.0, 1.0);
            rgb += transmittance * a * vec3(gray);
            transmittance *= 1.0-a;
        }
        t += dt;
    }
    color = vec4(rgb, 1.0-transmittance) * opacity;
}
)glsl";
QVector3D vector(const Point& p)
{
    return {float(p[0]), float(p[1]), float(p[2])};
}
float length(const Point& p)
{
    return vector(p).length();
}
}  // namespace
Fiber3DVolumeRenderer::Fiber3DVolumeRenderer()
{
    initializeOpenGLFunctions();
    glGenVertexArrays(1, &vao_);
    glGenTextures(1, &transfer_);
    program_ = std::make_unique<QOpenGLShaderProgram>();
    if (!program_->addShaderFromSourceCode(QOpenGLShader::Vertex, vertex) ||
        !program_->addShaderFromSourceCode(QOpenGLShader::Fragment, fragment) || !program_->link())
        error_ = QStringLiteral("Cannot initialize volumetric CT: %1").arg(program_->log());
}
void Fiber3DVolumeRenderer::clearImages()
{
    for (auto t : textures_)
        glDeleteTextures(1, &t.id);
    textures_.clear();
    images_.clear();
}
Fiber3DVolumeRenderer::~Fiber3DVolumeRenderer()
{
    clearImages();
    glDeleteTextures(1, &transfer_);
    glDeleteTextures(1, &depth_);
    glDeleteFramebuffers(1, &depthFbo_);
    glDeleteVertexArrays(1, &vao_);
}
bool Fiber3DVolumeRenderer::copyDepth(GLuint framebuffer, int width, int height)
{
    if (depthWidth_ != width || depthHeight_ != height) {
        glDeleteTextures(1, &depth_);
        glDeleteFramebuffers(1, &depthFbo_);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        GLint stencil = 0, bits = 0;
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE, &bits);
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE, &stencil);
        glGenTextures(1, &depth_);
        glBindTexture(GL_TEXTURE_2D, depth_);
        const GLenum internal = stencil ? (bits == 32 ? GL_DEPTH32F_STENCIL8 : GL_DEPTH24_STENCIL8)
                                        : (bits == 32 ? GL_DEPTH_COMPONENT32F : GL_DEPTH_COMPONENT24);
        glTexImage2D(GL_TEXTURE_2D, 0, internal, width, height, 0, stencil ? GL_DEPTH_STENCIL : GL_DEPTH_COMPONENT, stencil ? (bits == 32 ? GL_FLOAT_32_UNSIGNED_INT_24_8_REV : GL_UNSIGNED_INT_24_8) : GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glGenFramebuffers(1, &depthFbo_);
        glBindFramebuffer(GL_FRAMEBUFFER, depthFbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, stencil ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        const bool complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        if (!complete) {
            error_ = QStringLiteral("Cannot create CT volume depth buffer.");
            return false;
        }
        depthWidth_ = width;
        depthHeight_ = height;
    }
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, depthFbo_);
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    if (glGetError() != GL_NO_ERROR) {
        error_ = QStringLiteral("Cannot resolve CT volume depth.");
        return false;
    }
    return true;
}
bool Fiber3DVolumeRenderer::draw(
    const slices::Images& images,
    const std::array<CompositeParams, 3>& params,
    const QMatrix4x4& mvp,
    const Point& origin,
    GLuint framebuffer,
    int width,
    int height,
    float low,
    float high,
    float opacity,
    bool interactive,
    int maxRaySamples)
{
    if (!program_->isLinked())
        return false;
    if (images != images_) {
        clearImages();
        images_ = images;
        textures_.resize(images.size());
        error_.clear();
    }
    if (images.empty() || !error_.isEmpty())
        return false;
    glActiveTexture(GL_TEXTURE0);
    size_t budget = 4 * 1024 * 1024;
    bool uploading = false;
    for (size_t i = 0; i < images.size(); ++i) {
        auto& gpu = textures_[i];
        const auto& slab = *images[i]->slab;
        if (gpu.layers == slab.depth)
            continue;
        uploading = true;
        if (!budget)
            continue;
        if (!gpu.id) {
            GLint maxEdge = 0;
            glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &maxEdge);
            if (slab.width > maxEdge || slab.height > maxEdge || slab.depth > maxEdge) {
                error_ = QStringLiteral("CT volume exceeds this GPU's 3D texture limit. Choose a coarser Zarr level.");
                return false;
            }
            glGenTextures(1, &gpu.id);
            glBindTexture(GL_TEXTURE_3D, gpu.id);
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RG8, slab.width, slab.height, slab.depth, 0, GL_RG, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            for (auto direction : {GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R})
                glTexParameteri(GL_TEXTURE_3D, direction, GL_CLAMP_TO_EDGE);
        }
        const size_t layerBytes = size_t(slab.width) * slab.height * 2;
        const int layers = std::min(slab.depth - gpu.layers, int(std::max(size_t(1), budget / layerBytes)));
        glBindTexture(GL_TEXTURE_3D, gpu.id);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0);
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, gpu.layers, slab.width, slab.height, layers, GL_RG, GL_UNSIGNED_BYTE, slab.pixels.data() + gpu.layers * layerBytes);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        if (glGetError() != GL_NO_ERROR) {
            error_ = QStringLiteral("Cannot upload volumetric CT.");
            return false;
        }
        gpu.layers += layers;
        budget -= std::min(budget, layers * layerBytes);
    }
    glBindTexture(GL_TEXTURE_3D, 0);
    if (uploading)
        return true;
    if (opacity <= 0)
        return false;
    if (!copyDepth(framebuffer, width, height))
        return false;
    if (transferDirty_ || params != params_) {
        std::array<float, 256 * 3> values{};
        for (int i = 0; i < 3; ++i) {
            const auto& p = params[i];
            const auto lut = vc3d::volumetric::buildOpacityLut(p.alphaMin, p.alphaMax, p.alphaOpacity, p.tfGamma, p.isoCutoff, 1.f);
            std::copy(lut.begin(), lut.end(), values.begin() + 256 * i);
        }
        glBindTexture(GL_TEXTURE_2D, transfer_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 256, 3, 0, GL_RED, GL_FLOAT, values.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        params_ = params;
        transferDirty_ = false;
    }
    program_->bind();
    glBindVertexArray(vao_);
    for (size_t i = 0; i < images.size(); ++i) {
        const auto& slab = *images[i]->slab;
        QMatrix4x4 box;
        box.setColumn(0, QVector4D(vector(slab.right), 0));
        box.setColumn(1, QVector4D(vector(slab.down), 0));
        box.setColumn(2, QVector4D(vector(slab.through), 0));
        box.setColumn(3, QVector4D(float(slab.origin[0] - origin[0]), float(slab.origin[1] - origin[1]), float(slab.origin[2] - origin[2]), 1));
        program_->setUniformValue(qPrintable(QStringLiteral("toTexture[%1]").arg(i)), box.inverted());
        program_->setUniformValue(qPrintable(QStringLiteral("voxelSize[%1]").arg(i)), QVector3D(length(slab.right) / slab.width, length(slab.down) / slab.height, length(slab.through) / slab.depth));
        program_->setUniformValue(qPrintable(QStringLiteral("axis[%1]").arg(i)), images[i]->plane.axis);
        glActiveTexture(GL_TEXTURE0 + GLuint(i));
        glBindTexture(GL_TEXTURE_3D, textures_[i].id);
    }
    program_->setUniformValue("volume0", 0);
    program_->setUniformValue("volume1", 1);
    program_->setUniformValue("volume2", 2);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, depth_);
    program_->setUniformValue("depthImage", 3);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, transfer_);
    program_->setUniformValue("transfer", 4);
    program_->setUniformValue("inverseMvp", mvp.inverted());
    program_->setUniformValue("count", int(images.size()));
    program_->setUniformValue("windowRange", QVector2D(low, high));
    program_->setUniformValue("opacity", opacity);
    program_->setUniformValue("stepFactor", interactive ? 1.f : .5f);
    program_->setUniformValue("maxRaySamples", float(interactive ? 512 : std::clamp(maxRaySamples, 512, 4096)));
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    glDisable(GL_SAMPLE_ALPHA_TO_ONE);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    program_->release();
    glBindVertexArray(0);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0);
    if (glGetError() != GL_NO_ERROR)
        error_ = QStringLiteral("Cannot render volumetric CT.");
    return false;
}
}  // namespace vc3d::fibers3d
