#include "Fiber3DSurfaceRenderer.hpp"
#include <algorithm>
#include <cstddef>

namespace vc3d::fibers3d
{
Fiber3DSurfaceRenderer::Fiber3DSurfaceRenderer()
{
    if (!initializeOpenGLFunctions()) {
        error_ = QStringLiteral("Surface mesh requires OpenGL 3.3.");
        return;
    }
    const char* vertex = R"glsl(#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
uniform mat4 mvp;
uniform mat3 rotation;
out vec3 n;
void main() { gl_Position = mvp * vec4(position,1); n = rotation * normal; }
)glsl";
    const char* fragment = R"glsl(#version 330 core
in vec3 n;
uniform float opacity;
uniform bool multisample;
out vec4 color;
void main() {
    // Coverage transparency writes depth per covered sample, so front/back
    // faces and fibers occlude correctly without sorting millions of triangles.
    if (!multisample && opacity < 1.0) {
        const int bayer[16] = int[16](0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5);
        ivec2 p = ivec2(gl_FragCoord.xy) & 3;
        if (opacity <= (float(bayer[p.y*4+p.x])+0.5)/16.0) discard;
    }
    float lengthN = length(n);
    float light = lengthN > 0.0 ? abs(dot(n/lengthN, normalize(vec3(-0.3,0.5,1)))) : 1.0;
    color = vec4(vec3(0.65,0.77,0.73) * (0.4+0.6*light), multisample ? opacity : 1.0);
}
)glsl";
    if (!program_.addShaderFromSourceCode(QOpenGLShader::Vertex, vertex) ||
        !program_.addShaderFromSourceCode(QOpenGLShader::Fragment, fragment) || !program_.link()) {
        error_ = program_.log();
        return;
    }
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vertices_);
    glGenBuffers(1, &indices_);
}
Fiber3DSurfaceRenderer::~Fiber3DSurfaceRenderer()
{
    if (vao_)
        glDeleteVertexArrays(1, &vao_);
    if (vertices_)
        glDeleteBuffers(1, &vertices_);
    if (indices_)
        glDeleteBuffers(1, &indices_);
}
bool Fiber3DSurfaceRenderer::draw(std::shared_ptr<const surface::Mesh> mesh, const QMatrix4x4& matrix, const Point& origin, const QQuaternion& rotation, float opacity)
{
    if (!vao_)
        return false;
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vertices_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices_);
    if (mesh_ != mesh) {
        error_.clear();
        mesh_ = std::move(mesh);
        vertexBytes_ = indexBytes_ = 0;
        glBufferData(GL_ARRAY_BUFFER, mesh_ ? mesh_->vertices.size() * sizeof(surface::Vertex) : 0, nullptr, GL_STATIC_DRAW);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh_ ? mesh_->indices.size() * sizeof(uint32_t) : 0, nullptr, GL_STATIC_DRAW);
        if (glGetError() != GL_NO_ERROR)
            error_ = QStringLiteral("Cannot allocate surface mesh GPU buffers.");
    }
    if (!error_.isEmpty()) {
        glBindVertexArray(0);
        return false;
    }
    if (!mesh_ || mesh_->indices.empty()) {
        glBindVertexArray(0);
        return false;
    }
    size_t budget = 4 * 1024 * 1024;
    const auto upload = [&](GLenum target, const void* data, size_t size, size_t& offset) {
        const auto count = std::min(size - offset, budget);
        if (count)
            glBufferSubData(target, GLintptr(offset), GLsizeiptr(count), static_cast<const char*>(data) + offset);
        offset += count;
        budget -= count;
    };
    upload(GL_ARRAY_BUFFER, mesh_->vertices.data(), mesh_->vertices.size() * sizeof(surface::Vertex), vertexBytes_);
    upload(GL_ELEMENT_ARRAY_BUFFER, mesh_->indices.data(), mesh_->indices.size() * sizeof(uint32_t), indexBytes_);
    if (glGetError() != GL_NO_ERROR) {
        error_ = QStringLiteral("Cannot upload surface mesh GPU buffers.");
        glBindVertexArray(0);
        return false;
    }
    const bool pending = vertexBytes_ != mesh_->vertices.size() * sizeof(surface::Vertex) || indexBytes_ != mesh_->indices.size() * sizeof(uint32_t);
    if (!pending && opacity > 0) {
        QMatrix4x4 mvp = matrix;
        mvp.translate(float(mesh_->origin[0] - origin[0]), float(mesh_->origin[1] - origin[1]), float(mesh_->origin[2] - origin[2]));
        GLint samples = 0;
        glGetIntegerv(GL_SAMPLES, &samples);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        glDisable(GL_CULL_FACE);
        glDisable(GL_BLEND);
        glDisable(GL_SAMPLE_COVERAGE);
        glEnable(GL_MULTISAMPLE);
        if (samples > 1) {
            glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
            glEnable(GL_SAMPLE_ALPHA_TO_ONE);
        }
        program_.bind();
        program_.setUniformValue("mvp", mvp);
        program_.setUniformValue("rotation", rotation.toRotationMatrix());
        program_.setUniformValue("opacity", std::clamp(opacity, 0.f, 1.f));
        program_.setUniformValue("multisample", samples > 1);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(surface::Vertex), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(surface::Vertex), reinterpret_cast<void*>(offsetof(surface::Vertex, normal)));
        glDrawElements(GL_TRIANGLES, GLsizei(mesh_->indices.size()), GL_UNSIGNED_INT, nullptr);
        program_.release();
        glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
        glDisable(GL_SAMPLE_ALPHA_TO_ONE);
        glDisable(GL_DEPTH_TEST);
    }
    glBindVertexArray(0);
    return pending;
}
}  // namespace vc3d::fibers3d
