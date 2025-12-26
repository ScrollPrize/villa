/*
OpenABF
https://gitlab.com/educelab/OpenABF

Copyright 2025 EduceLab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

// clang-format off
// #include "OpenABF/Exceptions.hpp"


#include <stdexcept>
#include <string>

namespace OpenABF
{

/** @brief Solver exception */
class SolverException : public std::runtime_error
{
public:
    /** @brief Constructor with message */
    explicit SolverException(const char* msg) : std::runtime_error(msg) {}
    /** @brief Constructor with message */
    explicit SolverException(const std::string& msg) : std::runtime_error(msg)
    {
    }
};

/** @brief Solver exception */
class MeshException : public std::runtime_error
{
public:
    /** @brief Constructor with message */
    explicit MeshException(const char* msg) : std::runtime_error(msg) {}
    /** @brief Constructor with message */
    explicit MeshException(const std::string& msg) : std::runtime_error(msg) {}
};

}  // namespace OpenABF
// #include "OpenABF/Math.hpp"


#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace OpenABF
{

/** @brief Pi, templated for floating-point type */
template <class T>
constexpr T PI = T(3.1415926535897932385L);

/** @brief Inf, templated for floating-point type */
template <class T>
constexpr T INF = std::numeric_limits<T>::infinity();

/** @brief Vector dot product (inner product) */
template <typename T1, typename T2>
auto dot(const T1& a, const T2& b)
{
    using Ret = decltype(*std::begin(a));
    return std::inner_product(
        std::begin(a), std::end(a), std::begin(b), Ret(0));
}
/** @brief Vector cross product */
template <typename T1, typename T2>
auto cross(const T1& a, const T2& b) -> T1
{
    T1 c;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
    return c;
}

/** @brief Norm type enumeration */
enum class Norm {
    L1,  /**< \f$L_1\f$ norm */
    L2,  /**< \f$L_2\f$ norm */
    LInf /**< \f$L_{Inf}\f$ norm */
};

/** @brief Compute vector norm */
template <class Vector>
auto norm(const Vector& v, Norm norm = Norm::L2)
{
    using Ret = decltype(*std::begin(v));
    switch (norm) {
        case Norm::L1: {
            return std::accumulate(
                std::begin(v), std::end(v), Ret(0),
                [](auto a, auto b) { return a + std::abs(b); });
        }
        case Norm::L2: {
            auto sum = std::accumulate(
                std::begin(v), std::end(v), Ret(0),
                [](auto a, auto b) { return a + (b * b); });
            return std::sqrt(sum);
        }
        case Norm::LInf: {
            return std::abs(*std::max_element(
                std::begin(v), std::end(v),
                [](auto a, auto b) { return std::abs(a) < std::abs(b); }));
        }
    }
    throw std::invalid_argument("Invalid norm option");
}

/** @brief Normalize a vector (i.e. compute a unit vector) */
template <class Vector>
auto normalize(Vector v)
{
    return v / norm(v, Norm::L2);
}

/** @brief Compute the interior angle between two vectors */
template <class Vector1, class Vector2>
auto interior_angle(const Vector1& a, const Vector2& b)
{
    return std::acos(dot(a, b) / (norm(a, Norm::L2) * norm(b, Norm::L2)));
}

/** @brief Convert degrees to radians */
template <
    typename T = float,
    typename T2,
    std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
constexpr auto to_radians(T2 deg) -> T
{
    return deg * PI<T> / T(180);
}

/** @brief Convert radians to degrees */
template <
    typename T = float,
    typename T2,
    std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
constexpr auto to_degrees(T2 rad) -> T
{
    return rad * T(180) / PI<T>;
}

}  // namespace OpenABF
// #include "OpenABF/Vec.hpp"


#include <array>
#include <iostream>

// #include "OpenABF/Math.hpp"


namespace OpenABF
{
/**
 * @brief N-dimensional vector class
 *
 * Essentially a wrapper around std::array that makes it more convenient for
 * vector math purposes.
 *
 * @tparam T Element type
 * @tparam Dims Number of elements
 */
template <
    typename T,
    std::size_t Dims,
    std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
class Vec
{
    /** Underlying element storage */
    using Container = std::array<T, Dims>;

public:
    /** Element type */
    using value_type = T;
    /** Vector size type */
    using size_type = typename Container::size_type;
    /** Difference type */
    using difference_type = typename Container::difference_type;
    /** Reference type */
    using reference = value_type&;
    /** Const reference type */
    using const_reference = const value_type&;
    /** Pointer type */
    using pointer = value_type*;
    /** Const pointer type */
    using const_pointer = const value_type*;
    /** Iterator type */
    using iterator = typename Container::iterator;
    /** Const iterator type */
    using const_iterator = typename Container::const_iterator;
    /** Reverse iterator type */
    using reverse_iterator = typename Container::reverse_iterator;
    /** Const reverse iterator type */
    using const_reverse_iterator = typename Container::const_reverse_iterator;

    /** @brief Default constructor */
    Vec() { val_.fill(0); }

    /**
     * @brief Construct with element values
     *
     * The number of arguments provided must match Dims.
     */
    template <typename... Args>
    explicit Vec(Args... args)
    {
        static_assert(sizeof...(args) == Dims, "Incorrect number of arguments");
        std::size_t i{0};
        ((val_[i++] = args), ...);
    }

    /** @brief Copy constructor */
    template <typename Vector>
    explicit Vec(const Vector& vec)
    {
        std::copy(std::begin(vec), std::end(vec), val_.begin());
    }

    /** @brief Bounds-checked element access */
    constexpr reference at(size_type pos) { return val_.at(pos); }
    /** @brief Bounds-checked element access */
    constexpr const_reference at(size_type pos) const { return val_.at(pos); }
    /** @brief Element access */
    constexpr reference operator[](size_type i) { return val_[i]; }
    /** @brief Element access */
    constexpr const_reference operator[](size_type i) const { return val_[i]; }

    /** @brief First element */
    constexpr reference front() { return val_.front(); }
    /** @brief First element */
    constexpr const_reference front() const { return val_.front(); }
    /** @brief Last element */
    constexpr reference back() { return val_.back(); }
    /** @brief Last element */
    constexpr const_reference back() const { return val_.back(); }

    /** @brief Get a pointer to the first element of the raw data */
    constexpr pointer data() { return val_.data(); }
    /** @brief Get a pointer to the first element of the raw data */
    constexpr const_pointer data() const { return val_.data(); }

    /** @brief Get an iterator to the first element of the vector */
    constexpr iterator begin() noexcept { return val_.begin(); }
    /** @brief Get an iterator to the first element of the vector */
    constexpr const_iterator begin() const noexcept { return val_.begin(); }
    /** @brief Get an iterator to the first element of the vector */
    constexpr const_iterator cbegin() const noexcept { return val_.cbegin(); }

    /** @brief Get an iterator to one past the last element in the vector */
    constexpr iterator end() noexcept { return val_.end(); }
    /** @brief Get an iterator to one past the last element in the vector */
    constexpr const_iterator end() const noexcept { return val_.end(); }
    /** @brief Get an iterator to one past the last element in the vector */
    constexpr const_iterator cend() const noexcept { return val_.cend(); }

    /** @brief Get an iterator to the first element of the reverse vector */
    constexpr iterator rbegin() noexcept { return val_.rbegin(); }
    /** @brief Get an iterator to the first element of the vector */
    constexpr const_iterator rbegin() const noexcept { return val_.rbegin(); }
    /** @brief Get an iterator to the first element of the vector */
    constexpr const_iterator crbegin() const noexcept { return val_.crbegin(); }

    /**
     * @brief Get an iterator to one past the last element in the reverse vector
     */
    constexpr iterator rend() noexcept { return val_.rend(); }
    /**
     * @brief Get an iterator to one past the last element in the reverse vector
     */
    constexpr const_iterator rend() const noexcept { return val_.rend(); }
    /**
     * @brief Get an iterator to one past the last element in the reverse vector
     */
    constexpr const_iterator crend() const noexcept { return val_.crend(); }

    /** @brief Return whether the vector is empty (uninitialized) */
    constexpr bool empty() const noexcept { return val_.empty(); }
    /** @brief Return the number of elements in the vector */
    constexpr size_type size() const noexcept { return val_.size(); }

    /** @brief Fill the vector with a value */
    constexpr void fill(const T& value) { val_.fill(value); }
    /** @brief Swap this vector with another vector */
    constexpr void swap(Vec& other) noexcept { val_.swap(other.val_); }

    /** @brief Equality comparison operator */
    bool operator==(const Vec& rhs) const { return val_ == rhs.val_; }
    /** @brief Inequality comparison operator */
    bool operator!=(const Vec& rhs) const { return val_ != rhs.val_; }

    /** @brief Assignment operator */
    template <class Vector>
    Vec& operator=(const Vector& b)
    {
        std::size_t idx{0};
        for (auto& v : val_) {
            v = b[idx++];
        }
        return *this;
    }

    /** @brief Assignment operator for std::initializer_list */
    template <typename T2>
    Vec& operator=(const std::initializer_list<T2>& b)
    {
        auto it = b.begin();
        for (auto& v : val_) {
            v = *it;
            it++;
        }
        return *this;
    }

    /** @brief Addition assignment operator */
    template <class Vector>
    Vec& operator+=(const Vector& b)
    {
        std::size_t idx{0};
        for (auto& v : val_) {
            v += b[idx++];
        }
        return *this;
    }

    /** @brief Addition operator */
    template <class Vector>
    friend Vec operator+(Vec lhs, const Vector& rhs)
    {
        lhs += rhs;
        return lhs;
    }

    /** @brief Subtraction assignment operator */
    template <class Vector>
    Vec& operator-=(const Vector& b)
    {
        std::size_t idx{0};
        for (auto& v : val_) {
            v -= b[idx++];
        }
        return *this;
    }

    /** @brief Subtraction operator */
    template <class Vector>
    friend Vec operator-(Vec lhs, const Vector& rhs)
    {
        lhs -= rhs;
        return lhs;
    }

    /** @brief Multiplication assignment operator */
    template <
        typename T2,
        std::enable_if_t<std::is_arithmetic<T2>::value, bool> = true>
    Vec& operator*=(const T2& b)
    {
        for (auto& v : val_) {
            v *= b;
        }
        return *this;
    }

    /** @brief Multiplication operator */
    template <class Vector>
    friend Vec operator*(Vec lhs, const Vector& rhs)
    {
        lhs *= rhs;
        return lhs;
    }

    /** @brief Division assignment operator */
    template <
        typename T2,
        std::enable_if_t<std::is_arithmetic<T2>::value, bool> = true>
    Vec& operator/=(const T2& b)
    {
        for (auto& v : val_) {
            v /= b;
        }
        return *this;
    }

    /** @brief Division operator */
    template <class Vector>
    friend Vec operator/(Vec lhs, const Vector& rhs)
    {
        lhs /= rhs;
        return lhs;
    }

    /** @brief Compute the vector dot product (i.e. inner product) */
    template <class Vector>
    T dot(const Vector& v)
    {
        return OpenABF::dot(val_, v);
    }

    /** @brief Compute the vector cross product */
    template <class Vector, std::size_t D = Dims>
    std::enable_if_t<D == 3, Vec> cross(const Vector& v)
    {
        return OpenABF::cross(*this, v);
    }

    /** @brief Compute the vector magnitude */
    T magnitude() const { return OpenABF::norm(*this, Norm::L2); }

    /** @brief Return the unit vector of this vector */
    Vec unit() const { return OpenABF::normalize(*this); }

private:
    /** Values */
    Container val_{};
};

/** @brief 3D, 32-bit float vector */
using Vec3f = Vec<float, 3>;
/** @brief 3D, 64-bit float vector */
using Vec3d = Vec<double, 3>;

}  // namespace OpenABF

/** Debug: Print a vector to a std::ostream */
template <typename T, std::size_t Dims>
std::ostream& operator<<(std::ostream& os, const OpenABF::Vec<T, Dims>& vec)
{
    os << "[";
    std::size_t i{0};
    for (const auto& v : vec) {
        if (i++ > 0) {
            os << ", ";
        }
        os << v;
    }
    os << "]";
    return os;
}

// #include "OpenABF/HalfEdgeMesh.hpp"


#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <vector>

// #include "OpenABF/Exceptions.hpp"

// #include "OpenABF/Vec.hpp"


namespace OpenABF
{

namespace traits
{
/** @brief Default HalfEdgeMesh vertex traits */
template <typename T>
struct DefaultVertexTraits {
};

/** @brief Default HalfEdgeMesh edge traits */
template <typename T>
struct DefaultEdgeTraits {
    /** Edge internal angle */
    T alpha{0};
};

/** @brief Default HalfEdgeMesh face traits */
template <typename T>
struct DefaultFaceTraits {
};
}  // namespace traits

namespace detail
{
/** Debug: Print a vector of elements to a string */
template <typename T>
auto vec_to_string(const T& v) -> std::string
{
    std::ostringstream ss;
    ss << '[';
    for (std::size_t i = 0; i < std::size(v); ++i) {
        if (i != 0)
            ss << ", ";
        ss << std::begin(v)[i];
    }
    ss << ']';
    return ss.str();
}

/** Remove elements which meet the given predicate */
template <class ForwardContainer, class UnaryPred>
auto erase_if(ForwardContainer v, UnaryPred p)
{
    auto end = std::remove_if(std::begin(v), std::end(v), p);
    v.erase(end, std::end(v));
    return v;
}
}  // namespace detail

/**
 * @brief Compute the internal angles of a face
 *
 * Updates the current angle (DefaultEdgeTraits::alpha) with the internal angles
 * derived from the face's vertex positions. Useful if you want to reset a face
 * after being processed by ABF or ABFPlusPlus.
 *
 * @tparam FacePtr A Face-type pointer implementing DefaultEdgeTraits
 * @throws MeshException If interior angle is NaN or Inf
 */
template <class FacePtr>
void ComputeFaceAngles(FacePtr& face)
{
    for (auto& e : *face) {
        auto ab = e->next->vertex->pos - e->vertex->pos;
        auto ac = e->next->next->vertex->pos - e->vertex->pos;
        e->alpha = interior_angle(ab, ac);
        if (std::isnan(e->alpha) or std::isinf(e->alpha)) {
            auto msg = "Interior angle for edge " + std::to_string(e->idx) +
                       " is nan/inf";
            throw MeshException(msg);
        }
    }
}

/**
 * @brief Compute the internal angles for all faces in a mesh
 *
 * Runs ComputeFaceAngles on all faces in the mesh. Useful if you want to reset
 * a mesh after running through ABF or ABFPlusPlus.
 *
 * @tparam MeshPtr A Mesh-type pointer with faces implementing DefaultEdgeTraits
 */
template <class MeshPtr>
void ComputeMeshAngles(MeshPtr& mesh)
{
    for (auto& f : mesh->faces()) {
        ComputeFaceAngles(f);
    }
}

/** @brief Determines if mesh is open or closed */
template <class MeshPtr>
auto HasBoundary(const MeshPtr& mesh) -> bool
{
    for (const auto& v : mesh->vertices()) {
        if (v->is_boundary()) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Check if a mesh has unreferenced vertices
 *
 * @note This only checks if the vertex is associated with at least one edge.
 * A face is not currently guaranteed.
 */
template <class MeshPtr>
auto HasUnreferencedVertices(const MeshPtr& mesh) -> bool
{
    for (const auto& v : mesh->vertices()) {
        if (v->edges.empty()) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Get a list of unreferenced vertices
 *
 * @note This only checks if the vertex is associated with at least one edge.
 * A face is not currently guaranteed.
 */
template <class MeshPtr>
auto UnreferencedVertices(const MeshPtr& mesh) -> std::vector<std::size_t>
{
    std::vector<std::size_t> indices;
    for (const auto& v : mesh->vertices()) {
        if (v->edges.empty()) {
            indices.emplace_back(v->idx);
        }
    }
    return indices;
}

/** @brief Check if mesh is manifold */
template <class MeshPtr>
auto IsManifold(const MeshPtr& mesh) -> bool
{
    // insert_faces won't allow non-manifold edges, but vertices may still be
    // non-manifold if update_boundary was never called, so check those here
    for (const auto& v : mesh->vertices()) {
        if (not v->is_manifold()) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Find an edge path between two vertices
 *
 * Uses Dijkstra's algorithm to find the shortest path between two vertices.
 * Distance is measured using the edge lengths of the mesh. The returned mesh
 * is not guaranteed to be the _only_ shortest path (there may be many which
 * have the same length), but only the first discovered.
 *
 * If the returned list is empty, the endpoints are the same or a path between
 * the two endpoints does not exist (i.e. the mesh has multiple connected
 * components).
 *
 * @returns std::vector<EdgePtr>
 */
template <class MeshPtr>
auto FindEdgePath(const MeshPtr& mesh, std::size_t from, std::size_t to)
{
    using Mesh = std::remove_reference_t<decltype(*mesh)>;
    using Value = typename Mesh::type;
    using EdgePtr = typename Mesh::EdgePtr;

    // End points are the same
    if (from == to) {
        return std::vector<EdgePtr>{};
    }

    struct Node {
        using Ptr = std::shared_ptr<Node>;
        std::size_t idx{0};
        Value dist{INF<Value>};
        Ptr prev{nullptr};
        EdgePtr fromEdge{nullptr};
    };

    // Build a list of all nodes
    std::vector<typename Node::Ptr> nodes;
    nodes.reserve(mesh->num_vertices());
    for (std::size_t i = 0; i < mesh->num_vertices(); ++i) {
        auto n = std::make_shared<Node>();
        n->idx = i;
        if (i == from) {
            n->dist = 0;
        }
        nodes.push_back(n);
    }

    // Build a queue
    struct Compare {
        auto operator()(
            const typename Node::Ptr& p, const typename Node::Ptr& q) const
            -> bool
        {
            return p->dist > q->dist;
        }
    };
    using Queue = std::priority_queue<
        typename Node::Ptr, std::vector<typename Node::Ptr>, Compare>;
    Queue queue;
    queue.push(nodes[from]);

    typename Node::Ptr end{nullptr};
    while (not queue.empty()) {
        auto p = queue.top();
        queue.pop();
        if (p->idx == to) {
            end = p;
            break;
        }

        for (const auto& e : mesh->outgoing_edges(p->idx)) {
            const auto next = e->pair->vertex->idx;
            const auto d = p->dist + e->magnitude();

            if (d < nodes[next]->dist) {
                nodes[next]->dist = d;
                nodes[next]->prev = p;
                nodes[next]->fromEdge = e;
                queue.push(nodes[next]);
            }
        }
    }

    // Error: haven't found a path
    if (end == nullptr) {
        return std::vector<EdgePtr>{};
    }

    // Build edge path
    std::vector<EdgePtr> path;
    auto node = end;
    while (node->prev) {
        path.emplace_back(node->fromEdge);
        node = node->prev;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

/**
 * @brief Half-edge mesh class
 *
 * A half-edge mesh represents each edge as two oppositely oriented half-edges.
 * There is one half-edge for each face containing the original edge. For
 * example, if two faces share the edge **AB**, this will result in two
 * half-edges, **AB** and **BA**. If an edge **BC** lies on the mesh border
 * (i.e. it is only included in a single face), there will one be a single
 * half-edge created. This data structure makes it possible to easily traverse
 * the edges and faces adjacent to a vertex (the "wheel"), as well as to
 * traverse the edges of each face.
 *
 * For more information, see Chapter 12.1 in "Fundamentals of Computer
 * Graphics, Fourth edition", Marschner and Shirley (2015)
 * \cite marschner2015fundamentals.
 *
 * @tparam T Floating-point type for coordinates
 * @tparam Dim Dimensionality of vertex coordinates
 * @tparam VertexTraits Additional traits for vertices
 * @tparam EdgeTraits Additional traits for edges
 * @tparam FaceTraits Additional traits for face
 */
template <
    typename T,
    std::size_t Dim = 3,
    typename VertexTraits = traits::DefaultVertexTraits<T>,
    typename EdgeTraits = traits::DefaultEdgeTraits<T>,
    typename FaceTraits = traits::DefaultFaceTraits<T>>
class HalfEdgeMesh
{
public:
    /** Fundamental element type (e.g. float, double) */
    using type = T;

    /** Pointer type */
    using Pointer = std::shared_ptr<HalfEdgeMesh>;

    struct Vertex;
    struct Edge;
    struct Face;

    /** @brief Vertex pointer type */
    using VertPtr = std::shared_ptr<Vertex>;
    /** @brief Edge pointer type */
    using EdgePtr = std::shared_ptr<Edge>;
    /** @brief Edge pointer type */
    using FacePtr = std::shared_ptr<Face>;

private:
    /**
     * @brief Iterator for the edges of a face
     *
     * @tparam Const If true, is a const iterator
     */
    template <bool Const = false>
    class FaceIterator
    {
    public:
        /** Difference type */
        using difference_type = std::size_t;
        /** Value type */
        using value_type = EdgePtr;
        /** Pointer type */
        using pointer =
            std::conditional_t<Const, value_type const*, value_type*>;
        /** Reference type */
        using reference =
            std::conditional_t<Const, value_type const&, value_type&>;
        /** Iterator category */
        using iterator_category = std::input_iterator_tag;

        /** Default constructor == End iterator */
        FaceIterator() = default;
        /** Construct from head of triangle and current edge */
        explicit FaceIterator(const EdgePtr& head, const EdgePtr& current)
            : head_{head}, current_{current}
        {
        }

        /** Dereference operator */
        template <bool Const_ = Const>
        std::enable_if_t<Const_, reference> operator*() const
        {
            return current_;
        }

        /** Dereference operator */
        template <bool Const_ = Const>
        std::enable_if_t<not Const_, reference> operator*()
        {
            return current_;
        }

        /** Equality operator */
        auto operator==(const FaceIterator& other) const -> bool
        {
            return current_ == other.current_;
        }
        /** Inequality operator */
        auto operator!=(const FaceIterator& other) const -> bool
        {
            return !(*this == other);
        }
        /** Increment operator */
        auto operator++() -> FaceIterator&
        {
            // Already at end
            if (current_ == nullptr) {
                return *this;
            }

            // Get the next edge
            current_ = current_->next;
            // If back at head, done iterating
            if (current_ == head_) {
                current_ = nullptr;
            }
            return *this;
        }
    private:
        /** Pointer to beginning of face */
        EdgePtr head_;
        /** Current edge pointer */
        EdgePtr current_;
    };

public:
    /** @brief %Vertex type */
    struct Vertex : VertexTraits {
        /** @brief Default constructor */
        Vertex() = default;

        /** @brief Construct from position values */
        template <typename... Args>
        explicit Vertex(Args... args) : pos{args...}
        {
        }

        /** @brief Copy-construct inherited traits */
        Vertex(const Vertex& rhs) : VertexTraits(rhs), pos{rhs.pos} {}

        /** @brief Construct a new Vertex pointer */
        template <typename... Args>
        static auto New(Args&&... args) -> VertPtr
        {
            return std::make_shared<Vertex>(std::forward<Args>(args)...);
        }

        /**
         * @brief Get the edges of a vertex's wheel
         *
         * @throws MeshException If vertex is a boundary vertex.
         */
        auto wheel() const -> std::vector<EdgePtr>
        {
            std::vector<EdgePtr> ret;
            auto e = edge;
            do {
                if (not e->is_boundary()) {
                    ret.emplace_back(e);
                }
                e = e->pair->next;
            } while (e != edge);
            return ret;
        }

        /** @brief Unit vertex normal */
        auto normal() const -> Vec<T, Dim>
        {
            Vec<T, Dim> n{0, 0, 0};
            for (const auto& e : wheel()) {
                n += e->face->normal();
            }
            return normalize(n);
        }

        /** @brief Returns if vertex is on mesh boundary */
        [[nodiscard]] auto is_boundary() const -> bool
        {
            auto e = edge;
            do {
                if (e->is_boundary() or e->pair->is_boundary()) {
                    return true;
                }
                e = e->pair->next;
            } while (e != edge);
            return false;
        }

        /** @brief Returns if vertex is interior to mesh */
        [[nodiscard]] auto is_interior() const -> bool
        {
            return not is_boundary();
        }

        /** @brief Returns if vertex is unreferenced */
        [[nodiscard]] auto is_unreferenced() const -> bool
        {
            return edge == nullptr;
        }

        /** @brief Returns if vertex is manifold */
        [[nodiscard]] auto is_manifold() const -> bool
        {
            std::size_t boundaryCnt{0};
            auto out = mesh->outgoing_edges(idx);
            for (const auto& e : out) {
                if (e->is_boundary() or e->pair->is_boundary()) {
                    boundaryCnt++;
                }
            }
            return boundaryCnt == 0 or boundaryCnt == 2;
        }

        /** @brief Insertion index */
        std::size_t idx{0};
        /** @brief Vertex position */
        Vec<T, Dim> pos;
        /**
         * @brief Pointer to _an_ Edge with this vertex as its head
         *
         * @note There may be many such vertices.
         */
        EdgePtr edge;
        /** @brief Mesh to which this vertex belongs */
        HalfEdgeMesh* mesh{nullptr};
    };

    /** @brief %Edge type */
    struct Edge : EdgeTraits {
        Edge() = default;

        /** @brief Copy-construct inherited traits */
        Edge(const Edge& rhs) : EdgeTraits(rhs) {}

        /** @brief Construct a new Edge pointer */
        template <typename... Args>
        static auto New(Args&&... args) -> EdgePtr
        {
            return std::make_shared<Edge>(std::forward<Args>(args)...);
        }

        /** @brief Returns if edge is on mesh boundary */
        [[nodiscard]] auto is_boundary() const -> bool
        {
            return face == nullptr;
        }

        /** @brief Edge length */
        auto magnitude() -> T
        {
            return (pair->vertex->pos - vertex->pos).magnitude();
        }

        /** @brief This edge's adjacent half-edge */
        EdgePtr pair;
        /**
         * @brief The next edge in this edge's face
         *
         * If the edge is not assigned to a face, the next edge along the
         * boundary.
         */
        EdgePtr next;
        /**
         * @brief The previous edge in this edge's face
         *
         * If the edge is not assigned to a face, the previous edge along the
         * boundary.
         */
        EdgePtr prev;
        /** @brief The edge's vertex */
        VertPtr vertex;
        /** @brief The face containing this edge */
        FacePtr face;
        /** @brief Insertion index among all edges (including boundary) */
        std::size_t idx{0};
        /** @brief Insertion index among all face edges (excluding boundary) */
        std::optional<std::size_t> idxI;
        /** @brief Mesh to which this edge belongs */
        HalfEdgeMesh* mesh{nullptr};
    };

    /** @brief %Face type */
    struct Face : FaceTraits {
        /** Default constructor */
        Face() = default;

        /** @brief Copy-construct inherited traits */
        Face(const Face& rhs) : FaceTraits(rhs) {}

        /** @brief Construct a new Face pointer */
        template <typename... Args>
        static auto New(Args&&... args) -> FacePtr
        {
            return std::make_shared<Face>(std::forward<Args>(args)...);
        }

        /** Face edge iterator type */
        using iterator = FaceIterator<>;
        /** Face edge const iterator type */
        using const_iterator = FaceIterator<true>;
        /** @brief Returns an iterator over the edges of the face */
        iterator begin() { return iterator{head, head}; }
        /** @brief Returns the end iterator */
        iterator end() { return iterator(); }
        /** @brief Returns a const iterator over the edges of the face */
        const_iterator cbegin() const { return const_iterator{head, head}; }
        /** @brief Returns the const end iterator */
        const_iterator cend() const { return const_iterator(); }

        /** @brief Area of the face */
        auto area() const -> T
        {
            // Get the edge lengths
            std::array<T, 3> l{
                head->magnitude(), head->next->magnitude(),
                head->prev->magnitude()};

            // Sort the side lengths so that a >= b >= c
            std::sort(l.begin(), l.end(), std::greater<T>());

            // Calculate the area
            const auto& a = l[0];
            const auto& b = l[1];
            const auto& c = l[2];
            auto p =
                (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c));
            return 0.25 * std::sqrt(p);
        }

        /** @brief Face barycenter (center-of-mass) */
        auto barycenter() const -> Vec<T, 3>
        {
            return (head->vertex->pos + head->next->vertex->pos +
                    head->prev->vertex->pos) /
                   T(3);
        }

        /** @brief Unit face normal */
        auto normal() const -> Vec<T, Dim>
        {
            // Get the edge vectors
            auto e0 = head->prev->vertex->pos - head->vertex->pos;
            auto e1 = head->next->vertex->pos - head->vertex->pos;

            // Take the cross-product
            return normalize(e1.cross(e0));
        }

        /** @brief First edge in the face */
        EdgePtr head;
        /** @brief The next face in the mesh */
        FacePtr next;
        /** @brief Insertion index */
        std::size_t idx{0};
        /** @brief Mesh to which this vertex belongs */
        HalfEdgeMesh* mesh{nullptr};
    };

private:
    /** List of vertices */
    std::vector<VertPtr> verts_;
    /** List of faces */
    std::vector<FacePtr> faces_;
    /** List of all edges, indexed by the vertex's insertion index */
    std::multimap<std::size_t, EdgePtr> edges_;
    /** Number of all edges which border a face */
    std::size_t numFaceEdges_{0};

public:
    /** @brief Default constructor */
    HalfEdgeMesh() = default;

    /** @brief Destructor deallocating all element pointers */
    ~HalfEdgeMesh()
    {
        // Remove smart pointers from all items
        for (auto& v : verts_) {
            v->edge = nullptr;
        }
        for (auto& e : edges_) {
            e.second->pair = nullptr;
            e.second->next = nullptr;
            e.second->prev = nullptr;
            e.second->vertex = nullptr;
            e.second->face = nullptr;
        }
        for (auto& f : faces_) {
            f->head = nullptr;
            f->next = nullptr;
        }
        verts_.clear();
        edges_.clear();
        faces_.clear();
    }

    /** @brief Construct a new HalfEdgeMesh pointer */
    template <typename... Args>
    static Pointer New(Args... args)
    {
        return std::make_shared<HalfEdgeMesh>(std::forward<Args>(args)...);
    }

    /**
     * @brief Clone this mesh
     *
     * Returns a new mesh with the same structure as this mesh but not sharing
     * vertex, face, or edge elements.
     */
    auto clone() const -> Pointer
    {
        auto ret = HalfEdgeMesh::New();
        ret->verts_.reserve(verts_.size());
        ret->faces_.reserve(faces_.size());

        // Insert vertices
        for (const auto& v : verts_) {
            auto i = ret->insert_vertex(*v);
            ret->verts_[i]->edge = nullptr;
        }

        // Insert faces and edges
        for (const auto& f : faces_) {
            ret->clone_face_(f);
        }
        ret->update_boundary();
        return ret;
    }

    /**
     * @brief Insert a new vertex
     *
     * Accepts all arguments supported by the Vertex constructor.
     */
    template <typename... Args>
    auto insert_vertex(Args... args) -> std::size_t
    {
        auto vert = Vertex::New(std::forward<Args>(args)...);
        vert->mesh = this;
        vert->idx = verts_.size();
        verts_.push_back(vert);
        return vert->idx;
    }

    /**
     * @brief Insert new vertices from a list of Vertex-like objects
     *
     * A convenience function which adds multiple vertices to the mesh.
     */
    template <class VectorOfVectors>
    auto insert_vertices(const VectorOfVectors& v) -> std::vector<std::size_t>
    {
        std::vector<std::size_t> idxs;
        for (const auto& f : v) {
            idxs.emplace_back(insert_vertex(f));
        }
        return idxs;
    }

    /**
     * @copydoc insert_vertices(const VectorOfVectors&)
     */
    template <typename ValType>
    auto insert_vertices(
        std::initializer_list<std::initializer_list<ValType>> v)
        -> std::vector<std::size_t>
    {
        auto it = std::begin(v);
        std::vector<std::size_t> idxs;
        for (std::size_t i = 0; i < std::size(v); ++i) {
            idxs.emplace_back(insert_vertex(it[i]));
        }
        return idxs;
    }

    /**
     * @brief Insert a face from an ordered list of Vertex indices
     *
     * Accepts an iterable supporting range-based for loops.
     *
     * @param vector List of vertex indices
     * @throws std::out_of_range If one of the vertex indices is out of bounds.
     * @throws MeshException (1) If one of provided edges is already paired.
     * This indicates that the mesh is not 2-manifold. (2) If an edge has
     * zero length. This means the face has zero area. (3) If an edge's interior
     * angle is NaN or Inf.
     */
    template <class Vector>
    auto insert_face(Vector&& vector) -> std::size_t
    {
        return insert_face_(std::forward<Vector>(vector));
    }

    /**
     * @brief Insert a new face from an ordered list of Vertex indices
     *
     * This function inserts a face but **does not** update the mesh's boundary
     * connections. Make sure to call update_boundary() after all faces have
     * been inserted or use insert_faces() to insert and update in one step.
     *
     * @throws std::out_of_range If one of the vertex indices is out of bounds.
     * @throws MeshException If one of provided edges is already paired. This
     * indicates that the mesh is not 2-manifold.
     */
    template <typename... Args>
    auto insert_face(Args... args) -> std::size_t
    {
        static_assert(sizeof...(args) >= 3, "Faces require >= 3 indices");
        using Tuple = std::tuple<Args...>;
        using ElemT = std::tuple_element_t<0, Tuple>;
        return insert_face_(std::initializer_list<ElemT>{args...});
    }

    /**
     * @brief Insert new faces from a list of lists of Vertex indices
     *
     * A convenience function which adds multiple faces to the mesh (using
     * insert_face()) and updates the mesh boundary (using update_boundary())
     * when complete.
     */
    template <class VectorOfVectors>
    auto insert_faces(const VectorOfVectors& v) -> std::vector<std::size_t>
    {
        std::vector<std::size_t> idxs;
        for (const auto& f : v) {
            idxs.emplace_back(insert_face_(f));
        }

        update_boundary();
        return idxs;
    }

    /**
     * @copydoc insert_faces()
     */
    template <typename IdxType>
    auto insert_faces(std::initializer_list<std::initializer_list<IdxType>> v)
    {
        const auto it = std::begin(v);
        std::vector<std::size_t> idxs;
        for (std::size_t i = 0; i < std::size(v); ++i) {
            idxs.emplace_back(insert_face(it[i]));
        }
        update_boundary();
        return idxs;
    }

    /**
     * @brief Update the mesh boundary connections
     *
     * Because the mesh boundary may become temporarily non-traversable while
     * the mesh is being constructed, the mesh boundary connections should only
     * be updated after all faces have been added to the mesh. Call this
     * function after inserting faces with insert_face() or use insert_faces()
     * to construct the mesh and update the boundary in one step.
     */
    void update_boundary()
    {
        for (const auto& [_, edge] : edges_) {
            if (edge->is_boundary()) {
                // Get incoming boundary edges to the start point
                auto inBoundary = detail::erase_if(
                    incoming_edges(edge->vertex->idx),
                    [](const auto& e) { return not e->is_boundary(); });
                if (inBoundary.size() == 0 or inBoundary.size() > 1) {
                    const std::array<std::size_t, 2> idx{
                        edge->vertex->idx, edge->pair->vertex->idx};
                    throw MeshException(
                        "Cannot update mesh boundary along edge " +
                        detail::vec_to_string(idx) +
                        " due to non-manifold surface and/or inconsistent "
                        "winding order");
                }

                // Get outgoing boundary edges to the end point
                auto outBoundary = detail::erase_if(
                    outgoing_edges(edge->pair->vertex->idx),
                    [](const auto& e) { return not e->is_boundary(); });
                if (outBoundary.size() == 0 or outBoundary.size() > 1) {
                    const std::array<std::size_t, 2> idx{
                        edge->vertex->idx, edge->pair->vertex->idx};
                    throw MeshException(
                        "Cannot update mesh boundary along edge " +
                        detail::vec_to_string(idx) +
                        " due to non-manifold surface and/or inconsistent "
                        "winding order");
                }

                edge->prev = inBoundary[0];
                inBoundary[0]->next = edge;
                edge->next = outBoundary[0];
                outBoundary[0]->prev = edge;
            }
        }
    }

    /** @brief Get the list of vertices in insertion order */
    auto vertices() const -> std::vector<VertPtr> { return verts_; }

    /** @brief Get a vertex by index */
    auto vertex(std::size_t idx) const -> VertPtr { return verts_.at(idx); }

    /** @brief Get the list of face edges in insertion order */
    auto edges() const -> std::vector<EdgePtr>
    {
        std::vector<EdgePtr> edges;
        for (const auto& f : faces_) {
            for (const auto& e : *f) {
                edges.emplace_back(e);
            }
        }
        return edges;
    }

    /** @brief Find an existing edge with the provided end points */
    auto edge(std::size_t start, std::size_t end) -> EdgePtr
    {
        // Get edges with this start index
        const auto range = edges_.equal_range(start);

        // Loop over potential edges
        for (auto it = range.first; it != range.second; ++it) {
            const auto& e = it->second;
            if (e->pair->vertex->idx == end) {
                return e;
            }
        }
        return nullptr;
    }

    /**
     * @brief Get a boundary edge
     *
     * Returns the first boundary edge in the list of edges
     */
    auto boundary_edge() const -> EdgePtr
    {
        // Find a boundary edge
        for (const auto& e : edges_) {
            if (e.second->is_boundary()) {
                return e.second;
            }
        }
        return nullptr;
    }

    /**
     * @brief Build a list of all boundary edges
     *
     * Returns a list of lists of edges which lie on one of this mesh's
     * boundaries. One edge list is returned for each unique boundary. Meshes
     * with a single connected component may still have multiple boundaries
     * (i.e. if the mesh has holes).
     */
    auto boundaries() const -> std::vector<std::vector<EdgePtr>>
    {
        using Boundary = std::vector<EdgePtr>;
        Boundary boundary;
        std::vector<Boundary> boundaries;
        std::unordered_set<std::size_t> visited;
        for (const auto& e : edges_) {
            const auto edge = e.second;
            // Skip is not a boundary edge or already visited
            if (not edge->is_boundary() or visited.count(edge->idx) > 0) {
                continue;
            }

            // Create a new boundary
            boundary.clear();
            visited.insert(edge->idx);
            boundary.emplace_back(edge);
            auto test = edge->next;
            do {
                visited.insert(test->idx);
                boundary.emplace_back(test);
                test = test->next;
            } while (test != edge);
            boundaries.emplace_back(boundary);
        }
        return boundaries;
    }

    /** @brief Get the list of faces in insertion order */
    auto faces() const -> std::vector<FacePtr> { return faces_; }

    /** @brief Get a face by index */
    auto face(std::size_t idx) const -> FacePtr { return faces_.at(idx); }

    /**
     * @brief Get the number of connected components
     *
     * A connected component is a set of continuous, adjacent faces. If you
     * want to get the list of connected components, use connected_components().
     *
     * @see connected_components()
     */
    [[nodiscard]] auto num_connected_components() const -> std::size_t
    {
        std::size_t cnt{0};
        std::vector visited(num_faces(), false);
        std::queue<FacePtr> queue;
        // Iterate over the faces
        for (const auto& f : faces_) {
            // Skip faces we've visited
            if (visited[f->idx]) {
                continue;
            }

            // Start a new connected component
            queue.push(f);
            while (not queue.empty()) {
                // Get the top of the queue
                auto p = queue.front();
                queue.pop();
                // Mark as visited
                visited[p->idx] = true;
                // Add the neighbor faces to the queue
                for (const auto& e : *p) {
                    if (not e->pair->is_boundary()) {
                        auto n = e->pair->face;
                        if (not visited[n->idx]) {
                            queue.push(n);
                        }
                    }
                }
            }
            // Finished this component
            ++cnt;
        }
        return cnt;
    }

    /** @brief Get a list of connected components */
    auto connected_components() const -> std::vector<std::vector<FacePtr>>
    {
        // Tracking structures
        std::vector<std::vector<FacePtr>> components;
        std::vector visited(num_faces(), false);
        std::vector<FacePtr> current;
        std::queue<FacePtr> queue;

        // Iterate over the faces
        for (const auto& f : faces_) {
            // Skip faces we've visited
            if (visited[f->idx]) {
                continue;
            }

            // Start a new connected component
            current.clear();
            queue.push(f);
            while (not queue.empty()) {
                // Get the top of the queue
                auto p = queue.front();
                queue.pop();
                // Mark as visited
                visited[p->idx] = true;
                // Add to this connected component
                current.emplace_back(p);
                // Add the neighbor faces to the queue
                for (const auto& e : *p) {
                    if (not e->pair->is_boundary()) {
                        auto n = e->pair->face;
                        if (not visited[n->idx]) {
                            queue.push(n);
                        }
                    }
                }
            }
            // Add this component to the list
            components.emplace_back(current);
        }
        return components;
    }

    /** @brief Get the list of interior vertices in insertion order */
    auto vertices_interior() const -> std::vector<VertPtr>
    {
        std::vector<VertPtr> ret;
        std::copy_if(
            verts_.begin(), verts_.end(), std::back_inserter(ret),
            [](auto x) { return not x->is_boundary(); });
        return ret;
    }

    /** @brief Get the list of boundary vertices in insertion order */
    auto vertices_boundary() const -> std::vector<VertPtr>
    {
        std::vector<VertPtr> ret;
        std::copy_if(
            verts_.begin(), verts_.end(), std::back_inserter(ret),
            [](auto x) { return x->is_boundary(); });
        return ret;
    }

    /** @brief Get the number of vertices */
    [[nodiscard]] auto num_vertices() const -> std::size_t
    {
        return verts_.size();
    }

    /** @brief Get the number of interior vertices */
    [[nodiscard]] auto num_vertices_interior() const -> std::size_t
    {
        return std::accumulate(
            verts_.begin(), verts_.end(), std::size_t{0}, [](auto a, auto b) {
                return a + static_cast<std::size_t>(not b->is_boundary());
            });
    }

    /** @brief Get the number of edges */
    [[nodiscard]] auto num_edges() const -> std::size_t
    {
        std::size_t ret = 0;
        for (const auto& [_, e] : edges_) {
            if (not e->is_boundary()) {
                ++ret;
            }
        }
        return ret;
    }

    /** @brief Get the number of faces */
    [[nodiscard]] auto num_faces() const -> std::size_t
    {
        return faces_.size();
    }

    /**
     * @brief Split an edge in order to introduce a new boundary
     *
     * Disconnects a single, paired half-edge (i.e. a single edge between two
     * connected triangles) into two, unpaired half-edges, creating a
     * "hole" in the mesh's connectivity graph. Useful when you want to
     * introduce a boundary, or tear, into a mesh to improve parameterization.
     *
     * @note If the given edge intersects with an existing boundary, one
     * or both of your endpoint vertices will be duplicated. Be sure to take
     * this into account when converting the parameterized mesh to your
     * (per-wedge) UV map.
     *
     * @see split_path()
     */
    void split_edge(const EdgePtr& edge)
    {
        // Get forward and backward edge
        auto oldFwd = edge;
        auto oldBwd = oldFwd->pair;

        // Don't split boundary edge pairs
        if (oldFwd->is_boundary() and oldBwd->is_boundary()) {
            return;
        }

        // Get initial vertices
        auto oldStart = oldFwd->vertex;
        auto oldEnd = oldBwd->vertex;
        auto startOnBoundary = oldStart->is_boundary();
        auto endOnBoundary = oldEnd->is_boundary();

        // Get the new starting vertex for this edge
        VertPtr newStart;
        EdgePtr startIn, startOut;
        if (startOnBoundary) {
            auto newIdx = insert_vertex(oldStart->pos);
            newStart = verts_.at(newIdx);

            auto in = detail::erase_if(
                incoming_edges(oldStart->idx),
                [](auto e) { return not e->is_boundary(); });
            auto out = detail::erase_if(
                outgoing_edges(oldStart->idx),
                [](auto e) { return not e->is_boundary(); });
            if (in.size() == 0 or out.size() == 0) {
                throw MeshException("No incoming/outgoing edges");
            }
            if (in.size() > 1 or out.size() > 1) {
                throw MeshException("Too many incoming/outgoing edges");
            }
            startIn = in[0];
            startOut = out[0];
        } else {
            newStart = oldStart;
        }

        // Get the new ending vertex for this edge
        VertPtr newEnd;
        EdgePtr endIn, endOut;
        if (endOnBoundary) {
            auto newIdx = insert_vertex(oldEnd->pos);
            newEnd = verts_.at(newIdx);

            auto in = detail::erase_if(incoming_edges(oldEnd->idx), [](auto e) {
                return not e->is_boundary();
            });
            auto out = detail::erase_if(
                outgoing_edges(oldEnd->idx),
                [](auto e) { return not e->is_boundary(); });
            if (in.size() == 0 or out.size() == 0) {
                throw MeshException("No incoming/outgoing edges");
            }
            if (in.size() > 1 or out.size() > 1) {
                throw MeshException("Too many incoming/outgoing edges");
            }
            endIn = in[0];
            endOut = out[0];
        } else {
            newEnd = oldEnd;
        }

        // Create new edge pair
        auto newFwd = Edge::New();
        auto newBwd = Edge::New();
        newFwd->pair = newBwd;
        newBwd->pair = newFwd;
        newFwd->mesh = newBwd->mesh = this;

        // Assign vertices and add to mesh
        newFwd->vertex = newStart;
        newFwd->idx = edges_.size();
        edges_.emplace(newStart->idx, newFwd);
        newBwd->vertex = newEnd;
        newBwd->idx = edges_.size();
        edges_.emplace(newEnd->idx, newBwd);

        // Update vertices with edge if required
        if (not newStart->edge) {
            newStart->edge = newFwd;
        }
        if (not newEnd->edge) {
            newEnd->edge = newBwd;
        }

        // New forward takes old forward's face edge idx
        std::swap(newFwd->idxI, oldFwd->idxI);

        // Move old forward's face to new forward
        std::swap(newFwd->face, oldFwd->face);
        std::swap(newFwd->next, oldFwd->next);
        std::swap(newFwd->prev, oldFwd->prev);
        std::swap(newFwd->alpha, oldFwd->alpha);
        if (newFwd->face->head == oldFwd) {
            newFwd->face->head = newFwd;
        }

        // Update the face's edges
        newFwd->next->prev = newFwd;
        newFwd->prev->next = newFwd;
        newFwd->next->vertex = newEnd;
        newFwd->prev->pair->vertex = newStart;

        // Update new boundary edges' next/prev
        if (startOnBoundary) {
            startOut->vertex = newStart;
            newBwd->next = startOut;
            startOut->prev = newBwd;
            oldFwd->prev = startIn;
            startIn->next = oldFwd;
            for (auto e : newStart->wheel()) {
                e->vertex = newStart;
            }
        } else {
            newBwd->next = oldFwd;
            oldFwd->prev = newBwd;
        }
        if (endOnBoundary) {
            newBwd->prev = endIn;
            endIn->next = newBwd;
            oldFwd->next = endOut;
            endOut->prev = oldFwd;
            for (auto e : newEnd->wheel()) {
                e->vertex = newEnd;
            }
        } else {
            newBwd->prev = oldFwd;
            oldFwd->next = newBwd;
        }
    }

    /**
     * @brief Split a list of edges (path) to form a new boundary
     *
     * @note This function was designed to split a list of continuous edges
     * forming a path on the surface of the mesh. Providing otherwise can lead
     * to undefined behavior.
     *
     * @see split_edge()
     */
    void split_path(const std::vector<EdgePtr>& path)
    {
        // Split edges
        for (const auto& e : path) {
            split_edge(e);
        }
    }

    /**
     * @copybrief split_path
     *
     * This overload accepts a list of vertex indices forming a continuous path.
     *
     * @copydetails split_path
     */
    void split_path(const std::vector<std::size_t>& path)
    {
        // Convert index path to edge path
        std::vector<EdgePtr> edgePath;
        for (std::size_t i = 0; i < path.size() - 1; ++i) {
            auto e = this->edge(path[i], path[i + 1]);
            if (not e) {
                throw MeshException("Could not find edge");
            }
            edgePath.emplace_back(e);
        }

        split_path(edgePath);
    }

    /** @brief Get a list of outgoing edges from a specific vertex (by index) */
    auto outgoing_edges(const std::size_t idx) -> std::vector<EdgePtr>
    {
        const auto range = edges_.equal_range(idx);
        std::vector<EdgePtr> ret;
        ret.reserve(std::distance(range.first, range.second));
        std::transform(
            range.first, range.second, std::back_inserter(ret),
            [](auto it) { return it.second; });
        return ret;
    }

    /** @brief Get a list of incoming edges to a specific vertex (by index) */
    auto incoming_edges(const std::size_t idx) -> std::vector<EdgePtr>
    {
        auto outEdges = outgoing_edges(idx);
        std::vector<EdgePtr> ret;
        ret.reserve(outEdges.size());
        std::transform(
            outEdges.begin(), outEdges.end(), std::back_inserter(ret),
            [](auto e) { return e->pair; });
        return ret;
    }

private:
    /**
     * Face insertion implementation
     *
     * @param vector Iterable of vertex indices
     * @param face Pre-existing Face (only used when cloning)
     */
    template <class Vector>
    auto insert_face_(const Vector& vector, FacePtr face = nullptr)
        -> std::size_t
    {
        // Make a new face structure
        if (not face) {
            face = Face::New();
        }
        face->mesh = this;

        // Create a list of vertex pairs
        using IDType = std::size_t;
        using IDPair = std::pair<std::size_t, IDType>;
        std::vector<IDPair> endPts;
        for (std::size_t i = 0; i < std::size(vector); ++i) {
            auto nextIdx = i == std::size(vector) - 1 ? 0 : i + 1;
            endPts.emplace_back(
                std::begin(vector)[i], std::begin(vector)[nextIdx]);
        }

        // Create a new edge for every edge pair
        bool reverse{false};
        std::vector<EdgePtr> edges;
        for (const auto& [startIdx, endIdx] : endPts) {
            // See if this edge already exists
            auto thisEdge = this->edge(startIdx, endIdx);

            // If this edge doesn't exist, make it and its pair
            if (not thisEdge) {
                thisEdge = Edge::New();
                auto pair = Edge::New();
                thisEdge->pair = pair;
                pair->pair = thisEdge;
                thisEdge->mesh = pair->mesh = this;

                thisEdge->idx = edges_.size();
                edges_.emplace(startIdx, thisEdge);
                pair->idx = edges_.size();
                edges_.emplace(endIdx, pair);

                thisEdge->vertex = verts_.at(startIdx);
                if (not thisEdge->vertex->edge) {
                    thisEdge->vertex->edge = thisEdge;
                }

                pair->vertex = verts_.at(endIdx);
                if (not pair->vertex->edge) {
                    pair->vertex->edge = pair;
                }
            }

            // Reverse winding order
            if (reverse) {
                thisEdge = thisEdge->pair;
            }

            // If this edge has a face, try reversing the winding order
            if (thisEdge->face) {
                auto pair = thisEdge->pair;
                // Winding order error if already reversed
                // TODO: Theoretically could recursively flip winding order for
                //       adjacent faces which violate the order
                if (reverse) {
                    const auto msg =
                        "Winding order cannot be fixed for face"
                        "with vids=" +
                        detail::vec_to_string(vector);
                    throw MeshException(msg);
                }
                // Non-manifold error if the pair is already assigned
                if (pair->face) {
                    const auto msg =
                        "Attempted to add non-manifold face along "
                        "edge with vids=" +
                        detail::vec_to_string(vector);
                    throw MeshException(msg);
                }
                reverse = true;
                thisEdge = pair;

                // If reversing, update existing visited edges with the pair
                for (auto& e : edges) {
                    pair = e->pair;
                    // If pair already has a face, then manifold error
                    if (pair->face) {
                        const auto msg =
                            "Attempted to add non-manifold face "
                            "along edge with vids=[" +
                            std::to_string(startIdx) + ", " +
                            std::to_string(endIdx) + "]";
                        throw MeshException(msg);
                    }
                    e = pair;
                }
            }
            thisEdge->face = face;

            // Set the head edge for this face
            if (not face->head) {
                face->head = thisEdge;
            }

            // Store the edges for next/prev later
            edges.push_back(thisEdge);
        }

        // Update next/previous
        for (std::size_t i = 0; i < edges.size(); ++i) {
            auto edge = edges[i];
            const auto prevIdx = i == 0 ? edges.size() - 1 : i - 1;
            const auto nextIdx = i == edges.size() - 1 ? 0 : i + 1;
            edge->next = reverse ? edges[prevIdx] : edges[nextIdx];
            edge->prev = reverse ? edges[nextIdx] : edges[prevIdx];
            if (not edge->idxI) {
                edge->idxI = numFaceEdges_;
                ++numFaceEdges_;
            }
        }

        // Sanity check: edge lengths
        for (const auto& e : *face) {
            if (norm(e->next->vertex->pos - e->vertex->pos) == 0.0) {
                auto msg = "Zero-length edge (" +
                           std::to_string(e->vertex->idx) + ", " +
                           std::to_string(e->next->vertex->idx) + ")";
                throw MeshException(msg);
            }
        }

        // Compute angles for edges in face
        ComputeFaceAngles(face);

        // Give this face an idx and link the previous face with this one
        face->idx = faces_.size();
        if (not faces_.empty()) {
            faces_.back()->next = face;
        }
        faces_.emplace_back(face);
        return face->idx;
    }

    /**
     * Extra steps which need to be run before insert_face_ when cloning a face
     * from an existing mesh
     *
     * @param face Existing face from the mesh being cloned
     */
    auto clone_face_(const FacePtr& face)
    {
        // Copy the existing face
        auto f = Face::New(*face);

        // Pre-make all edges
        std::vector<std::size_t> idxs;
        for (const auto& e : *face) {
            auto startIdx = e->vertex->idx;
            auto endIdx = e->pair->vertex->idx;
            idxs.emplace_back(startIdx);

            // Make sure we haven't created this edge and pair already
            auto outEdge = this->edge(startIdx, endIdx);

            // Copy all inherited properties
            if (not outEdge) {
                outEdge = Edge::New(*e);
                auto inEdge = Edge::New(*e->pair);

                outEdge->pair = inEdge;
                inEdge->pair = outEdge;
                outEdge->mesh = inEdge->mesh = this;

                outEdge->idx = edges_.size();
                edges_.emplace(startIdx, outEdge);
                inEdge->idx = edges_.size();
                edges_.emplace(endIdx, inEdge);

                outEdge->vertex = verts_.at(startIdx);
                if (not outEdge->vertex->edge) {
                    outEdge->vertex->edge = outEdge;
                }

                inEdge->vertex = verts_.at(endIdx);
                if (not inEdge->vertex->edge) {
                    inEdge->vertex->edge = inEdge;
                }
            }
        }

        // Create the face
        return insert_face_(idxs, f);
    }
};
}  // namespace OpenABF

// #include "OpenABF/ABF.hpp"


#include <cmath>

#include <Eigen/SparseLU>

// #include "OpenABF/Exceptions.hpp"

// #include "OpenABF/HalfEdgeMesh.hpp"

// #include "OpenABF/Math.hpp"


namespace OpenABF
{

namespace traits
{
/** @brief ABF and ABFPlusPlus vertex traits */
template <typename T>
struct ABFVertexTraits : DefaultVertexTraits<T> {
    /** Lagrange Multiplier: Planarity constraint */
    T lambda_plan{0};
    /** Lagrange Multiplier: Reconstruction constraint */
    T lambda_len{1};
};

/** @brief ABF and ABFPlusPlus edge traits */
template <typename T>
struct ABFEdgeTraits : DefaultEdgeTraits<T> {
    /** 3D angle */
    T beta{0};
    /** Optimal (i.e. target) angle */
    T phi{0};
    /** Angle weight */
    T weight{0};
    /** Sin of alpha, because it's used a lot */
    T alpha_sin{0};
    /** Cos of alpha, because it's used a lot */
    T alpha_cos{0};
};

/** @brief ABF and ABFPlusPlus face traits */
template <typename T>
struct ABFFaceTraits : DefaultFaceTraits<T> {
    /** Lagrange Multiplier: Triangle validity constraint */
    T lambda_tri{0};
};
}  // namespace traits

/** @brief %ABF and ABF++ implementation details */
namespace detail::ABF
{

/** @brief A HalfEdgeMesh with the %ABF traits */
template <typename T>
using Mesh = HalfEdgeMesh<
    T,
    3,
    traits::ABFVertexTraits<T>,
    traits::ABFEdgeTraits<T>,
    traits::ABFFaceTraits<T>>;

/** @brief Initialize the %ABF angles and weights from the edge alpha values */
template <typename T, class MeshPtr>
void InitializeAnglesAndWeights(MeshPtr& m)
{
    // Initialize and bound angle properties
    static constexpr auto MinAngle = PI<T> / T(180);
    static constexpr auto MaxAngle = PI<T> - MinAngle;
    for (auto& e : m->edges()) {
        e->alpha = e->beta = e->phi =
            std::min(std::max(e->alpha, MinAngle), MaxAngle);
        e->alpha_sin = std::sin(e->alpha);
        e->alpha_cos = std::cos(e->alpha);
        e->weight = T(1) / (e->phi * e->phi);
    }

    // Update weights for interior vertices
    for (auto& v : m->vertices_interior()) {
        auto wheel = v->wheel();
        auto angle_sum = std::accumulate(
            wheel.begin(), wheel.end(), T(0),
            [](auto a, auto b) { return a + b->beta; });
        for (auto& e : wheel) {
            e->phi *= 2 * PI<T> / angle_sum;
            e->weight = T(1) / (e->phi * e->phi);
        }
    }
}

/** @brief Compute ∇CTri w.r.t LambdaTri == CTri */
template <
    typename T,
    class FacePtr,
    std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
auto TriGrad(const FacePtr& f) -> T
{
    T g = -PI<T>;
    for (const auto& e : *f) {
        g += e->alpha;
    }
    return g;
}

/** @brief Compute ∇CPlan w.r.t LambdaPlan == CPlan */
template <
    typename T,
    class VertPtr,
    std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
auto PlanGrad(const VertPtr& v) -> T
{
    auto edges = v->wheel();
    T g = -2 * PI<T>;
    for (const auto& e : v->wheel()) {
        g += e->alpha;
    }
    return g;
}

/** @brief Compute ∇CLen w.r.t LambdaLen == CLen */
template <
    typename T,
    class VertPtr,
    std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
auto LenGrad(const VertPtr& vertex) -> T
{
    T p1{1};
    T p2{1};
    for (const auto& e : vertex->wheel()) {
        p1 *= e->next->alpha_sin;
        p2 *= e->next->next->alpha_sin;
    }
    return p1 - p2;
}

/** @brief Compute ∇CLen w.r.t edge->alpha */
template <
    typename T,
    class VertPtr,
    class EdgePtr,
    std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
auto LenGrad(const VertPtr& vertex, const EdgePtr& edge) -> T
{
    T p1{1};
    T p2{1};
    for (const auto& a : vertex->wheel()) {
        auto b = a->next;
        if (b == edge) {
            p1 *= b->alpha_cos;
            p2 = T(0);
        } else {
            p1 *= b->alpha_sin;
        }

        auto c = a->next->next;
        if (c == edge) {
            p1 = T(0);
            p2 *= c->alpha_cos;
        } else {
            p2 *= c->alpha_sin;
        }
    }
    return p1 - p2;
}

/** @brief Compute ∇F w.r.t an edge's alpha */
template <
    typename T,
    class EdgePtr,
    std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
auto AlphaGrad(const EdgePtr& edge) -> T
{
    // δE/δα
    auto g = (edge->alpha - edge->phi) * edge->weight;
    // δCTri/δα
    g += edge->face->lambda_tri;
    for (const auto& e : *edge->face) {
        // Skip boundary vertices
        if (e->vertex->is_boundary()) {
            continue;
        }
        if (e == edge) {
            // δCPlan/δα
            g += e->vertex->lambda_plan;
        } else {
            // δCLen/δα
            auto d = LenGrad<T>(e->vertex, edge);
            d *= e->vertex->lambda_len;
            g += d;
        }
    }
    return g;
}

/** @brief Compute ∇F w.r.t all parameters */
template <
    typename T,
    class MeshPtr,
    std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
auto Gradient(const MeshPtr& mesh) -> T
{
    T g{0};
    for (const auto& f : mesh->faces()) {
        // AlphaGrad for all edges
        for (const auto& e : *f) {
            auto gAlpha = AlphaGrad<T>(e);
            g += gAlpha * gAlpha;
        }
        // TriGrad for all faces
        auto gTri = TriGrad<T>(f);
        g += gTri * gTri;
    }

    // PlanGrad and LenGrad for all interior vertices
    for (const auto& v : mesh->vertices_interior()) {
        auto gPlan = PlanGrad<T>(v);
        g += gPlan * gPlan;

        auto gLen = LenGrad<T>(v);
        g += gLen * gLen;
    }
    return g;
}
}  // namespace detail::ABF

/**
 * @brief Compute parameterized interior angles using Angle-based flattening
 *
 * Iteratively computes a new set of interior angles which minimize the total
 * angular error of the parameterized mesh. This follows the standard
 * angled-based flattening formulation, which directly solves for the objective
 * functions and constraints. ABFPlusPlus is generally preferred as it
 * dramatically simplifies the size of the solved problem without introducing
 * more error.
 *
 * This class **does not** compute a parameterized mesh. Rather, it calculates
 * the optimal interior angles for such a mesh. To convert this information
 * into a full parameterization, pass the processed HalfEdgeMesh to
 * AngleBasedLSCM.
 *
 * Implements "Parameterization of faceted surfaces for meshing using
 * angle-based flattening" by Sheffer and de Sturler (2001)
 * \cite sheffer2001abf.
 *
 * @tparam T Floating-point type
 * @tparam MeshType HalfEdgeMesh type which implements the ABF traits
 * @tparam Solver A solver implementing the
 * [Eigen Sparse solver
 * concept](https://eigen.tuxfamily.org/dox-devel/group__TopicSparseSystems.html)
 * and templated on Eigen::SparseMatrix<T>
 */
template <
    typename T,
    class MeshType = detail::ABF::Mesh<T>,
    class Solver =
        Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>>,
    std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
class ABF
{
public:
    /** @brief Mesh type alias */
    using Mesh = MeshType;

    /** @brief Set the maximum number of iterations */
    void setMaxIterations(const std::size_t it) { maxIters_ = it; }

    /**
     * @brief Get the mesh gradient
     *
     * **Note:** Result is only valid after running compute().
     */
    auto gradient() const -> T { return grad_; }

    /**
     * @brief Get the number of iterations of the last computation
     *
     * **Note:** Result is only valid after running compute().
     */
    [[nodiscard]] auto iterations() const -> std::size_t { return iters_; }

    /** @copydoc ABF::Compute */
    void compute(typename Mesh::Pointer& mesh)
    {
        Compute(mesh, iters_, grad_, maxIters_);
    }

    /**
     * @brief Compute parameterized interior angles
     *
     * @throws SolverException If matrix cannot be decomposed or if solver fails
     * to find a solution.
     * @throws MeshException If mesh gradient cannot be calculated.
     */
    static void Compute(
        typename Mesh::Pointer& mesh,
        std::size_t& iters,
        T& gradient,
        const std::size_t maxIters = 10)
    {
        using namespace detail::ABF;

        // Initialize angles and weights
        InitializeAnglesAndWeights<T>(mesh);

        // while ||∇F(x)|| > ε
        gradient = Gradient<T>(mesh);
        if (std::isnan(gradient) or std::isinf(gradient)) {
            throw MeshException("Mesh gradient cannot be computed");
        }
        auto gradDelta = INF<T>;
        iters = 0;
        while (gradient > 0.001 and gradDelta > 0.001 and iters < maxIters) {
            if (std::isnan(gradient) or std::isinf(gradient)) {
                throw MeshException("Mesh gradient cannot be computed");
            }
            // Typedefs
            using Triplet = Eigen::Triplet<T>;
            using SparseMatrix = Eigen::SparseMatrix<T>;
            using DenseVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

            // Helpful parameters
            auto vCnt = mesh->num_vertices();
            auto vIntCnt = mesh->num_vertices_interior();
            auto edgeCnt = mesh->num_edges();
            auto faceCnt = mesh->num_faces();

            //// RHS ////
            // b1 = -alpha gradient
            std::vector<Triplet> triplets;
            std::size_t idx{0};
            for (const auto& e : mesh->edges()) {
                triplets.emplace_back(idx, 0, -AlphaGrad<T>(e));
                ++idx;
            }

            // b2 = -lambda gradient
            // lambda tri
            for (const auto& f : mesh->faces()) {
                triplets.emplace_back(idx, 0, -TriGrad<T>(f));
                ++idx;
            }
            // lambda plan and lambda len
            for (const auto& v : mesh->vertices_interior()) {
                triplets.emplace_back(idx, 0, -PlanGrad<T>(v));
                triplets.emplace_back(vIntCnt + idx, 0, -LenGrad<T>(v));
                ++idx;
            }
            SparseMatrix b(edgeCnt + faceCnt + 2 * vIntCnt, 1);
            b.reserve(triplets.size());
            b.setFromTriplets(triplets.begin(), triplets.end());

            // vertex idx -> interior vertex idx permutation
            std::map<std::size_t, std::size_t> vIdx2vIntIdx;
            std::size_t newIdx{0};
            for (const auto& v : mesh->vertices_interior()) {
                vIdx2vIntIdx[v->idx] = newIdx++;
            }

            ///// LHS /////
            // Lambda = diag(2/w)
            // v.weight == 1/w, so Lambda is diag(2*weight)
            // We only need Lambda Inverse, so this is 1 / 2*weight
            triplets.clear();
            idx = 0;
            for (const auto& e : mesh->edges()) {
                triplets.emplace_back(idx, idx, 2 * e->weight);
                ++idx;
            }

            // J
            // Jacobian of the CTri constraints
            for (idx = 0; idx < faceCnt; idx++) {
                auto row = idx + edgeCnt;
                auto col = 3 * idx;
                triplets.emplace_back(row, col, 1);
                triplets.emplace_back(row, col + 1, 1);
                triplets.emplace_back(row, col + 2, 1);

                // Jt
                triplets.emplace_back(col, row, 1);
                triplets.emplace_back(col + 1, row, 1);
                triplets.emplace_back(col + 2, row, 1);
            }
            for (const auto& v : mesh->vertices_interior()) {
                auto row = idx + edgeCnt;
                for (const auto& e0 : v->wheel()) {
                    // Jacobian of the CPlan constraint
                    triplets.emplace_back(row, e0->idxI.value(), 1);
                    triplets.emplace_back(e0->idxI.value(), row, 1);

                    // Jacobian of the CLen constraint
                    auto e1 = e0->next;
                    auto e2 = e1->next;
                    auto d1 = LenGrad<T>(v, e1);
                    auto d2 = LenGrad<T>(v, e2);
                    triplets.emplace_back(vIntCnt + row, e1->idxI.value(), d1);
                    triplets.emplace_back(vIntCnt + row, e2->idxI.value(), d2);
                    triplets.emplace_back(e1->idxI.value(), vIntCnt + row, d1);
                    triplets.emplace_back(e2->idxI.value(), vIntCnt + row, d2);
                }
                ++idx;
            }
            auto Asize = edgeCnt + faceCnt + 2 * vIntCnt;
            SparseMatrix A(Asize, Asize);
            A.reserve(triplets.size());
            A.setFromTriplets(triplets.begin(), triplets.end());

            A.makeCompressed();
            Solver solver;
            solver.compute(A);
            if (solver.info() != Eigen::ComputationInfo::Success) {
                throw SolverException("ABF: Failed to solve A");
            }
            DenseVector delta = solver.solve(b);
            if (solver.info() != Eigen::ComputationInfo::Success) {
                throw SolverException("ABF: Failed to solve b");
            }

            // alpha += delta_alpha
            // Update sin and cos
            idx = 0;
            for (auto& e : mesh->edges()) {
                e->alpha += delta(idx++, 0);
                e->alpha = std::min(std::max(e->alpha, T(0)), PI<T>);
                e->alpha_sin = std::sin(e->alpha);
                e->alpha_cos = std::cos(e->alpha);
            }

            // lambda += delta_lambda
            for (auto& f : mesh->faces()) {
                f->lambda_tri += delta(idx++, 0);
            }
            for (auto& v : mesh->vertices_interior()) {
                auto intIdx = vIdx2vIntIdx.at(v->idx);
                v->lambda_plan += delta(idx + intIdx, 0);
                v->lambda_len += delta(idx + vIntCnt + intIdx, 0);
                idx++;
            }

            // Recalculate gradient for next iteration
            auto newGrad = detail::ABF::Gradient<T>(mesh);
            gradDelta = std::abs(newGrad - gradient);
            gradient = newGrad;
            iters++;
        }
    }

    /** @copydoc ABF::Compute */
    static void Compute(typename Mesh::Pointer& mesh)
    {
        std::size_t iters{0};
        T grad{0};
        Compute(mesh, iters, grad);
    }

protected:
    /** Gradient */
    T grad_{0};
    /** Number of executed iterations */
    std::size_t iters_{0};
    /** Max iterations */
    std::size_t maxIters_{10};
};

}  // namespace OpenABF
// #include "OpenABF/ABFPlusPlus.hpp"


#include <cmath>

#include <Eigen/SparseLU>

// #include "OpenABF/ABF.hpp"

// #include "OpenABF/Exceptions.hpp"

// #include "OpenABF/HalfEdgeMesh.hpp"

// #include "OpenABF/Math.hpp"


namespace OpenABF
{

/**
 * @brief Compute parameterized interior angles using ABF++
 *
 * Iteratively computes a new set of interior angles which minimize the total
 * angular error of the parameterized mesh. This follows the ABF++ formulation,
 * which solves a 5x smaller system of equations than standard ABF at the
 * expense of more iterations.
 *
 * This class **does not** compute a parameterized mesh. Rather, it calculates
 * the optimal interior angles for such a mesh. To convert this information
 * into a full parameterization, pass the processed HalfEdgeMesh to
 * AngleBasedLSCM.
 *
 * Implements "ABF++: Fast and Robust Angle Based Flattening" by Sheffer
 * _et al._ (2005) \cite sheffer2005abf++.
 *
 * @tparam T Floating-point type
 * @tparam MeshType HalfEdgeMesh type which implements the ABF traits
 * @tparam Solver A solver implementing the
 * [Eigen Sparse solver
 * concept](https://eigen.tuxfamily.org/dox-devel/group__TopicSparseSystems.html)
 * and templated on Eigen::SparseMatrix<T>
 */
template <
    typename T,
    class MeshType = detail::ABF::Mesh<T>,
    class Solver =
        Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>>,
    std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
class ABFPlusPlus
{
public:
    /** @brief Mesh type alias */
    using Mesh = MeshType;

    /** @brief Set the maximum number of iterations */
    void setMaxIterations(std::size_t it) { maxIters_ = it; }

    /**
     * @brief Get the mesh gradient
     *
     * **Note:** Result is only valid after running compute().
     */
    auto gradient() const -> T { return grad_; }

    /**
     * @brief Get the number of iterations of the last computation
     *
     * **Note:** Result is only valid after running compute().
     */
    [[nodiscard]] auto iterations() const -> std::size_t { return iters_; }

    /** @copydoc ABFPlusPlus::Compute */
    void compute(typename Mesh::Pointer& mesh)
    {
        Compute(mesh, iters_, grad_, maxIters_);
    }

    /**
     * @brief Compute parameterized interior angles
     *
     * @throws SolverException If matrix cannot be decomposed or if solver fails
     * to find a solution.
     * @throws MeshException If mesh gradient cannot be calculated.
     */
    static void Compute(
        typename Mesh::Pointer& mesh,
        std::size_t& iters,
        T& gradient,
        const std::size_t maxIters = 10)
    {
        using namespace detail::ABF;

        // Initialize angles and weights
        InitializeAnglesAndWeights<T>(mesh);

        // while ||∇F(x)|| > ε
        gradient = Gradient<T>(mesh);
        if (std::isnan(gradient) or std::isinf(gradient)) {
            throw MeshException("Mesh gradient cannot be computed");
        }
        auto gradDelta = INF<T>;
        iters = 0;
        while (gradient > 0.001 and gradDelta > 0.001 and iters < maxIters) {
            if (std::isnan(gradient) or std::isinf(gradient)) {
                throw MeshException("Mesh gradient cannot be computed");
            }
            // Typedefs
            using Triplet = Eigen::Triplet<T>;
            using SparseMatrix = Eigen::SparseMatrix<T>;
            using DenseVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

            // Helpful parameters
            auto vIntCnt = mesh->num_vertices_interior();
            auto edgeCnt = mesh->num_edges();
            auto faceCnt = mesh->num_faces();

            // b1 = -alpha gradient
            std::vector<Triplet> triplets;
            std::size_t idx{0};
            for (const auto& e : mesh->edges()) {
                triplets.emplace_back(idx, 0, -AlphaGrad<T>(e));
                ++idx;
            }
            SparseMatrix b1(edgeCnt, 1);
            b1.reserve(triplets.size());
            b1.setFromTriplets(triplets.begin(), triplets.end());

            // b2 = -lambda gradient
            triplets.clear();
            idx = 0;
            // lambda tri
            for (const auto& f : mesh->faces()) {
                triplets.emplace_back(idx, 0, -TriGrad<T>(f));
                idx++;
            }
            // lambda plan and lambda len
            for (const auto& v : mesh->vertices_interior()) {
                triplets.emplace_back(idx, 0, -PlanGrad<T>(v));
                triplets.emplace_back(vIntCnt + idx, 0, -LenGrad<T>(v));
                idx++;
            }
            SparseMatrix b2(faceCnt + 2 * vIntCnt, 1);
            b2.reserve(triplets.size());
            b2.setFromTriplets(triplets.begin(), triplets.end());

            // vertex idx -> interior vertex idx permutation
            std::map<std::size_t, std::size_t> vIdx2vIntIdx;
            std::size_t newIdx{0};
            for (const auto& v : mesh->vertices_interior()) {
                vIdx2vIntIdx[v->idx] = newIdx++;
            }

            // Compute J1 + J2
            triplets.clear();
            idx = 0;
            // Jacobian of the CTri constraints
            for (; idx < faceCnt; idx++) {
                triplets.emplace_back(idx, 3 * idx, 1);
                triplets.emplace_back(idx, 3 * idx + 1, 1);
                triplets.emplace_back(idx, 3 * idx + 2, 1);
            }
            for (const auto& v : mesh->vertices_interior()) {
                for (const auto& e0 : v->wheel()) {
                    // Jacobian of the CPlan constraint
                    triplets.emplace_back(idx, e0->idxI.value(), 1);

                    // Jacobian of the CLen constraint
                    auto e1 = e0->next;
                    auto e2 = e1->next;
                    auto d1 = LenGrad<T>(v, e1);
                    auto d2 = LenGrad<T>(v, e2);
                    triplets.emplace_back(vIntCnt + idx, e1->idxI.value(), d1);
                    triplets.emplace_back(vIntCnt + idx, e2->idxI.value(), d2);
                }
                ++idx;
            }
            SparseMatrix J(faceCnt + 2 * vIntCnt, 3 * faceCnt);
            J.reserve(triplets.size());
            J.setFromTriplets(triplets.begin(), triplets.end());

            // Lambda = diag(2/w)
            // v.weight == 1/w, so LambdaInv is diag(2*weight)
            // We only need Lambda Inverse, so this is 1 / 2*weight
            triplets.clear();
            idx = 0;
            for (const auto& e : mesh->edges()) {
                triplets.emplace_back(idx, idx, T(1) / (2 * e->weight));
                ++idx;
            }
            SparseMatrix LambdaInv(edgeCnt, edgeCnt);
            LambdaInv.reserve(edgeCnt);
            LambdaInv.setFromTriplets(triplets.begin(), triplets.end());

            // solve Eq. 16
            auto bstar = J * LambdaInv * b1 - b2;
            auto JLiJt = J * LambdaInv * J.transpose();

            SparseMatrix LambdaStarInv = JLiJt.block(0, 0, faceCnt, faceCnt);
            for (int k = 0; k < LambdaStarInv.outerSize(); ++k) {
                for (typename SparseMatrix::InnerIterator it(LambdaStarInv, k);
                     it; ++it) {
                    it.valueRef() = 1.F / it.value();
                }
            }
            auto Jstar = JLiJt.block(faceCnt,0,2*vIntCnt,faceCnt);
            auto JstarT = JLiJt.block(0,faceCnt,faceCnt, 2*vIntCnt);
            auto Jstar2 = JLiJt.block(faceCnt,faceCnt,2*vIntCnt, 2*vIntCnt);
            auto bstar1 = bstar.block(0, 0, faceCnt, 1);
            auto bstar2 = bstar.block(faceCnt, 0, 2*vIntCnt, 1);

            // (J* Lam*^-1 J*^t - J**) delta_lambda_2 = J* Lam*^-1 b*_1 - b*_2
            SparseMatrix A = Jstar * LambdaStarInv * JstarT - Jstar2;
            SparseMatrix b = Jstar * LambdaStarInv * bstar1 - bstar2;
            A.makeCompressed();
            Solver solver;
            solver.compute(A);
            if (solver.info() != Eigen::ComputationInfo::Success) {
                throw SolverException("ABF++: Failed to solve A");
            }
            auto deltaLambda2 = solver.solve(b);
            if (solver.info() != Eigen::ComputationInfo::Success) {
                throw SolverException("ABF++: Failed to solve b");
            }

            // Compute Eq. 17 -> delta_lambda_1
            auto deltaLambda1 =
                LambdaStarInv * (bstar1 - JstarT * deltaLambda2);

            // Construct deltaLambda
            DenseVector deltaLambda(
                deltaLambda1.rows() + deltaLambda2.rows(), 1);
            deltaLambda << DenseVector(deltaLambda1), DenseVector(deltaLambda2);

            // Compute Eq. 10 -> delta_alpha
            DenseVector deltaAlpha =
                LambdaInv * (b1 - J.transpose() * deltaLambda);

            // lambda += delta_lambda
            for (auto& f : mesh->faces()) {
                f->lambda_tri += deltaLambda(f->idx, 0);
            }
            for (auto& v : mesh->vertices_interior()) {
                auto intIdx = vIdx2vIntIdx.at(v->idx);
                v->lambda_plan += deltaLambda(faceCnt + intIdx, 0);
                v->lambda_len += deltaLambda(faceCnt + vIntCnt + intIdx, 0);
            }

            // alpha += delta_alpha
            // Update sin and cos
            idx = 0;
            for (auto& e : mesh->edges()) {
                e->alpha += deltaAlpha(idx++, 0);
                e->alpha = std::min(std::max(e->alpha, T(0)), PI<T>);
                e->alpha_sin = std::sin(e->alpha);
                e->alpha_cos = std::cos(e->alpha);
            }

            // Recalculate gradient for next iteration
            auto newGrad = Gradient<T>(mesh);
            gradDelta = std::abs(newGrad - gradient);
            gradient = newGrad;
            iters++;
        }
    }

    /** @brief Compute parameterized interior angles */
    static void Compute(typename Mesh::Pointer& mesh)
    {
        std::size_t iters{0};
        T grad{0};
        Compute(mesh, iters, grad);
    }

private:
    /** Gradient */
    T grad_{0};
    /** Number of executed iterations */
    std::size_t iters_{0};
    /** Max iterations */
    std::size_t maxIters_{10};
};

}  // namespace OpenABF
// #include "OpenABF/AngleBasedLSCM.hpp"


#include <cmath>
#include <map>
#include <type_traits>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>

// #include "OpenABF/Exceptions.hpp"

// #include "OpenABF/HalfEdgeMesh.hpp"

// #include "OpenABF/Math.hpp"


namespace OpenABF
{

namespace detail
{
/** Check if type is an instance of a template type: False */
template <class T, template <class...> class U>
constexpr bool is_instance_of_v = std::false_type{};

/** Check if type is an instance of a template type: True */
template <template <class...> class U, class... Vs>
constexpr bool is_instance_of_v<U<Vs...>, U> = std::true_type{};

/** Solve least squares using A'Ab  */
template <
    class SparseMatrix,
    class DenseMatrix,
    class Solver,
    std::enable_if_t<
        !is_instance_of_v<Solver, Eigen::LeastSquaresConjugateGradient>,
        bool> = false>
auto SolveLeastSquares(SparseMatrix A, SparseMatrix b) -> DenseMatrix
{
    // Setup AtA and solver
    SparseMatrix AtA = A.transpose() * A;
    AtA.makeCompressed();
    Solver solver;
    solver.compute(AtA);
    if (solver.info() != Eigen::ComputationInfo::Success) {
        throw SolverException("AB-LSCM: Failed to solve AtA");
    }

    // Setup Atb
    SparseMatrix Atb = A.transpose() * b;

    // Solve AtAx = AtAb
    DenseMatrix x = solver.solve(Atb);

    return x;
}

/** Solve least squares with LeastSquaresConjugateGradient */
template <
    class SparseMatrix,
    class DenseMatrix,
    class Solver,
    std::enable_if_t<
        is_instance_of_v<Solver, Eigen::LeastSquaresConjugateGradient>,
        bool> = true>
auto SolveLeastSquares(SparseMatrix A, SparseMatrix b) -> DenseMatrix
{
    // Solve
    Solver solver(A);
    DenseMatrix x = solver.solve(b);
    if (solver.info() != Eigen::ComputationInfo::Success) {
        throw SolverException("AB-LSCM: Failed to solve for b");
    }

    return x;
}

}  // namespace detail

/**
 * @brief Compute parameterized mesh using Angle-based LSCM
 *
 * Computes a least-squares conformal parameterization of a mesh. Unlike the
 * original LSCM algorithm, this class ignores the 3D vertex positions and
 * instead uses the angle associated with the mesh's edge trait
 * (MeshType::EdgeTraits::alpha) to calculate the initial per-triangle edge
 * lengths. Without previously modifying the angles of the provided mesh, this
 * class produces the same result as a vertex-based LSCM implementation.
 * However, by first processing the mesh with a parameterized angle optimizer,
 * such as ABFPlusPlus, the parameterization can be improved, sometimes
 * significantly.
 *
 * Implements the angle-based variant of "Least squares conformal maps for
 * automatic texture atlas generation" by Lévy _et al._ (2002)
 * \cite levy2002lscm.
 *
 * @tparam T Floating-point type
 * @tparam MeshType HalfEdgeMesh type which implements the default mesh traits
 * @tparam Solver A solver implementing the
 * [Eigen Sparse solver
 * concept](https://eigen.tuxfamily.org/dox-devel/group__TopicSparseSystems.html)
 * and templated on Eigen::SparseMatrix<T>
 */
template <
    typename T,
    class MeshType = HalfEdgeMesh<T>,
    class Solver =
        Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>>,
    std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
class AngleBasedLSCM
{
public:
    /** @brief Mesh type alias */
    using Mesh = MeshType;

    /** @copydoc AngleBasedLSCM::Compute */
    void compute(typename Mesh::Pointer& mesh) const { Compute(mesh); }

    /**
     * @brief Compute the parameterized mesh
     *
     * @throws MeshException If pinned vertex is not on boundary.
     * @throws SolverException If matrix cannot be decomposed or if solver fails
     * to find a solution.
     */
    static void Compute(typename Mesh::Pointer& mesh)
    {
        using Triplet = Eigen::Triplet<T>;
        using SparseMatrix = Eigen::SparseMatrix<T>;
        using DenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        // Pinned vertex selection
        // Get the end points of a boundary edge
        auto p0 = mesh->vertices_boundary()[0];
        auto e = p0->edge;
        do {
            if (e->pair->is_boundary()) {
                break;
            }
            e = e->pair->next;
        } while (e != p0->edge);
        if (e == p0->edge and not e->pair->is_boundary()) {
            throw MeshException("Pinned vertex not on boundary");
        }
        auto p1 = e->next->vertex;

        // Map selected edge to closest XY axis
        // Use sign to select direction
        auto pinVec = p1->pos - p0->pos;
        auto dist = norm(pinVec);
        pinVec /= dist;
        p0->pos = {T(0), T(0), T(0)};
        auto maxElem = std::max_element(pinVec.begin(), pinVec.end());
        auto maxAxis = std::distance(pinVec.begin(), maxElem);
        dist = std::copysign(dist, *maxElem);
        if (maxAxis == 0) {
            p1->pos = {dist, T(0), T(0)};
        } else {
            p1->pos = {T(0), dist, T(0)};
        }

        // For convenience
        auto numFaces = mesh->num_faces();
        auto numVerts = mesh->num_vertices();
        auto numFixed = 2;
        auto numFree = numVerts - numFixed;

        // Permutation for free vertices
        // This helps us find a vert's row in the solution matrix
        std::map<std::size_t, std::size_t> freeIdxTable;
        for (const auto& v : mesh->vertices()) {
            if (v == p0 or v == p1) {
                continue;
            }
            auto newIdx = freeIdxTable.size();
            freeIdxTable[v->idx] = newIdx;
        }

        // Setup pinned bFixed
        std::vector<Triplet> tripletsB;
        tripletsB.emplace_back(0, 0, p0->pos[0]);
        tripletsB.emplace_back(1, 0, p0->pos[1]);
        tripletsB.emplace_back(2, 0, p1->pos[0]);
        tripletsB.emplace_back(3, 0, p1->pos[1]);
        SparseMatrix bFixed(2 * numFixed, 1);
        bFixed.reserve(tripletsB.size());
        bFixed.setFromTriplets(tripletsB.begin(), tripletsB.end());

        // Setup variables matrix
        // Are only solving for free vertices, so push pins in special matrix
        std::vector<Triplet> tripletsA;
        tripletsB.clear();
        for (const auto& f : mesh->faces()) {
            auto e0 = f->head;
            auto e1 = e0->next;
            auto e2 = e1->next;
            auto sin0 = std::sin(e0->alpha);
            auto sin1 = std::sin(e1->alpha);
            auto sin2 = std::sin(e2->alpha);

            // Find the max sin idx
            std::vector<T> sins{sin0, sin1, sin2};
            auto sinMaxElem = std::max_element(sins.begin(), sins.end());
            auto sinMaxIdx = std::distance(sins.begin(), sinMaxElem);

            // Rotate the edge order of the face so last angle is largest
            if (sinMaxIdx == 0) {
                auto temp = e0;
                e0 = e1;
                e1 = e2;
                e2 = temp;
                sin0 = sins[1];
                sin1 = sins[2];
                sin2 = sins[0];
            } else if (sinMaxIdx == 1) {
                auto temp = e2;
                e2 = e1;
                e1 = e0;
                e0 = temp;
                sin0 = sins[2];
                sin1 = sins[0];
                sin2 = sins[1];
            }

            auto ratio = (sin2 == T(0)) ? T(1) : sin1 / sin2;
            auto cosine = std::cos(e0->alpha) * ratio;
            auto sine = sin0 * ratio;

            // If pin0 or pin1, put in fixedB matrix, else put in A
            auto row = 2 * f->idx;
            if (e0->vertex == p0) {
                tripletsB.emplace_back(row, 0, cosine - T(1));
                tripletsB.emplace_back(row, 1, -sine);
                tripletsB.emplace_back(row + 1, 0, sine);
                tripletsB.emplace_back(row + 1, 1, cosine - T(1));
            } else if (e0->vertex == p1) {
                tripletsB.emplace_back(row, 2, cosine - T(1));
                tripletsB.emplace_back(row, 3, -sine);
                tripletsB.emplace_back(row + 1, 2, sine);
                tripletsB.emplace_back(row + 1, 3, cosine - T(1));
            } else {
                auto freeIdx = freeIdxTable.at(e0->vertex->idx);
                tripletsA.emplace_back(row, 2 * freeIdx, cosine - T(1));
                tripletsA.emplace_back(row, 2 * freeIdx + 1, -sine);
                tripletsA.emplace_back(row + 1, 2 * freeIdx, sine);
                tripletsA.emplace_back(row + 1, 2 * freeIdx + 1, cosine - T(1));
            }

            if (e1->vertex == p0) {
                tripletsB.emplace_back(row, 0, -cosine);
                tripletsB.emplace_back(row, 1, sine);
                tripletsB.emplace_back(row + 1, 0, -sine);
                tripletsB.emplace_back(row + 1, 1, -cosine);
            } else if (e1->vertex == p1) {
                tripletsB.emplace_back(row, 2, -cosine);
                tripletsB.emplace_back(row, 3, sine);
                tripletsB.emplace_back(row + 1, 2, -sine);
                tripletsB.emplace_back(row + 1, 3, -cosine);
            } else {
                auto freeIdx = freeIdxTable.at(e1->vertex->idx);
                tripletsA.emplace_back(row, 2 * freeIdx, -cosine);
                tripletsA.emplace_back(row, 2 * freeIdx + 1, sine);
                tripletsA.emplace_back(row + 1, 2 * freeIdx, -sine);
                tripletsA.emplace_back(row + 1, 2 * freeIdx + 1, -cosine);
            }

            if (e2->vertex == p0) {
                tripletsB.emplace_back(row, 0, T(1));
                tripletsB.emplace_back(row + 1, 1, T(1));
            } else if (e2->vertex == p1) {
                tripletsB.emplace_back(row, 2, T(1));
                tripletsB.emplace_back(row + 1, 3, T(1));
            } else {
                auto freeIdx = freeIdxTable.at(e2->vertex->idx);
                tripletsA.emplace_back(row, 2 * freeIdx, T(1));
                tripletsA.emplace_back(row + 1, 2 * freeIdx + 1, T(1));
            }
        }
        SparseMatrix A(2 * numFaces, 2 * numFree);
        A.reserve(tripletsA.size());
        A.setFromTriplets(tripletsA.begin(), tripletsA.end());

        SparseMatrix bFree(2 * numFaces, 2 * numFixed);
        bFree.reserve(tripletsB.size());
        bFree.setFromTriplets(tripletsB.begin(), tripletsB.end());

        // Calculate rhs from free and fixed matrices
        SparseMatrix b = bFree * bFixed * -1;

        // Solve for x
        auto x =
            detail::SolveLeastSquares<SparseMatrix, DenseMatrix, Solver>(A, b);

        // Assign solution to UV coordinates
        // Pins are already updated, so these are free vertices
        for (const auto& v : mesh->vertices()) {
            if (v == p0 or v == p1) {
                continue;
            }
            auto newIdx = 2 * freeIdxTable.at(v->idx);
            v->pos[0] = x(newIdx, 0);
            v->pos[1] = x(newIdx + 1, 0);
            v->pos[2] = T(0);
        }
    }
};

}  // namespace OpenABF

// #include "OpenABF/MeshIO.hpp"


#include <filesystem>
#include <fstream>

// #include "OpenABF/MeshIOFormats.hpp"


#include <charconv>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <vector>

// #include "OpenABF/MeshIOUtils.hpp"


#include <algorithm>
#include <cctype>
#include <charconv>
#include <locale>
#include <string_view>
#include <vector>

namespace OpenABF::io_utils
{

/** @brief Compare two string_views, ignoring case */
static auto icase_compare(const std::string_view a, const std::string_view b)
    -> bool
{
    // not the same length
    if (a.length() != b.length()) {
        return false;
    }

    // iterate over the characters
    for (std::size_t i = 0; i < a.length(); ++i) {
        if (std::tolower(a[i]) != std::tolower(b[i])) {
            return false;
        }
    }

    // success
    return true;
}

/** @brief Left trim */
static auto trim_left(std::string_view s) -> std::string_view
{
    const auto& loc = std::locale();
    const auto* start = std::find_if_not(
        std::begin(s), std::end(s),
        [&loc](auto ch) -> bool { return std::isspace(ch, loc); });
    s.remove_prefix(std::distance(std::begin(s), start));
    return s;
}

/** @brief Right trim */
static auto trim_right(std::string_view s) -> std::string_view
{
    const auto& loc = std::locale();
    const auto* start =
        std::find_if_not(s.rbegin(), s.rend(), [&loc](auto ch) -> bool {
            return std::isspace(ch, loc);
        }).base();
    s.remove_suffix(std::distance(start, std::end(s)));
    return s;
}

/** @brief Trim from both ends */
static auto trim(std::string_view s) -> std::string_view
{
    s = trim_left(s);
    s = trim_right(s);
    return s;
}

/**
 * @brief Split a string by a delimiter
 *
 * When provided conflicting delimiters, the largest delimiter will take
 * precedence:
 *
 * ```{.cpp}
 * split("a->b->c", "-", "->");  // returns {"a", "b", "c"}
 * ```
 */
template <typename... Ds>
static auto split(std::string_view s, const Ds&... ds)
    -> std::vector<std::string_view>
{
    constexpr std::string_view DEFAULT_DELIM{" "};

    // Build delimiters list
    std::vector<std::string_view> delimiters;
    if (sizeof...(ds) > 0) {
        delimiters = {ds...};
    } else {
        delimiters.emplace_back(DEFAULT_DELIM);
    }

    // Get a list of all delimiter start pos and sizes
    std::vector<
        std::pair<std::string_view::size_type, std::string_view::size_type>>
        delimPos;
    for (const auto& delim : delimiters) {
        auto b = s.find(delim, 0);
        while (b != std::string_view::npos) {
            delimPos.emplace_back(b, delim.size());
            b = s.find(delim, b + delim.size());
        }
    }

    // Sort the delimiter start positions by first and largest
    std::sort(
        delimPos.begin(), delimPos.end(),
        [](const auto& l, const auto& r) { return l.second > r.second; });
    std::sort(
        delimPos.begin(), delimPos.end(),
        [](const auto& l, const auto& r) { return l.first < r.first; });

    // Split string
    std::vector<std::string_view> tokens;
    std::string_view::size_type begin{0};
    for (const auto& [end, size] : delimPos) {
        // ignore nested delimiters
        if (end < begin) {
            continue;
        }
        // get from begin to delim start
        if (auto t = s.substr(begin, end - begin); not t.empty()) {
            tokens.emplace_back(t);
        }
        begin = end + size;
    }
    if (auto t = s.substr(begin); not t.empty()) {
        tokens.emplace_back(t);
    }

    return tokens;
}

/**
 * @brief Convenience wrapper around std::to_chars for converting numerics to
 * std::string_view
 *
 * Useful during file writing operations when you're reusing a buffer, but
 * don't want to duplicate the error checking code of using std::to_chars.
 */
template <typename T>
auto to_string_view(const T& a, char* buf, const std::size_t& bufSize)
{
    auto res = std::to_chars(buf, buf + bufSize, a);
    if (res.ec != std::errc()) {
        throw std::runtime_error(std::make_error_code(res.ec).message());
    }
    return std::string_view(buf, res.ptr - buf);
}

/**
 * @brief Convert a string to a numeric type.
 *
 * A drop-in replacement for the `std:sto` family of functions which uses
 * `std::from_chars` for conversion. Like `std::sto`, throws exceptions when
 * conversion fails or if the converted value is out of range of the result
 * type.
 *
 * @throws std::invalid_argument If string cannot be converted to the result
 * type.
 * @throws std::result_out_of_range If converted value is out of range for the
 * result type.
 * @tparam T Requested numeric type
 * @tparam Args Parameter pack type
 * @param str Value to convert
 * @param args Extra parameters passed directly to `std::to_chars`
 * @return Converted value
 */
template <typename T, typename... Args>
auto to_numeric(const std::string_view str, Args... args) -> T
{
    T val;
    const auto* first = std::data(str);
    const auto* last = std::data(str) + std::size(str);
    auto [ptr, ec] = std::from_chars(first, last, val, args...);
    if (ec == std::errc::invalid_argument) {
        throw std::invalid_argument("Conversion could not be performed");
    }
    if (ec == std::errc::result_out_of_range) {
        throw std::out_of_range("Value out of range for the result type");
    }
    return val;
}

/**
 * @copybrief to_numeric
 *
 * Template specialization as fallback when the compiler does not support
 * `std::from_chars` for floating point types. Converts the input to a
 * `std::string` and passes to the appropriate `std::sto` function.
 */
template <>
inline auto to_numeric<float>(const std::string_view str) -> float
{
    return std::stof(std::string(str));
}

/** @copydoc to_numeric<float> */
template <>
inline auto to_numeric<double>(const std::string_view str) -> double
{
    return std::stod(std::string(str));
}

/** @copydoc to_numeric<float> */
template <>
inline auto to_numeric<long double>(const std::string_view str) -> long double
{
    return std::stold(std::string(str));
}
}  // namespace OpenABF::io_utils


namespace OpenABF::io_formats
{

/**
 * @brief Utility function for checking whether a given path matches one of the
 * accepted file extensions for a given format.
 */
template <typename PluginType>
static auto is_file_type(const std::filesystem::path& path)
{
    auto ext = path.extension().string();
    if (ext[0] == '.') {
        ext = ext.substr(1);
    }
    for (const auto& opt : PluginType::Extensions()) {
        if (io_utils::icase_compare(ext, opt)) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Wavefront %OBJ file format
 *
 * Read support:
 *   - Vertices (v)
 *   - Faces (f) but only the first element (vertex ID) is stored
 *
 * Write support:
 *   - Vertices (v)
 *   - Vertex normals (vn)
 *   - Faces (f) but only v//vn constructions
 *
 * @see [Object Files (.obj) by Paul
 * Bourke](https://paulbourke.net/dataformats/obj/)
 */
struct OBJ {
    /** @brief List of recognized file format extensions */
    static auto Extensions() -> std::vector<std::string_view>
    {
        return {"obj"};
    }

    /** Read the file stream into the provided object */
    template <typename MeshType>
    static auto Read(std::istream& is, MeshType& mesh)
    {
        using namespace io_utils;
        using T = typename MeshType::type;

        // Iterate the lines
        for (std::string line; std::getline(is, line);) {
            // Remove everything after a comment
            line = line.substr(0, line.find('#'));

            // Trim leading/trailing empty space
            auto line_view = trim(line);

            // Skip empty lines
            if (line_view.empty()) {
                continue;
            }

            // Split by part
            const auto parts = split(line_view);

            // Handle vertices
            if (parts[0] == "v") {
                std::vector<T> v;
                std::transform(
                    parts.begin() + 1, parts.end(), std::back_inserter(v),
                    to_numeric<T>);
                mesh.insert_vertex(v);
            }

            // Handle faces (v attribute only)
            else if (parts[0] == "f") {
                std::vector<std::size_t> indices;
                std::transform(
                    parts.begin() + 1, parts.end(), std::back_inserter(indices),
                    [](const auto& p) {
                        return to_numeric<std::size_t>(split(p, "/")[0]) - 1;
                    });
                mesh.insert_face(indices);
            }
        }
        mesh.update_boundary();
    }

    /** Write the provided object to the given file stream */
    template <typename MeshType>
    static void Write(std::ostream& os, MeshType& mesh)
    {
        // Character buffer
        constexpr auto bufSize = 128;
        char buf[bufSize];

        // Write vertices
        for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
            const auto v = mesh.vertex(i);
            // write vertex position
            os << "v";
            for (const auto& a : v->pos) {
                auto res = std::to_chars(buf, buf + bufSize, a);
                if (res.ec != std::errc()) {
                    throw std::runtime_error(
                        std::make_error_code(res.ec).message());
                }
                os << ' ' << std::string_view(buf, res.ptr - buf);
            }
            os << "\n";

            // write vertex normal
            os << "vn";
            for (const auto& a : v->normal()) {
                auto res = std::to_chars(buf, buf + bufSize, a);
                if (res.ec != std::errc()) {
                    throw std::runtime_error(
                        std::make_error_code(res.ec).message());
                }
                os << ' ' << std::string_view(buf, res.ptr - buf);
            }
            os << "\n";
        }

        // Write faces
        for (std::size_t i = 0; i < mesh.num_faces(); ++i) {
            const auto f = mesh.face(i);
            os << "f";
            for (const auto& e : *f) {
                auto res =
                    std::to_chars(buf, buf + bufSize, e->vertex->idx + 1);
                if (res.ec != std::errc()) {
                    throw std::runtime_error(
                        std::make_error_code(res.ec).message());
                }
                // write vertex and normal IDs
                const auto id = std::string_view(buf, res.ptr - buf);
                os << ' ' << id << "//" << id;
            }
            os << "\n";
        }
    }
};

/**
 * @brief %PLY Polygon file format
 *
 * Read support:
 *   - Vertex properties:
 *     - (float) x, y, z
 *   - Face properties:
 *     - (property list uchar int) vertex_index
 *
 * Write support:
 *   - Formats: ASCII
 *   - Vertex properties:
 *     - (float) x, y, z, nx, ny, nz
 *   - Face properties:
 *     - (list uchar int) vertex_index
 *
 * @see [PLY - Polygon File Format by Paul
 * Bourke](https://paulbourke.net/dataformats/ply/)
 */
struct PLY {
    /** @brief List of recognized file format extensions */
    static auto Extensions() -> std::vector<std::string_view>
    {
        return {"ply"};
    }

    /** Read the file stream into the provided object */
    template <typename MeshType>
    static auto Read(std::istream& is, MeshType& mesh)
    {
        using namespace io_utils;
        using T = typename MeshType::type;

        //// Parse header ////
        // Validate the type
        std::string line;
        std::getline(is, line);
        if (line != "ply") {
            throw std::runtime_error("File header does not begin with ply");
        }
        // Read the format line
        std::getline(is, line);
        const auto fmtParts = split(line);
        if (fmtParts[0] != "format") {
            throw std::runtime_error("File header missing format declaration");
        }
        if (fmtParts[1] != "ascii") {
            const auto fmt =
                std::string(fmtParts[1]) + " " + std::string(fmtParts[2]);
            throw std::runtime_error("Unsupported ply format: " + fmt);
        }

        // property = (label, type)
        struct Property {
            bool is_list{false};
            std::string list_count_type;
            std::string label;
            std::string type;
        };
        // element = (label, no. of elements, property list)
        struct Element {
            std::string label;
            std::uint32_t count{0};
            std::vector<Property> properties;
        };
        // list of elements
        std::vector<Element> elements;

        // Read the remaining header lines until end_header
        while (std::getline(is, line)) {
            // Trim leading/trailing empty space
            auto line_view = trim(line);

            // Skip empty lines
            if (line_view.empty()) {
                continue;
            }

            // Split by part
            const auto parts = split(line_view);

            // Handle comments (skip)
            if (parts[0] == "comment") {
                continue;
            }

            // Handle elements
            if (parts[0] == "element") {
                elements.push_back(
                    {.label = std::string(parts[1]),
                     .count = to_numeric<std::uint32_t>(parts[2])});
            }

            // Handle properties for the most recent element
            else if (parts[0] == "property") {
                if (parts[1] == "list") {
                    elements.back().properties.push_back(
                        {.is_list = true,
                         .list_count_type = std::string(parts[2]),
                         .label = std::string(parts[4]),
                         .type = std::string(parts[3])});
                } else {
                    elements.back().properties.push_back(
                        {.label = std::string(parts[2]),
                         .type = std::string(parts[1])});
                }
            }

            // Handle the end of the header
            else if (parts[0] == "end_header") {
                break;
            }
        }

        // Set up vertex map: v[n] -> property[m]
        // Probably unnecessary
        std::array<std::size_t, 3> vmap{};
        auto v_elem = std::find_if(
            elements.begin(), elements.end(),
            [](const auto& e) { return e.label == "vertex"; });
        if (v_elem == elements.end()) {
            throw std::runtime_error("Did not find vertex element");
        }
        for (auto i = 0; i < v_elem->properties.size(); ++i) {
            if (const auto& prop = v_elem->properties[i]; prop.label == "x") {
                vmap[0] = i;
            } else if (prop.label == "y") {
                vmap[1] = i;
            } else if (prop.label == "z") {
                vmap[2] = i;
            }
        }

        // Iterate the lines of the body
        constexpr auto max_line = std::numeric_limits<std::streamsize>::max();
        for (const auto e : elements) {
            // Iterate the element lines
            for (auto i = 0; i < e.count; i++) {
                // parse vertex line
                if (e.label == "vertex") {
                    std::getline(is, line);
                    const auto line_view = trim(line);
                    const auto parts = split(line_view);
                    mesh.insert_vertex(
                        to_numeric<T>(parts[vmap[0]]),
                        to_numeric<T>(parts[vmap[1]]),
                        to_numeric<T>(parts[vmap[2]]));
                }

                // parse face line
                else if (e.label == "face") {
                    std::getline(is, line);
                    const auto line_view = trim(line);
                    const auto parts = split(line_view);
                    if (parts[0] != "3") {
                        throw std::runtime_error(
                            "Unsupported number of vertices in face: " +
                            std::string(parts[0]));
                    }
                    mesh.insert_face(
                        to_numeric<std::size_t>(parts[1]),
                        to_numeric<std::size_t>(parts[2]),
                        to_numeric<std::size_t>(parts[3]));
                }

                // ignore unrecognized element
                else {
                    is.ignore(max_line, is.widen('\n'));
                }
            }
        }
        mesh.update_boundary();
    }

    /** Write the provided object to the given file stream */
    template <typename MeshType>
    static void Write(std::ostream& os, MeshType& mesh)
    {
        using namespace io_utils;

        // Character buffer
        constexpr auto bufSize = 128;
        char buf[bufSize];

        // Write header
        os << "ply" << '\n';
        os << "format ascii 1.0" << '\n';
        os << "comment OpenABF PLY IO" << '\n';
        // Vertex element
        os << "element vertex ";
        os << to_string_view(mesh.num_vertices(), buf, bufSize) << '\n';
        os << "property float x" << '\n';
        os << "property float y" << '\n';
        os << "property float z" << '\n';
        os << "property float nx" << '\n';
        os << "property float ny" << '\n';
        os << "property float nz" << '\n';
        // Face element
        os << "element face ";
        os << to_string_view(mesh.num_faces(), buf, bufSize) << '\n';
        os << "property list uchar int vertex_indices" << '\n';
        os << "end_header" << '\n';

        // Write vertices
        for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
            const auto v = mesh.vertex(i);
            // write vertex position
            bool is_first{true};
            for (const auto& a : v->pos) {
                if (not is_first) {
                    os << ' ';
                }
                os << to_string_view(a, buf, bufSize);
                is_first = false;
            }
            for (const auto& a : v->normal()) {
                os << ' ' << to_string_view(a, buf, bufSize);
            }
            os << '\n';
        }

        // Write faces
        for (std::size_t i = 0; i < mesh.num_faces(); ++i) {
            // Only supports triangular faces
            os << '3';
            const auto f = mesh.face(i);
            for (const auto& e : *f) {
                os << ' ' << to_string_view(e->vertex->idx, buf, bufSize);
            }
            os << '\n';
        }
    }
};
}  // namespace OpenABF::io_formats

namespace OpenABF
{

/** @brief Load a HalfEdgeMesh from a file */
template <class MeshType>
auto ReadMesh(const std::filesystem::path& path)
{
    // Open the file
    std::ifstream file(path, std::ios::in);
    if (not file.is_open()) {
        throw std::runtime_error(
            "Cannot open file for reading: " + path.string());
    }

    // Read the mesh
    auto result = MeshType::New();
    if (io_formats::is_file_type<io_formats::OBJ>(path)) {
        io_formats::OBJ::Read(file, *result);
    } else if (io_formats::is_file_type<io_formats::PLY>(path)) {
        io_formats::PLY::Read(file, *result);
    } else {
        throw std::runtime_error(
            "Unsupported file type: " + path.extension().string());
    }

    return result;
}

/** @brief Write a HalfEdgeMesh to a file */
template <class MeshPtr>
void WriteMesh(const std::filesystem::path& path, const MeshPtr& mesh)
{
    // Open the file
    std::ofstream file(path, std::ios::out);
    if (not file.is_open()) {
        throw std::runtime_error(
            "Cannot open file for writing: " + path.string());
    }

    // Write the mesh
    if (io_formats::is_file_type<io_formats::OBJ>(path)) {
        io_formats::OBJ::Write(file, *mesh);
    } else if (io_formats::is_file_type<io_formats::PLY>(path)) {
        io_formats::PLY::Write(file, *mesh);
    } else {
        throw std::runtime_error(
            "Unsupported file type: " + path.extension().string());
    }

    // Close file
    file.flush();
    file.close();
    if (file.fail()) {
        throw std::runtime_error("Failed to write file: " + path.string());
    }
}

}  // namespace OpenABF

// clang-format on