#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

// Minimal 3D array replacing xt::xtensor<T,3,column_major>.
// Column-major layout: element(z,y,x) = data[z + y*nz + x*nz*ny]
// (first index varies fastest, matching the existing calc_off convention).
template <typename T>
struct Array3D {
    std::vector<T> data_;
    std::array<size_t, 3> shape_ = {0, 0, 0};  // {z, y, x}

    Array3D() = default;

    explicit Array3D(std::array<size_t, 3> shape)
        : data_(shape[0] * shape[1] * shape[2]), shape_(shape) {}

    Array3D(std::array<size_t, 3> shape, T fill)
        : data_(shape[0] * shape[1] * shape[2], fill), shape_(shape) {}

    // Construct from initializer-list shape: Array3D<T>({z, y, x})
    Array3D(std::initializer_list<size_t> dims) {
        auto it = dims.begin();
        shape_[0] = *it++; shape_[1] = *it++; shape_[2] = *it;
        data_.resize(shape_[0] * shape_[1] * shape_[2]);
    }

    T& operator()(size_t z, size_t y, size_t x) {
        return data_[z + y * shape_[0] + x * shape_[0] * shape_[1]];
    }
    const T& operator()(size_t z, size_t y, size_t x) const {
        return data_[z + y * shape_[0] + x * shape_[0] * shape_[1]];
    }

    const std::array<size_t, 3>& shape() const { return shape_; }
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }

    void fill(T val) {
        std::fill(data_.begin(), data_.end(), val);
    }

    // Resize (data is zeroed)
    void resize(std::array<size_t, 3> shape) {
        shape_ = shape;
        data_.assign(shape[0] * shape[1] * shape[2], T{});
    }

    // Extract a sub-region [z0,z1) x [y0,y1) x [x0,x1)
    Array3D subarray(size_t z0, size_t z1, size_t y0, size_t y1, size_t x0, size_t x1) const {
        Array3D out({z1 - z0, y1 - y0, x1 - x0});
        for (size_t x = x0; x < x1; x++)
            for (size_t y = y0; y < y1; y++)
                for (size_t z = z0; z < z1; z++)
                    out(z - z0, y - y0, x - x0) = (*this)(z, y, x);
        return out;
    }
};
