#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <system_error>
#include <algorithm>
#include <cstring>

#include "vc/tracer/GrowSeg.hpp"
#include "z5/factory.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include "xtensor/xadapt.hpp"

namespace py = pybind11;

namespace {

std::filesystem::path to_path(py::handle obj, const char* name, bool required = true) {
    if (obj.is_none()) {
        if (required) {
            std::ostringstream oss;
            oss << name << " cannot be None";
            throw py::value_error(oss.str().c_str());
        }
        return {};
    }

    static py::object fspath = py::module_::import("os").attr("fspath");
    py::object py_path = fspath(obj);
    return std::filesystem::path(py_path.cast<std::string>());
}

cv::Vec3d to_origin(py::handle obj) {
    if (obj.is_none()) {
        return cv::Vec3d{0.0, 0.0, 0.0};
    }

    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    if (py::len(seq) != 3) {
        throw py::value_error("origin must be an iterable of length 3");
    }

    return cv::Vec3d{
        seq[0].cast<double>(),
        seq[1].cast<double>(),
        seq[2].cast<double>()
    };
}

cv::Vec3f to_vec3f(py::handle obj, const char* name) {
    if (obj.is_none()) {
        return cv::Vec3f{0.0f, 0.0f, 0.0f};
    }

    if (!py::isinstance<py::sequence>(obj)) {
        std::ostringstream oss;
        oss << name << " must be an iterable of length 3";
        throw py::value_error(oss.str());
    }

    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    if (py::len(seq) != 3) {
        std::ostringstream oss;
        oss << name << " must have exactly three elements";
        throw py::value_error(oss.str());
    }

    return cv::Vec3f{
        seq[0].cast<float>(),
        seq[1].cast<float>(),
        seq[2].cast<float>()
    };
}

cv::Size to_size(py::handle obj) {
    if (!py::isinstance<py::sequence>(obj)) {
        throw py::value_error("size must be a sequence of two positive integers");
    }

    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    if (py::len(seq) != 2) {
        throw py::value_error("size must have exactly two entries (rows, cols)");
    }

    const int rows = seq[0].cast<int>();
    const int cols = seq[1].cast<int>();
    if (rows <= 0 || cols <= 0) {
        throw py::value_error("size entries must be positive integers");
    }

    return cv::Size{cols, rows};
}

nlohmann::json to_json(py::handle obj);

nlohmann::json map_dict(py::dict dict_obj) {
    nlohmann::json result = nlohmann::json::object();
    for (auto item : dict_obj) {
        auto key = py::str(item.first);
        result[key.cast<std::string>()] = to_json(item.second);
    }
    return result;
}

nlohmann::json map_sequence(py::handle iterable) {
    nlohmann::json result = nlohmann::json::array();
    for (auto item : iterable) {
        result.push_back(to_json(item));
    }
    return result;
}

nlohmann::json to_json(py::handle obj) {
    if (obj.is_none()) {
        return nullptr;
    }

    if (py::isinstance<py::bool_>(obj)) {
        return obj.cast<bool>();
    }
    if (py::isinstance<py::int_>(obj)) {
        return obj.cast<long long>();
    }
    if (py::isinstance<py::float_>(obj)) {
        return obj.cast<double>();
    }
    if (py::isinstance<py::str>(obj)) {
        return obj.cast<std::string>();
    }
    if (py::isinstance<py::dict>(obj)) {
        return map_dict(obj.cast<py::dict>());
    }
    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        return map_sequence(obj);
    }
    if (py::isinstance<py::bytes>(obj)) {
        return obj.cast<std::string>();
    }

    if (py::hasattr(obj, "__json__")) {
        py::object json_obj = obj.attr("__json__")();
        return to_json(json_obj);
    }

    py::type type_obj = py::type::of(obj);
    std::string type_name = py::str(type_obj).cast<std::string>();
    std::string message = "Object of type '" + type_name + "' is not JSON serializable";
    throw py::type_error(message.c_str());
}

py::object to_python(const nlohmann::json& value) {
    if (value.is_null()) {
        return py::none();
    }
    if (value.is_boolean()) {
        return py::bool_(value.get<bool>());
    }
    if (value.is_number_integer()) {
        return py::int_(value.get<long long>());
    }
    if (value.is_number_unsigned()) {
        return py::int_(value.get<unsigned long long>());
    }
    if (value.is_number_float()) {
        return py::float_(value.get<double>());
    }
    if (value.is_string()) {
        return py::str(value.get<std::string>());
    }
    if (value.is_array()) {
        py::list list;
        for (const auto& item : value) {
            list.append(to_python(item));
        }
        return list;
    }
    if (value.is_object()) {
        py::dict dict;
        for (const auto& item : value.items()) {
            dict[py::str(item.key())] = to_python(item.value());
        }
        return dict;
    }
    return py::none();
}

struct TempZarrVolume {
    std::filesystem::path root;
    std::shared_ptr<z5::Dataset> dataset;

    ~TempZarrVolume() {
        dataset.reset();
        if (root.empty()) {
            return;
        }
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }
};

struct ArrayVolumeSpec {
    py::array_t<uint8_t, py::array::c_style> array;
    float voxel_size;
    std::optional<std::array<size_t, 3>> chunk_shape;
    std::optional<std::filesystem::path> cache_root;
};

std::filesystem::path make_temp_volume_dir() {
    static std::atomic<uint64_t> counter{0};
    const auto base = std::filesystem::temp_directory_path();
    for (int attempt = 0; attempt < 128; ++attempt) {
        const auto unique = counter.fetch_add(1, std::memory_order_relaxed);
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
        const auto ticks = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
        std::ostringstream oss;
        oss << "vc_numpy_volume_" << ticks << "_" << unique << "_" << attempt;
        const std::filesystem::path candidate = base / oss.str();
        std::error_code ec;
        if (!std::filesystem::exists(candidate, ec)) {
            return candidate;
        }
        if (ec && ec.value() != ENOENT) {
            throw std::system_error(ec, "Failed to probe temporary directory for NumPy volume input");
        }
    }
    throw std::runtime_error("Unable to create temporary directory for NumPy volume input after multiple attempts");
}

std::optional<std::filesystem::path> parse_normal_grid_path(py::handle obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }

    if (py::isinstance<py::str>(obj) || py::hasattr(obj, "__fspath__")) {
        return to_path(obj, "normal_grid");
    }

    throw py::value_error("normal_grid must be None or a path-like object");
}

ArrayVolumeSpec parse_array_volume(py::handle volume) {
    if (!py::isinstance<py::tuple>(volume)) {
        throw py::type_error("expected volume tuple (array, voxel_size, [chunk_shape], [cache_dir])");
    }

    py::tuple tuple = volume.cast<py::tuple>();
    if (tuple.size() < 2 || tuple.size() > 4) {
        throw py::value_error("volume tuple must be (array, voxel_size, [chunk_shape], [cache_dir])");
    }

    py::handle array_obj = tuple[0];
    py::handle voxel_obj = tuple[1];

    py::array_t<uint8_t, py::array::c_style> array = py::array_t<uint8_t, py::array::c_style>::ensure(array_obj);
    if (!array) {
        throw py::value_error("volume array must be a C-contiguous numpy.uint8 array");
    }
    if (array.ndim() != 3) {
        throw py::value_error("volume array must be 3-dimensional (z, y, x)");
    }

    float voxel_size = 0.0f;
    try {
        voxel_size = voxel_obj.cast<float>();
    } catch (const py::cast_error&) {
        throw py::value_error("voxel_size must be convertible to float");
    }
    if (!(voxel_size > 0.0f)) {
        throw py::value_error("voxel_size must be positive");
    }

    std::optional<std::array<size_t, 3>> chunk_shape;
    if (tuple.size() >= 3 && !tuple[2].is_none()) {
        py::sequence seq = tuple[2].cast<py::sequence>();
        if (py::len(seq) != 3) {
            throw py::value_error("chunk_shape must contain exactly three integers");
        }
        std::array<size_t, 3> parsed{};
        for (size_t i = 0; i < 3; ++i) {
            const ssize_t value = seq[i].cast<ssize_t>();
            if (value <= 0) {
                throw py::value_error("chunk_shape entries must be positive integers");
            }
            parsed[i] = static_cast<size_t>(value);
        }
        chunk_shape = parsed;
    }

    std::optional<std::filesystem::path> cache_root;
    if (tuple.size() == 4 && !tuple[3].is_none()) {
        cache_root = to_path(tuple[3], "cache_dir", false);
    }

    return ArrayVolumeSpec{
        std::move(array),
        voxel_size,
        chunk_shape,
        cache_root
    };
}

std::shared_ptr<TempZarrVolume> materialize_numpy_volume(const ArrayVolumeSpec& spec) {
    py::buffer_info info = spec.array.request();
    if (info.ndim != 3) {
        throw py::value_error("volume array must be 3-dimensional (z, y, x)");
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (info.shape[axis] <= 0) {
            throw py::value_error("volume array dimensions must be positive");
        }
        if (info.strides[axis] <= 0) {
            throw py::value_error("volume array strides must be positive");
        }
    }
    if (info.itemsize != sizeof(uint8_t)) {
        throw py::value_error("volume array must have uint8 item size");
    }

    const std::vector<size_t> shape{
        static_cast<size_t>(info.shape[0]),
        static_cast<size_t>(info.shape[1]),
        static_cast<size_t>(info.shape[2])
    };

    std::vector<size_t> chunk_shape(3);
    if (spec.chunk_shape) {
        std::copy(spec.chunk_shape->begin(), spec.chunk_shape->end(), chunk_shape.begin());
    } else {
        auto pick_chunk = [](size_t extent) -> size_t {
            constexpr size_t target = 64;
            return std::max<size_t>(1, std::min(extent, target));
        };
        chunk_shape[0] = pick_chunk(shape[0]);
        chunk_shape[1] = pick_chunk(shape[1]);
        chunk_shape[2] = pick_chunk(shape[2]);
    }

    const auto data_ptr = static_cast<uint8_t*>(info.ptr);
    if (data_ptr == nullptr) {
        throw std::runtime_error("volume array pointer is null");
    }

    const size_t element_count = static_cast<size_t>(info.size);
    const std::vector<size_t> element_strides{
        static_cast<size_t>(info.strides[0] / static_cast<ssize_t>(info.itemsize)),
        static_cast<size_t>(info.strides[1] / static_cast<ssize_t>(info.itemsize)),
        static_cast<size_t>(info.strides[2] / static_cast<ssize_t>(info.itemsize))
    };

    auto holder = std::make_shared<TempZarrVolume>();
    holder->root = make_temp_volume_dir();

    {
        py::gil_scoped_release release;

        z5::filesystem::handle::File file(holder->root);
        z5::createFile(file, true);

        const nlohmann::json compressor = {
            {"cname", "zstd"},
            {"clevel", 1},
            {"shuffle", 0}
        };

        holder->dataset = z5::createDataset(
            file,
            "0",
            "uint8",
            shape,
            chunk_shape,
            std::string("blosc"),
            compressor);

        z5::types::ShapeType offset = {0, 0, 0};
        auto tensor = xt::adapt(
            data_ptr,
            element_count,
            xt::no_ownership(),
            shape,
            element_strides);
        z5::multiarray::writeSubarray<uint8_t>(*holder->dataset, tensor, offset.begin());

        nlohmann::json meta;
        meta["voxelsize"] = spec.voxel_size;
        std::ofstream meta_stream(holder->root / "meta.json");
        if (!meta_stream) {
            throw std::runtime_error("failed to create meta.json for NumPy volume input");
        }
        meta_stream << meta.dump(2);
    }

    return holder;
}

bool is_array_volume(py::handle volume) {
    return py::isinstance<py::tuple>(volume);
}

std::shared_ptr<VCCollection> parse_corrections(py::handle obj) {
    if (obj.is_none()) {
        return nullptr;
    }

    auto corrections = std::make_shared<VCCollection>();

    const bool treat_as_path = py::isinstance<py::str>(obj) || (!py::isinstance<py::dict>(obj) && py::hasattr(obj, "__fspath__"));
    if (treat_as_path) {
        const auto path = to_path(obj, "corrections");
        if (!corrections->loadFromJSON(path.string())) {
            std::ostringstream oss;
            oss << "failed to load corrections from '" << path.string() << "'";
            throw py::value_error(oss.str().c_str());
        }
        return corrections;
    }

    nlohmann::json payload = to_json(obj);
    if (!corrections->loadFromJson(payload)) {
        throw py::value_error("corrections payload has unexpected structure");
    }

    return corrections;
}

py::tuple vec3_to_tuple(const cv::Vec3f& vec) {
    return py::make_tuple(vec[0], vec[1], vec[2]);
}

py::tuple vec2_to_tuple(const cv::Vec2f& vec) {
    return py::make_tuple(vec[0], vec[1]);
}

py::tuple rect3d_to_tuple(const Rect3D& rect) {
    return py::make_tuple(vec3_to_tuple(rect.low), vec3_to_tuple(rect.high));
}

py::array_t<float> mat_vec3f_to_numpy(const cv::Mat_<cv::Vec3f>& points) {
    if (points.empty()) {
        return py::array_t<float>({0, 0, 3});
    }

    const int rows = points.rows;
    const int cols = points.cols;
    py::array_t<float> result({rows, cols, 3});
    float* dst = result.mutable_data();
    for (int r = 0; r < rows; ++r) {
        const cv::Vec3f* src_row = points.ptr<cv::Vec3f>(r);
        float* dst_row = dst + static_cast<size_t>(r) * cols * 3;
        std::memcpy(dst_row, src_row, static_cast<size_t>(cols) * sizeof(cv::Vec3f));
    }
    return result;
}

py::array_t<float> quad_surface_points_to_numpy(const QuadSurface& surface) {
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points) {
        return py::array_t<float>({0, 0, 3});
    }
    return mat_vec3f_to_numpy(*points);
}

} // namespace

PYBIND11_MODULE(vc_tracing, m) {
    m.doc() = "Pybind11 bindings for volume-cartographer tracing routines";

    py::class_<QuadSurface, std::unique_ptr<QuadSurface>>(m, "QuadSurface")
        .def_property_readonly(
            "uuid",
            [](const QuadSurface& surface) -> py::object {
                if (surface.id.empty()) {
                    return py::none();
                }
                return py::str(surface.id);
            })
        .def_property_readonly(
            "path",
            [](const QuadSurface& surface) -> py::object {
                if (surface.path.empty()) {
                    return py::none();
                }
                return py::str(surface.path.string());
            })
        .def_property_readonly(
            "scale",
            [](const QuadSurface& surface) {
                return vec2_to_tuple(surface.scale());
            })
        .def_property_readonly(
            "grid_shape",
            [](const QuadSurface& surface) {
                const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
                if (!points) {
                    return py::make_tuple(0, 0);
                }
                return py::make_tuple(points->rows, points->cols);
            })
        .def_property_readonly(
            "bounds",
            [](QuadSurface& surface) {
                const cv::Size size = surface.size();
                return py::make_tuple(size.height, size.width);
            })
        .def_property_readonly(
            "bbox",
            [](QuadSurface& surface) {
                return rect3d_to_tuple(surface.bbox());
            })
        .def_property_readonly(
            "meta",
            [](QuadSurface& surface) -> py::object {
                if (!surface.meta) {
                    return py::none();
                }
                return to_python(*surface.meta);
            })
        .def(
            "points",
            [](const QuadSurface& surface) {
                return quad_surface_points_to_numpy(surface);
            })
        .def(
            "save",
            [](QuadSurface& surface,
               py::object path_obj,
               std::optional<std::string> uuid,
               bool force_overwrite) {
                const auto path = to_path(path_obj, "path");
                if (uuid && !uuid->empty()) {
                    surface.save(path.string(), *uuid, force_overwrite);
                } else {
                    surface.save(path, force_overwrite);
                }
            },
            py::arg("path"),
            py::arg("uuid") = py::none(),
            py::arg("force_overwrite") = false)
        .def(
            "save_overwrite",
            [](QuadSurface& surface) {
                surface.saveOverwrite();
            })
        .def(
            "gen",
            [](QuadSurface& surface,
               py::object size_obj,
               py::object ptr_obj,
               double scale,
               py::object offset_obj,
               bool with_normals) -> py::object {
                if (scale <= 0.0) {
                    throw py::value_error("scale must be positive");
                }

                cv::Size size = to_size(size_obj);
                cv::Vec3f ptr = to_vec3f(ptr_obj, "ptr");
                cv::Vec3f offset = to_vec3f(offset_obj, "offset");

                cv::Mat_<cv::Vec3f> coords;
                cv::Mat_<cv::Vec3f> normals;
                cv::Mat_<cv::Vec3f>* normals_ptr = with_normals ? &normals : nullptr;

                surface.gen(&coords, normals_ptr, size, ptr, static_cast<float>(scale), offset);

                py::array_t<float> coords_array = mat_vec3f_to_numpy(coords);
                if (!with_normals) {
                    return coords_array;
                }

                py::array_t<float> normals_array = mat_vec3f_to_numpy(normals);
                return py::make_tuple(coords_array, normals_array);
            },
            py::arg("size"),
            py::arg("ptr") = py::none(),
            py::arg("scale") = 1.0,
            py::arg("offset") = py::none(),
            py::arg("with_normals") = false);

    m.def(
        "load_quadmesh",
        [](py::object path_obj, bool ignore_mask) {
            const auto path = to_path(path_obj, "path");
            const int flags = ignore_mask ? SURF_LOAD_IGNORE_MASK : 0;
            QuadSurface* surface = load_quad_from_tifxyz(path.string(), flags);
            if (!surface) {
                std::ostringstream oss;
                oss << "failed to load quadmesh from '" << path.string() << "'";
                throw std::runtime_error(oss.str());
            }
            return std::unique_ptr<QuadSurface>(surface);
        },
        py::arg("path"),
        py::arg("ignore_mask") = false);

    m.def(
        "grow_seg_from_seed",
        [](py::object volume,
           py::object params,
           py::object origin,
           py::object resume,
           py::object corrections,
           py::object output_dir,
           py::object meta,
           py::object cache_root,
           py::object voxel_size,
           py::object normal_grid) {
            GrowSegRequest request;

            std::shared_ptr<TempZarrVolume> array_volume_holder;
            std::optional<std::filesystem::path> tuple_cache_root;
            std::shared_ptr<VCCollection> corrections_holder;
            std::optional<std::filesystem::path> normal_grid_path_override = parse_normal_grid_path(normal_grid);

            if (is_array_volume(volume)) {
                ArrayVolumeSpec spec = parse_array_volume(volume);
                if (!voxel_size.is_none()) {
                    throw py::value_error("voxel_size argument must be omitted when passing a NumPy volume tuple");
                }
                array_volume_holder = materialize_numpy_volume(spec);
                request.volume_path = array_volume_holder->root;
                request.dataset = array_volume_holder->dataset;
                request.voxel_size_override = spec.voxel_size;
                if (spec.cache_root) {
                    tuple_cache_root = *spec.cache_root;
                }
            } else {
                request.volume_path = to_path(volume, "volume");
                if (!voxel_size.is_none()) {
                    try {
                        request.voxel_size_override = voxel_size.cast<float>();
                    } catch (const py::cast_error&) {
                        throw py::value_error("voxel_size must be convertible to float");
                    }
                }
            }

            nlohmann::json params_json = params.is_none() ? nlohmann::json::object() : to_json(params);

            if (normal_grid_path_override) {
                if (params_json.contains("normal_grid_path") && params_json["normal_grid_path"].is_string()) {
                    const std::string existing_path = params_json["normal_grid_path"].get<std::string>();
                    if (existing_path != normal_grid_path_override->string()) {
                        throw py::value_error("normal_grid argument conflicts with params['normal_grid_path']");
                    }
                }
                params_json["normal_grid_path"] = normal_grid_path_override->string();
            }

            request.params = std::move(params_json);
            request.origin = to_origin(origin);
            request.output_dir = to_path(output_dir, "output_dir");
            request.meta = meta.is_none() ? nlohmann::json::object() : to_json(meta);

            if (!resume.is_none()) {
                request.resume_surface_path = to_path(resume, "resume");
            }
            if (!corrections.is_none()) {
                corrections_holder = parse_corrections(corrections);
                request.corrections = corrections_holder.get();
            }

            std::optional<std::filesystem::path> effective_cache_root = tuple_cache_root;
            if (!cache_root.is_none()) {
                effective_cache_root = to_path(cache_root, "cache_root", false);
            }
            if (effective_cache_root) {
                request.cache_root_override = *effective_cache_root;
            }

            GrowSegResult result;
            {
                py::gil_scoped_release release;
                result = run_grow_seg_from_seed(request);
            }

            if (!result.surface) {
                throw std::runtime_error("grow_seg_from_seed returned empty surface");
            }

            double area_cm2 = result.area_cm2;
            if (area_cm2 <= 0.0 && result.surface->meta && result.surface->meta->contains("area_cm2")) {
                try {
                    area_cm2 = (*result.surface->meta)["area_cm2"].get<double>();
                } catch (const std::exception&) {
                    area_cm2 = 0.0;
                }
            }

            std::unique_ptr<QuadSurface> surface_ptr = std::move(result.surface);
            py::object surface_meta = py::none();
            if (surface_ptr->meta) {
                surface_meta = to_python(*surface_ptr->meta);
            }
            const std::string surface_path = surface_ptr->path.string();

            py::dict response;
            response["area_cm2"] = area_cm2;
            response["voxel_size"] = result.voxel_size;
            response["output_dir"] = request.output_dir.string();
            response["surface_path"] = surface_path;
            response["surface_meta"] = surface_meta;
            response["surface"] = py::cast(std::move(surface_ptr));

            return response;
        },
        py::arg("volume"),
        py::arg("params"),
        py::arg("origin") = py::none(),
        py::arg("resume") = py::none(),
        py::arg("corrections") = py::none(),
        py::arg("output_dir") = py::none(),
        py::arg("meta") = py::none(),
        py::arg("cache_root") = py::none(),
        py::arg("voxel_size") = py::none(),
        py::arg("normal_grid") = py::none());
}
