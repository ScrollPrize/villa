#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <Python.h>

#include <Eigen/Core>
#include <igl/AABB.h>
#include <igl/WindingNumberAABB.h>
#include <igl/signed_distance.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXiR = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowVector3d = Eigen::Matrix<double, 1, 3>;

MatrixXdR tensor_to_vertices(torch::Tensor vertices) {
	TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "vertices must be (N, 3)");
	vertices = vertices.to(torch::kFloat64).cpu().contiguous();
	MatrixXdR V(vertices.size(0), 3);
	const double* src = vertices.data_ptr<double>();
	std::copy(src, src + vertices.numel(), V.data());
	return V;
}

MatrixXiR tensor_to_faces(torch::Tensor faces) {
	TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces must be (M, 3)");
	faces = faces.to(torch::kInt64).cpu().contiguous();
	MatrixXiR F(faces.size(0), 3);
	const int64_t* src = faces.data_ptr<int64_t>();
	for (int64_t i = 0; i < faces.numel(); ++i) {
		F.data()[i] = static_cast<int>(src[i]);
	}
	return F;
}

bool is_inside_at(
	const igl::WindingNumberAABB<double, int>& hier,
	const RowVector3d& q
) {
	const double winding = hier.winding_number(q.transpose());
	return winding > 0.5;
}

double surface_distance_at(
	const igl::AABB<MatrixXdR, 3>& tree,
	const MatrixXdR& V,
	const MatrixXiR& F,
	const RowVector3d& q
) {
	double sqrd = 0.0;
	int face_i = -1;
	RowVector3d closest;
	sqrd = tree.squared_distance(V, F, q, face_i, closest);
	return std::sqrt(std::max(0.0, sqrd));
}

uint8_t encode_depth(double depth, double depth_max) {
	if (!(depth > 0.0) || !(depth_max > 0.0) || !std::isfinite(depth_max)) {
		return static_cast<uint8_t>(0);
	}
	const double normalized = std::min(1.0, std::max(0.0, depth / depth_max));
	const long q = std::lround(255.0 * std::sqrt(normalized));
	return static_cast<uint8_t>(std::min<long>(255, std::max<long>(0, q)));
}

int default_thread_count(int requested_threads) {
	if (requested_threads > 0) {
		return requested_threads;
	}
	const int torch_threads = at::get_num_threads();
	if (torch_threads > 1) {
		return torch_threads;
	}
	const unsigned int hw = std::thread::hardware_concurrency();
	return static_cast<int>(std::max(1u, hw));
}

void print_progress_line(
	const std::string& label,
	const std::string& pass,
	int64_t done,
	int64_t total,
	const std::chrono::steady_clock::time_point& start,
	bool final,
	bool final_complete,
	const std::string& unit
) {
	const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	const double rate = elapsed > 0.0 ? static_cast<double>(done) / elapsed : 0.0;
	std::cout
		<< "[cyl_outside] " << label << " " << pass << ": ";
	if (total > 0) {
		const double pct = 100.0 * static_cast<double>(done) / static_cast<double>(total);
		const int bar_width = 28;
		const int filled = std::min(
			bar_width,
			std::max(0, static_cast<int>(std::round((pct / 100.0) * static_cast<double>(bar_width))))
		);
		std::cout << "[";
		for (int i = 0; i < bar_width; ++i) {
			std::cout << (i < filled ? "#" : "-");
		}
		std::cout
			<< "] "
			<< std::fixed << std::setprecision(1) << pct << "% "
			<< done << "/" << total << " " << unit;
	} else {
		std::cout
			<< done << " " << unit;
	}
	std::cout
		<< " elapsed=" << std::setprecision(1) << elapsed << "s"
		<< " rate=" << std::setprecision(0) << rate << " " << unit << "/s"
		<< (final ? (final_complete ? " done" : " stopped") : "")
		<< std::endl;
}

class ProgressPrinter {
public:
	ProgressPrinter(
		const std::string& label,
		const std::string& pass,
		int64_t total,
		const std::atomic<int64_t>& done,
		bool enabled,
		const std::string& unit = "vox",
		bool partial_ok = false
	):
		label_(label),
		pass_(pass),
		total_(total),
		done_(done),
		enabled_(enabled),
		unit_(unit),
		partial_ok_(partial_ok),
		stop_(false),
		start_(std::chrono::steady_clock::now())
	{
		if (enabled_) {
			print_progress_line(label_, pass_, 0, total_, start_, false, false, unit_);
			thread_ = std::thread([this]() {
				auto last = std::chrono::steady_clock::now();
				while (!stop_.load()) {
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
					if (stop_.load()) {
						break;
					}
					const auto now = std::chrono::steady_clock::now();
					if (std::chrono::duration<double>(now - last).count() < 2.0) {
						continue;
					}
					last = now;
					print_progress_line(label_, pass_, done_.load(), total_, start_, false, false, unit_);
				}
			});
		}
	}

	~ProgressPrinter() {
		if (!enabled_) {
			return;
		}
		stop_.store(true);
		if (thread_.joinable()) {
			thread_.join();
		}
		const int64_t done = done_.load();
		print_progress_line(label_, pass_, done, total_, start_, true, partial_ok_ || done >= total_, unit_);
	}

private:
	std::string label_;
	std::string pass_;
	int64_t total_;
	const std::atomic<int64_t>& done_;
	bool enabled_;
	std::string unit_;
	bool partial_ok_;
	std::atomic<bool> stop_;
	std::chrono::steady_clock::time_point start_;
	std::thread thread_;
};

template <typename Fn>
bool parallel_chunks(int64_t total, int n_threads, int64_t grain, std::atomic<bool>& cancel, Fn fn) {
	const int64_t chunk_count = std::max<int64_t>(1, (total + grain - 1) / grain);
	n_threads = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(n_threads, chunk_count)));
	std::atomic<int64_t> next(0);
	std::atomic<int> active(n_threads);
	std::vector<std::thread> workers;
	workers.reserve(static_cast<size_t>(n_threads));
	for (int ti = 0; ti < n_threads; ++ti) {
		workers.emplace_back([&, ti]() {
			while (true) {
				if (cancel.load(std::memory_order_relaxed)) {
					break;
				}
				const int64_t begin = next.fetch_add(grain);
				if (begin >= total) {
					break;
				}
				const int64_t end = std::min(total, begin + grain);
				fn(begin, end, ti);
			}
			active.fetch_sub(1);
		});
	}
	bool interrupted = false;
	while (active.load() > 0) {
		if (!interrupted && PyErr_CheckSignals() != 0) {
			interrupted = true;
			cancel.store(true);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	for (std::thread& worker : workers) {
		worker.join();
	}
	return interrupted;
}

void throw_if_cancelled(
	std::atomic<bool>& cancel,
	const std::string& label,
	const std::string& phase,
	bool progress_enabled
) {
	if (!cancel.load(std::memory_order_relaxed) && PyErr_CheckSignals() == 0) {
		return;
	}
	cancel.store(true);
	if (progress_enabled) {
		std::cout
			<< "[cyl_outside] " << label
			<< ": interrupted during " << phase << "; cancelling"
			<< std::endl;
	}
	if (!PyErr_Occurred()) {
		PyErr_SetNone(PyExc_KeyboardInterrupt);
	}
	throw pybind11::error_already_set();
}

struct DepthScanStats {
	double depth_max = 0.0;
	int64_t inside_count = 0;
	int64_t voxels_classified = 0;
	int64_t voxels_skipped_by_chunk_mask = 0;
	int64_t certified_inside_chunks = 0;
	int64_t winding_chunks = 0;
	int64_t exact_distance_chunks = 0;
	int64_t blended_chunks = 0;
	int64_t coarse_chunks = 0;
	int64_t exact_distance_voxels = 0;
	int64_t coarse_voxels = 0;
	int64_t total_chunks = 0;
	int64_t center_inside_chunks = 0;
	int64_t surface_seed_chunks = 0;
	int64_t initial_candidate_chunks = 0;
	int64_t growth_iterations = 0;
	int64_t processed_chunks = 0;
	int64_t chunks_with_inside = 0;
	double center_prepass_elapsed = 0.0;
	double growth_elapsed = 0.0;
	double pass_cpu_seconds = 0.0;
	double distance_cpu_seconds = 0.0;
	double chunk_probe_distance_seconds = 0.0;
	bool used_chunked = false;
	bool used_fallback = false;
};

struct ChunkBounds {
	int64_t x0 = 0;
	int64_t x1 = 0;
	int64_t y0 = 0;
	int64_t y1 = 0;
	int64_t z0 = 0;
	int64_t z1 = 0;
};

int64_t flat_voxel_index(int64_t x, int64_t y, int64_t z, int64_t X, int64_t Y) {
	return (z * Y + y) * X + x;
}

int64_t flat_chunk_index(int64_t cx, int64_t cy, int64_t cz, int64_t CX, int64_t CY) {
	return (cz * CY + cy) * CX + cx;
}

ChunkBounds chunk_bounds_for_coords(
	int64_t cx,
	int64_t cy,
	int64_t cz,
	int64_t X,
	int64_t Y,
	int64_t Z,
	int64_t cs
) {
	ChunkBounds b;
	b.x0 = cx * cs;
	b.x1 = std::min<int64_t>(X, b.x0 + cs);
	b.y0 = cy * cs;
	b.y1 = std::min<int64_t>(Y, b.y0 + cs);
	b.z0 = cz * cs;
	b.z1 = std::min<int64_t>(Z, b.z0 + cs);
	return b;
}

RowVector3d grid_point(
	const double* origin,
	const double* spacing,
	double x,
	double y,
	double z
) {
	return RowVector3d(
		origin[0] + x * spacing[0],
		origin[1] + y * spacing[1],
		origin[2] + z * spacing[2]
	);
}

RowVector3d voxel_point(
	const double* origin,
	const double* spacing,
	int64_t x,
	int64_t y,
	int64_t z
) {
	return grid_point(
		origin,
		spacing,
		static_cast<double>(x),
		static_cast<double>(y),
		static_cast<double>(z)
	);
}

RowVector3d chunk_domain_center(
	const ChunkBounds& b,
	const double* origin,
	const double* spacing
) {
	return grid_point(
		origin,
		spacing,
		0.5 * static_cast<double>(b.x0 + b.x1),
		0.5 * static_cast<double>(b.y0 + b.y1),
		0.5 * static_cast<double>(b.z0 + b.z1)
	);
}

double chunk_domain_half_diagonal(const ChunkBounds& b, const double* spacing) {
	const double hx = 0.5 * static_cast<double>(b.x1 - b.x0) * spacing[0];
	const double hy = 0.5 * static_cast<double>(b.y1 - b.y0) * spacing[1];
	const double hz = 0.5 * static_cast<double>(b.z1 - b.z0) * spacing[2];
	return std::sqrt(hx * hx + hy * hy + hz * hz);
}

int64_t chunk_voxel_count(const ChunkBounds& b) {
	return (b.x1 - b.x0) * (b.y1 - b.y0) * (b.z1 - b.z0);
}

double smoothstep(double edge0, double edge1, double x) {
	if (!(edge1 > edge0)) {
		return x >= edge1 ? 1.0 : 0.0;
	}
	const double t = std::min(1.0, std::max(0.0, (x - edge0) / (edge1 - edge0)));
	return t * t * (3.0 - 2.0 * t);
}

double trilinear_value(const std::array<double, 8>& values, double tx, double ty, double tz) {
	tx = std::min(1.0, std::max(0.0, tx));
	ty = std::min(1.0, std::max(0.0, ty));
	tz = std::min(1.0, std::max(0.0, tz));
	const double c00 = values[0] * (1.0 - tx) + values[1] * tx;
	const double c10 = values[2] * (1.0 - tx) + values[3] * tx;
	const double c01 = values[4] * (1.0 - tx) + values[5] * tx;
	const double c11 = values[6] * (1.0 - tx) + values[7] * tx;
	const double c0 = c00 * (1.0 - ty) + c10 * ty;
	const double c1 = c01 * (1.0 - ty) + c11 * ty;
	return c0 * (1.0 - tz) + c1 * tz;
}

double bubble_weight(double tx, double ty, double tz) {
	auto axis = [](double t) {
		t = std::min(1.0, std::max(0.0, t));
		return 4.0 * t * (1.0 - t);
	};
	return axis(tx) * axis(ty) * axis(tz);
}

double positive_finite_depth(double depth) {
	if (!(depth > 0.0) || !std::isfinite(depth)) {
		return 0.0;
	}
	return depth;
}

int64_t count_marked(const std::vector<uint8_t>& values) {
	int64_t count = 0;
	for (uint8_t value : values) {
		if (value != 0) {
			++count;
		}
	}
	return count;
}

void dilate_marked_chunks(
	const std::vector<uint8_t>& src,
	std::vector<uint8_t>& dst,
	int64_t CZ,
	int64_t CY,
	int64_t CX
) {
	for (int64_t cz = 0; cz < CZ; ++cz) {
		for (int64_t cy = 0; cy < CY; ++cy) {
			for (int64_t cx = 0; cx < CX; ++cx) {
				const int64_t ci = flat_chunk_index(cx, cy, cz, CX, CY);
				if (src[static_cast<size_t>(ci)] == 0) {
					continue;
				}
				for (int dz = -1; dz <= 1; ++dz) {
					const int64_t nz = cz + dz;
					if (nz < 0 || nz >= CZ) {
						continue;
					}
					for (int dy = -1; dy <= 1; ++dy) {
						const int64_t ny = cy + dy;
						if (ny < 0 || ny >= CY) {
							continue;
						}
						for (int dx = -1; dx <= 1; ++dx) {
							const int64_t nx = cx + dx;
							if (nx < 0 || nx >= CX) {
								continue;
							}
							dst[static_cast<size_t>(flat_chunk_index(nx, ny, nz, CX, CY))] = 1;
						}
					}
				}
			}
		}
	}
}

void dilate_one_chunk_into(
	int64_t ci,
	std::vector<uint8_t>& dst,
	int64_t CZ,
	int64_t CY,
	int64_t CX
) {
	const int64_t cx = ci % CX;
	const int64_t yz = ci / CX;
	const int64_t cy = yz % CY;
	const int64_t cz = yz / CY;
	for (int dz = -1; dz <= 1; ++dz) {
		const int64_t nz = cz + dz;
		if (nz < 0 || nz >= CZ) {
			continue;
		}
		for (int dy = -1; dy <= 1; ++dy) {
			const int64_t ny = cy + dy;
			if (ny < 0 || ny >= CY) {
				continue;
			}
			for (int dx = -1; dx <= 1; ++dx) {
				const int64_t nx = cx + dx;
				if (nx < 0 || nx >= CX) {
					continue;
				}
				dst[static_cast<size_t>(flat_chunk_index(nx, ny, nz, CX, CY))] = 1;
			}
		}
	}
}

bool voxel_range_for_coord_range(
	double lo,
	double hi,
	double origin,
	double spacing,
	int64_t dim,
	int64_t& out0,
	int64_t& out1
) {
	const double v0 = (lo - origin) / spacing;
	const double v1 = (hi - origin) / spacing;
	int64_t i0 = static_cast<int64_t>(std::floor(v0));
	int64_t i1 = static_cast<int64_t>(std::ceil(v1));
	if (i1 < 0 || i0 >= dim) {
		return false;
	}
	i0 = std::max<int64_t>(0, std::min<int64_t>(dim - 1, i0));
	i1 = std::max<int64_t>(0, std::min<int64_t>(dim - 1, i1));
	if (i1 < i0) {
		return false;
	}
	out0 = i0;
	out1 = i1;
	return true;
}

DepthScanStats scan_full_volume(
	const igl::WindingNumberAABB<double, int>& hier,
	const igl::AABB<MatrixXdR, 3>& tree,
	const MatrixXdR& V,
	const MatrixXiR& F,
	const double* origin,
	const double* spacing,
	int64_t Z,
	int64_t Y,
	int64_t X,
	int n_threads,
	float* depth_ptr,
	std::atomic<bool>& cancel,
	const std::string& progress_label,
	const std::string& pass_label,
	bool progress_enabled
) {
	const int64_t total = Z * Y * X;
	std::vector<double> thread_max(static_cast<size_t>(n_threads), 0.0);
	std::vector<int64_t> thread_inside(static_cast<size_t>(n_threads), 0);
	std::vector<int64_t> thread_classified(static_cast<size_t>(n_threads), 0);
	std::vector<double> thread_pass_seconds(static_cast<size_t>(n_threads), 0.0);
	std::vector<double> thread_distance_seconds(static_cast<size_t>(n_threads), 0.0);
	std::atomic<int64_t> max_done(0);
	const auto pass_start = std::chrono::steady_clock::now();
	{
		ProgressPrinter progress(progress_label, pass_label, total, max_done, progress_enabled);
		parallel_chunks(total, n_threads, 2048, cancel, [&](int64_t begin, int64_t end, int ti) {
			const auto chunk_start = std::chrono::steady_clock::now();
			double local_max = 0.0;
			int64_t local_inside = 0;
			int64_t processed = 0;
			double local_distance_seconds = 0.0;
			for (int64_t n = begin; n < end; ++n) {
				if ((processed & 63) == 0 && cancel.load(std::memory_order_relaxed)) {
					break;
				}
				const int64_t x = n % X;
				const int64_t yz = n / X;
				const int64_t y = yz % Y;
				const int64_t z = yz / Y;
				const RowVector3d q(
					origin[0] + static_cast<double>(x) * spacing[0],
					origin[1] + static_cast<double>(y) * spacing[1],
					origin[2] + static_cast<double>(z) * spacing[2]
				);
				double depth = 0.0;
				if (is_inside_at(hier, q)) {
					const auto distance_start = std::chrono::steady_clock::now();
					depth = surface_distance_at(tree, V, F, q);
					local_distance_seconds += std::chrono::duration<double>(
						std::chrono::steady_clock::now() - distance_start
					).count();
				}
				const float depth_f = static_cast<float>(depth);
				depth_ptr[n] = depth_f;
				local_max = std::max(local_max, static_cast<double>(depth_f));
				if (depth_f > 0.0f) {
					++local_inside;
				}
				++processed;
			}
			thread_max[static_cast<size_t>(ti)] = std::max(thread_max[static_cast<size_t>(ti)], local_max);
			thread_inside[static_cast<size_t>(ti)] += local_inside;
			thread_classified[static_cast<size_t>(ti)] += processed;
			thread_distance_seconds[static_cast<size_t>(ti)] += local_distance_seconds;
			thread_pass_seconds[static_cast<size_t>(ti)] += std::chrono::duration<double>(
				std::chrono::steady_clock::now() - chunk_start
			).count();
			max_done.fetch_add(processed);
		});
	}
	DepthScanStats stats;
	stats.growth_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - pass_start).count();
	for (const double value : thread_max) {
		stats.depth_max = std::max(stats.depth_max, value);
	}
	for (const int64_t value : thread_inside) {
		stats.inside_count += value;
	}
	for (const int64_t value : thread_classified) {
		stats.voxels_classified += value;
	}
	for (const double value : thread_pass_seconds) {
		stats.pass_cpu_seconds += value;
	}
	for (const double value : thread_distance_seconds) {
		stats.distance_cpu_seconds += value;
	}
	return stats;
}

DepthScanStats scan_chunked_volume(
	const igl::WindingNumberAABB<double, int>& hier,
	const igl::AABB<MatrixXdR, 3>& tree,
	const MatrixXdR& V,
	const MatrixXiR& F,
	const double* origin,
	const double* spacing,
	int64_t Z,
	int64_t Y,
	int64_t X,
	int n_threads,
	int chunk_size,
	double deep_interp_chunks,
	double deep_blend_chunks,
	float* depth_ptr,
	std::atomic<bool>& cancel,
	const std::string& progress_label,
	bool progress_enabled
) {
	const int64_t total = Z * Y * X;
	const int64_t cs = static_cast<int64_t>(std::max(1, chunk_size));
	const int64_t CX = (X + cs - 1) / cs;
	const int64_t CY = (Y + cs - 1) / cs;
	const int64_t CZ = (Z + cs - 1) / cs;
	const int64_t total_chunks = CX * CY * CZ;
	const double max_spacing = std::max({spacing[0], spacing[1], spacing[2]});
	const double deep_threshold = std::max(0.0, deep_interp_chunks) * static_cast<double>(cs) * max_spacing;
	const double deep_blend_width = std::max(0.0, deep_blend_chunks) * static_cast<double>(cs) * max_spacing;
	const bool coarse_enabled = deep_interp_chunks > 0.0;
	DepthScanStats stats;
	stats.used_chunked = true;
	stats.total_chunks = total_chunks;
	stats.voxels_skipped_by_chunk_mask = total;

	std::fill(depth_ptr, depth_ptr + total, 0.0f);

	std::vector<uint8_t> center_inside(static_cast<size_t>(total_chunks), 0);
	std::vector<int64_t> thread_center_inside(static_cast<size_t>(n_threads), 0);
	std::atomic<int64_t> center_done(0);
	const auto center_start = std::chrono::steady_clock::now();
	{
		ProgressPrinter progress(
			progress_label,
			"chunk center winding prepass",
			total_chunks,
			center_done,
			progress_enabled,
			"chunks"
		);
		parallel_chunks(total_chunks, n_threads, 64, cancel, [&](int64_t begin, int64_t end, int ti) {
			int64_t local_inside = 0;
			int64_t processed = 0;
			for (int64_t ci = begin; ci < end; ++ci) {
				if ((processed & 15) == 0 && cancel.load(std::memory_order_relaxed)) {
					break;
				}
				const int64_t cx = ci % CX;
				const int64_t yz = ci / CX;
				const int64_t cy = yz % CY;
				const int64_t cz = yz / CY;
				const ChunkBounds b = chunk_bounds_for_coords(cx, cy, cz, X, Y, Z, cs);
				const RowVector3d q = chunk_domain_center(b, origin, spacing);
				if (is_inside_at(hier, q)) {
					center_inside[static_cast<size_t>(ci)] = 1;
					++local_inside;
				}
				++processed;
			}
			thread_center_inside[static_cast<size_t>(ti)] += local_inside;
			center_done.fetch_add(processed);
		});
	}
	stats.center_prepass_elapsed = std::chrono::duration<double>(
		std::chrono::steady_clock::now() - center_start
	).count();
	for (const int64_t value : thread_center_inside) {
		stats.center_inside_chunks += value;
	}

	throw_if_cancelled(cancel, progress_label, "chunk center prepass", progress_enabled);

	std::vector<uint8_t> surface_seed(static_cast<size_t>(total_chunks), 0);
	for (int64_t fi = 0; fi < F.rows(); ++fi) {
		const int i0 = F(fi, 0);
		const int i1 = F(fi, 1);
		const int i2 = F(fi, 2);
		const double xmin = std::min({V(i0, 0), V(i1, 0), V(i2, 0)});
		const double xmax = std::max({V(i0, 0), V(i1, 0), V(i2, 0)});
		const double ymin = std::min({V(i0, 1), V(i1, 1), V(i2, 1)});
		const double ymax = std::max({V(i0, 1), V(i1, 1), V(i2, 1)});
		const double zmin = std::min({V(i0, 2), V(i1, 2), V(i2, 2)});
		const double zmax = std::max({V(i0, 2), V(i1, 2), V(i2, 2)});
		int64_t vx0 = 0, vx1 = 0, vy0 = 0, vy1 = 0, vz0 = 0, vz1 = 0;
		if (!voxel_range_for_coord_range(xmin, xmax, origin[0], spacing[0], X, vx0, vx1) ||
			!voxel_range_for_coord_range(ymin, ymax, origin[1], spacing[1], Y, vy0, vy1) ||
			!voxel_range_for_coord_range(zmin, zmax, origin[2], spacing[2], Z, vz0, vz1)) {
			continue;
		}
		for (int64_t cz = vz0 / cs; cz <= vz1 / cs; ++cz) {
			for (int64_t cy = vy0 / cs; cy <= vy1 / cs; ++cy) {
				for (int64_t cx = vx0 / cs; cx <= vx1 / cs; ++cx) {
					surface_seed[static_cast<size_t>(flat_chunk_index(cx, cy, cz, CX, CY))] = 1;
				}
			}
		}
		if ((fi & 255) == 0) {
			throw_if_cancelled(cancel, progress_label, "surface chunk seeding", progress_enabled);
		}
	}
	stats.surface_seed_chunks = count_marked(surface_seed);

	std::vector<uint8_t> candidate(static_cast<size_t>(total_chunks), 0);
	dilate_marked_chunks(center_inside, candidate, CZ, CY, CX);
	dilate_marked_chunks(surface_seed, candidate, CZ, CY, CX);
	stats.initial_candidate_chunks = count_marked(candidate);

	std::vector<uint8_t> processed(static_cast<size_t>(total_chunks), 0);
	std::vector<double> thread_max(static_cast<size_t>(n_threads), 0.0);
	std::vector<int64_t> thread_inside(static_cast<size_t>(n_threads), 0);
	std::vector<int64_t> thread_classified(static_cast<size_t>(n_threads), 0);
	std::vector<int64_t> thread_processed_chunks(static_cast<size_t>(n_threads), 0);
	std::vector<int64_t> thread_chunks_with_inside(static_cast<size_t>(n_threads), 0);
	std::vector<int64_t> thread_certified_inside_chunks(static_cast<size_t>(n_threads), 0);
	std::vector<int64_t> thread_winding_chunks(static_cast<size_t>(n_threads), 0);
	std::vector<int64_t> thread_exact_distance_chunks(static_cast<size_t>(n_threads), 0);
	std::vector<int64_t> thread_blended_chunks(static_cast<size_t>(n_threads), 0);
	std::vector<int64_t> thread_coarse_chunks(static_cast<size_t>(n_threads), 0);
	std::vector<int64_t> thread_exact_distance_voxels(static_cast<size_t>(n_threads), 0);
	std::vector<int64_t> thread_coarse_voxels(static_cast<size_t>(n_threads), 0);
	std::vector<double> thread_pass_seconds(static_cast<size_t>(n_threads), 0.0);
	std::vector<double> thread_distance_seconds(static_cast<size_t>(n_threads), 0.0);
	std::vector<double> thread_chunk_probe_distance_seconds(static_cast<size_t>(n_threads), 0.0);
	std::vector<std::vector<int64_t>> thread_inside_chunks(static_cast<size_t>(n_threads));
	std::atomic<int64_t> growth_done(0);
	const auto growth_start = std::chrono::steady_clock::now();
	{
		ProgressPrinter progress(
			progress_label,
			"candidate/growth winding+inside-distance",
			0,
			growth_done,
			progress_enabled,
			"vox classified",
			true
		);
		while (true) {
			throw_if_cancelled(cancel, progress_label, "candidate/growth scheduling", progress_enabled);
			std::vector<int64_t> scan_list;
			scan_list.reserve(static_cast<size_t>(std::min<int64_t>(total_chunks, 4096)));
			for (int64_t ci = 0; ci < total_chunks; ++ci) {
				if (candidate[static_cast<size_t>(ci)] != 0 && processed[static_cast<size_t>(ci)] == 0) {
					processed[static_cast<size_t>(ci)] = 1;
					scan_list.push_back(ci);
				}
			}
			if (scan_list.empty()) {
				break;
			}
			++stats.growth_iterations;
			for (std::vector<int64_t>& chunks : thread_inside_chunks) {
				chunks.clear();
			}
			parallel_chunks(static_cast<int64_t>(scan_list.size()), n_threads, 1, cancel, [&](int64_t begin, int64_t end, int ti) {
				const auto chunk_start = std::chrono::steady_clock::now();
				double local_max = 0.0;
				int64_t local_inside = 0;
				int64_t local_classified = 0;
				int64_t local_processed_chunks = 0;
				int64_t local_chunks_with_inside = 0;
				int64_t local_certified_inside_chunks = 0;
				int64_t local_winding_chunks = 0;
				int64_t local_exact_distance_chunks = 0;
				int64_t local_blended_chunks = 0;
				int64_t local_coarse_chunks = 0;
				int64_t local_exact_distance_voxels = 0;
				int64_t local_coarse_voxels = 0;
				double local_distance_seconds = 0.0;
				double local_probe_distance_seconds = 0.0;
				std::vector<int64_t>& local_inside_chunks = thread_inside_chunks[static_cast<size_t>(ti)];
				auto query_distance = [&](const RowVector3d& q, bool probe) {
					const auto distance_start = std::chrono::steady_clock::now();
					const double depth = surface_distance_at(tree, V, F, q);
					const double elapsed = std::chrono::duration<double>(
						std::chrono::steady_clock::now() - distance_start
					).count();
					if (probe) {
						local_probe_distance_seconds += elapsed;
					} else {
						local_distance_seconds += elapsed;
					}
					return depth;
				};
				auto write_depth = [&](int64_t n, double depth, int64_t& chunk_inside) {
					const float depth_f = static_cast<float>(positive_finite_depth(depth));
					depth_ptr[n] = depth_f;
					local_max = std::max(local_max, static_cast<double>(depth_f));
					if (depth_f > 0.0f) {
						++local_inside;
						++chunk_inside;
					}
				};
				for (int64_t si = begin; si < end; ++si) {
					if (cancel.load(std::memory_order_relaxed)) {
						break;
					}
					const int64_t ci = scan_list[static_cast<size_t>(si)];
					const int64_t cx = ci % CX;
					const int64_t yz = ci / CX;
					const int64_t cy = yz % CY;
					const int64_t cz = yz / CY;
					const ChunkBounds b = chunk_bounds_for_coords(cx, cy, cz, X, Y, Z, cs);
					int64_t chunk_inside = 0;
					int64_t chunk_processed = 0;
					bool used_certified = false;
					bool used_blended = false;
					bool used_coarse_only = false;
					std::array<double, 8> corner_depths{};
					double center_correction = 0.0;
					const int64_t voxel_count = chunk_voxel_count(b);

					if (center_inside[static_cast<size_t>(ci)] != 0) {
						const RowVector3d center_q = chunk_domain_center(b, origin, spacing);
						const double center_distance = positive_finite_depth(query_distance(center_q, true));
						const double half_diag = chunk_domain_half_diagonal(b, spacing);
						if (center_distance > half_diag) {
							used_certified = true;
							++local_certified_inside_chunks;
							const double chunk_distance_lower = center_distance - half_diag;
							const double chunk_distance_upper = center_distance + half_diag;
							used_coarse_only = coarse_enabled &&
								chunk_distance_lower >= deep_threshold + deep_blend_width;
							used_blended = coarse_enabled && !used_coarse_only &&
								chunk_distance_upper > deep_threshold;
							if (used_coarse_only || used_blended) {
								for (int zbit = 0; zbit <= 1; ++zbit) {
									for (int ybit = 0; ybit <= 1; ++ybit) {
										for (int xbit = 0; xbit <= 1; ++xbit) {
											const int idx = xbit + 2 * ybit + 4 * zbit;
											const RowVector3d q = grid_point(
												origin,
												spacing,
												static_cast<double>(xbit ? b.x1 : b.x0),
												static_cast<double>(ybit ? b.y1 : b.y0),
												static_cast<double>(zbit ? b.z1 : b.z0)
											);
											corner_depths[static_cast<size_t>(idx)] =
												positive_finite_depth(query_distance(q, true));
										}
									}
								}
								center_correction = center_distance -
									trilinear_value(corner_depths, 0.5, 0.5, 0.5);
							}
						}
					}

					auto coarse_depth_at = [&](int64_t x, int64_t y, int64_t z) {
						const double tx = static_cast<double>(x - b.x0) / static_cast<double>(b.x1 - b.x0);
						const double ty = static_cast<double>(y - b.y0) / static_cast<double>(b.y1 - b.y0);
						const double tz = static_cast<double>(z - b.z0) / static_cast<double>(b.z1 - b.z0);
						const double depth = trilinear_value(corner_depths, tx, ty, tz) +
							bubble_weight(tx, ty, tz) * center_correction;
						return positive_finite_depth(depth);
					};

					if (used_certified && used_coarse_only) {
						++local_coarse_chunks;
						local_coarse_voxels += voxel_count;
						for (int64_t z = b.z0; z < b.z1; ++z) {
							for (int64_t y = b.y0; y < b.y1; ++y) {
								for (int64_t x = b.x0; x < b.x1; ++x) {
									if ((chunk_processed & 63) == 0 && cancel.load(std::memory_order_relaxed)) {
										break;
									}
									const int64_t n = flat_voxel_index(x, y, z, X, Y);
									write_depth(n, coarse_depth_at(x, y, z), chunk_inside);
									++chunk_processed;
								}
								if (cancel.load(std::memory_order_relaxed)) {
									break;
								}
							}
							if (cancel.load(std::memory_order_relaxed)) {
								break;
							}
						}
					} else if (used_certified && used_blended) {
						++local_blended_chunks;
						local_coarse_voxels += voxel_count;
						for (int64_t z = b.z0; z < b.z1; ++z) {
							for (int64_t y = b.y0; y < b.y1; ++y) {
								for (int64_t x = b.x0; x < b.x1; ++x) {
									if ((chunk_processed & 63) == 0 && cancel.load(std::memory_order_relaxed)) {
										break;
									}
									const int64_t n = flat_voxel_index(x, y, z, X, Y);
									const double exact_depth = positive_finite_depth(query_distance(
										voxel_point(origin, spacing, x, y, z),
										false
									));
									++local_exact_distance_voxels;
									const double coarse_depth = coarse_depth_at(x, y, z);
									const double w = smoothstep(
										deep_threshold,
										deep_threshold + deep_blend_width,
										exact_depth
									);
									write_depth(n, exact_depth * (1.0 - w) + coarse_depth * w, chunk_inside);
									++chunk_processed;
								}
								if (cancel.load(std::memory_order_relaxed)) {
									break;
								}
							}
							if (cancel.load(std::memory_order_relaxed)) {
								break;
							}
						}
					} else if (used_certified) {
						++local_exact_distance_chunks;
						for (int64_t z = b.z0; z < b.z1; ++z) {
							for (int64_t y = b.y0; y < b.y1; ++y) {
								for (int64_t x = b.x0; x < b.x1; ++x) {
									if ((chunk_processed & 63) == 0 && cancel.load(std::memory_order_relaxed)) {
										break;
									}
									const int64_t n = flat_voxel_index(x, y, z, X, Y);
									write_depth(n, query_distance(voxel_point(origin, spacing, x, y, z), false), chunk_inside);
									++local_exact_distance_voxels;
									++chunk_processed;
								}
								if (cancel.load(std::memory_order_relaxed)) {
									break;
								}
							}
							if (cancel.load(std::memory_order_relaxed)) {
								break;
							}
						}
					} else {
						++local_winding_chunks;
						for (int64_t z = b.z0; z < b.z1; ++z) {
							for (int64_t y = b.y0; y < b.y1; ++y) {
								for (int64_t x = b.x0; x < b.x1; ++x) {
									if ((chunk_processed & 63) == 0 && cancel.load(std::memory_order_relaxed)) {
										break;
									}
									const int64_t n = flat_voxel_index(x, y, z, X, Y);
									double depth = 0.0;
									const RowVector3d q = voxel_point(origin, spacing, x, y, z);
									if (is_inside_at(hier, q)) {
										depth = query_distance(q, false);
										++local_exact_distance_voxels;
									}
									write_depth(n, depth, chunk_inside);
									++chunk_processed;
								}
								if (cancel.load(std::memory_order_relaxed)) {
									break;
								}
							}
							if (cancel.load(std::memory_order_relaxed)) {
								break;
							}
						}
					}
					local_classified += chunk_processed;
					growth_done.fetch_add(chunk_processed);
					++local_processed_chunks;
					if (chunk_inside > 0) {
						++local_chunks_with_inside;
						local_inside_chunks.push_back(ci);
					}
				}
				thread_max[static_cast<size_t>(ti)] = std::max(thread_max[static_cast<size_t>(ti)], local_max);
				thread_inside[static_cast<size_t>(ti)] += local_inside;
				thread_classified[static_cast<size_t>(ti)] += local_classified;
				thread_processed_chunks[static_cast<size_t>(ti)] += local_processed_chunks;
				thread_chunks_with_inside[static_cast<size_t>(ti)] += local_chunks_with_inside;
				thread_certified_inside_chunks[static_cast<size_t>(ti)] += local_certified_inside_chunks;
				thread_winding_chunks[static_cast<size_t>(ti)] += local_winding_chunks;
				thread_exact_distance_chunks[static_cast<size_t>(ti)] += local_exact_distance_chunks;
				thread_blended_chunks[static_cast<size_t>(ti)] += local_blended_chunks;
				thread_coarse_chunks[static_cast<size_t>(ti)] += local_coarse_chunks;
				thread_exact_distance_voxels[static_cast<size_t>(ti)] += local_exact_distance_voxels;
				thread_coarse_voxels[static_cast<size_t>(ti)] += local_coarse_voxels;
				thread_distance_seconds[static_cast<size_t>(ti)] += local_distance_seconds;
				thread_chunk_probe_distance_seconds[static_cast<size_t>(ti)] += local_probe_distance_seconds;
				thread_pass_seconds[static_cast<size_t>(ti)] += std::chrono::duration<double>(
					std::chrono::steady_clock::now() - chunk_start
				).count();
			});
			throw_if_cancelled(cancel, progress_label, "candidate/growth chunk processing", progress_enabled);
			for (const std::vector<int64_t>& chunks : thread_inside_chunks) {
				for (int64_t ci : chunks) {
					dilate_one_chunk_into(ci, candidate, CZ, CY, CX);
				}
			}
		}
	}
	stats.growth_elapsed = std::chrono::duration<double>(
		std::chrono::steady_clock::now() - growth_start
	).count();
	for (const double value : thread_max) {
		stats.depth_max = std::max(stats.depth_max, value);
	}
	for (const int64_t value : thread_inside) {
		stats.inside_count += value;
	}
	for (const int64_t value : thread_classified) {
		stats.voxels_classified += value;
	}
	for (const int64_t value : thread_processed_chunks) {
		stats.processed_chunks += value;
	}
	for (const int64_t value : thread_chunks_with_inside) {
		stats.chunks_with_inside += value;
	}
	for (const int64_t value : thread_certified_inside_chunks) {
		stats.certified_inside_chunks += value;
	}
	for (const int64_t value : thread_winding_chunks) {
		stats.winding_chunks += value;
	}
	for (const int64_t value : thread_exact_distance_chunks) {
		stats.exact_distance_chunks += value;
	}
	for (const int64_t value : thread_blended_chunks) {
		stats.blended_chunks += value;
	}
	for (const int64_t value : thread_coarse_chunks) {
		stats.coarse_chunks += value;
	}
	for (const int64_t value : thread_exact_distance_voxels) {
		stats.exact_distance_voxels += value;
	}
	for (const int64_t value : thread_coarse_voxels) {
		stats.coarse_voxels += value;
	}
	for (const double value : thread_pass_seconds) {
		stats.pass_cpu_seconds += value;
	}
	for (const double value : thread_distance_seconds) {
		stats.distance_cpu_seconds += value;
	}
	for (const double value : thread_chunk_probe_distance_seconds) {
		stats.chunk_probe_distance_seconds += value;
	}
	stats.voxels_skipped_by_chunk_mask = std::max<int64_t>(0, total - stats.voxels_classified);
	return stats;
}

} // namespace

pybind11::tuple build_inside_depth_volume(
	torch::Tensor vertices,
	torch::Tensor faces,
	torch::Tensor origin_xyz,
	torch::Tensor spacing_xyz,
	torch::Tensor shape_zyx,
	const std::string& progress_label,
	int requested_threads,
	int chunk_size,
	double deep_interp_chunks,
	double deep_blend_chunks
) {
	MatrixXdR V = tensor_to_vertices(vertices);
	MatrixXiR F = tensor_to_faces(faces);
	TORCH_CHECK(V.rows() >= 4, "vertices must contain a closed shell mesh");
	TORCH_CHECK(F.rows() >= 4, "faces must contain a closed shell mesh");

	origin_xyz = origin_xyz.to(torch::kFloat64).cpu().contiguous();
	spacing_xyz = spacing_xyz.to(torch::kFloat64).cpu().contiguous();
	shape_zyx = shape_zyx.to(torch::kInt64).cpu().contiguous();
	TORCH_CHECK(origin_xyz.numel() == 3, "origin must have 3 values");
	TORCH_CHECK(spacing_xyz.numel() == 3, "spacing must have 3 values");
	TORCH_CHECK(shape_zyx.numel() == 3, "shape must have 3 values");

	const double* origin = origin_xyz.data_ptr<double>();
	const double* spacing = spacing_xyz.data_ptr<double>();
	const int64_t Z = shape_zyx.data_ptr<int64_t>()[0];
	const int64_t Y = shape_zyx.data_ptr<int64_t>()[1];
	const int64_t X = shape_zyx.data_ptr<int64_t>()[2];
	TORCH_CHECK(Z > 0 && Y > 0 && X > 0, "shape values must be positive");
	TORCH_CHECK(spacing[0] > 0.0 && spacing[1] > 0.0 && spacing[2] > 0.0, "spacing values must be positive");

	const auto total_start = std::chrono::steady_clock::now();
	const int64_t total = Z * Y * X;
	const int n_threads = default_thread_count(requested_threads);
	const bool progress_enabled = !progress_label.empty();
	std::atomic<bool> cancel(false);
	if (progress_enabled) {
		std::cout
			<< "[cyl_outside] " << progress_label
			<< ": building field shape=(" << Z << "," << Y << "," << X << ")"
			<< " voxels=" << total
			<< " vertices=" << V.rows()
			<< " faces=" << F.rows()
			<< " depth_temp="
			<< std::fixed << std::setprecision(2)
			<< (static_cast<double>(total) * static_cast<double>(sizeof(float)) / (1024.0 * 1024.0 * 1024.0))
			<< "GiB"
			<< " threads=" << n_threads
			<< " torch_threads=" << at::get_num_threads()
			<< std::endl;
		if (requested_threads <= 0 && at::get_num_threads() <= 1) {
			std::cout
				<< "[cyl_outside] " << progress_label
				<< ": torch intra-op threads is 1; using hardware_concurrency fallback. "
				<< "Set cyl_outside_threads or LASAGNA_CYL_OUTSIDE_THREADS to override."
				<< std::endl;
		}
		std::cout
			<< "[cyl_outside] " << progress_label
			<< ": preparing libigl acceleration structures..."
			<< std::endl;
	}
	throw_if_cancelled(cancel, progress_label, "startup", progress_enabled);
	const auto prep_start = std::chrono::steady_clock::now();
	igl::AABB<MatrixXdR, 3> tree;
	tree.init(V, F);
	igl::WindingNumberAABB<double, int> hier(V, F);
	hier.grow();
	const double prep_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - prep_start).count();
	if (progress_enabled) {
		std::cout
			<< "[cyl_outside] " << progress_label
			<< ": libigl acceleration ready elapsed="
			<< std::fixed << std::setprecision(1)
			<< prep_elapsed
			<< "s"
			<< std::endl;
	}
	throw_if_cancelled(cancel, progress_label, "libigl acceleration setup", progress_enabled);
	auto depth_tmp = torch::empty({total}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
	float* depth_ptr = depth_tmp.data_ptr<float>();
	DepthScanStats scan_stats;
	if (chunk_size <= 0) {
		if (progress_enabled) {
			std::cout
				<< "[cyl_outside] " << progress_label
				<< ": chunk_size=" << chunk_size << "; using full voxel scan"
				<< std::endl;
		}
		scan_stats = scan_full_volume(
			hier, tree, V, F, origin, spacing, Z, Y, X, n_threads,
			depth_ptr, cancel, progress_label, "pass 1/2 full winding+inside-distance", progress_enabled
		);
		throw_if_cancelled(cancel, progress_label, "full winding+inside-distance pass", progress_enabled);
	} else {
		if (progress_enabled) {
			std::cout
				<< "[cyl_outside] " << progress_label
				<< ": using chunked scan chunk_size=" << chunk_size
				<< " deep_interp_chunks=" << deep_interp_chunks
				<< " deep_blend_chunks=" << deep_blend_chunks
				<< std::endl;
		}
		scan_stats = scan_chunked_volume(
			hier, tree, V, F, origin, spacing, Z, Y, X, n_threads, chunk_size,
			deep_interp_chunks, deep_blend_chunks,
			depth_ptr, cancel, progress_label, progress_enabled
		);
		throw_if_cancelled(cancel, progress_label, "chunked winding+inside-distance pass", progress_enabled);
		if (scan_stats.inside_count == 0) {
			if (progress_enabled) {
				std::cout
					<< "[cyl_outside] " << progress_label
					<< ": WARNING chunked scan found zero inside voxels; rerunning full scan"
					<< std::endl;
			}
			DepthScanStats fallback_stats = scan_full_volume(
				hier, tree, V, F, origin, spacing, Z, Y, X, n_threads,
				depth_ptr, cancel, progress_label, "fallback full winding+inside-distance", progress_enabled
			);
			throw_if_cancelled(cancel, progress_label, "fallback full winding+inside-distance pass", progress_enabled);
			fallback_stats.used_chunked = true;
			fallback_stats.used_fallback = true;
			fallback_stats.total_chunks = scan_stats.total_chunks;
			fallback_stats.center_inside_chunks = scan_stats.center_inside_chunks;
			fallback_stats.surface_seed_chunks = scan_stats.surface_seed_chunks;
			fallback_stats.initial_candidate_chunks = scan_stats.initial_candidate_chunks;
			fallback_stats.growth_iterations = scan_stats.growth_iterations;
			fallback_stats.processed_chunks = scan_stats.processed_chunks;
			fallback_stats.chunks_with_inside = scan_stats.chunks_with_inside;
			fallback_stats.certified_inside_chunks = scan_stats.certified_inside_chunks;
			fallback_stats.winding_chunks = scan_stats.winding_chunks;
			fallback_stats.exact_distance_chunks = scan_stats.exact_distance_chunks;
			fallback_stats.blended_chunks = scan_stats.blended_chunks;
			fallback_stats.coarse_chunks = scan_stats.coarse_chunks;
			fallback_stats.exact_distance_voxels = scan_stats.exact_distance_voxels;
			fallback_stats.coarse_voxels = scan_stats.coarse_voxels;
			fallback_stats.voxels_skipped_by_chunk_mask = scan_stats.voxels_skipped_by_chunk_mask;
			fallback_stats.center_prepass_elapsed = scan_stats.center_prepass_elapsed;
			fallback_stats.chunk_probe_distance_seconds = scan_stats.chunk_probe_distance_seconds;
			scan_stats = fallback_stats;
		}
	}
	const double depth_max = scan_stats.depth_max;
	const int64_t inside_count = scan_stats.inside_count;

	auto out = torch::empty({1, Z, Y, X}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
	uint8_t* out_ptr = out.data_ptr<uint8_t>();
	std::atomic<int64_t> encode_done(0);
	const auto encode_start = std::chrono::steady_clock::now();
	{
		ProgressPrinter progress(progress_label, "pass 2/2 encode", total, encode_done, progress_enabled);
		parallel_chunks(total, n_threads, 2048, cancel, [&](int64_t begin, int64_t end, int /*ti*/) {
			int64_t processed = 0;
			for (int64_t n = begin; n < end; ++n) {
				if ((processed & 4095) == 0 && cancel.load(std::memory_order_relaxed)) {
					break;
				}
				out_ptr[n] = encode_depth(static_cast<double>(depth_ptr[n]), depth_max);
				++processed;
			}
			encode_done.fetch_add(processed);
		});
	}
	throw_if_cancelled(cancel, progress_label, "encode pass", progress_enabled);
	const double encode_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - encode_start).count();
	if (progress_enabled) {
		const double total_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
		const double pass1_elapsed = scan_stats.growth_elapsed;
		const double pass1_rate = pass1_elapsed > 0.0 ? static_cast<double>(scan_stats.voxels_classified) / pass1_elapsed : 0.0;
		const double encode_rate = encode_elapsed > 0.0 ? static_cast<double>(total) / encode_elapsed : 0.0;
		const int64_t skipped_distance = total - inside_count;
		const double distance_share = scan_stats.pass_cpu_seconds > 0.0
			? std::min(1.0, std::max(0.0, scan_stats.distance_cpu_seconds / scan_stats.pass_cpu_seconds))
			: 0.0;
		const double chunk_probe_distance_share = scan_stats.pass_cpu_seconds > 0.0
			? std::min(1.0, std::max(0.0, scan_stats.chunk_probe_distance_seconds / scan_stats.pass_cpu_seconds))
			: 0.0;
		const double distance_wall_est = pass1_elapsed * distance_share;
		const double chunk_probe_distance_wall_est = pass1_elapsed * chunk_probe_distance_share;
		const double winding_wall_est = std::max(0.0, pass1_elapsed - distance_wall_est - chunk_probe_distance_wall_est);
		std::cout
			<< "[cyl_outside] " << progress_label
			<< ": timing mode="
			<< (scan_stats.used_chunked ? (scan_stats.used_fallback ? "chunked+fallback" : "chunked") : "full")
			<< " prep=" << std::fixed << std::setprecision(2) << prep_elapsed << "s"
			<< " center_prepass_winding=" << scan_stats.center_prepass_elapsed << "s"
			<< " candidate_growth_winding~=" << winding_wall_est << "s"
			<< " inside_distance~=" << distance_wall_est << "s"
			<< " chunk_probe_distance~=" << chunk_probe_distance_wall_est << "s"
			<< " scan=" << pass1_elapsed << "s"
			<< " (" << std::setprecision(0) << pass1_rate << " vox/s)"
			<< " encode=" << std::setprecision(2) << encode_elapsed << "s"
			<< " (" << std::setprecision(0) << encode_rate << " vox/s)"
			<< " total=" << std::setprecision(2) << total_elapsed << "s"
			<< std::endl;
		std::cout
			<< "[cyl_outside] " << progress_label
			<< ": distance queries inside=" << inside_count
			<< " skipped_outside=" << skipped_distance
			<< " inside_frac=" << std::fixed << std::setprecision(4)
			<< (total > 0 ? static_cast<double>(inside_count) / static_cast<double>(total) : 0.0)
			<< " distance_cpu=" << std::setprecision(2) << scan_stats.distance_cpu_seconds << "s"
			<< " chunk_probe_distance_cpu=" << scan_stats.chunk_probe_distance_seconds << "s"
			<< " scan_cpu=" << scan_stats.pass_cpu_seconds << "s"
			<< " depth_max=" << std::setprecision(3) << depth_max
			<< std::endl;
		if (scan_stats.used_chunked) {
			std::cout
				<< "[cyl_outside] " << progress_label
				<< ": chunk stats total_chunks=" << scan_stats.total_chunks
				<< " center_inside_chunks=" << scan_stats.center_inside_chunks
				<< " surface_seed_chunks=" << scan_stats.surface_seed_chunks
				<< " initial_candidate_chunks=" << scan_stats.initial_candidate_chunks
				<< " growth_iterations=" << scan_stats.growth_iterations
				<< " processed_chunks=" << scan_stats.processed_chunks
				<< " chunks_with_inside=" << scan_stats.chunks_with_inside
				<< " processed_chunk_frac=" << std::fixed << std::setprecision(4)
				<< (scan_stats.total_chunks > 0
					? static_cast<double>(scan_stats.processed_chunks) / static_cast<double>(scan_stats.total_chunks)
					: 0.0)
				<< " certified_inside_chunks=" << scan_stats.certified_inside_chunks
				<< " winding_chunks=" << scan_stats.winding_chunks
				<< " exact_distance_chunks=" << scan_stats.exact_distance_chunks
				<< " blended_chunks=" << scan_stats.blended_chunks
				<< " coarse_chunks=" << scan_stats.coarse_chunks
				<< " voxels_classified=" << scan_stats.voxels_classified
				<< " voxels_skipped_by_chunk_mask=" << scan_stats.voxels_skipped_by_chunk_mask
				<< " inside_voxels=" << scan_stats.inside_count
				<< " exact_distance_voxels=" << scan_stats.exact_distance_voxels
				<< " coarse_voxels=" << scan_stats.coarse_voxels
				<< std::endl;
		}
	}

	return pybind11::make_tuple(out, depth_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def(
		"build_inside_depth_volume",
		&build_inside_depth_volume,
		pybind11::arg("vertices"),
		pybind11::arg("faces"),
		pybind11::arg("origin_xyz"),
		pybind11::arg("spacing_xyz"),
		pybind11::arg("shape_zyx"),
		pybind11::arg("progress_label") = "",
		pybind11::arg("requested_threads") = 0,
		pybind11::arg("chunk_size") = 8,
		pybind11::arg("deep_interp_chunks") = 10.0,
		pybind11::arg("deep_blend_chunks") = 2.0,
		"Build uint8 inside-depth volume from a capped shell mesh"
	);
}
