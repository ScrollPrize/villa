#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <Python.h>

#include <Eigen/Core>
#include <igl/AABB.h>
#include <igl/WindingNumberAABB.h>
#include <igl/signed_distance.h>

#include <algorithm>
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
	bool final
) {
	const double pct = total > 0 ? 100.0 * static_cast<double>(done) / static_cast<double>(total) : 100.0;
	const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	const double rate = elapsed > 0.0 ? static_cast<double>(done) / elapsed : 0.0;
	const int bar_width = 28;
	const int filled = std::min(bar_width, std::max(0, static_cast<int>(std::round((pct / 100.0) * static_cast<double>(bar_width)))));
	std::cout
		<< "[cyl_outside] " << label << " " << pass << ": "
		<< "[";
	for (int i = 0; i < bar_width; ++i) {
		std::cout << (i < filled ? "#" : "-");
	}
	std::cout
		<< "] "
		<< std::fixed << std::setprecision(1) << pct << "% "
		<< done << "/" << total << " vox"
		<< " elapsed=" << std::setprecision(1) << elapsed << "s"
		<< " rate=" << std::setprecision(0) << rate << " vox/s"
		<< (final ? (done >= total ? " done" : " stopped") : "")
		<< std::endl;
}

class ProgressPrinter {
public:
	ProgressPrinter(
		const std::string& label,
		const std::string& pass,
		int64_t total,
		const std::atomic<int64_t>& done,
		bool enabled
	):
		label_(label),
		pass_(pass),
		total_(total),
		done_(done),
		enabled_(enabled),
		stop_(false),
		start_(std::chrono::steady_clock::now())
	{
		if (enabled_) {
			print_progress_line(label_, pass_, 0, total_, start_, false);
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
					print_progress_line(label_, pass_, done_.load(), total_, start_, false);
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
		print_progress_line(label_, pass_, done_.load(), total_, start_, true);
	}

private:
	std::string label_;
	std::string pass_;
	int64_t total_;
	const std::atomic<int64_t>& done_;
	bool enabled_;
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

} // namespace

pybind11::tuple build_inside_depth_volume(
	torch::Tensor vertices,
	torch::Tensor faces,
	torch::Tensor origin_xyz,
	torch::Tensor spacing_xyz,
	torch::Tensor shape_zyx,
	const std::string& progress_label,
	int requested_threads
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
	std::vector<double> thread_max(static_cast<size_t>(n_threads), 0.0);
	std::vector<int64_t> thread_inside(static_cast<size_t>(n_threads), 0);
	std::vector<double> thread_pass_seconds(static_cast<size_t>(n_threads), 0.0);
	std::vector<double> thread_distance_seconds(static_cast<size_t>(n_threads), 0.0);
	std::atomic<int64_t> max_done(0);
	const auto pass1_start = std::chrono::steady_clock::now();
	{
		ProgressPrinter progress(progress_label, "pass 1/2 winding+inside-distance", total, max_done, progress_enabled);
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
			thread_distance_seconds[static_cast<size_t>(ti)] += local_distance_seconds;
			thread_pass_seconds[static_cast<size_t>(ti)] += std::chrono::duration<double>(
				std::chrono::steady_clock::now() - chunk_start
			).count();
			max_done.fetch_add(processed);
		});
	}
	throw_if_cancelled(cancel, progress_label, "winding+inside-distance pass", progress_enabled);
	const double pass1_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - pass1_start).count();
	double depth_max = 0.0;
	int64_t inside_count = 0;
	double pass_cpu_seconds = 0.0;
	double distance_cpu_seconds = 0.0;
	for (const double value : thread_max) {
		if (value > depth_max) {
			depth_max = value;
		}
	}
	for (const int64_t value : thread_inside) {
		inside_count += value;
	}
	for (const double value : thread_pass_seconds) {
		pass_cpu_seconds += value;
	}
	for (const double value : thread_distance_seconds) {
		distance_cpu_seconds += value;
	}

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
		const double pass1_rate = pass1_elapsed > 0.0 ? static_cast<double>(total) / pass1_elapsed : 0.0;
		const double encode_rate = encode_elapsed > 0.0 ? static_cast<double>(total) / encode_elapsed : 0.0;
		const int64_t skipped_distance = total - inside_count;
		const double distance_share = pass_cpu_seconds > 0.0
			? std::min(1.0, std::max(0.0, distance_cpu_seconds / pass_cpu_seconds))
			: 0.0;
		const double distance_wall_est = pass1_elapsed * distance_share;
		const double winding_wall_est = pass1_elapsed - distance_wall_est;
		std::cout
			<< "[cyl_outside] " << progress_label
			<< ": timing prep=" << std::fixed << std::setprecision(2) << prep_elapsed << "s"
			<< " winding_classify~=" << winding_wall_est << "s"
			<< " inside_distance~=" << distance_wall_est << "s"
			<< " pass1=" << pass1_elapsed << "s"
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
			<< " distance_cpu=" << std::setprecision(2) << distance_cpu_seconds << "s"
			<< " pass1_cpu=" << pass_cpu_seconds << "s"
			<< " depth_max=" << std::setprecision(3) << depth_max
			<< std::endl;
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
		"Build uint8 inside-depth volume from a capped shell mesh"
	);
}
