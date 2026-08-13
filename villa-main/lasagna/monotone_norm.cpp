#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <limits>

namespace py = pybind11;

namespace {

void compute_monotone_normalized(const float *in, float *out, int H, int W)
{
	const int N = H * W;

	std::vector<int> next_up(N, -1);
	std::vector<int> next_down(N, -1);

	const int dx[8] = { -1, 1, 0, 0, -1, -1, 1, 1 };
	const int dy[8] = { 0, 0, -1, 1, -1, 1, -1, 1 };

	// Precompute steepest ascent and descent neighbor for each pixel.
	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			const int idx = y * W + x;
			const float v = in[idx];

			float best_up_val = -std::numeric_limits<float>::infinity();
			int best_up_idx = -1;
			float best_down_val = std::numeric_limits<float>::infinity();
			int best_down_idx = -1;

			for (int k = 0; k < 8; ++k) {
				const int nx = x + dx[k];
				const int ny = y + dy[k];
				if (nx < 0 || nx >= W || ny < 0 || ny >= H) {
					continue;
				}
				const int nidx = ny * W + nx;
				const float nv = in[nidx];

				if (nv > v && nv > best_up_val) {
					best_up_val = nv;
					best_up_idx = nidx;
				}
				if (nv < v && nv < best_down_val) {
					best_down_val = nv;
					best_down_idx = nidx;
				}
			}

			next_up[idx] = best_up_idx;
			next_down[idx] = best_down_idx;
		}
	}

	// Follow ascent/descent chains with path compression to find extrema.
	std::vector<int> max_root(N, -1);
	std::vector<int> min_root(N, -1);

	for (int i = 0; i < N; ++i) {
		int cur = i;
		std::vector<int> path;
		path.reserve(64);
		while (true) {
			if (max_root[cur] != -1) {
				break;
			}
			path.push_back(cur);
			int nxt = next_up[cur];
			if (nxt == -1) {
				break;
			}
			cur = nxt;
		}
		int root = (max_root[cur] != -1) ? max_root[cur] : cur;
		for (int idx : path) {
			max_root[idx] = root;
		}
		// Also ensure the final node has its root set.
		if (max_root[cur] == -1) {
			max_root[cur] = root;
		}
	}

	for (int i = 0; i < N; ++i) {
		int cur = i;
		std::vector<int> path;
		path.reserve(64);
		while (true) {
			if (min_root[cur] != -1) {
				break;
			}
			path.push_back(cur);
			int nxt = next_down[cur];
			if (nxt == -1) {
				break;
			}
			cur = nxt;
		}
		int root = (min_root[cur] != -1) ? min_root[cur] : cur;
		for (int idx : path) {
			min_root[idx] = root;
		}
		// Also ensure the final node has its root set.
		if (min_root[cur] == -1) {
			min_root[cur] = root;
		}
	}

	// Compute normalized value in [0,1] for each pixel.
	// Map reachable minimum -> 1 and reachable maximum -> 0:
	// t = (d_max - d) / (d_max - d_min)
	constexpr float eps = 1e-6f;
	for (int i = 0; i < N; ++i) {
		const float d = in[i];
		const float dmax = in[max_root[i]];
		const float dmin = in[min_root[i]];
		const float denom = dmax - dmin;
		if (denom > eps) {
			out[i] = (dmax - d) / denom;
		} else {
			out[i] = 0.5f;
		}
	}
}

} // namespace

py::array_t<float> monotone_normalized(py::array_t<float, py::array::c_style | py::array::forcecast> input)
{
	py::buffer_info buf = input.request();
	if (buf.ndim != 2) {
		throw std::runtime_error("Input array must be 2D");
	}
	const int H = static_cast<int>(buf.shape[0]);
	const int W = static_cast<int>(buf.shape[1]);
	const float *in = static_cast<const float *>(buf.ptr);

	py::array_t<float> output({H, W});
	py::buffer_info out_buf = output.request();
	float *out = static_cast<float *>(out_buf.ptr);

	compute_monotone_normalized(in, out, H, W);
	return output;
}

PYBIND11_MODULE(monotone_norm, m)
{
	m.doc() = "Monotone normalized distance transform";
	m.def("compute", &monotone_normalized, "Compute monotone-normalized DT");
}