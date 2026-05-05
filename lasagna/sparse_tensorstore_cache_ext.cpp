#include <torch/extension.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "tensorstore/context.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/tensorstore.h"

namespace {

constexpr int64_t kChunkSize = 32;
constexpr int64_t kPadded = kChunkSize + 2;

template <typename Status>
[[noreturn]] void ThrowStatus(const char* where, const Status& status) {
    throw std::runtime_error(std::string(where) + ": " + status.ToString());
}

struct Bounds {
    int64_t rz0, rz1, ry0, ry1, rx0, rx1;
    int64_t dz0, dy0, dx0;
    int64_t dz, dy, dx;
};

using ReadArray = tensorstore::SharedArray<const uint8_t, tensorstore::dynamic_rank>;
using ReadFuture = tensorstore::Future<ReadArray>;

struct PendingChunk {
    int64_t cz = 0;
    int64_t cy = 0;
    int64_t cx = 0;
    std::vector<ReadFuture> reads;
};

class TensorStoreSparseChunkGroupCache {
public:
    TensorStoreSparseChunkGroupCache(
        std::vector<std::string> channels,
        std::string zarr_path,
        std::vector<int64_t> vol_shape_zyx,
        std::vector<int64_t> channel_indices,
        bool is_3d_zarr,
        int64_t device_index,
        int64_t cache_pool_bytes,
        int64_t file_io_threads,
        int64_t data_copy_threads)
        : channels_(std::move(channels)),
          zarr_path_(std::move(zarr_path)),
          vol_shape_zyx_(std::move(vol_shape_zyx)),
          channel_indices_(std::move(channel_indices)),
          is_3d_zarr_(is_3d_zarr),
          device_index_(device_index),
          cache_pool_bytes_(cache_pool_bytes),
          file_io_threads_(file_io_threads),
          data_copy_threads_(data_copy_threads) {
        if (vol_shape_zyx_.size() != 3) {
            throw std::runtime_error("vol_shape_zyx must have 3 entries");
        }
        if (!is_3d_zarr_ && channel_indices_.size() != channels_.size()) {
            throw std::runtime_error("channel_indices must match channels for 4D zarr");
        }
        const int64_t Z = vol_shape_zyx_[0];
        const int64_t Y = vol_shape_zyx_[1];
        const int64_t X = vol_shape_zyx_[2];
        chunk_grid_ = {
            (Z + kChunkSize - 1) / kChunkSize,
            (Y + kChunkSize - 1) / kChunkSize,
            (X + kChunkSize - 1) / kChunkSize,
        };

        auto device = device_index_ >= 0
            ? torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(device_index_))
            : torch::Device(torch::kCPU);
        chunk_table_ = torch::zeros({chunk_grid_[0], chunk_grid_[1], chunk_grid_[2]},
                                    torch::TensorOptions().dtype(torch::kInt64).device(device));

        auto context_result = tensorstore::Context::FromJson({
            {"cache_pool", {{"total_bytes_limit", cache_pool_bytes_}}},
            {"file_io_concurrency", {{"limit", file_io_threads_}}},
            {"data_copy_concurrency", {{"limit", data_copy_threads_}}},
        });
        if (!context_result.ok()) {
            ThrowStatus("TensorStore context", context_result.status());
        }
        context_ = *context_result;

        nlohmann::json spec = {
            {"driver", "zarr"},
            {"kvstore", {{"driver", "file"}, {"path", zarr_path_}}},
            {"recheck_cached_data", "open"},
        };
        auto open_result = tensorstore::Open<uint8_t>(
            spec,
            context_,
            tensorstore::OpenMode::open,
            tensorstore::ReadWriteMode::read).result();
        if (!open_result.ok()) {
            ThrowStatus("TensorStore open", open_result.status());
        }
        store_ = *open_result;

        const double table_mib = static_cast<double>(chunk_grid_[0] * chunk_grid_[1] * chunk_grid_[2] * 8) / (1024.0 * 1024.0);
        std::cout << "[sparse_cache_cpp] ";
        for (size_t i = 0; i < channels_.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << channels_[i];
        }
        std::cout << ": chunk_grid=" << chunk_grid_[0] << "x" << chunk_grid_[1] << "x" << chunk_grid_[2]
                  << " vol=" << Z << "x" << Y << "x" << X
                  << " table=" << table_mib << "MiB"
                  << " cache_pool=" << (cache_pool_bytes_ / (1024 * 1024)) << "MiB"
                  << " file_io=" << file_io_threads_
                  << " data_copy=" << data_copy_threads_
                  << " recheck_cached_data=open" << std::endl;
    }

    torch::Tensor chunk_table() const {
        return chunk_table_;
    }

    void prefetch_coords(torch::Tensor coords_cpu) {
        if (coords_cpu.numel() == 0) {
            return;
        }
        coords_cpu = coords_cpu.to(torch::kCPU).contiguous();
        if (coords_cpu.dtype() != torch::kInt64 || coords_cpu.dim() != 2 || coords_cpu.size(1) != 3) {
            throw std::runtime_error("coords must be an int64 CPU tensor of shape (N, 3)");
        }
        auto acc = coords_cpu.accessor<int64_t, 2>();
        for (int64_t i = 0; i < coords_cpu.size(0); ++i) {
            PendingChunk pending;
            pending.cz = acc[i][0];
            pending.cy = acc[i][1];
            pending.cx = acc[i][2];
            SubmitReads(pending);
            pending_.push_back(std::move(pending));
        }
    }

    void sync() {
        if (pending_.empty()) {
            last_sync_new_ = 0;
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t n = static_cast<int64_t>(pending_.size());
        const int64_t C = static_cast<int64_t>(channels_.size());
        auto cpu_batch = torch::empty(
            {n, C, kPadded, kPadded, kPadded},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(device_index_ >= 0));
        auto cpu_ptr = cpu_batch.data_ptr<uint8_t>();
        std::vector<int64_t> cz(n), cy(n), cx(n), ptrs(n);

        const int64_t chunk_bytes = C * kPadded * kPadded * kPadded;
        for (int64_t i = 0; i < n; ++i) {
            uint8_t* dst = cpu_ptr + i * chunk_bytes;
            std::memset(dst, 0, static_cast<size_t>(chunk_bytes));
            FinishChunk(pending_[static_cast<size_t>(i)], dst);
            cz[i] = pending_[static_cast<size_t>(i)].cz;
            cy[i] = pending_[static_cast<size_t>(i)].cy;
            cx[i] = pending_[static_cast<size_t>(i)].cx;
        }
        pending_.clear();

        auto device = device_index_ >= 0
            ? torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(device_index_))
            : torch::Device(torch::kCPU);
        auto gpu_batch = cpu_batch.to(device, /*non_blocking=*/device_index_ >= 0);
        batches_.push_back(gpu_batch);

        const int64_t base_ptr = static_cast<int64_t>(gpu_batch.data_ptr<uint8_t>());
        for (int64_t i = 0; i < n; ++i) {
            ptrs[i] = base_ptr + i * chunk_bytes;
        }
        auto idx_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
        auto cz_t = torch::from_blob(cz.data(), {n}, torch::TensorOptions().dtype(torch::kInt64)).clone().to(idx_opts.device());
        auto cy_t = torch::from_blob(cy.data(), {n}, torch::TensorOptions().dtype(torch::kInt64)).clone().to(idx_opts.device());
        auto cx_t = torch::from_blob(cx.data(), {n}, torch::TensorOptions().dtype(torch::kInt64)).clone().to(idx_opts.device());
        auto ptr_t = torch::from_blob(ptrs.data(), {n}, torch::TensorOptions().dtype(torch::kInt64)).clone().to(idx_opts.device());
        chunk_table_.index_put_({cz_t, cy_t, cx_t}, ptr_t);

        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        last_sync_new_ = n;
        total_new_chunks_ += n;
        total_fetch_ms_ += ms;
    }

    void end_iteration() {
        ++iter_count_;
    }

    void print_summary() const {
        const double ms_per_it = iter_count_ > 0 ? total_fetch_ms_ / iter_count_ : 0.0;
        const double ms_per_chunk = total_new_chunks_ > 0 ? total_fetch_ms_ / total_new_chunks_ : 0.0;
        std::cout << "[sparse_cache_cpp] chunks=" << total_new_chunks_
                  << " in " << iter_count_ << "it"
                  << " (" << ms_per_it << "ms/it, " << ms_per_chunk << "ms/chunk)"
                  << " total=" << loaded_chunks()
                  << " (" << loaded_mib() << "MiB)" << std::endl;
    }

    int64_t loaded_chunks() const {
        return chunk_table_.ne(0).sum().item<int64_t>();
    }

    double loaded_mib() const {
        const int64_t C = static_cast<int64_t>(channels_.size());
        return static_cast<double>(loaded_chunks() * C * kPadded * kPadded * kPadded) / (1024.0 * 1024.0);
    }

private:
    Bounds ChunkBounds(int64_t cz, int64_t cy, int64_t cx) const {
        const int64_t Z = vol_shape_zyx_[0];
        const int64_t Y = vol_shape_zyx_[1];
        const int64_t X = vol_shape_zyx_[2];
        const int64_t gz0 = cz * kChunkSize - 1;
        const int64_t gy0 = cy * kChunkSize - 1;
        const int64_t gx0 = cx * kChunkSize - 1;
        Bounds b;
        b.rz0 = std::max<int64_t>(0, gz0);
        b.ry0 = std::max<int64_t>(0, gy0);
        b.rx0 = std::max<int64_t>(0, gx0);
        b.rz1 = std::min<int64_t>(Z, gz0 + kPadded);
        b.ry1 = std::min<int64_t>(Y, gy0 + kPadded);
        b.rx1 = std::min<int64_t>(X, gx0 + kPadded);
        b.dz0 = b.rz0 - gz0;
        b.dy0 = b.ry0 - gy0;
        b.dx0 = b.rx0 - gx0;
        b.dz = b.rz1 - b.rz0;
        b.dy = b.ry1 - b.ry0;
        b.dx = b.rx1 - b.rx0;
        return b;
    }

    void SubmitReads(PendingChunk& pending) {
        const Bounds b = ChunkBounds(pending.cz, pending.cy, pending.cx);
        if (b.dz <= 0 || b.dy <= 0 || b.dx <= 0) {
            return;
        }
        if (is_3d_zarr_) {
            std::array<tensorstore::Index, 3> start{b.rz0, b.ry0, b.rx0};
            std::array<tensorstore::Index, 3> stop{b.rz1, b.ry1, b.rx1};
            auto view = store_ | tensorstore::Dims(0, 1, 2).HalfOpenInterval(start, stop);
            auto view_result = view.result();
            if (!view_result.ok()) ThrowStatus("TensorStore slice", view_result.status());
            pending.reads.push_back(tensorstore::Read(*view_result, tensorstore::ContiguousLayoutOrder::c));
        } else {
            for (int64_t ch_idx : channel_indices_) {
                auto ch_view = store_ | tensorstore::Dims(0).IndexSlice(ch_idx);
                auto ch_result = ch_view.result();
                if (!ch_result.ok()) ThrowStatus("TensorStore channel slice", ch_result.status());
                std::array<tensorstore::Index, 3> start{b.rz0, b.ry0, b.rx0};
                std::array<tensorstore::Index, 3> stop{b.rz1, b.ry1, b.rx1};
                auto view = *ch_result | tensorstore::Dims(0, 1, 2).HalfOpenInterval(start, stop);
                auto view_result = view.result();
                if (!view_result.ok()) ThrowStatus("TensorStore slice", view_result.status());
                pending.reads.push_back(tensorstore::Read(*view_result, tensorstore::ContiguousLayoutOrder::c));
            }
        }
    }

    void Copy3dArrayToChunk(const ReadArray& arr, uint8_t* dst, int64_t channel, const Bounds& b) const {
        const uint8_t* src = arr.data();
        const int64_t src_dy = b.dy;
        const int64_t src_dx = b.dx;
        const int64_t C = static_cast<int64_t>(channels_.size());
        for (int64_t z = 0; z < b.dz; ++z) {
            for (int64_t y = 0; y < b.dy; ++y) {
                const uint8_t* src_row = src + (z * src_dy + y) * src_dx;
                uint8_t* dst_row = dst + (((channel * kPadded + (b.dz0 + z)) * kPadded + (b.dy0 + y)) * kPadded + b.dx0);
                std::memcpy(dst_row, src_row, static_cast<size_t>(b.dx));
            }
        }
        (void)C;
    }

    void FinishChunk(const PendingChunk& pending, uint8_t* dst) const {
        const Bounds b = ChunkBounds(pending.cz, pending.cy, pending.cx);
        if (b.dz <= 0 || b.dy <= 0 || b.dx <= 0 || pending.reads.empty()) {
            return;
        }
        if (is_3d_zarr_) {
            auto arr_result = pending.reads[0].result();
            if (!arr_result.ok()) ThrowStatus("TensorStore read", arr_result.status());
            Copy3dArrayToChunk(*arr_result, dst, 0, b);
            return;
        }
        for (size_t ch = 0; ch < pending.reads.size(); ++ch) {
            auto arr_result = pending.reads[ch].result();
            if (!arr_result.ok()) ThrowStatus("TensorStore read", arr_result.status());
            Copy3dArrayToChunk(*arr_result, dst, static_cast<int64_t>(ch), b);
        }
    }

    std::vector<std::string> channels_;
    std::string zarr_path_;
    std::vector<int64_t> vol_shape_zyx_;
    std::vector<int64_t> channel_indices_;
    bool is_3d_zarr_ = false;
    int64_t device_index_ = -1;
    int64_t cache_pool_bytes_ = 0;
    int64_t file_io_threads_ = 0;
    int64_t data_copy_threads_ = 0;
    std::array<int64_t, 3> chunk_grid_{0, 0, 0};
    tensorstore::Context context_;
    tensorstore::TensorStore<uint8_t, tensorstore::dynamic_rank, tensorstore::ReadWriteMode::read> store_;
    torch::Tensor chunk_table_;
    std::vector<PendingChunk> pending_;
    std::vector<torch::Tensor> batches_;
    int64_t iter_count_ = 0;
    int64_t total_new_chunks_ = 0;
    double total_fetch_ms_ = 0.0;
    int64_t last_sync_new_ = 0;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<TensorStoreSparseChunkGroupCache>(m, "TensorStoreSparseChunkGroupCache")
        .def(pybind11::init<std::vector<std::string>, std::string, std::vector<int64_t>,
                            std::vector<int64_t>, bool, int64_t, int64_t, int64_t, int64_t>())
        .def("chunk_table", &TensorStoreSparseChunkGroupCache::chunk_table)
        .def("prefetch_coords", &TensorStoreSparseChunkGroupCache::prefetch_coords)
        .def("sync", &TensorStoreSparseChunkGroupCache::sync)
        .def("end_iteration", &TensorStoreSparseChunkGroupCache::end_iteration)
        .def("print_summary", &TensorStoreSparseChunkGroupCache::print_summary)
        .def("loaded_chunks", &TensorStoreSparseChunkGroupCache::loaded_chunks)
        .def("loaded_mib", &TensorStoreSparseChunkGroupCache::loaded_mib);
}
