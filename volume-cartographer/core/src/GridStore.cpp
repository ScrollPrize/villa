#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/LineSegList.hpp"

#include <unordered_set>
#include <fstream>
#include <stdexcept>
#include <numeric>

#include <arpa/inet.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace vc::core::util {

namespace {
constexpr uint32_t GRIDSTORE_MAGIC = 0x56434753; // "VCGS"
constexpr uint32_t GRIDSTORE_VERSION = 2;
}

struct MmappedData {
    int fd = -1;
    void* data = MAP_FAILED;
    size_t size = 0;

    ~MmappedData() {
        if (data != MAP_FAILED) {
            munmap(data, size);
        }
        if (fd != -1) {
            close(fd);
        }
    }
};

class GridStore::GridStoreImpl {
public:
    GridStoreImpl(const cv::Rect& bounds, int cell_size)
        : bounds_(bounds), cell_size_(cell_size), read_only_(false) {
        grid_size_ = cv::Size(
            (bounds.width + cell_size - 1) / cell_size,
            (bounds.height + cell_size - 1) / cell_size
        );
        grid_.resize(grid_size_.width * grid_size_.height);
    }

    void add(const std::vector<cv::Point>& points) {
        if (read_only_) {
            throw std::runtime_error("Cannot add to a read-only GridStore.");
        }
        if (points.size() < 2) return;

        int handle = storage_.size();
        storage_.emplace_back(std::make_shared<LineSegList>(points));

        std::unordered_set<int> relevant_buckets;
        for (const auto& p : points) {
            cv::Point grid_pos = (p - bounds_.tl()) / cell_size_;
            if (grid_pos.x >= 0 && grid_pos.x < grid_size_.width &&
                grid_pos.y >= 0 && grid_pos.y < grid_size_.height) {
                int index = grid_pos.y * grid_size_.width + grid_pos.x;
                relevant_buckets.insert(index);
            }
        }

        for (int index : relevant_buckets) {
            grid_[index].push_back(handle);
        }
    }

    std::vector<std::shared_ptr<std::vector<cv::Point>>> get(const cv::Rect& query_rect) const {
        std::unordered_set<int> handles;
        cv::Rect clamped_rect = query_rect & bounds_;

        cv::Point start = (clamped_rect.tl() - bounds_.tl()) / cell_size_;
        cv::Point end = (clamped_rect.br() - bounds_.tl()) / cell_size_;

        for (int y = start.y; y <= end.y; ++y) {
            for (int x = start.x; x <= end.x; ++x) {
                int index = y * grid_size_.width + x;
                if (index >= 0 && index < grid_.size()) {
                    handles.insert(grid_[index].begin(), grid_[index].end());
                }
            }
        }

        std::vector<std::shared_ptr<std::vector<cv::Point>>> result;
        result.reserve(handles.size());
        for (int handle : handles) {
            result.push_back(storage_[handle]->get());
        }
        return result;
    }

    std::vector<std::shared_ptr<std::vector<cv::Point>>> get_all() const {
        std::vector<std::shared_ptr<std::vector<cv::Point>>> result;
        result.reserve(storage_.size());
        for (const auto& seg_list : storage_) {
            result.push_back(seg_list->get());
        }
        return result;
    }

    cv::Size size() const {
        return bounds_.size();
    }

    size_t get_memory_usage() const {
        size_t grid_memory = grid_.capacity() * sizeof(std::vector<int>);
        for (const auto& cell : grid_) {
            grid_memory += cell.capacity() * sizeof(int);
        }
        size_t storage_memory = storage_.capacity() * sizeof(std::shared_ptr<LineSegList>);
        // This doesn't account for the memory inside LineSegList, which is complex to calculate here.
        // A more accurate implementation would require a get_memory_usage() method in LineSegList.
        return grid_memory + storage_memory;
    }

    size_t numSegments() const {
        size_t count = 0;
        for (const auto& seg_list : storage_) {
            if (seg_list->num_points() > 0) {
                count += seg_list->num_points() - 1;
            }
        }
        return count;
    }

    size_t numNonEmptyBuckets() const {
        size_t count = 0;
        for (const auto& bucket : grid_) {
            if (!bucket.empty()) {
                count++;
            }
        }
        return count;
    }

    void save(const std::string& path) const {
        std::string meta_str = meta_.dump();
        size_t header_size = 13 * sizeof(uint32_t);
        size_t buckets_size = get_all_buckets_size();
        size_t paths_size = get_all_seglist_size();
        size_t meta_size = meta_str.size();
        size_t total_size = header_size + buckets_size + paths_size + meta_size;

        std::vector<char> buffer(total_size);

        // Serialize paths and store offsets
        std::unordered_map<int, uint32_t> path_offsets;
        char* paths_start = buffer.data() + header_size + buckets_size;
        char* current_path_ptr = paths_start;
        for (size_t i = 0; i < storage_.size(); ++i) {
            path_offsets[i] = current_path_ptr - paths_start;
            current_path_ptr = write_seglist(current_path_ptr, *storage_[i]);
        }

        // Serialize buckets
        char* current_bucket_ptr = buffer.data() + header_size;
        for (const auto& bucket : grid_) {
            current_bucket_ptr = write_bucket(current_bucket_ptr, bucket, path_offsets);
        }

        // Write header
        char* header_ptr = buffer.data();
        uint32_t magic = htonl(GRIDSTORE_MAGIC);
        uint32_t version = htonl(GRIDSTORE_VERSION);
        uint32_t bounds_x = htonl(bounds_.x);
        uint32_t bounds_y = htonl(bounds_.y);
        uint32_t bounds_width = htonl(bounds_.width);
        uint32_t bounds_height = htonl(bounds_.height);
        uint32_t cell_size = htonl(cell_size_);
        uint32_t num_buckets = htonl(grid_.size());
        uint32_t num_paths = htonl(storage_.size());
        uint32_t buckets_offset = htonl(header_size);
        uint32_t paths_offset = htonl(header_size + buckets_size);
        uint32_t json_meta_offset = htonl(header_size + buckets_size + paths_size);
        uint32_t json_meta_size = htonl(meta_size);

        memcpy(header_ptr, &magic, sizeof(magic)); header_ptr += sizeof(magic);
        memcpy(header_ptr, &version, sizeof(version)); header_ptr += sizeof(version);
        memcpy(header_ptr, &bounds_x, sizeof(bounds_x)); header_ptr += sizeof(bounds_x);
        memcpy(header_ptr, &bounds_y, sizeof(bounds_y)); header_ptr += sizeof(bounds_y);
        memcpy(header_ptr, &bounds_width, sizeof(bounds_width)); header_ptr += sizeof(bounds_width);
        memcpy(header_ptr, &bounds_height, sizeof(bounds_height)); header_ptr += sizeof(bounds_height);
        memcpy(header_ptr, &cell_size, sizeof(cell_size)); header_ptr += sizeof(cell_size);
        memcpy(header_ptr, &num_buckets, sizeof(num_buckets)); header_ptr += sizeof(num_buckets);
        memcpy(header_ptr, &num_paths, sizeof(num_paths)); header_ptr += sizeof(num_paths);
        memcpy(header_ptr, &buckets_offset, sizeof(buckets_offset)); header_ptr += sizeof(buckets_offset);
        memcpy(header_ptr, &paths_offset, sizeof(paths_offset)); header_ptr += sizeof(paths_offset);
        memcpy(header_ptr, &json_meta_offset, sizeof(json_meta_offset)); header_ptr += sizeof(json_meta_offset);
        memcpy(header_ptr, &json_meta_size, sizeof(json_meta_size)); header_ptr += sizeof(json_meta_size);

        // In-line verification
        {
            const char* buffer_start = buffer.data();
            const char* buffer_end = buffer_start + buffer.size();

            // 1. Verify Header
            if (ntohl(*reinterpret_cast<const uint32_t*>(buffer_start)) != GRIDSTORE_MAGIC) throw std::runtime_error("Header verification failed: magic mismatch.");
            if (ntohl(*reinterpret_cast<const uint32_t*>(buffer_start + 4)) != GRIDSTORE_VERSION) throw std::runtime_error("Header verification failed: version mismatch.");
            // ... (add more header checks if desired)

            // 2. Deserialize Paths
            std::vector<std::shared_ptr<LineSegList>> deserialized_storage;
            deserialized_storage.reserve(storage_.size());
            const char* read_paths_ptr = buffer_start + ntohl(paths_offset);
            for (size_t i = 0; i < storage_.size(); ++i) {
                std::shared_ptr<LineSegList> seglist;
                read_paths_ptr = read_seglist(read_paths_ptr, buffer_end, seglist);
                deserialized_storage.push_back(seglist);
            }

            // 3. Deserialize Buckets (requires mapping offsets back to handles)
            std::unordered_map<uint32_t, int> offset_to_handle;
            const char* temp_paths_ptr = buffer_start + ntohl(paths_offset);
            for (size_t i = 0; i < deserialized_storage.size(); ++i) {
                uint32_t offset = (temp_paths_ptr - (buffer_start + ntohl(paths_offset)));
                offset_to_handle[offset] = i;
                
                // Advance pointer to the next seglist by reading its size
                if (temp_paths_ptr + 3 * sizeof(uint32_t) > buffer_end) throw std::runtime_error("Verification failed: path header out of bounds.");
                uint32_t num_offsets_bytes = ntohl(*reinterpret_cast<const uint32_t*>(temp_paths_ptr + 2 * sizeof(uint32_t)));
                temp_paths_ptr += 3 * sizeof(uint32_t) + num_offsets_bytes;
            }

            std::vector<std::vector<int>> deserialized_grid(grid_.size());
            const char* read_bucket_ptr = buffer_start + ntohl(buckets_offset);
            for (auto& bucket : deserialized_grid) {
                if (read_bucket_ptr + sizeof(uint32_t) > buffer_end) throw std::runtime_error("Verification failed: bucket header out of bounds.");
                uint32_t num_indices = ntohl(*reinterpret_cast<const uint32_t*>(read_bucket_ptr)); read_bucket_ptr += sizeof(uint32_t);
                bucket.resize(num_indices);
                if (read_bucket_ptr + num_indices * sizeof(uint32_t) > buffer_end) throw std::runtime_error("Verification failed: bucket data out of bounds.");
                for (uint32_t i = 0; i < num_indices; ++i) {
                    uint32_t path_offset = ntohl(*reinterpret_cast<const uint32_t*>(read_bucket_ptr)); read_bucket_ptr += sizeof(uint32_t);
                    bucket[i] = offset_to_handle.at(path_offset);
                }
            }

            // 4. Compare
            if (grid_ != deserialized_grid) {
                throw std::runtime_error("Bucket serialization verification failed: data mismatch.");
            }

            if (storage_.size() != deserialized_storage.size()) {
                throw std::runtime_error("Seglist serialization verification failed: size mismatch.");
            }

            for (size_t i = 0; i < storage_.size(); ++i) {
                const auto& original = storage_[i];
                const auto& deserialized = deserialized_storage[i];
                if (original->start_point() != deserialized->start_point()) {
                    throw std::runtime_error("Seglist verification failed: start point mismatch.");
                }
                if (original->compressed_data_size() != deserialized->compressed_data_size()) {
                    throw std::runtime_error("Seglist verification failed: data size mismatch.");
                }
                if (memcmp(original->compressed_data(), deserialized->compressed_data(), original->compressed_data_size()) != 0) {
                    throw std::runtime_error("Seglist verification failed: data mismatch.");
                }
            }
        }

        // Write metadata
        char* meta_start = buffer.data() + header_size + buckets_size + paths_size;
        memcpy(meta_start, meta_str.data(), meta_size);

        // Write buffer to file
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for writing: " + path);
        }
        file.write(buffer.data(), buffer.size());
    }

    void load_mmap(const std::string& path) {
        read_only_ = true;
        mmapped_data_ = std::make_unique<MmappedData>();

        mmapped_data_->fd = open(path.c_str(), O_RDONLY);
        if (mmapped_data_->fd == -1) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        struct stat sb;
        if (fstat(mmapped_data_->fd, &sb) == -1) {
            throw std::runtime_error("Failed to stat file: " + path);
        }
        mmapped_data_->size = sb.st_size;

        if (mmapped_data_->size == 0) {
            // Handle empty file: Grid is already empty, just set bounds and return.
            bounds_ = cv::Rect();
            cell_size_ = 1; // Avoid division by zero
            grid_size_ = cv::Size(0,0);
            return;
        }

        mmapped_data_->data = mmap(NULL, mmapped_data_->size, PROT_READ, MAP_PRIVATE, mmapped_data_->fd, 0);
        if (mmapped_data_->data == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap file: " + path);
        }

        const char* current = static_cast<const char*>(mmapped_data_->data);
        const char* end = current + mmapped_data_->size;

        // 1. Read Header
        size_t min_header_size = 11 * sizeof(uint32_t);
        if (mmapped_data_->size < min_header_size) {
            throw std::runtime_error("Invalid GridStore file: too small for header.");
        }
        uint32_t magic = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        uint32_t version = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        if (magic != GRIDSTORE_MAGIC) {
            throw std::runtime_error("Invalid GridStore file: magic mismatch.");
        }
        if (version > GRIDSTORE_VERSION) {
            throw std::runtime_error("GridStore file is a newer version than this reader supports.");
        }

        bounds_.x = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        bounds_.y = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        bounds_.width = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        bounds_.height = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        cell_size_ = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        uint32_t num_buckets = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        uint32_t num_paths = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        uint32_t buckets_offset = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        uint32_t paths_offset = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        
        uint32_t json_meta_offset = 0;
        uint32_t json_meta_size = 0;
        if (version >= 2) {
            if (mmapped_data_->size < 13 * sizeof(uint32_t)) {
                throw std::runtime_error("Invalid GridStore v2 file: too small for extended header.");
            }
            json_meta_offset = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
            json_meta_size = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        }

        grid_size_ = cv::Size(
            (bounds_.width + cell_size_ - 1) / cell_size_,
            (bounds_.height + cell_size_ - 1) / cell_size_
        );

        // 2. Read Paths and build offset map
        storage_.reserve(num_paths);
        std::unordered_map<uint32_t, int> offset_to_handle;
        const char* paths_start = static_cast<const char*>(mmapped_data_->data) + paths_offset;
        const char* current_path_ptr = paths_start;
        for (uint32_t i = 0; i < num_paths; ++i) {
            uint32_t current_offset = current_path_ptr - paths_start;
            int handle = storage_.size();
            
            std::shared_ptr<LineSegList> seglist;
            current_path_ptr = read_seglist(current_path_ptr, end, seglist);
            storage_.push_back(seglist);
            offset_to_handle[current_offset] = handle;
        }

        // 3. Read Buckets
        grid_.assign(num_buckets, std::vector<int>());
        const char* buckets_start = static_cast<const char*>(mmapped_data_->data) + buckets_offset;
        const char* current_bucket_ptr = buckets_start;
        for (uint32_t i = 0; i < num_buckets; ++i) {
            if (current_bucket_ptr + sizeof(uint32_t) > end) throw std::runtime_error("Invalid GridStore file: unexpected end in bucket header.");
            uint32_t num_indices = ntohl(*reinterpret_cast<const uint32_t*>(current_bucket_ptr)); current_bucket_ptr += sizeof(uint32_t);
            
            grid_[i].reserve(num_indices);
            if (current_bucket_ptr + num_indices * sizeof(uint32_t) > end) throw std::runtime_error("Invalid GridStore file: bucket indices out of bounds.");
            for (uint32_t j = 0; j < num_indices; ++j) {
                uint32_t path_offset = ntohl(*reinterpret_cast<const uint32_t*>(current_bucket_ptr)); current_bucket_ptr += sizeof(uint32_t);
                grid_[i].push_back(offset_to_handle.at(path_offset));
            }
        }
        // 4. Read Metadata
        if (version >= 2 && json_meta_size > 0) {
            const char* meta_start = static_cast<const char*>(mmapped_data_->data) + json_meta_offset;
            if (meta_start + json_meta_size > end) {
                throw std::runtime_error("Invalid GridStore file: metadata out of bounds.");
            }
            std::string meta_str(meta_start, json_meta_size);
            meta_ = nlohmann::json::parse(meta_str);
        }
    }
    nlohmann::json meta_;

private:
    char* write_bucket(char* current, const std::vector<int>& bucket, const std::unordered_map<int, uint32_t>& path_offsets) const {
        uint32_t num_indices = htonl(bucket.size());
        memcpy(current, &num_indices, sizeof(num_indices));
        current += sizeof(num_indices);
        for (int handle : bucket) {
            uint32_t offset = path_offsets.at(handle);
            uint32_t net_offset = htonl(offset);
            memcpy(current, &net_offset, sizeof(net_offset));
            current += sizeof(net_offset);
        }
        return current;
    }

    char* write_seglist(char* current, const LineSegList& seglist) const {
        cv::Point start = seglist.start_point();
        uint32_t start_x = htonl(start.x);
        uint32_t start_y = htonl(start.y);
        uint32_t num_offsets = htonl(seglist.compressed_data_size());

        memcpy(current, &start_x, sizeof(start_x));
        current += sizeof(start_x);
        memcpy(current, &start_y, sizeof(start_y));
        current += sizeof(start_y);
        memcpy(current, &num_offsets, sizeof(num_offsets));
        current += sizeof(num_offsets);
        
        memcpy(current, seglist.compressed_data(), seglist.compressed_data_size());
        current += seglist.compressed_data_size();
        return current;
    }

    const char* read_bucket(const char* current, const char* end, std::vector<int>& bucket) const {
        if (current + sizeof(uint32_t) > end) throw std::runtime_error("Invalid GridStore file: unexpected end in bucket header.");
        uint32_t num_indices = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);

        bucket.resize(num_indices);
        if (current + num_indices * sizeof(uint32_t) > end) throw std::runtime_error("Invalid GridStore file: bucket indices out of bounds.");
        for (uint32_t i = 0; i < num_indices; ++i) {
            bucket[i] = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        }
        return current;
    }

    const char* read_seglist(const char* current, const char* end, std::shared_ptr<LineSegList>& seglist) const {
        if (current + 3 * sizeof(uint32_t) > end) throw std::runtime_error("Invalid GridStore file: unexpected end in seglist header.");
        uint32_t start_x = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        uint32_t start_y = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        uint32_t num_offsets = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);

        if (current + num_offsets > end) throw std::runtime_error("Invalid GridStore file: seglist offsets out of bounds.");
        
        cv::Point start(start_x, start_y);
        const int8_t* offsets_ptr = reinterpret_cast<const int8_t*>(current);
        current += num_offsets;

        seglist = std::make_shared<LineSegList>(start, offsets_ptr, num_offsets);
        return current;
    }

    size_t get_all_buckets_size() const {
        size_t total_size = 0;
        for (const auto& bucket : grid_) {
            total_size += sizeof(uint32_t); // num_indices
            total_size += bucket.size() * sizeof(uint32_t); // handles
        }
        return total_size;
    }

    size_t get_all_seglist_size() const {
        size_t total_size = 0;
        for (const auto& seglist : storage_) {
            total_size += sizeof(uint32_t); // start.x
            total_size += sizeof(uint32_t); // start.y
            total_size += sizeof(uint32_t); // num_offsets
            total_size += seglist->compressed_data_size();
        }
        return total_size;
    }

    cv::Rect bounds_;
    int cell_size_;
    cv::Size grid_size_;
    std::vector<std::vector<int>> grid_;
    std::vector<std::shared_ptr<LineSegList>> storage_;
    bool read_only_;
    std::unique_ptr<MmappedData> mmapped_data_;
};

GridStore::GridStore(const cv::Rect& bounds, int cell_size)
    : pimpl_(std::make_unique<GridStoreImpl>(bounds, cell_size)) {}

GridStore::GridStore(const std::string& path)
    : pimpl_(std::make_unique<GridStoreImpl>(cv::Rect(), 1)) { // Use a dummy cell_size to avoid division by zero
    pimpl_->load_mmap(path);
    meta = pimpl_->meta_;
}

GridStore::~GridStore() = default;

void GridStore::add(const std::vector<cv::Point>& points) {
    pimpl_->add(points);
}

std::vector<std::shared_ptr<std::vector<cv::Point>>> GridStore::get(const cv::Rect& query_rect) const {
    return pimpl_->get(query_rect);
}

std::vector<std::shared_ptr<std::vector<cv::Point>>> GridStore::get(const cv::Point2f& center, float radius) const {
    int x = static_cast<int>(center.x - radius);
    int y = static_cast<int>(center.y - radius);
    int size = static_cast<int>(radius * 2);
    return get(cv::Rect(x, y, size, size));
}

std::vector<std::shared_ptr<std::vector<cv::Point>>> GridStore::get_all() const {
    return pimpl_->get_all();
}

cv::Size GridStore::size() const {
    return pimpl_->size();
}

size_t GridStore::get_memory_usage() const {
    return pimpl_->get_memory_usage();
}

size_t GridStore::numSegments() const {
    return pimpl_->numSegments();
}

size_t GridStore::numNonEmptyBuckets() const {
    return pimpl_->numNonEmptyBuckets();
}

void GridStore::save(const std::string& path) const {
    pimpl_->meta_ = meta;
    pimpl_->save(path);
}

void GridStore::load_mmap(const std::string& path) {
    pimpl_->load_mmap(path);
}

}
