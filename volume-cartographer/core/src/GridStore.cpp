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
constexpr uint32_t GRIDSTORE_VERSION = 3;
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
        std::vector<std::shared_ptr<std::vector<cv::Point>>> result;
        cv::Rect clamped_rect = query_rect & bounds_;

        cv::Point start = (clamped_rect.tl() - bounds_.tl()) / cell_size_;
        cv::Point end = (clamped_rect.br() - bounds_.tl()) / cell_size_;

        if (read_only_) {
            std::unordered_set<size_t> offsets;
            for (int y = start.y; y <= end.y; ++y) {
                for (int x = start.x; x <= end.x; ++x) {
                    int index = y * grid_size_.width + x;
                    auto bucket_ptr = get_bucket_offsets(index);
                    if (bucket_ptr && !bucket_ptr->empty()) {
                        offsets.insert(bucket_ptr->begin(), bucket_ptr->end());
                    }
                }
            }
            result.reserve(offsets.size());
            for (size_t offset : offsets) {
                result.push_back(get_seglist_from_offset(offset)->get());
            }
        } else {
            std::unordered_set<int> handles;
            for (int y = start.y; y <= end.y; ++y) {
                for (int x = start.x; x <= end.x; ++x) {
                    int index = y * grid_size_.width + x;
                    if (index >= 0 && index < grid_.size()) {
                        handles.insert(grid_[index].begin(), grid_[index].end());
                    }
                }
            }
            result.reserve(handles.size());
            for (int handle : handles) {
                result.push_back(storage_[handle]->get());
            }
        }
        return result;
    }

    std::vector<std::shared_ptr<std::vector<cv::Point>>> get_all() const {
        std::vector<std::shared_ptr<std::vector<cv::Point>>> result;
        if (read_only_) {
            std::unordered_set<size_t> all_offsets;
            size_t num_buckets = grid_size_.width * grid_size_.height;
            for(size_t i = 0; i < num_buckets; ++i) {
                auto bucket_ptr = get_bucket_offsets(i);
                if (bucket_ptr) {
                    all_offsets.insert(bucket_ptr->begin(), bucket_ptr->end());
                }
            }
            result.reserve(all_offsets.size());
            for (const auto& offset : all_offsets) {
                result.push_back(get_seglist_from_offset(offset)->get());
            }
        } else {
            result.reserve(storage_.size());
            for (const auto& seg_list : storage_) {
                result.push_back(seg_list->get());
            }
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
        size_t storage_memory = 0;
        if (read_only_) {
            // In read-only mode, storage is just offsets, which are part of grid_offsets_
            storage_memory = 0;
        } else {
            storage_memory = storage_.capacity() * sizeof(std::shared_ptr<LineSegList>);
            for (const auto& seg : storage_) {
                storage_memory += seg->get_memory_usage();
            }
        }
        return grid_memory + storage_memory;
    }

    size_t numSegments() const {
        if (read_only_) {
            std::unordered_set<size_t> all_offsets;
            size_t num_buckets = grid_size_.width * grid_size_.height;
            for(size_t i = 0; i < num_buckets; ++i) {
                auto bucket_ptr = get_bucket_offsets(i);
                if (bucket_ptr) {
                    all_offsets.insert(bucket_ptr->begin(), bucket_ptr->end());
                }
            }
            size_t count = 0;
            for (const auto& offset : all_offsets) {
                auto seg_list = get_seglist_from_offset(offset);
                if (seg_list->num_points() > 0) {
                    count += seg_list->num_points() - 1;
                }
            }
            return count;
        } else {
            size_t count = 0;
            for (const auto& seg_list : storage_) {
                if (seg_list->num_points() > 0) {
                    count += seg_list->num_points() - 1;
                }
            }
            return count;
        }
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
        if (read_only_) {
            throw std::runtime_error("Cannot save a read-only GridStore. Load the data into a new, writable GridStore instance first.");
        }

        std::string meta_str = meta_.dump();
        size_t header_size = 13 * sizeof(uint32_t);

        // 1. Serialize all paths and record their offsets
        std::vector<char> paths_buffer;
        std::unordered_map<int, uint32_t> handle_to_offset;
        for (size_t i = 0; i < storage_.size(); ++i) {
            handle_to_offset[i] = paths_buffer.size();
            const auto& seglist = storage_[i];
            size_t current_size = paths_buffer.size();
            paths_buffer.resize(current_size + 3 * sizeof(uint32_t) + seglist->compressed_data_size());
            write_seglist(paths_buffer.data() + current_size, *seglist);
        }
        size_t paths_size = paths_buffer.size();

        // 2. Create bucket structures for v3
        std::vector<uint32_t> bucket_path_indices;
        bucket_path_indices.reserve(grid_.size() + 1);
        std::vector<uint32_t> bucket_paths_flat;
        uint32_t current_path_idx_counter = 0;
        for (const auto& bucket : grid_) {
            bucket_path_indices.push_back(current_path_idx_counter);
            for (int handle : bucket) {
                bucket_paths_flat.push_back(handle_to_offset.at(handle));
            }
            current_path_idx_counter += bucket.size();
        }
        bucket_path_indices.push_back(current_path_idx_counter);

        size_t bucket_indices_size = bucket_path_indices.size() * sizeof(uint32_t);
        size_t bucket_paths_flat_size = bucket_paths_flat.size() * sizeof(uint32_t);
        size_t meta_size = meta_str.size();
        size_t total_size = header_size + bucket_indices_size + bucket_paths_flat_size + paths_size + meta_size;

        std::vector<char> buffer(total_size);
        char* current_ptr = buffer.data();

        // 3. Define offsets for v3
        uint32_t bucket_indices_offset = header_size;
        uint32_t bucket_paths_offset = bucket_indices_offset + bucket_indices_size;
        uint32_t paths_offset = bucket_paths_offset + bucket_paths_flat_size;
        uint32_t json_meta_offset = paths_offset + paths_size;

        // 4. Write header
        uint32_t magic = htonl(GRIDSTORE_MAGIC);
        uint32_t version = htonl(GRIDSTORE_VERSION);
        uint32_t bounds_x = htonl(bounds_.x);
        uint32_t bounds_y = htonl(bounds_.y);
        uint32_t bounds_width = htonl(bounds_.width);
        uint32_t bounds_height = htonl(bounds_.height);
        uint32_t cell_size = htonl(cell_size_);
        uint32_t num_buckets = htonl(grid_.size());
        uint32_t num_paths = htonl(storage_.size());
        uint32_t json_meta_size = htonl(meta_size);

        memcpy(current_ptr, &magic, sizeof(magic)); current_ptr += sizeof(magic);
        memcpy(current_ptr, &version, sizeof(version)); current_ptr += sizeof(version);
        memcpy(current_ptr, &bounds_x, sizeof(bounds_x)); current_ptr += sizeof(bounds_x);
        memcpy(current_ptr, &bounds_y, sizeof(bounds_y)); current_ptr += sizeof(bounds_y);
        memcpy(current_ptr, &bounds_width, sizeof(bounds_width)); current_ptr += sizeof(bounds_width);
        memcpy(current_ptr, &bounds_height, sizeof(bounds_height)); current_ptr += sizeof(bounds_height);
        memcpy(current_ptr, &cell_size, sizeof(cell_size)); current_ptr += sizeof(cell_size);
        memcpy(current_ptr, &num_buckets, sizeof(num_buckets)); current_ptr += sizeof(num_buckets);
        memcpy(current_ptr, &num_paths, sizeof(num_paths)); current_ptr += sizeof(num_paths);
        uint32_t net_bucket_indices_offset = htonl(bucket_indices_offset);
        memcpy(current_ptr, &net_bucket_indices_offset, sizeof(net_bucket_indices_offset)); current_ptr += sizeof(net_bucket_indices_offset);
        uint32_t net_paths_offset = htonl(paths_offset);
        memcpy(current_ptr, &net_paths_offset, sizeof(net_paths_offset)); current_ptr += sizeof(net_paths_offset);
        uint32_t net_json_meta_offset = htonl(json_meta_offset);
        memcpy(current_ptr, &net_json_meta_offset, sizeof(net_json_meta_offset)); current_ptr += sizeof(net_json_meta_offset);
        memcpy(current_ptr, &json_meta_size, sizeof(json_meta_size)); current_ptr += sizeof(json_meta_size);

        // 5. Write v3 bucket structures
        char* bucket_indices_start = buffer.data() + bucket_indices_offset;
        for (uint32_t idx : bucket_path_indices) {
            uint32_t net_idx = htonl(idx);
            memcpy(bucket_indices_start, &net_idx, sizeof(net_idx));
            bucket_indices_start += sizeof(net_idx);
        }

        char* bucket_paths_start = buffer.data() + bucket_paths_offset;
        for (uint32_t offset : bucket_paths_flat) {
            uint32_t net_offset = htonl(offset);
            memcpy(bucket_paths_start, &net_offset, sizeof(net_offset));
            bucket_paths_start += sizeof(net_offset);
        }

        // 6. Write paths data
        memcpy(buffer.data() + paths_offset, paths_buffer.data(), paths_size);

        // 7. Write metadata
        memcpy(buffer.data() + json_meta_offset, meta_str.data(), meta_size);

        // 8. Write buffer to file
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for writing: " + path);
        }
        file.write(buffer.data(), buffer.size());
        file.close();

        // 9. In-line verification by reloading the saved file
        {
            GridStore reloaded_store(path);
            auto original_paths = this->get_all();
            auto reloaded_paths = reloaded_store.get_all();

            if (original_paths.size() != reloaded_paths.size()) {
                throw std::runtime_error("Verification failed: path count mismatch. Original: " + std::to_string(original_paths.size()) + ", Reloaded: " + std::to_string(reloaded_paths.size()));
            }

            auto points_to_string_set = [](const std::vector<std::shared_ptr<std::vector<cv::Point>>>& paths) {
                std::multiset<std::string> string_set;
                for (const auto& path_ptr : paths) {
                    std::stringstream ss;
                    for (const auto& p : *path_ptr) {
                        ss << p.x << "," << p.y << ";";
                    }
                    string_set.insert(ss.str());
                }
                return string_set;
            };

            auto original_set = points_to_string_set(original_paths);
            auto reloaded_set = points_to_string_set(reloaded_paths);

            if (original_set != reloaded_set) {
                 throw std::runtime_error("Verification failed: path data mismatch after reload.");
            }
        }
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
            throw std::runtime_error("GridStore file version " + std::to_string(version) + " is newer than supported version " + std::to_string(GRIDSTORE_VERSION) + ".");
        }
        if (version < 1) {
             throw std::runtime_error("GridStore file version " + std::to_string(version) + " is older than minimum supported version 1.");
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
                throw std::runtime_error("Invalid GridStore v2+ file: too small for extended header.");
            }
            json_meta_offset = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
            json_meta_size = ntohl(*reinterpret_cast<const uint32_t*>(current)); current += sizeof(uint32_t);
        }

        grid_size_ = cv::Size(
            (bounds_.width + cell_size_ - 1) / cell_size_,
            (bounds_.height + cell_size_ - 1) / cell_size_
        );

        paths_offset_in_file_ = paths_offset;
        buckets_offset_in_file_ = buckets_offset;
        file_version_ = version;

        if (version <= 2) {
            // Legacy v1/v2 loading: Read bucket descriptors
            grid_bucket_descriptors_.resize(num_buckets);
            const char* buckets_start = static_cast<const char*>(mmapped_data_->data) + buckets_offset;
            const char* current_bucket_ptr = buckets_start;
            for (uint32_t i = 0; i < num_buckets; ++i) {
                if (current_bucket_ptr + sizeof(uint32_t) > end) throw std::runtime_error("Invalid GridStore file: unexpected end in bucket header.");
                uint32_t num_indices = ntohl(*reinterpret_cast<const uint32_t*>(current_bucket_ptr));
                grid_bucket_descriptors_[i] = { (size_t)(current_bucket_ptr - buckets_start), num_indices };
                current_bucket_ptr += sizeof(uint32_t) + num_indices * sizeof(uint32_t);
                if (current_bucket_ptr > end) throw std::runtime_error("Invalid GridStore file: bucket data out of bounds during descriptor reading.");
            }
        }
        // For v3, we don't need to read descriptors. The bucket indices are read on-demand.

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

    const char* read_seglist_header_and_data(const char* current, const char* end, std::shared_ptr<LineSegList>& seglist) const {
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

    std::shared_ptr<LineSegList> get_seglist_from_offset(size_t offset) const {
        const char* paths_start = static_cast<const char*>(mmapped_data_->data) + paths_offset_in_file_;
        const char* end = static_cast<const char*>(mmapped_data_->data) + mmapped_data_->size;
        std::shared_ptr<LineSegList> seglist;
        read_seglist_header_and_data(paths_start + offset, end, seglist);
        return seglist;
    }

    std::shared_ptr<std::vector<size_t>> get_bucket_offsets(int index) const {
        // Acquire lock to check for existence
        bucket_mutex_.lock();
        auto it = grid_offsets_.find(index);
        if (it != grid_offsets_.end()) {
            auto ptr = it->second;
            bucket_mutex_.unlock();
            return ptr;
        }
        // If not found, release the lock before loading
        bucket_mutex_.unlock();

        // Perform expensive I/O without holding the lock
        auto bucket_ptr = std::make_shared<std::vector<size_t>>();
        if (file_version_ <= 2) {
            if (index >= 0 && index < grid_bucket_descriptors_.size()) {
                const auto& descriptor = grid_bucket_descriptors_[index];
                if (descriptor.second > 0) {
                    const char* buckets_start = static_cast<const char*>(mmapped_data_->data) + buckets_offset_in_file_;
                    const char* current_bucket_ptr = buckets_start + descriptor.first;
                    
                    current_bucket_ptr += sizeof(uint32_t); // Skip num_indices
                    
                    bucket_ptr->reserve(descriptor.second);
                    for (uint32_t j = 0; j < descriptor.second; ++j) {
                        uint32_t path_offset = ntohl(*reinterpret_cast<const uint32_t*>(current_bucket_ptr)); current_bucket_ptr += sizeof(uint32_t);
                        bucket_ptr->push_back(path_offset);
                    }
                }
            }
        } else { // Version 3
            const char* data_start = static_cast<const char*>(mmapped_data_->data);
            const uint32_t* bucket_indices = reinterpret_cast<const uint32_t*>(data_start + buckets_offset_in_file_);
            
            uint32_t start_idx = ntohl(bucket_indices[index]);
            uint32_t end_idx = ntohl(bucket_indices[index + 1]);
            uint32_t count = end_idx - start_idx;

            if (count > 0) {
                const uint32_t* header_ptr = reinterpret_cast<const uint32_t*>(data_start);
                uint32_t num_buckets = ntohl(header_ptr[7]);
                // In V3, header[8] is the total number of paths in the storage, not the number of paths in all buckets combined.
                // The total number of path offsets in the flat list is given by the last element of the bucket_indices array.
                const uint32_t* bucket_indices = reinterpret_cast<const uint32_t*>(data_start + buckets_offset_in_file_);
                uint32_t total_path_indices = ntohl(bucket_indices[num_buckets]);

                if (start_idx + count > total_path_indices) {
                    throw std::runtime_error("Bucket data is out of bounds of the flat path offset list.");
                }

                size_t bucket_indices_size = (num_buckets + 1) * sizeof(uint32_t);
                const uint32_t* path_offsets_flat = reinterpret_cast<const uint32_t*>(data_start + buckets_offset_in_file_ + bucket_indices_size);

                bucket_ptr->reserve(count);
                for (uint32_t i = 0; i < count; ++i) {
                    bucket_ptr->push_back(ntohl(path_offsets_flat[start_idx + i]));
                }
            }
        }
        
        // Re-acquire lock to safely insert the new bucket
        std::lock_guard<std::mutex> lock(bucket_mutex_);
        it = grid_offsets_.find(index);
        if (it != grid_offsets_.end()) {
            // Another thread created it. Use the existing one.
            return it->second;
        } else {
            // We are the first. Insert our loaded bucket.
            grid_offsets_.emplace(index, bucket_ptr);
            return bucket_ptr;
        }
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
    mutable std::unordered_map<int, std::shared_ptr<std::vector<size_t>>> grid_offsets_;
    std::vector<std::pair<size_t, size_t>> grid_bucket_descriptors_;
    std::vector<std::shared_ptr<LineSegList>> storage_;
    bool read_only_;
    uint32_t file_version_ = 0;
    uint32_t paths_offset_in_file_;
    uint32_t buckets_offset_in_file_;
    std::unique_ptr<MmappedData> mmapped_data_;
    mutable std::mutex bucket_mutex_;
    mutable std::mutex seglist_mutex_;
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
