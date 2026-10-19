#pragma once
#include "../core.hpp"
namespace vcr::augment::cuda {
struct DeviceInput { void* data; Shape shape; char kind; std::size_t bytes; int device; };
std::shared_ptr<void> begin_device(const std::map<std::string,DeviceInput>&, Sample&, bool inplace, bool async = false,
                                   const std::map<std::string,DeviceInput>& destinations = {});
std::map<std::string,DeviceInput> device_outputs(void*);
std::map<std::string,std::vector<DeviceInput>> device_pyramids(void*);
void copy_batch_output(void*, const std::vector<DeviceInput>&);
// Marks a session's arrays as shared through DLPack (destruction then syncs).
void export_outputs(void*);
// Raises deferred device checks; the caller first ensures completion.
void check_device(void*);
// Download all sessions' status flags with one stream synchronization.
void check_devices(const std::vector<std::shared_ptr<void>>&);
// Frees (stream-ordered) a session's arrays once copied elsewhere.
void release_outputs(void*, const std::vector<std::string>& keys);
void finish_device(void*);
}
