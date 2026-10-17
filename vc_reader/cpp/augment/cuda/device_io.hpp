#pragma once
#include "../core.hpp"
namespace vcr::augment::cuda {
struct DeviceInput { void* data; Shape shape; char kind; std::size_t bytes; int device; };
std::shared_ptr<void> begin_device(const std::map<std::string,DeviceInput>&, Sample&, bool inplace, bool async = false);
std::map<std::string,DeviceInput> device_outputs(void*);
// Marks a session's arrays as shared through DLPack (destruction then syncs).
void export_outputs(void*);
// Raises deferred device checks; the caller first ensures completion.
void check_device(void*);
void finish_device(void*);
}
