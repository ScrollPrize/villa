#pragma once
// Prospective guard for utils::ZarrArray::read_raw, NOT a universal I/O hook.
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace selected_read {
using Bytes = std::vector<std::byte>;
struct Expected { bool absent; std::size_t size; std::string sha256; };
class Guard {
 static constexpr std::size_t kMaxVerifiedChunkBytes = 32u * 1024u * 1024u;
 std::filesystem::path volume_, group_;
 std::map<std::string, Expected> manifest_;
 std::function<std::string(const Bytes&)> digest_;
 std::size_t max_bytes_;
 [[noreturn]] static void fail(const char* why) { throw std::runtime_error(why); }
public:
 Guard(std::filesystem::path volume, std::map<std::string, Expected> manifest,
       std::function<std::string(const Bytes&)> digest, std::size_t max_bytes)
 : volume_(std::filesystem::canonical(volume)), group_(volume_/"0"),
   manifest_(std::move(manifest)), digest_(std::move(digest)), max_bytes_(max_bytes) {
  // The retained CT manifest uses 2 MiB chunks. A 32 MiB hard ceiling leaves
  // 16x headroom while keeping malformed manifests from requesting unbounded
  // per-read allocations. The schema is for this guarded CT path, not arbitrary
  // user-selected Zarr chunk sizes.
  if (!max_bytes_ || max_bytes_ > kMaxVerifiedChunkBytes || !digest_) fail("guard configuration");
  for (const auto& [k,v]:manifest_) {
   std::filesystem::path p(k);
   if (k.empty() || p.is_absolute() || p.lexically_normal()!=p) fail("manifest key");
   for (auto& part:p) if (part==".." || part==".") fail("manifest escape");
   if (v.absent ? (v.size!=0 || !v.sha256.empty()) :
       (v.size>max_bytes_ || v.sha256.size()!=64 ||
        v.sha256.find_first_not_of("0123456789abcdef")!=std::string::npos)) fail("manifest value");
  }
 }
 bool protects(const std::filesystem::path& root) const {
  auto r=std::filesystem::weakly_canonical(root);
  auto rel=r.lexically_relative(volume_);
  return r==volume_ || (!rel.empty() && *rel.begin()!="..");
 }
 // Call this on EVERY read_raw before choosing store_/filesystem branches.
 // Returns the exact compressed byte vector subsequently given to the decoder.
 std::optional<Bytes> read(const std::filesystem::path& root, const std::string& key,
                           bool sharded=false, bool alternate_store=false) const {
  if (!protects(root)) fail("guard called outside protected volume");
  if (sharded || alternate_store) fail("unsupported protected storage bypass");
  if (std::filesystem::weakly_canonical(root)!=group_) fail("unapproved CT level");
  std::filesystem::path relative(key);
  if (key.empty() || relative.is_absolute() || relative.lexically_normal()!=relative) fail("read key");
  for (auto& part:relative) if (part==".." || part==".") fail("read escape");
  auto row=manifest_.find(key);
  if (row==manifest_.end()) fail("unlisted CT read");
  auto file=group_/relative;
  // Reject symlink components observed before open. The caller must keep the
  // input tree immutable for the guarded operation; the digest authenticates
  // the exact owned bytes returned even if the filesystem path is later changed.
  auto cursor=group_;
  for (auto& part:relative) { cursor/=part; if (std::filesystem::is_symlink(std::filesystem::symlink_status(cursor))) fail("chunk symlink"); }
  const auto& expected=row->second;
  if (expected.absent) {
   if (std::filesystem::exists(file)) fail("source-absent chunk appeared");
   return std::nullopt;
  }
  if (!std::filesystem::is_regular_file(file)) fail("present chunk missing/nonregular");
  if (std::filesystem::file_size(file)!=expected.size) fail("chunk size mismatch");
  std::ifstream stream(file,std::ios::binary);
  if (!stream) fail("chunk open");
  Bytes bytes(expected.size);
  if (expected.size && !stream.read(reinterpret_cast<char*>(bytes.data()),bytes.size())) fail("chunk short read");
  char extra;
  if (stream.get(extra) || !stream.eof()) fail("chunk changed/error during read");
  if (digest_(bytes)!=expected.sha256) fail("chunk SHA256 mismatch");
  return bytes; // immutable owned snapshot; no hash-then-reopen window.
 }
 // Must be called at entry of mmap/shard/alternate-store APIs, before any I/O.
 void reject_bypass(const std::filesystem::path& root) const {
  if (protects(root)) fail("protected bypass API forbidden");
 }
};
}
