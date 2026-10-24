#pragma once
#include "verified_chunk_read.hpp"
#include <cstdlib>
#include <mutex>
#include <memory>
#include <cstdio>
extern "C" unsigned char* SHA256(const unsigned char*,std::size_t,unsigned char*);
namespace selected_read {
// Opt-in integrity evidence is mandatory. A decoder may catch C++ exceptions
// and substitute zeroes, so evidence infrastructure failures must not unwind.
[[noreturn]] inline void evidence_fatal() noexcept {
 std::fputs("selected-read mandatory evidence failure\n",stderr);
 std::fflush(stderr);std::_Exit(86);
}
inline std::string sha256(const Bytes& b) {
 unsigned char digest[32];SHA256(reinterpret_cast<const unsigned char*>(b.data()),b.size(),digest);
 const char* hex="0123456789abcdef";std::string out;
 for(auto x:digest){out+=hex[x>>4];out+=hex[x&15];}return out;
}
class Runtime {
 std::unique_ptr<Guard> guard_;
 std::filesystem::path ledger_;
 std::mutex ledger_mutex_;
 std::size_t events_=0;
 void event(const std::string& status,const std::string& key,std::size_t size) {
  try {
  std::lock_guard<std::mutex> lock(ledger_mutex_);
  if(++events_>65536)throw std::runtime_error("bounded verification ledger exceeded");
  utils::Json e=utils::Json::object();e["status"]=status;e["key"]=key;e["bytes"]=uint64_t(size);
  std::ofstream f(ledger_,std::ios::app);f<<e.dump()<<'\n';f.flush();
  if(!f)throw std::runtime_error("verification ledger write");
  } catch(...) { evidence_fatal(); }
 }
public:
 Runtime() {
  const char* manifest=std::getenv("VC_VERIFIED_CT_MANIFEST");
  const char* expected=std::getenv("VC_VERIFIED_CT_MANIFEST_SHA256");
  const char* ledger=std::getenv("VC_VERIFIED_CT_LEDGER");
  if(!manifest && !expected && !ledger) return; // Stock mode only when every opt-in variable is absent.
  if(!manifest || !expected || !ledger) evidence_fatal();
  std::ifstream f(manifest,std::ios::binary|std::ios::ate);
  auto size=f.tellg();if(!f || size<0 || size>32*1024*1024)throw std::runtime_error("bounded manifest");
  Bytes bytes(static_cast<std::size_t>(size));f.seekg(0);
  if(size && !f.read(reinterpret_cast<char*>(bytes.data()),size))throw std::runtime_error("manifest read");
  if(sha256(bytes)!=expected)throw std::runtime_error("manifest SHA256");
  auto j=utils::Json::parse(std::string(reinterpret_cast<char*>(bytes.data()),bytes.size()));
  if(j.at("schema").get_string()!="verified-ct-read-manifest-v1")throw std::runtime_error("manifest schema");
  std::map<std::string,Expected> records;
  for(const auto& r:j.at("objects")) {
   auto key=r.at("key").get_string();
   if(!records.emplace(key,Expected{r.at("absent").get_bool(),r.at("bytes").get_size_t(),r.at("sha256").get_string()}).second)
    throw std::runtime_error("duplicate CT manifest key");
  }
  if(records.empty())throw std::runtime_error("empty CT manifest");
  ledger_=ledger;
  guard_=std::make_unique<Guard>(j.at("volume_root").get_string(),std::move(records),sha256,j.at("max_chunk_bytes").get_size_t());
  event("INITIALIZED",expected,0);
 }
 bool protects(const std::filesystem::path& root)const{return guard_ && guard_->protects(root);}
 void reject_store() {
  if(guard_){try{event("REJECTED_STORE","alternate-store",0);}catch(...){};throw std::runtime_error("alternate store forbidden in guarded qualification");}
 }
 std::optional<Bytes> read(const std::filesystem::path& root,const std::string& key,bool shard,bool store) {
  try {
   auto bytes=guard_->read(root,key,shard,store);
   event(bytes?"VERIFIED_PRESENT":"EXPLICIT_ABSENT",key,bytes?bytes->size():0);
   return bytes;
  } catch(...) {
   // Preserve the primary verification failure if evidence storage also fails.
   try{event("REJECTED",key,0);}catch(...){}
   throw;
  }
 }
 void reject(const std::filesystem::path& root) {
  if(protects(root)){try{event("REJECTED_BYPASS",root.string(),0);}catch(...){};guard_->reject_bypass(root);}
 }
};
inline Runtime& runtime(){
 static Runtime value=[]() -> Runtime {
  // A missing env retains stock behavior; an applied but invalid opt-in cannot
  // silently fall back after a renderer catches initialization exceptions.
  try {return Runtime();} catch(...) {evidence_fatal();}
 }();
 return value;
}
}
