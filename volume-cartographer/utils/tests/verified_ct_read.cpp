#include "utils/zarr.hpp"
#include <iostream>
int main(int argc,char**argv) {
 if(argc!=3)return 2;
 try {
  std::string mode=argv[2];
  if(mode=="store" || mode=="store_registry" || mode=="store_stock" || mode=="store_registry_stock") {
   auto store=std::make_shared<utils::FileSystemStore>(std::filesystem::path(argv[1]));
   auto array = (mode=="store_registry" || mode=="store_registry_stock")
       ? utils::ZarrArray::open(store,"0",utils::ZarrArray::CodecRegistry{})
       : utils::ZarrArray::open(store,"0",utils::ZarrArray::Codec{});
   if(mode=="store_stock" || mode=="store_registry_stock") {
    std::array<std::size_t,3> index{0,0,0};
    auto bytes=array.read_chunk_encoded(index);
    if(!bytes || std::string(reinterpret_cast<const char*>(bytes->data()),bytes->size())!="abc") return 4;
    std::cout<<"abc\n";return 0;
   }
   std::cerr<<"unexpected Store open accepted\n";return 3;
  }
  auto array=utils::ZarrArray::open(std::filesystem::path(argv[1]),utils::ZarrArray::Codec{});
  std::array<std::size_t,3> index{0,0,0};
  if(mode=="shard") {array.read_whole_shard(index);std::cerr<<"unexpected bypass accepted\n";return 3;}
  auto bytes=mode=="storage"?array.read_storage_object(index):array.read_chunk_encoded(index);
  if(mode=="ledgerfail") {
   auto ledger=std::filesystem::path(std::getenv("VC_VERIFIED_CT_LEDGER"));
   std::filesystem::remove(ledger);std::filesystem::create_directory(ledger);
   array.read_chunk_encoded(index);return 3;
  }
  if(mode=="eventlimit") {
   for(int n=0;n<65536;++n)array.read_chunk_encoded(index);
   return 3;
  }
  if(!bytes){std::cout<<"EXPLICIT_ABSENT\n";return 0;}
  std::cout<<std::string(reinterpret_cast<const char*>(bytes->data()),bytes->size())<<'\n';
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return std::string(argv[2])=="swallow"?0:7;}
}
