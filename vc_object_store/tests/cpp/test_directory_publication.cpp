#include "TestSupport.hpp"
#include <vc_reader/publication.hpp>
#include <barrier>
#include <fstream>
#include <future>
#include <iostream>

namespace fs=std::filesystem;
void put(const fs::path& path,const std::string& value){std::ofstream out(path);out<<value;require(bool(out),"write test payload");}
std::string get(const fs::path& path){std::ifstream in(path);std::string value;std::getline(in,value);return value;}

int main() try
{
    TemporaryDirectory temp;
    const auto source=temp.path/"source",target=temp.path/"target";
    fs::create_directory(source);fs::create_directory(target);put(source/"new","new");
    mustFail([&]{vcr::files::publishDirectory(source,target,false);});
    require(fs::is_empty(target)&&get(source/"new")=="new","no-replace clobbered existing empty directory");
    fs::remove(target);vcr::files::publishDirectory(source,target,false);
    require(!fs::exists(source)&&get(target/"new")=="new","new directory publication failed");
    // Preserve the distinct overwrite API: merge regular files while leaving
    // unrelated siblings/subdirectories in the destination intact.
    fs::create_directory(source);put(source/"new","updated");put(target/"sibling","retain");
    fs::create_directory(target/"nested");put(target/"nested"/"child","retain");
    vcr::files::publishDirectory(source,target,true);
    require(get(target/"new")=="updated"&&get(target/"sibling")=="retain"&&
        get(target/"nested"/"child")=="retain","overwrite directory merge semantics changed");
    put(temp.path/"file-source","source");put(temp.path/"file-target","target");
    mustFail([&]{vcr::files::publishPath(temp.path/"file-source",temp.path/"file-target",false);});
    require(get(temp.path/"file-target")=="target","no-replace file clobbered target");
#ifndef _WIN32
    fs::create_symlink(temp.path/"absent",temp.path/"dangling");
    mustFail([&]{vcr::files::publishPath(temp.path/"file-source",temp.path/"dangling",false);});
    require(fs::is_symlink(fs::symlink_status(temp.path/"dangling")),"no-replace replaced dangling symlink");
#endif
#if defined(__linux__) || defined(__APPLE__)
    const auto replacement=temp.path/"replacement";fs::create_directory(replacement);put(replacement/"new-generation","fresh");
    vcr::files::publishPath(replacement,target,true);
    require(get(target/"new-generation")=="fresh"&&!fs::exists(target/"sibling")&&
        get(replacement/"sibling")=="retain","whole-directory exchange did not preserve displaced generation");
#endif
    // No shared application mutex: race a competing directory creator against
    // publication. Exactly one may succeed, even when the competitor is empty.
    for(int i=0;i<64;++i){
        const auto staged=temp.path/("race-source-"+std::to_string(i));
        const auto final=temp.path/("race-target-"+std::to_string(i));
        fs::create_directory(staged);put(staged/"value","payload");
        std::barrier start(2);
        auto producer=std::async(std::launch::async,[&]{start.arrive_and_wait();try{
            vcr::files::publishPath(staged,final,false);return true;
        }catch(const fs::filesystem_error& e){if(e.code()!=std::errc::file_exists)throw;return false;}});
        start.arrive_and_wait();const bool competitor=fs::create_directory(final);const bool published=producer.get();
        require(competitor!=published,"both competing directory publishers succeeded");
        if(competitor)require(fs::is_empty(final)&&fs::exists(staged/"value"),"publication replaced competitor directory");
        else require(get(final/"value")=="payload"&&!fs::exists(staged),"winning publication lost its payload");
    }
    std::cout<<"directory publication tests passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
