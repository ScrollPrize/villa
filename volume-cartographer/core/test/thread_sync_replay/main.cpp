#include "thread_sync_replay/RenderValgrindCli.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 2 || std::string(argv[1]) != "evaluate-render") {
        std::cerr << "usage: bench_thread_sync_replay evaluate-render ...\n";
        return 2;
    }
    try {
        return vc::thread_sync_replay::renderValgrindCli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "bench_thread_sync_replay: " << error.what() << '\n';
        return 1;
    }
}
