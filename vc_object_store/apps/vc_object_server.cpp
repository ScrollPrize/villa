// Host one object store for clients without filesystem access to it.
#include "vc_object_store/server/Server.hpp"

#include <iostream>
#include <string>

namespace {
int usage()
{
    std::cerr << "usage: vc_object_server STORE.vcobj [--listen ADDRESS] [--port N] [--read-workers N]\n"
                 "    [--max-payload-mib N] [--max-staged-mib N] [--staging-dir DIR]\n"
                 "Serves STORE over HTTP until SIGINT/SIGTERM. There is no authentication: the\n"
                 "default listen address is 127.0.0.1 (reach it through an SSH tunnel); use a\n"
                 "trusted private network address only. Standalone writers must be closed first.\n"
                 "Uploads are staged in DIR (default STORE.vcobj.staging); staged payloads are dropped at startup.\n";
    return 2;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || std::string(argv[1]) == "--help") return usage();
    vc::objects::server::ServerOptions options;
    options.port = 8750;
    try {
        for (int i = 2; i < argc; ++i) {
            const std::string flag = argv[i];
            if (i + 1 >= argc) return usage();
            const std::string value = argv[++i];
            if (flag == "--listen") options.listenAddress = value;
            else if (flag == "--port") options.port = static_cast<uint16_t>(std::stoul(value));
            else if (flag == "--read-workers") options.readWorkers = std::stoi(value);
            else if (flag == "--max-payload-mib") options.maxPayloadBytes = std::stoull(value) << 20;
            else if (flag == "--max-staged-mib") options.maxStagedBytes = std::stoull(value) << 20;
            else if (flag == "--staging-dir") options.stagingDirectory = value;
            else return usage();
        }
        vc::objects::server::Server server(argv[1], options);
        std::cerr << "vc_object_server: serving " << argv[1] << " on " << options.listenAddress << ":" << server.port() << "\n";
        server.run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vc_object_server: " << error.what() << "\n";
        return 1;
    }
}
