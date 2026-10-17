#include "vc/core/graph/GeometryExport.hpp"
#include <iostream>

int main(int argc, char** argv)
try {
    std::filesystem::path path, destination;
    std::string object;
    std::optional<std::string> revision;
    bool overwrite = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--db") path = next();
        else if (arg == "--uuid") object = next();
        else if (arg == "--revision") revision = next();
        else if (arg == "--output") destination = next();
        else if (arg == "--overwrite") overwrite = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: vc_graph_export --db graph.vcgr --uuid UUID --output PATH [options]\n"
                "  --revision UUID  export a historical revision (default: current)\n"
                "  --overwrite      replace the entire destination atomically\n"
                "Exports one patch directory, fiber JSON, single-collection JSON, or\n"
                "single-track packed directory. The output parent must already exist.\n"
                "The graph opens read-only; original source files are not needed.\n";
            return 0;
        } else throw std::runtime_error("unknown argument: " + arg);
    }
    if (path.empty() || object.empty() || destination.empty())
        throw std::runtime_error("specify --db, --uuid and --output; see --help");
    vc::graph::OpenOptions options; options.readOnly = true;
    auto db = vc::graph::GraphDb::open(path, options);
    const auto exported = vc::graph::exportGeometry(db, object, destination, revision, overwrite);
    std::cout << "Exported revision " << exported << " to " << destination << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "vc_graph_export: " << error.what() << '\n';
    return 1;
}
