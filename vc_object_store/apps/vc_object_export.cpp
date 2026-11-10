#include "vc_object_store/objects/GeometryExport.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include <iostream>

int main(int argc, char** argv)
try {
    std::filesystem::path path, destination;
    std::string object;
    std::optional<int64_t> version;
    std::optional<std::string> generation, owner;
    bool overwrite = false, validOnly = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--db") path = next();
        else if (arg == "--uuid") object = next();
        else if (arg == "--version") version = std::stoll(next());
        else if (arg == "--generation") generation = next();
        else if (arg == "--store") owner = next();
        else if (arg == "--output") destination = next();
        else if (arg == "--overwrite") overwrite = true;
        else if (arg == "--valid-only") validOnly = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: vc_object_export --db objects.vcobj --uuid UUID --output PATH [options]\n"
                "  --version N      export a recorded object version (default: current)\n"
                "  --generation G   history generation of --version (default: current owner's)\n"
                "  --store UUID     explicit source owner; requires --version and --generation\n"
                "  --overwrite      replace the entire destination atomically\n"
                "  --valid-only     surfaces: write invalid vertices as missing, no mask.tif\n"
                "Exports one surface directory, fiber JSON, single-collection JSON, or\n"
                "single-track packed directory. The output parent must already exist.\n"
                "The object store opens read-only; original source files are not needed.\n";
            return 0;
        } else throw std::runtime_error("unknown argument: " + arg);
    }
    if (path.empty() || object.empty() || destination.empty())
        throw std::runtime_error("specify --db, --uuid and --output; see --help");
    vc::objects::OpenOptions options; options.readOnly = true;
    auto db = vc::objects::ObjectStore::open(path, options);
    if (owner && (!version || !generation)) throw std::invalid_argument("--store needs --version and --generation");
    const auto read = db.read();
    const auto exact = owner ? std::optional(vc::objects::ObjectVersion{*owner,*generation,object,*version}) : std::nullopt;
    std::optional<vc::objects::ObjectVersion> selected;
    if (version || generation || exact) {
        const auto state = vc::objects::LocalStoreReader(db, read).selectedObject(object,
            owner ? std::nullopt : version, owner ? std::nullopt : generation, exact);
        if (!state) throw std::invalid_argument("export_geometry: object version not found");
        selected = state->reference();
    }
    const auto exported = vc::objects::exportGeometry(db, read, object, destination, selected, overwrite, validOnly);
    std::cout << "Exported version " << exported.version << " (generation " << exported.generation << ") to "
              << destination << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "vc_object_export: " << error.what() << '\n';
    return 1;
}
