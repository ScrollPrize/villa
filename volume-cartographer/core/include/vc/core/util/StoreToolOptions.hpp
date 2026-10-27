#pragma once
// Store mode of the command-line tools that parse options with
// boost::program_options: --store, --volume-uuid and --tag.
#include "vc/core/util/StoreSurfaces.hpp"

#include <boost/program_options.hpp>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

struct StoreToolOptions {
    std::filesystem::path store;
    // The volume stored surfaces are read in (through the store's view of them:
    // their binding or a registration) and new surfaces are bound to; none:
    // stored coordinates.
    std::optional<std::string> volume;
    std::vector<std::string> tags;

    // --store (described by `storeHelp`), --volume-uuid and, for tools that
    // create surfaces, --tag.
    static void add(boost::program_options::options_description& desc, const std::string& storeHelp,
                    bool creates = true)
    {
        namespace po = boost::program_options;
        desc.add_options()("store", po::value<std::string>(), storeHelp.c_str())(
            "volume-uuid", po::value<std::string>(),
            "Volume whose voxels stored surfaces are read in, through the store's view of them, and that new "
            "surfaces are bound to (store mode)");
        if (creates)
            desc.add_options()("tag", po::value<std::vector<std::string>>()->composing(),
                               "Tag for the created surface (store mode, repeatable)");
    }

    // The store options given; nullopt without --store. --volume-uuid or --tag
    // without --store throws.
    static std::optional<StoreToolOptions> from(const boost::program_options::variables_map& vm)
    {
        if (!vm.count("store")) {
            if (vm.count("volume-uuid") || vm.count("tag"))
                throw std::runtime_error("--volume-uuid and --tag need --store");
            return std::nullopt;
        }
        StoreToolOptions options;
        options.store = vm["store"].as<std::string>();
        if (vm.count("volume-uuid")) options.volume = vm["volume-uuid"].as<std::string>();
        if (vm.count("tag")) options.tags = vm["tag"].as<std::vector<std::string>>();
        return options;
    }
};
