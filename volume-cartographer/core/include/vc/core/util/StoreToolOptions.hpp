#pragma once
// Store mode of the command-line tools: one contract for all of them.
//   --store PATH        the object store (local or a hosted replica)
//   --volume-uuid ID    stored surfaces are read in that volume's voxels, through
//                       the store's view of them (their binding or a
//                       registration), and new surfaces are bound to it; none:
//                       stored coordinates
//   --surface UUID      a stored input surface (repeatable, in order)
//   --name NAME         the name of the surface the tool creates
//   --tag TAG           a tag for the created surfaces (repeatable)
// Tools parsing with boost::program_options add() the options and read them
// from() the variables map; others take() them out of their arguments.
#include "vc/core/util/StoreSurfaces.hpp"

#include <boost/program_options.hpp>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

struct StoreToolOptions {
    std::filesystem::path store;
    std::optional<std::string> volume;
    std::vector<std::string> surfaces;
    std::optional<std::string> name;
    std::vector<std::string> tags;

    // Which of the options a tool takes besides --store and --volume-uuid:
    // stored inputs (--surface), and whether it creates surfaces (--tag) and
    // names the one it creates (--name).
    struct Takes {
        bool surfaces = true;
        bool creates = true;
        bool named = false;
    };

    static void add(boost::program_options::options_description& desc, const std::string& storeHelp, Takes takes)
    {
        namespace po = boost::program_options;
        desc.add_options()("store", po::value<std::string>(), storeHelp.c_str())(
            "volume-uuid", po::value<std::string>(),
            "Volume whose voxels stored surfaces are read in, through the store's view of them, and that new "
            "surfaces are bound to (store mode)");
        if (takes.surfaces)
            desc.add_options()("surface", po::value<std::vector<std::string>>()->composing(),
                               "Stored input surface UUID (store mode, repeatable)");
        if (takes.creates)
            desc.add_options()("tag", po::value<std::vector<std::string>>()->composing(),
                               "Tag for the created surfaces (store mode, repeatable)");
        if (takes.named)
            desc.add_options()("name", po::value<std::string>(), "Name of the created surface (store mode)");
    }

    // The store options given; nullopt without --store. Other store options
    // without --store throw.
    static std::optional<StoreToolOptions> from(const boost::program_options::variables_map& vm)
    {
        StoreToolOptions options;
        if (vm.count("store")) options.store = vm["store"].as<std::string>();
        if (vm.count("volume-uuid")) options.volume = vm["volume-uuid"].as<std::string>();
        if (vm.count("surface")) options.surfaces = vm["surface"].as<std::vector<std::string>>();
        if (vm.count("name")) options.name = vm["name"].as<std::string>();
        if (vm.count("tag")) options.tags = vm["tag"].as<std::vector<std::string>>();
        return checked(std::move(options));
    }

    // The same for tools without boost::program_options: each option, as
    // `--option=VALUE` or `--option VALUE`, is removed from `args` (the
    // arguments after the program name); the rest keep their order.
    static std::optional<StoreToolOptions> take(std::vector<std::string>& args, Takes takes)
    {
        StoreToolOptions options;
        std::vector<std::string> rest;
        for (size_t i = 0; i < args.size(); ++i) {
            const auto& arg = args[i];
            const auto eq = arg.find('=');
            const auto key = arg.substr(0, eq);
            std::vector<std::string>* repeated = nullptr;
            const bool known = key == "--store" || key == "--volume-uuid" || (key == "--surface" && takes.surfaces) ||
                               (key == "--tag" && takes.creates) || (key == "--name" && takes.named);
            if (!known) {
                rest.push_back(arg);
                continue;
            }
            if (key == "--surface") repeated = &options.surfaces;
            if (key == "--tag") repeated = &options.tags;
            std::string value;
            if (eq != std::string::npos) value = arg.substr(eq + 1);
            else if (i + 1 < args.size()) value = args[++i];
            else throw std::invalid_argument(key + " needs a value");
            if (value.empty()) throw std::invalid_argument(key + " needs a value");
            if (repeated) repeated->push_back(value);
            else if (key == "--store") options.store = value;
            else if (key == "--volume-uuid") options.volume = value;
            else options.name = value;
        }
        args = std::move(rest);
        return checked(std::move(options));
    }

private:
    static std::optional<StoreToolOptions> checked(StoreToolOptions options)
    {
        if (!options.store.empty()) return options;
        if (options.volume || !options.surfaces.empty() || options.name || !options.tags.empty())
            throw std::runtime_error("--volume-uuid, --surface, --name and --tag need --store");
        return std::nullopt;
    }
};
