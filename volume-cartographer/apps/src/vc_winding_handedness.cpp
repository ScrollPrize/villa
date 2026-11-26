// vc_winding_handedness: show or state the scroll's winding direction on a
// store's umbilicus.
//
//   vc_winding_handedness <store.vcobj> [--volume <id>]
//       print what the umbilicus of the volume states and who stated it
//   vc_winding_handedness <store.vcobj> [--volume <id>] --set catalog
//       look the direction up in the open-data catalog (its orientation of
//       the volume) and keep it in the store
//   vc_winding_handedness <store.vcobj> [--volume <id>] --set +1|-1
//       state it by hand, for a volume the catalog does not orient
//
// +1: the winding index grows with atan2(dy, dx) about the umbilicus; -1:
// against it. --volume may be left out when the store holds one umbilicus.
#include <vc_object_store/objects/ObjectStore.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/VolumeUmbilicus.hpp>
#include <vc_reader/catalog.hpp>

#include <nlohmann/json.hpp>

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;

std::string declaredVolume(const vc::objects::ObjectRecord& record)
{
    const auto meta = Json::parse(record.meta.empty() ? std::string("{}") : record.meta, nullptr, false);
    if (!meta.is_object()) return {};
    const auto coordinates = meta.find("coordinates");
    if (coordinates == meta.end() || !coordinates->is_object()) return {};
    return coordinates->value("volume_uuid", std::string{});
}

// The catalog's direction for a volume id, with the catalog's name of the volume.
std::pair<int, std::string> catalogHandedness(const std::string& volume)
{
    const auto loaded = vcr::catalog::loadCatalog();
    const vcr::catalog::OpenDataSample* sample = nullptr;
    const vcr::catalog::OpenDataVolume* found = nullptr;
    for (const auto& candidate : loaded.manifest.samples)
        if (const auto* entry = candidate.findVolume(volume)) {
            if (found) throw std::runtime_error("the catalog lists volume " + volume + " under several samples");
            sample = &candidate;
            found = entry;
        }
    if (!found) throw std::runtime_error("the open-data catalog lists no volume " + volume);
    const auto handedness = vcr::catalog::windingHandedness(found->orientation);
    if (!handedness) {
        std::string missing;
        if (!found->orientation.zTopToBottom) missing += " z_direction_is_top_to_bottom";
        if (!found->orientation.leftHandedCoordinates) missing += " left_handed_coordinates";
        throw std::runtime_error("the open-data catalog does not orient volume " + sample->id + "/" + volume +
                                 " (unset:" + missing + "); state the direction by hand with --set +1 or --set -1");
    }
    return {*handedness, "catalog " + sample->id + "/" + volume};
}

int run(const std::vector<std::string>& args)
{
    std::string path, volume, set;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--volume" && i + 1 < args.size()) volume = args[++i];
        else if (args[i] == "--set" && i + 1 < args.size()) set = args[++i];
        else if (path.empty() && !args[i].starts_with("--")) path = args[i];
        else throw std::invalid_argument("unexpected argument " + args[i]);
    }
    if (path.empty()) throw std::invalid_argument("usage: vc_winding_handedness <store.vcobj> [--volume <id>] [--set catalog|+1|-1]");
    if (!set.empty() && set != "catalog" && set != "+1" && set != "1" && set != "-1")
        throw std::invalid_argument("--set takes catalog, +1 or -1");

    vc::objects::OpenOptions options;
    options.readOnly = set.empty();
    options.author = "vc_winding_handedness";
    auto store = vc::objects::ObjectStore::open(path, options);
    const auto umbilicusOf = [&]() {
        const auto txn = store.read();
        const vc::objects::LocalStoreReader reader(store, txn);
        std::vector<vc::objects::ObjectRecord> records;
        if (volume.empty()) {
            vc::objects::ObjectSelection selection;
            selection.kind = "umbilicus";
            records = reader.selectObjects(selection);
        } else {
            records = vc::objects::volumeUmbilici(reader, volume);
        }
        if (records.empty()) throw std::runtime_error(volume.empty() ? "the store holds no umbilicus" : "no umbilicus maps to volume " + volume);
        if (records.size() > 1)
            throw std::runtime_error(volume.empty() ? "the store holds several umbilici: name the volume with --volume"
                                                    : "several umbilici map to volume " + volume);
        return records.front();
    };
    auto record = umbilicusOf();
    if (volume.empty()) volume = declaredVolume(record);
    if (!set.empty()) {
        int handedness = 0;
        std::string source = "manual";
        if (set == "catalog") {
            if (volume.empty()) throw std::runtime_error("the umbilicus declares no volume: name it with --volume");
            std::tie(handedness, source) = catalogHandedness(volume);
        } else {
            handedness = set == "-1" ? -1 : 1;
        }
        auto txn = store.write();
        vc::objects::setUmbilicusWindingHandedness(store, txn, record, handedness, source);
        txn.commit();
        record = umbilicusOf();
    }
    const auto stated = vc::objects::umbilicusWindingHandedness(record);
    Json out{{"store", path}, {"volume", volume}, {"umbilicus", record.name.empty() ? record.uuid : record.name},
             {"winding_handedness", stated ? Json(*stated) : Json()},
             {"source", vc::objects::umbilicusWindingHandednessSource(record)}};
    std::cout << out.dump() << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    try {
        return run(std::vector<std::string>(argv + 1, argv + argc));
    } catch (const std::exception& problem) {
        std::cerr << "vc_winding_handedness: " << problem.what() << '\n';
        return 1;
    }
}
