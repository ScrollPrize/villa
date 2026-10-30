// What vc_reader's Python layer binds: how access resolves, errors at open, array
// attributes, group listing and saved state, and volume metadata kept once.
#include "vc_reader/catalog.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/array.hpp"
#include "vc_reader/zarr_metadata.hpp"
#include "vc_reader/zarr_write.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using namespace vcr::zarr;

namespace {
void require(bool value, const std::string& message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <class Error, class F>
bool raises(F&& call)
{
    try {
        call();
    } catch (const Error&) {
        return true;
    }
    return false;
}

void write(const fs::path& path, const std::string& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

void array(const fs::path& path)
{
    CreateOptions create;
    create.shape = create.chunks = {2, 2, 2};
    create.compressor = "";
    Writer::create(path, create);
}

void clearAws()
{
    for (const char* name : {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_REGION", "AWS_DEFAULT_REGION",
                             "AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3"})
        unsetenv(name);
}

void access(const fs::path& root)
{
    clearAws();
    setenv("AWS_SHARED_CREDENTIALS_FILE", (root / "no-credentials").c_str(), 1);
    setenv("AWS_CONFIG_FILE", (root / "no-config").c_str(), 1);
    const std::string open = "s3://vesuvius-challenge-open-data/v.zarr", other = "s3://vesuvius-challenge/v.zarr";
    const std::string bucket = "s3://my.private.bucket/v.zarr";  // a dotted name avoids asking AWS for its region
    require(resolveAccess(bucket).kind == Access::Kind::Anonymous, "auto without credentials is anonymous");
    require(raises<vcr::HttpDenied>([&] { resolveAccess(bucket, AuthMode::Aws); }), "aws without credentials is refused");

    setenv("AWS_ENDPOINT_URL", "http://localhost:1", 1);
    setenv("AWS_ACCESS_KEY_ID", "key", 1);
    setenv("AWS_SECRET_ACCESS_KEY", "secret", 1);
    const auto signedAccess = resolveAccess(bucket);
    require(signedAccess.kind == Access::Kind::S3 && signedAccess.fromEnvironment, "auto signs a private bucket");
    require(signedAccess.url(bucket) == "http://localhost:1/my.private.bucket/v.zarr", "endpoint URLs are path-style");
    for (const auto& location : {open, other, std::string("https://example.com/v.zarr")})
        require(resolveAccess(location).kind == Access::Kind::Anonymous, "public channels stay anonymous: " + location);
    require(resolveAccess(bucket, AuthMode::Auto, true).kind == Access::Kind::Anonymous, "offline needs no keys");
    require(resolveAccess(bucket, AuthMode::Anonymous).kind == Access::Kind::Anonymous, "anonymous mode");
    require(raises<std::invalid_argument>([&] { resolveAccess("https://example.com/v", AuthMode::Aws); }), "aws needs s3://");
    require(raises<std::invalid_argument>([&] { s3Access("https://example.com/v", {}); }), "S3 keys need s3://");

    auto copy = signedAccess;
    copy.auth.user = copy.auth.secret = "";
    reloadEnvironmentKeys(copy);
    require(copy.auth.user == "key" && copy.auth.secret == "secret", "environment keys read again");
    clearAws();
    require(raises<vcr::HttpDenied>([&] { reloadEnvironmentKeys(copy); }), "environment keys gone");

    require(isPublicLocation(open) && isPublicLocation("https://data.aws.ash2txt.org/samples/x") && !isPublicLocation(bucket),
            "public locations");
    require(isRemote("sftp://host/x") && isRemote("s3+us-west-2://bucket/x") && !isRemote("/data/x"), "remote schemes");
    require(raises<std::invalid_argument>([&] { listChildren({}, "https://example.com/v"); }), "plain HTTP has no listing");

    write(root / "auth.json", R"({"auth": {"user": "u", "password": ""}})");
    require(loadBasicAuth(root / "auth.json") == std::pair<std::string, std::string>{"u", ""}, "basic auth file");
    write(root / "bad.json", R"({"password": "p"})");
    require(raises<std::invalid_argument>([&] { loadBasicAuth(root / "bad.json"); }), "basic auth needs a user");
    require(raises<fs::filesystem_error>([&] { loadBasicAuth(root / "absent.json"); }), "basic auth file missing");
}

void errors(const fs::path& root)
{
    const OpenOptions options;
    require(raises<NotFound>([&] { openArray((root / "absent").string(), options); }), "missing array");
    require(raises<NotFound>([&] { Group::open((root / "absent").string(), options); }), "missing group");
    const auto read = metadataReader({});
    require(!read((root / "absent").string(), ".zarray"), "a missing object reads as absent");
    array(root / "plain");
    require(!read((root / "plain" / ".zarray").string(), "x"), "a file as a parent reads as absent");
    if (geteuid() != 0) {  // root reads anything
        fs::create_directories(root / "locked");
        write(root / "locked" / ".zarray", "{}");
        chmod((root / "locked" / ".zarray").c_str(), 0);
        require(raises<fs::filesystem_error>([&] { read((root / "locked").string(), ".zarray"); }),
                "an unreadable object is an error, not absent");
    }
}

void groups(const fs::path& root)
{
    const auto store = root / "store.zarr";
    write(store / ".zgroup", R"({"zarr_format": 2})");
    write(store / ".zattrs", R"({"multiscales": [{"axes": ["z", "y", "x"], "datasets": [
        {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]},
        {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [2, 4, 4]}]}]}]})");
    array(store / "0");
    array(store / "1");
    write(store / "1" / ".zattrs", R"({"note": "one"})");
    write(store / "labels" / ".zgroup", R"({"zarr_format": 2})");
    fs::create_directories(store / "not-a-node");

    OpenOptions options;
    auto group = Group::open(store.string(), options);
    require(group.keys() == std::vector<std::string>{"0", "1", "labels"}, "local keys skip non-nodes");
    require(group.levels() == std::vector<std::string>{"0", "1"}, "levels");
    require(group.placement("1").scale == std::array<double, 3>{2, 4, 4}, "group placement");
    require(levelPlacement(group.attributes(), "1").scale == std::array<double, 3>{2, 4, 4}, "placement from attributes");
    require(levelPlacement(Json::object(), "3").scale[0] == 8 && levelPlacement(Json::object(), "x").scale[0] == 1,
            "numbered levels without OME");

    // A 2-D image: the leading axis stays the identity.
    const auto image = levelPlacement(Json::parse(R"({"multiscales": [{"axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}]},
                     {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [2, 2]}]}]}]})"), "1");
    require(image.scale == std::array<double, 3>{1, 2, 2} && image.offset == std::array<double, 3>{0, 0.5, 0.5}, "2-D placement");

    auto level = group.array("1");
    require(level.store == store.string() && level.levelKey == "1" && !level.attributesRead, "a level knows its store");
    require(arrayAttributes(level, options).value("note", "") == "one" && level.attributesRead, "attributes on first use");
    fs::remove(store / "1" / ".zattrs");
    require(arrayAttributes(level, options).value("note", "") == "one", "attributes read once");
    level.close();

    ArrayHints hints;
    hints.attributes = Json{{"known", 1}};
    auto direct = openArray((store / "0").string(), options, hints);
    require(direct.store == store.string() && direct.levelKey == "0", "a numbered path is a level of its store");
    require(arrayAttributes(direct, options).value("known", 0) == 1, "attributes given at open");

    group.node("0");
    group.node("missing");
    auto restored = Group::fromState(group.state(), options);
    require(restored.attributes() == group.attributes() && restored.version() == group.version(), "state keeps the attributes");
    fs::remove_all(store / "0");
    require(restored.node("0") && restored.node("0")->first == Group::NodeKind::Array, "state keeps lookups");
    require(!restored.node("missing"), "state keeps absences");
    require(raises<std::invalid_argument>([&] { group.node("../x"); }), "invalid child path");

    // Listing works over consolidated metadata, without a directory scan.
    write(root / "c.zarr" / ".zgroup", R"({"zarr_format": 2})");
    write(root / "c.zarr" / ".zmetadata", R"({"metadata": {"a/.zarray": {}, "b/.zgroup": {}, "a/deep/.zarray": {}}})");
    require(Group::open((root / "c.zarr").string(), options).keys() == std::vector<std::string>{"a", "b"}, "consolidated keys");
}

const char* kManifest = R"({"samples": {"S": {"sample": {"id": "S", "properties": {}},
  "volumes": {"vol": {"id": "vol", "scan_id": "scan", "suffix": "masked", "properties": {"pixel_size_um": 7.91, "energy_keV": 88.0,
    "data_format": "uint8"},
    "data": [{"type": "zarr", "origins": [{"path": "S/volumes/vol.zarr", "access_roots": [
      {"type": "s3", "url": "s3://vesuvius-challenge-open-data/", "usage": "public-read"}]}]}]}}}}})";

void metadata(const fs::path& root)
{
    forgetVolumeMetadata();
    setenv("VC_READER_CACHE_DIR", (root / "catalog-cache").c_str(), 1);
    const std::string source(vcr::catalog::kDefaultManifestUrl);
    const auto manifestFile = vcr::catalog::catalogCacheFile(source);
    require(manifestFile.parent_path() == root / "catalog-cache", "catalog cache directory");
    write(manifestFile, kManifest);
    const auto now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    write(manifestFile.string() + ".info.json", Json{{"source", source}, {"checked_at", now}}.dump());

    const std::string url = "s3://vesuvius-challenge-open-data/S/volumes/vol.zarr";
    std::map<std::string, std::string> documents{{url + "/meta.json", R"({"voxelsize": 2.5})"}, {url + "/.zattrs", R"({"a": 1})"}};
    int requests = 0;
    bool down = false;
    OpenOptions options;
    options.cacheDirs = {(root / "cache").string()};
    options.metadata = [&](const std::string& location, const std::string& name) -> std::optional<std::string> {
        ++requests;
        if (down)
            throw std::runtime_error("network down");
        const auto it = documents.find(location + "/" + name);
        return it == documents.end() ? std::nullopt : std::optional<std::string>(it->second);
    };

    auto first = volumeMetadata(url, options);
    require(first.voxelSizeUm == 2.5, "the store's documents win");
    require(first.energyKeV == 88.0 && first.volumeId == "vol" && first.sampleId == "S" && first.name == "masked", "catalog fills the rest");
    require(first.documents.contains("catalog") && first.documents.at(".zattrs").value("a", 0) == 1, "sources are kept");
    const auto kept = decodedCachePath(root / "cache", url) / ".vc_reader_metadata.json";
    require(fs::is_regular_file(kept), "the result is kept with the volume");

    const auto asked = requests;
    require(volumeMetadata(url, options).voxelSizeUm == 2.5 && requests == asked, "this process does not ask again");
    forgetVolumeMetadata();
    down = true;
    require(volumeMetadata(url, options).energyKeV == 88.0 && requests == asked, "a kept copy is trusted");
    auto offlineOptions = options;
    offlineOptions.offline = true;
    require(volumeMetadata(url, offlineOptions).voxelSizeUm == 2.5, "offline reads the kept copy");
    require(raises<std::invalid_argument>([&] { volumeMetadata(url, offlineOptions, true); }), "refresh needs the network");

    // A failed first read is not kept as if the store had no documents.
    const std::string url2 = "s3://vesuvius-challenge-open-data/S/volumes/other.zarr";
    require(!volumeMetadata(url2, options).voxelSizeUm, "a failed read says nothing");
    require(!fs::exists(decodedCachePath(root / "cache", url2) / ".vc_reader_metadata.json"), "a failed read is not kept");

    // Outside the public channels the catalog is not consulted. Without one to refresh,
    // refresh needs no network beyond the store's own documents.
    fs::remove(manifestFile);
    down = false;
    const std::string url3 = "https://example.com/private.zarr";
    documents[url3 + "/meta.json"] = R"({"voxelsize": 9})";
    const auto keptPrivate = decodedCachePath(root / "cache", url3) / ".vc_reader_metadata.json";
    require(volumeMetadata(url3, options).voxelSizeUm == 9 && fs::is_regular_file(keptPrivate), "private store");
    documents[url3 + "/meta.json"] = R"({"voxelsize": 10})";
    require(volumeMetadata(url3, options).voxelSizeUm == 9, "kept");
    down = true;
    require(volumeMetadata(url3, options, true).voxelSizeUm == 9, "a failed refresh returns what is kept");
    down = false;
    require(volumeMetadata(url3, options, true).voxelSizeUm == 10, "refresh rereads");
    forgetVolumeMetadata();
    require(volumeMetadata(url3, options).voxelSizeUm == 10, "refresh replaced the kept copy");
    forgetVolumeMetadata();
    write(keptPrivate, "{damaged");
    require(volumeMetadata(url3, options).voxelSizeUm == 10, "a damaged copy is read again");

    // No disk cache: the process remembers. A read-only directory still answers.
    const std::string url4 = "https://example.com/memory.zarr";
    documents[url4 + "/meta.json"] = R"({"voxelsize": 4})";
    auto noCache = options;
    noCache.cacheDirs.clear();
    require(volumeMetadata(url4, noCache).voxelSizeUm == 4, "no disk cache");
    documents.erase(url4 + "/meta.json");
    require(volumeMetadata(url4, noCache).voxelSizeUm == 4, "the process remembers");
    const std::string url5 = "https://example.com/readonly.zarr";
    documents[url5 + "/meta.json"] = R"({"voxelsize": 5})";
    auto readOnly = options;
    readOnly.cacheDirs = {"/proc/not-writable"};
    require(volumeMetadata(url5, readOnly).voxelSizeUm == 5, "a read-only cache directory still answers");

    // VC3D's cache root: kept in the volume's own folder when it has one.
    const auto vc3dRoot = root / "vc3d-root";
    const std::string url6 = "https://example.com/vc3d.zarr";
    fs::create_directories(vc3dRoot / vc3dVolumeId(url6));
    write(root / "vc3d-config" / "VC3D.ini", "[viewer]\nremote_cache_dir=" + vc3dRoot.string() + "\n");
    setenv("VC3D_CONFIG_DIR", (root / "vc3d-config").c_str(), 1);
    documents[url6 + "/meta.json"] = R"({"voxelsize": 6})";
    auto vc3d = options;
    vc3d.cacheDirs = {"vc3d"};
    require(volumeMetadata(url6, vc3d).voxelSizeUm == 6, "VC3D cache");
    require(fs::is_regular_file(vc3dRoot / vc3dVolumeId(url6) / ".vc_reader_metadata.json"), "kept with VC3D's volume");
    unsetenv("VC3D_CONFIG_DIR");

    // Local stores are read at each call and leave nothing behind.
    const auto local = root / "local.zarr";
    write(local / "meta.json", R"({"voxelsize": 1})");
    OpenOptions plain;
    require(volumeMetadata(local.string(), plain).voxelSizeUm == 1, "local store");
    write(local / "meta.json", R"({"voxelsize": 2})");
    require(volumeMetadata(local.string(), plain).voxelSizeUm == 2, "local stores are not memoized");
    require(!fs::exists(local / ".vc_reader_metadata.json"), "local stores keep nothing");

    const auto round = storeMetadataFromJson(storeMetadataJson(first));
    require(round.voxelSizeUm == first.voxelSizeUm && round.volumeId == first.volumeId && round.documents == first.documents, "JSON form");
    unsetenv("VC_READER_CACHE_DIR");
    forgetVolumeMetadata();
}
}  // namespace

int main()
{
    const auto root = fs::temp_directory_path() / ("vcr-native-open-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    struct Cleanup {
        fs::path path;
        ~Cleanup()
        {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    } cleanup{root};
    try {
        access(root);
        errors(root);
        groups(root);
        metadata(root);
        std::cout << "Native opening, access and volume metadata passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
