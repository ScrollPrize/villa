#include "TestSupport.hpp"
#include "vc_object_store/objects/UmbilicusFile.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
using namespace vc::objects;

// Reading umbilicus files and deciding the frame their coordinates index, the
// way Import Umbilicus and vc_fiber_constraints bind one to a volume.
namespace {
std::filesystem::path write(const TemporaryDirectory& dir, const std::string& name, const std::string& content)
{
    const auto path = dir.path / name;
    std::ofstream(path) << content;
    return path;
}
bool near(double a, double b) { return std::abs(a - b) <= 1e-12 * std::max(1.0, std::abs(b)); }

// The real PHercParis4 configuration: umbilicus.json stamps the 9.6 µm ds2 grid
// while annotation runs on the 2.4 µm store, level-0 shape 32693 x 32693 x 75784.
// Only z is an exact multiple: 32693/4 = 8173.25 was rounded up to 8174.
UmbilicusFile stampedPHercParis4()
{
    UmbilicusFile file;
    file.points = {{1, 2, 3}};
    file.volumeWidth = 8174;
    file.volumeHeight = 8174;
    file.volumeSlices = 18946;
    file.voxelSizeUm = 9.6;
    return file;
}
UmbilicusBinding bindTo(const UmbilicusFile& file, std::array<double, 3> xyz, std::optional<double> voxelSizeUm,
                      std::optional<UmbilicusTargetGrid> named = std::nullopt)
{
    return umbilicusBinding(file, {xyz, voxelSizeUm}, named);
}
// A stated frame, taken as stated: `unitsPerVoxel` file units per target voxel.
void requireStated(const UmbilicusBinding& binding, double unitsPerVoxel, const char* message)
{
    require(!binding.refusal && !binding.confirm && binding.storedUnitsPerVoxel &&
                near(*binding.storedUnitsPerVoxel, unitsPerVoxel), message);
}
}

int main() try {
    TemporaryDirectory dir;
    {
        // Formats: JSON arrays of [z,y,x], {z,y,x} objects, "points" and
        // "control_points" wrappers, text/CSV with comments; points come back XYZ.
        require(readUmbilicusFile(write(dir, "a.json", "[[0,5,10],[10,6,11]]")).points.front() == vc::geometry::Point{10, 5, 0},
                "bare [z,y,x] arrays read as XYZ");
        require(readUmbilicusFile(write(dir, "o.json", R"([{"z":0,"y":5,"x":10},{"z":5,"y":6,"x":11}])")).points.size() == 2,
                "z/y/x objects parse");
        require(readUmbilicusFile(write(dir, "w.json", R"({"points":[[0,5,10],[5,6,11]]})")).points.size() == 2,
                "points wrapper parses");
        const auto control = readUmbilicusFile(write(dir, "c.json",
            R"({"control_points": [{"x": 10, "y": 20, "z": 5, "score": 100},{"x": 12, "y": 22, "z": 15, "score": 100}],)"
            R"("metadata": {"total_points": 2, "timestamp": "2026-04-20T13:16:29Z"}})"));
        require(control.points.size() == 2 && control.points[1] == vc::geometry::Point{12, 22, 15}, "control_points parse");
        require(!control.voxelSizeUm && !control.volume && !control.volumeWidth && control.metadataErrors.empty(),
                "metadata without frame keys states nothing and is no error");
        const auto text = readUmbilicusFile(write(dir, "t.txt", "# comment\n0, 5, 10\n\n  10, 6, 11  \n20, 7, 12\n"));
        require(text.points.size() == 3 && text.points.front() == vc::geometry::Point{10, 5, 0}, "text lines parse");
        for (const auto& [name, content] : {std::pair<const char*, const char*>{"m.txt", "0, 5\n"}, {"x.txt", "0, 5, 10, 99\n"},
                                            {"n.txt", "0, hello, 10\n"}, {"e.txt", "\n# only a comment\n"},
                                            {"r.json", R"({"other":"value"})"}, {"p.json", R"({"points":"nope"})"},
                                            {"s.json", "[[0,5]]"}, {"k.json", R"([{"z":0,"y":5}])"}})
            mustFail([&] { (void)readUmbilicusFile(write(dir, name, content)); });
        mustFail([&] { (void)readUmbilicusFile(dir.path / "missing.txt"); });
    }
    {
        // Frame metadata: every field is reported; a malformed one is listed
        // and costs only itself; an incomplete dimension triplet is malformed.
        const std::string points = R"("control_points": [{"x": 1, "y": 2, "z": 3, "score": 100}]})";
        const auto stamped = readUmbilicusFile(write(dir, "f.json",
            R"({"metadata": {"voxelsize_um": 9.6, "volume": "scroll.zarr", "volume_width": 8174,)"
            R"( "volume_height": 8174, "volume_slices": 18946},)" + points));
        require(stamped.metadataErrors.empty() && stamped.voxelSizeUm == 9.6 && stamped.volume == "scroll.zarr" &&
                    stamped.volumeWidth == 8174 && stamped.volumeHeight == 8174 && stamped.volumeSlices == 18946,
                "a stamped file reports every field");
        const auto partial = readUmbilicusFile(write(dir, "d.json",
            R"({"metadata": {"volume_width": 8174, "volume_slices": 18946},)" + points));
        require(partial.points.size() == 1 && partial.metadataErrors ==
                    std::vector<std::string>{"volume dimensions: an incomplete triplet declares nothing; missing volume_height"},
                "an incomplete triplet is malformed, points still load");
        const auto badWidth = readUmbilicusFile(write(dir, "b.json",
            R"({"metadata": {"volume_width": -5, "volume_height": 8174},)" + points));
        require(badWidth.metadataErrors == std::vector<std::string>{"volume_width: expected a positive integer, got -5"},
                "a rejected dimension is blamed once");
        const auto badVoxel = readUmbilicusFile(write(dir, "v.json",
            R"({"metadata": {"voxelsize_um": "nine point six", "volume": "scroll.zarr", "volume_width": 8174,)"
            R"( "volume_height": 8174, "volume_slices": 18946},)" + points));
        require(badVoxel.metadataErrors.size() == 1 && !badVoxel.voxelSizeUm && badVoxel.volumeWidth == 8174,
                "a malformed voxel size costs only itself");
        require(bindTo(badVoxel, {8174, 8174, 18946}, 9.6).refusal.has_value(), "malformed frame metadata is refused");
    }
    {
        // Stamped dimensions: one integer factor that carries every axis onto the
        // target within its floor/ceil rounding, else refused (never averaged,
        // never a voxel-size consolation).
        requireStated(bindTo(stampedPHercParis4(), {32693, 32693, 75784}, 2.4), 0.25, "PHercParis4 resolves to exactly 4");
        requireStated(bindTo(stampedPHercParis4(), {8174, 8174, 18946}, 9.6), 1.0, "an identical grid gives exactly 1");
        requireStated(bindTo(stampedPHercParis4(), {32699, 32699, 75781}, std::nullopt), 0.25,
                      "axes rounded in opposite directions still agree");
        requireStated(bindTo(stampedPHercParis4(), {32696, 32696, 75787}, std::nullopt), 0.25, "floor of the window fits");
        require(bindTo(stampedPHercParis4(), {32696, 32696, 75800}, std::nullopt).refusal.has_value(), "past the window refuses");
        require(bindTo(stampedPHercParis4(), {32696, 32696, 75026}, 2.4).refusal.has_value(), "a two percent spread refuses");
        require(bindTo(stampedPHercParis4(), {32696, 32696, 40000}, 2.4).refusal.has_value(), "a cropped axis refuses");
        requireStated(bindTo(stampedPHercParis4(), {24522, 24522, 56838}, std::nullopt), 1.0 / 3.0, "non-power-of-two factors");
        UmbilicusFile fine = stampedPHercParis4();
        fine.volumeWidth = fine.volumeHeight = 32693;
        fine.volumeSlices = 75784;
        fine.voxelSizeUm = 2.4;
        requireStated(bindTo(fine, {8174, 8174, 18946}, 9.6), 4.0, "a coarser target works with a real residual");
        UmbilicusFile small;
        small.points = {{1, 2, 3}};
        small.volumeWidth = small.volumeHeight = 100;
        small.volumeSlices = 1000;
        require(bindTo(small, {51, 51, 500}, std::nullopt).refusal.has_value(), "small grids get no free tolerance");
        UmbilicusFile aspect = stampedPHercParis4();
        aspect.volumeHeight = 4000;
        require(bindTo(aspect, {32693, 32693, 75784}, 2.4).refusal.has_value(),
                "dimensions that are not a rescale refuse even with a voxel size");
    }
    {
        // A stamped voxel size alone converts against the target's.
        UmbilicusFile voxelOnly;
        voxelOnly.points = {{1, 2, 3}};
        voxelOnly.voxelSizeUm = 9.6;
        requireStated(bindTo(voxelOnly, {32696, 32696, 75784}, 2.4), 0.25, "voxel sizes convert when dimensions are absent");
        require(bindTo(voxelOnly, {0, 0, 0}, std::nullopt).refusal.has_value(), "a stated frame with nothing to check refuses");
    }
    {
        // An unstamped file states nothing: a reading inferred from the grid it
        // fills, or none, is only offered for confirmation.
        UmbilicusFile filling;
        filling.points = {{100, 100, 400}, {120, 120, 18000}};
        const auto inferred = bindTo(filling, {32696, 32696, 75784}, std::nullopt);
        require(!inferred.refusal && inferred.confirm && inferred.storedUnitsPerVoxel &&
                    near(*inferred.storedUnitsPerVoxel, 0.25), "a filled grid is inferred, to confirm");
        UmbilicusFile shortSpan;
        shortSpan.points = {{100, 100, 400}, {120, 120, 900}};
        const auto unknown = bindTo(shortSpan, {32696, 32696, 75784}, std::nullopt);
        require(!unknown.refusal && unknown.confirm && !unknown.storedUnitsPerVoxel, "a short span states and infers nothing");
    }
    {
        // A stamp is checked against the volume it names; a disproven stamp
        // refuses even when the points would infer a scale.
        UmbilicusFile named = stampedPHercParis4();
        named.volume = "s1_2um_ds2.zarr";
        const UmbilicusTargetGrid annotation{{32693, 32693, 75784}, 2.4};
        requireStated(bindTo(named, annotation.xyz, 2.4, annotation), 0.25, "consistent through a rescale");
        requireStated(bindTo(named, annotation.xyz, 2.4, UmbilicusTargetGrid{{8174, 8174, 18946}, 9.6}), 0.25,
                      "consistent with the stamped grid itself");
        require(bindTo(named, annotation.xyz, 2.4, UmbilicusTargetGrid{{32696, 32696, 40000}, 2.4}).refusal.has_value(),
                "a named grid the stamp cannot rescale onto refuses");
        UmbilicusFile wrongVoxel = named;
        wrongVoxel.voxelSizeUm = 19.2;
        require(bindTo(wrongVoxel, annotation.xyz, 2.4, annotation).refusal.has_value(),
                "a voxel size the named volume disproves refuses");
        requireStated(bindTo(named, annotation.xyz, 2.4, UmbilicusTargetGrid{annotation.xyz, 2.399}), 0.25,
                      "metadata round-trip slack is no contradiction");
        UmbilicusFile voxelOnly;
        voxelOnly.points = {{1, 2, 3}};
        voxelOnly.volume = "s1_8um.zarr";
        voxelOnly.voxelSizeUm = 9.6;
        requireStated(bindTo(voxelOnly, annotation.xyz, 2.4, annotation), 0.25,
                      "a voxel-size-only stamp is converted, not contradicted");
    }
    std::cout << "umbilicus file tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
