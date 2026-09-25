#include <QtTest/QtTest>

#include "LineAnnotationDatasetSets.hpp"

#include <string>
#include <vector>

using namespace vc3d::line_annotation;

namespace
{

// The PHerc1451 open-data project as attached on 2026-09-24.
std::vector<ProjectVolumeInfo> pherc1451Volumes()
{
    const std::string fibers = "/cache/las-sd1-40cc618f.lasagna.json";
    const std::string lasagna = "/cache/lasagna-20260724.lasagna.json";
    return {
        {"20250521151225", "20250521151225-8.640um-1.2m-116keV-masked",
         {"vc-open-data-volume-id:20250521151225", "vc-open-data-preferred-source",
          "vc-open-data-source-coordinate-level:0", "vc-open-data-voxel-size-um:8.640000"},
         {16000, 5500, 5500}, 8.64},
        {"20260319101107", "20260319101107-2.399um-0.2m-78keV-masked",
         {"vc-open-data-volume-id:20260319101107", "vc-open-data-preferred-source",
          "vc-open-data-source-coordinate-level:0", "vc-open-data-voxel-size-um:2.399000"},
         {59944, 20812, 20812}, 2.399},
        {"surface-m7", "20260319101107-surface-20260413222639-surface-m7-L2-th0.2",
         {"prediction", "surface-prediction", "vc-open-data-volume-id:20260319101107",
          "vc-open-data-source-coordinate-level:2", "vc-open-data-voxel-size-um:9.596000"},
         {14986, 5203, 5203}, 9.596},
        {"20260319101107-vc-base-2-x", "masked.zarr [source L2, 9.596000 um]",
         {"vc-open-data-volume-id:20260319101107", "vc-open-data-virtual-source",
          "vc-open-data-source-coordinate-level:2", "vc-open-data-voxel-size-um:9.596000"},
         {14986, 5203, 5203}, 9.596},
        {"lasagna-cos", "PHerc1451-20260319101107-lasagna-20260724_cos.ome",
         {"vc-lasagna-manifest:" + lasagna, "vc-lasagna-group:cos"}, {59944, 20811, 20811}, 0.0},
        {"lasagna-nx", "PHerc1451-20260319101107-lasagna-20260724_nx.ome",
         {"vc-lasagna-manifest:" + lasagna, "vc-lasagna-group:nx"}, {59944, 20811, 20811}, 0.0},
        {"20260319101107-vc-base-1-x", "masked.zarr [source L1, 4.798000 um]",
         {"vc-open-data-volume-id:20260319101107", "vc-open-data-virtual-source",
          "vc-open-data-source-coordinate-level:1", "vc-open-data-voxel-size-um:4.798000"},
         {29972, 10406, 10406}, 4.798},
        {"fiber-presence", "PHerc1451-20260319101107-las-sd1-40cc618f_presence.ome",
         {"vc-lasagna-manifest:" + fibers, "vc-lasagna-group:presence"}, {59944, 20816, 20816}, 0.0},
        {"fiber-nx", "PHerc1451-20260319101107-las-sd1-40cc618f_nx.ome",
         {"vc-lasagna-manifest:" + fibers, "vc-lasagna-group:nx"}, {59944, 20816, 20816}, 0.0},
    };
}

std::vector<ClassifiedVolume> classifyAll(const std::vector<ProjectVolumeInfo>& infos)
{
    return classifyProjectVolumes(infos, {"/cache/las-sd1-40cc618f.lasagna.json"});
}

std::vector<std::string> labels(const std::vector<VolumeSelectorOption>& options)
{
    std::vector<std::string> out;
    for (const auto& o : options) out.push_back(o.label);
    return out;
}

}  // namespace

class TestLineAnnotationDatasetSets : public QObject {
    Q_OBJECT

private slots:
    void classifiesByTags();
    void groupsScansWithTheirLevelsFinestFirst();
    void resolvesDatasetScanByTagOrByFrame();
    void picksTheNewestApplicableDataset();
    void derivesTheDefaultScan();
    void listsTheSelectedSetWithReadableLabels();
    void listsEveryVolumeInAllMode();
    void untaggedLocalProjectStillWorks();
    void untaggedPublishedNamesAreRecognised();
    void severalSurfacesAndForeignSurfacesAreLabelled();
    void channelsTakeTheirDatasetsScan();
};

void TestLineAnnotationDatasetSets::classifiesByTags()
{
    const auto volumes = classifyAll(pherc1451Volumes());
    QCOMPARE(volumes[1].kind, ProjectVolumeKind::RawScan);
    QCOMPARE(volumes[1].scanKey, std::string("20260319101107"));
    QCOMPARE(volumes[1].level, 0);
    QCOMPARE(volumes[6].kind, ProjectVolumeKind::RawScan);
    QCOMPARE(volumes[6].level, 1);
    QVERIFY(volumes[6].virtualSource);
    QCOMPARE(volumes[2].kind, ProjectVolumeKind::SurfacePrediction);
    QCOMPARE(volumes[2].scanKey, std::string("20260319101107"));
    QCOMPARE(volumes[4].kind, ProjectVolumeKind::Lasagna);
    QCOMPARE(volumes[4].channel, std::string("cos"));
    QCOMPARE(volumes[7].kind, ProjectVolumeKind::Fiber);
    QCOMPARE(volumes[7].channel, std::string("presence"));
    QCOMPARE(volumes[7].manifestLocation, std::string("/cache/las-sd1-40cc618f.lasagna.json"));
}

void TestLineAnnotationDatasetSets::groupsScansWithTheirLevelsFinestFirst()
{
    const auto scans = rawScanOptions(classifyAll(pherc1451Volumes()));
    QCOMPARE(scans.size(), std::size_t{2});
    QCOMPARE(scans[0].scanKey, std::string("20260319101107"));
    QCOMPARE(scans[0].label, std::string("20260319101107, 2.399 \xC2\xB5m"));
    QCOMPARE(scans[0].levels.size(), std::size_t{3});
    QCOMPARE(scans[0].levels[0].first, 0);
    QCOMPARE(scans[0].levels[1].second, std::string("20260319101107-vc-base-1-x"));
    QCOMPARE(scans[0].levels[2].first, 2);
    QVERIFY(scans[0].level0ShapeZYX == (std::array<std::size_t, 3>{59944, 20812, 20812}));
    QCOMPARE(scans[1].scanKey, std::string("20250521151225"));
    QCOMPARE(scans[1].label, std::string("20250521151225, 8.64 \xC2\xB5m"));
    QCOMPARE(scanVolumeIdAtLevel(scans[0], 1), std::string("20260319101107-vc-base-1-x"));
    // A level the scan lacks falls back to its finest one.
    QCOMPARE(scanVolumeIdAtLevel(scans[0], 3), std::string("20260319101107"));
    QCOMPARE(scanVolumeIdAtLevel(scans[1], 1), std::string("20250521151225"));
    QCOMPARE(rawScanLevelLabel(scans[0], 0), std::string("Level 0 (2.399 \xC2\xB5m)"));
    QCOMPARE(rawScanLevelLabel(scans[0], 1), std::string("Level 1 (4.798 \xC2\xB5m)"));
    QCOMPARE(rawScanLevelLabel(scans[0], 2), std::string("Level 2 (9.596 \xC2\xB5m)"));
    const auto volumes = classifyAll(pherc1451Volumes());
    QCOMPARE(*rawScanLevelOfVolume(volumes, "20260319101107-vc-base-1-x"), 1);
    QCOMPARE(*rawScanLevelOfVolume(volumes, "20260319101107"), 0);
    QVERIFY(!rawScanLevelOfVolume(volumes, "fiber-presence").has_value());
    QVERIFY(!rawScanLevelOfVolume(volumes, "nope").has_value());
}

void TestLineAnnotationDatasetSets::resolvesDatasetScanByTagOrByFrame()
{
    const auto scans = rawScanOptions(classifyAll(pherc1451Volumes()));
    // Open-data entries carry the scan id.
    QCOMPARE(datasetScanKey({"a.json", {"vc-open-data-volume-id:20260319101107"}, std::nullopt}, scans),
             std::string("20260319101107"));
    // Local entries are placed by their manifest frame, with the one-voxel
    // convention tolerance (the lasagna manifest records 20811).
    QCOMPARE(datasetScanKey({"b.json", {}, std::array<std::size_t, 3>{59944, 20811, 20811}}, scans),
             std::string("20260319101107"));
    QCOMPARE(datasetScanKey({"c.json", {}, std::array<std::size_t, 3>{16000, 5500, 5500}}, scans),
             std::string("20250521151225"));
    // Unknown frame, or no frame: applies to any scan.
    QVERIFY(datasetScanKey({"d.json", {}, std::array<std::size_t, 3>{1000, 1000, 1000}}, scans).empty());
    QVERIFY(datasetScanKey({"e.json", {}, std::nullopt}, scans).empty());
    QVERIFY(datasetAppliesToScan("", "20260319101107"));
    QVERIFY(datasetAppliesToScan("20260319101107", "20260319101107"));
    QVERIFY(!datasetAppliesToScan("20250521151225", "20260319101107"));
}

void TestLineAnnotationDatasetSets::picksTheNewestApplicableDataset()
{
    const auto scans = rawScanOptions(classifyAll(pherc1451Volumes()));
    const std::vector<DatasetInfo> fibers{
        {"old.json", {"vc-open-data-volume-id:20260319101107", "vc-open-data-lasagna-model-id:20260801084232"}, std::nullopt},
        {"new.json", {"vc-open-data-volume-id:20260319101107", "vc-open-data-lasagna-model-id:20260915212757"}, std::nullopt},
        {"other-scan.json", {"vc-open-data-volume-id:20250521151225", "vc-open-data-lasagna-model-id:20261001000000"}, std::nullopt},
    };
    const auto newest = newestDatasetForScan(fibers, scans, "20260319101107");
    QVERIFY(newest.has_value());
    QCOMPARE(*newest, std::string("new.json"));
    const auto forOverview = newestDatasetForScan(fibers, scans, "20250521151225");
    QVERIFY(forOverview.has_value());
    QCOMPARE(*forOverview, std::string("other-scan.json"));
    QVERIFY(!newestDatasetForScan({}, scans, "20260319101107").has_value());
    // Without model ids the last applicable entry wins.
    const std::vector<DatasetInfo> plain{{"first.json", {}, std::nullopt}, {"second.json", {}, std::nullopt}};
    QCOMPARE(*newestDatasetForScan(plain, scans, "20260319101107"), std::string("second.json"));
}

void TestLineAnnotationDatasetSets::derivesTheDefaultScan()
{
    const auto scans = rawScanOptions(classifyAll(pherc1451Volumes()));
    QCOMPARE(defaultRawScanKey(scans, "20250521151225", "20260319101107", {}), std::string("20250521151225"));
    QCOMPARE(defaultRawScanKey(scans, "", "20260319101107", {}), std::string("20260319101107"));
    QCOMPARE(defaultRawScanKey(scans, "", "", {"20250521151225", "20250521151225"}), std::string("20250521151225"));
    // Nothing recorded: the finest scan.
    QCOMPARE(defaultRawScanKey(scans, "", "", {}), std::string("20260319101107"));
    // A dataset scan that is not attached does not win.
    QCOMPARE(defaultRawScanKey(scans, "unknown", "", {}), std::string("20260319101107"));
    QVERIFY(defaultRawScanKey({}, "", "", {}).empty());
}

void TestLineAnnotationDatasetSets::listsTheSelectedSetWithReadableLabels()
{
    const auto volumes = classifyAll(pherc1451Volumes());
    const auto scans = rawScanOptions(volumes);
    const auto options = volumeSelectorOptions(
        volumes, scans, "20260319101107", "/cache/lasagna-20260724.lasagna.json",
        "/cache/las-sd1-40cc618f.lasagna.json", "surface-m7", "20260319101107");
    const std::vector<std::string> expected{
        "raw scan, 2.399 \xC2\xB5m",
        "lasagna, cos",
        "lasagna, nx",
        "fiber, nx",
        "fiber, presence",
        "surface",
    };
    QVERIFY(labels(options) == expected);
    // The scan is listed once, as its level-0 volume by default...
    QCOMPARE(options[0].id, std::string("20260319101107"));
    QCOMPARE(options[0].tooltip, std::string("20260319101107-2.399um-0.2m-78keV-masked (20260319101107)"));
    // ...or as the level the workspace is on, so picking it does not change level.
    const auto atL1 = volumeSelectorOptions(
        volumes, scans, "20260319101107", "/cache/lasagna-20260724.lasagna.json",
        "/cache/las-sd1-40cc618f.lasagna.json", "surface-m7", "20260319101107-vc-base-1-x", 1);
    QVERIFY(labels(atL1) == expected);
    QCOMPARE(atL1[0].id, std::string("20260319101107-vc-base-1-x"));
    // A level the other scan lacks: it is listed at its finest level.
    const auto overviewAtL1 = volumeSelectorOptions(
        volumes, scans, "20250521151225", "", "", "", "20260319101107-vc-base-1-x", 1);
    QCOMPARE(overviewAtL1[0].label, std::string("raw scan, 8.64 \xC2\xB5m"));
    QCOMPARE(overviewAtL1[0].id, std::string("20250521151225"));

    // The other scan's set has no datasets; the current volume from the first
    // set is still listed, marked, so the selector is never empty of it.
    const auto overview = volumeSelectorOptions(
        volumes, scans, "20250521151225", "", "", "", "20260319101107");
    const std::vector<std::string> expectedOverview{
        "raw scan, 8.64 \xC2\xB5m",
        "raw scan, 2.399 \xC2\xB5m (not in this set)",
    };
    QVERIFY(labels(overview) == expectedOverview);
}

void TestLineAnnotationDatasetSets::listsEveryVolumeInAllMode()
{
    // Advanced mode: every volume, project order, raw names, twins included.
    const auto volumes = classifyAll(pherc1451Volumes());
    const auto options = rawVolumeSelectorOptions(volumes);
    QCOMPARE(options.size(), volumes.size());
    QCOMPARE(options[0].label, std::string("20250521151225-8.640um-1.2m-116keV-masked"));
    QCOMPARE(options[3].label, std::string("masked.zarr [source L2, 9.596000 um]"));
    QCOMPARE(options[3].id, std::string("20260319101107-vc-base-2-x"));
    QCOMPARE(options[3].tooltip, std::string("20260319101107-vc-base-2-x"));
}

void TestLineAnnotationDatasetSets::untaggedLocalProjectStillWorks()
{
    // A hand-made local project: one scan folder, no tags, plus fiber channels
    // attached from a local manifest.
    const std::vector<ProjectVolumeInfo> infos{
        {"scan-abc", "PHerc0332_scan", {}, {14000, 8000, 8000}, 7.91},
        {"pres", "presence.ome", {"vc-lasagna-manifest:/data/f.lasagna.json", "vc-lasagna-group:presence"},
         {14000, 8000, 8000}, 0.0},
    };
    // A custom-named folder matches no published pattern; with no other scan
    // in the project it is promoted to the scan.
    const auto volumes = classifyProjectVolumes(infos, {"/data/f.lasagna.json"});
    const auto scans = rawScanOptions(volumes);
    QCOMPARE(scans.size(), std::size_t{1});
    QCOMPARE(scans[0].scanKey, std::string("scan-abc"));
    QCOMPARE(scans[0].label, std::string("scan-abc, 7.91 \xC2\xB5m"));
    QCOMPARE(datasetScanKey({"/data/f.lasagna.json", {}, std::array<std::size_t, 3>{14000, 8000, 8000}}, scans),
             std::string("scan-abc"));
    const auto options = volumeSelectorOptions(volumes, scans, "scan-abc", "", "/data/f.lasagna.json", "", "scan-abc");
    const std::vector<std::string> expected{"raw scan, 7.91 \xC2\xB5m", "fiber, presence"};
    QVERIFY(labels(options) == expected);
    // No voxel size at all: the name stands in. A lone custom-named folder is
    // still a scan (the project has nothing else to offer).
    const auto bare = classifyProjectVolumes({{"v1", "local.zarr", {}, {10, 10, 10}, 0.0}}, {});
    QCOMPARE(bare[0].kind, ProjectVolumeKind::RawScan);
    QCOMPARE(volumeLabel(bare[0], 0.0), std::string("raw scan, local.zarr"));
}

void TestLineAnnotationDatasetSets::untaggedPublishedNamesAreRecognised()
{
    // PHerc0332 on nvme43 and PHercParis4 on nvme2: hand-attached folders
    // with published names and no tags; the Lasagna channels do carry their
    // group tags because they were attached through the manifest.
    const std::string las = "/nvme/las_008.lasagna.json";
    const std::vector<ProjectVolumeInfo> infos{
        {"20251211183505-2.399um-0.2m-78keV-masked-abc", "20251211183505-2.399um-0.2m-78keV-masked.zarr",
         {}, {33592, 15761, 15761}, 2.399},
        {"20251211183505-surface-20260413222639-surface-m7-L2-th0.2-def",
         "20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr", {}, {8398, 3941, 3941}, 0.0},
        {"20260411134726-ink3d-20260428123845-v3-78k-fullsup-ghi",
         "20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr", {}, {1, 1, 1}, 0.0},
        {"las-cos", "las_008_cos.ome.zarr", {"vc-lasagna-manifest:" + las, "vc-lasagna-group:cos"},
         {33592, 15761, 15761}, 0.0},
    };
    const auto volumes = classifyProjectVolumes(infos, {});
    QCOMPARE(volumes[0].kind, ProjectVolumeKind::RawScan);
    QCOMPARE(volumes[0].scanKey, std::string("20251211183505"));
    QCOMPARE(volumes[1].kind, ProjectVolumeKind::SurfacePrediction);
    QCOMPARE(volumes[1].scanKey, std::string("20251211183505"));
    QCOMPARE(volumes[2].kind, ProjectVolumeKind::Other);
    QCOMPARE(volumes[2].scanKey, std::string("20260411134726"));
    QCOMPARE(volumes[3].kind, ProjectVolumeKind::Lasagna);
    const auto scans = rawScanOptions(volumes);
    QCOMPARE(scans.size(), std::size_t{1});
    QCOMPARE(scans[0].label, std::string("20251211183505, 2.399 \xC2\xB5m"));
    // The dataset is placed by its manifest frame (0332 records 15762 for a
    // 15761 scan; the +-1 tolerance covers it).
    QCOMPARE(datasetScanKey({las, {}, std::array<std::size_t, 3>{33592, 15762, 15762}}, scans),
             std::string("20251211183505"));
    const auto surface = defaultSurfaceVolumeId(volumes, "20251211183505");
    QVERIFY(surface.has_value());
    QCOMPARE(*surface, std::string("20251211183505-surface-20260413222639-surface-m7-L2-th0.2-def"));
    const auto set = volumeSelectorOptions(volumes, scans, "20251211183505", las, "", *surface,
                                           "20251211183505-2.399um-0.2m-78keV-masked-abc");
    const std::vector<std::string> expected{"raw scan, 2.399 \xC2\xB5m", "lasagna, cos", "surface"};
    QVERIFY(labels(set) == expected);
    // The ink volume is an extra: never in the set, only in the advanced list.
    const auto all = rawVolumeSelectorOptions(volumes);
    QCOMPARE(all[2].label, std::string("20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr"));
    // A scan whose voxel size is only in its name.
    const auto named = classifyProjectVolumes({{"x", "20250728140407-9.362um-1.2m-113keV-masked.zarr", {}, {1, 1, 1}, 0.0}}, {});
    QCOMPARE(named[0].kind, ProjectVolumeKind::RawScan);
    QVERIFY(named[0].voxelSizeUm > 9.36 && named[0].voxelSizeUm < 9.37);
}

void TestLineAnnotationDatasetSets::severalSurfacesAndForeignSurfacesAreLabelled()
{
    // Open-data PHerc0139: two surface predictions of the 9.362 um overview
    // scan, none of the 2.399 um scan the fiber model ran on.
    const std::vector<ProjectVolumeInfo> infos{
        {"20250728140407", "20250728140407-9.362um-1.2m-113keV-masked",
         {"vc-open-data-volume-id:20250728140407", "vc-open-data-voxel-size-um:9.362000"}, {1, 1, 1}, 9.362},
        {"s1", "20250728140407-surface-20250701154204-surface-recto-090",
         {"prediction", "surface-prediction", "vc-open-data-volume-id:20250728140407"}, {1, 1, 1}, 0.0},
        {"s2", "20250728140407-surface-20260413222639-surface-m7-L0-th0.2",
         {"prediction", "surface-prediction", "vc-open-data-volume-id:20250728140407"}, {1, 1, 1}, 0.0},
        {"20260102150214", "20260102150214-2.399um-0.2m-78keV-masked",
         {"vc-open-data-volume-id:20260102150214", "vc-open-data-voxel-size-um:2.399000"}, {1, 1, 1}, 2.399},
    };
    const auto volumes = classifyProjectVolumes(infos, {});
    const auto scans = rawScanOptions(volumes);
    // The fine scan has no surface prediction of its own: none defaults.
    QVERIFY(!defaultSurfaceVolumeId(volumes, "20260102150214").has_value());
    const auto fine = volumeSelectorOptions(volumes, scans, "20260102150214", "", "", "", "20260102150214");
    const std::vector<std::string> fineExpected{"raw scan, 2.399 \xC2\xB5m"};
    QVERIFY(labels(fine) == fineExpected);
    // The overview scan has two: the newest run wins the default, and the set
    // lists exactly the selected one as "surface".
    const auto newest = defaultSurfaceVolumeId(volumes, "20250728140407");
    QVERIFY(newest.has_value());
    QCOMPARE(*newest, std::string("s2"));
    QCOMPARE(surfaceMenuLabel(volumes[1]), std::string("20250701154204-surface-recto-090"));
    const auto overview = volumeSelectorOptions(volumes, scans, "20250728140407", "", "", "s1", "20250728140407");
    const std::vector<std::string> overviewExpected{"raw scan, 9.362 \xC2\xB5m", "surface"};
    QVERIFY(labels(overview) == overviewExpected);
    QCOMPARE(overview[1].id, std::string("s1"));
    // A recorded surface of another scan is not listed.
    const auto foreign = volumeSelectorOptions(volumes, scans, "20260102150214", "", "", "s1", "20260102150214");
    QVERIFY(labels(foreign) == fineExpected);
}

void TestLineAnnotationDatasetSets::channelsTakeTheirDatasetsScan()
{
    auto volumes = classifyAll(pherc1451Volumes());
    const auto scans = rawScanOptions(volumes);
    const std::vector<DatasetInfo> datasets{
        {"/cache/lasagna-20260724.lasagna.json", {"vc-open-data-volume-id:20260319101107"}, std::nullopt},
        {"/cache/las-sd1-40cc618f.lasagna.json", {"vc-open-data-volume-id:20260319101107"}, std::nullopt},
    };
    assignDatasetScansToChannels(volumes, datasets, scans);
    for (const auto& v : volumes) {
        if (v.kind == ProjectVolumeKind::Lasagna || v.kind == ProjectVolumeKind::Fiber) {
            QCOMPARE(v.scanKey, std::string("20260319101107"));
        }
    }
}

QTEST_APPLESS_MAIN(TestLineAnnotationDatasetSets)
#include "test_line_annotation_dataset_sets.moc"
