// Committed Spiral fits read from the object store (SpiralStoredFits): one
// generation written both as a published preview artifact's winding map and
// as a committed surface must read back as the same windings and components,
// and the store's surface must keep the generation's points.
#include "SpiralStoredFits.hpp"

#include "vc/core/util/ImageMat.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/SurfaceCreate.hpp>

#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <opencv2/imgcodecs.hpp>

#include <cstring>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace vc::objects;
using Json = nlohmann::json;
using vc3d::spiral::PreviewComponent;

namespace {
void require(bool value, const std::string& message)
{
    if (!value) throw std::runtime_error(message);
}

template <typename Fn>
void requireThrows(Fn&& fn, const std::string& fragment, const std::string& what)
{
    try {
        fn();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(fragment) != std::string::npos,
                what + ": unexpected error: " + error.what());
        return;
    }
    throw std::runtime_error(what + ": no error");
}

const std::string kVolume = "20260411134726";
constexpr int kRows = 5, kColumns = 7;

// A flattened generation: windings 4 (columns 0-2) and 5 (columns 3-6); the
// vertex (0, 0) has no coordinates and no winding.
struct Generation {
    cv::Mat_<cv::Vec3f> points = cv::Mat_<cv::Vec3f>(kRows, kColumns);
    cv::Mat_<int32_t> windings = cv::Mat_<int32_t>(kRows, kColumns);
    Json bounds = Json::array({
        {{"winding", 4}, {"row_begin", 0}, {"row_end", kRows}, {"column_begin", 0}, {"column_end", 3}},
        {{"winding", 5}, {"row_begin", 0}, {"row_end", kRows}, {"column_begin", 3}, {"column_end", kColumns}}});
    Json windingIds = Json::array({4, 5});

    Generation()
    {
        for (int row = 0; row < kRows; ++row)
            for (int column = 0; column < kColumns; ++column) {
                points(row, column) = cv::Vec3f(10.f + 2.5f * column, 20.f + 1.5f * row, 30.f + row + column);
                windings(row, column) = column < 3 ? 4 : 5;
            }
        // Row 0 still spans column 0 through winding 4's other rows.
        points(0, 0) = cv::Vec3f(-1.f, -1.f, -1.f);
        windings(0, 0) = -1;
    }
};

QJsonArray qtArray(const Json& value)
{
    return QJsonDocument::fromJson(QByteArray::fromStdString(Json{{"v", value}}.dump()))
        .object().value(QStringLiteral("v")).toArray();
}

GeometryArray int32Channel(const std::string& name, const cv::Mat_<int32_t>& values)
{
    GeometryArray result{name, "<i4", {values.rows, values.cols}, {}};
    result.bytes.resize(values.total() * sizeof(int32_t));
    std::memcpy(result.bytes.data(), values.ptr(), result.bytes.size());
    return result;
}

struct Fixture {
    std::shared_ptr<ObjectStoreSession> session = std::make_shared<ObjectStoreSession>();
    AttachmentIdentity attachment;

    explicit Fixture(const fs::path& path)
    {
        OpenOptions options; options.author = "spiral-stored-fits-test";
        { auto created = ObjectStore::create(path, options); }
        attachment = session->attach(path, options);
    }

    std::string commit(const Generation& generation, std::vector<std::string> tags, Json bounds,
                       double scaleFactor = 1.0, const std::string& name = "s1-spiral-g1")
    {
        const std::string uuid = util::newUuid();
        vc::geometry::Surface surface;
        surface.rows = kRows;
        surface.columns = kColumns;
        surface.scale = {0.05, 0.05};
        for (const auto& p : generation.points) surface.points.push_back({p[0], p[1], p[2]});
        surface.channels.push_back(int32Channel("winding_id", generation.windings));
        const Json meta{{"winding_bounds", bounds}, {"component_winding_ids", generation.windingIds},
                        {"source_fit_iteration", 120}};
        auto request = session->newRequest(attachment);
        request.expectedObjects = {{uuid, std::nullopt}};
        request.operations = createSurfaceOperations(uuid, name, tags, volumeCoordinates(kVolume, scaleFactor),
                                                     SurfaceProvenance{"spiral-fitting", Json::object(), {kVolume}}, meta,
                                                     vc::geometry::encode(surface));
        session->submit(attachment, request);
        return uuid;
    }
};

void aStoredFitReadsAsItsArtifact(const fs::path& directory)
{
    const Generation generation;
    // The artifact's winding map: float32, as lasagna_publish writes it (here
    // OpenCV stands in for that writer), read as loadPreview reads it.
    const fs::path mapPath = directory / "winding-ids.tif";
    cv::Mat floats;
    generation.windings.convertTo(floats, CV_32F);
    require(cv::imwrite(mapPath.string(), floats), "the artifact winding map is written");
    QString error;
    const auto artifactWindings = vc3d::spiral::windingIdsFromImage(
        vc::core::util::imageMat(vcr::sources::readImagePage(mapPath.string(), 0)), &error);
    require(artifactWindings.has_value(), "the artifact map decodes: " + error.toStdString());
    const auto artifactComponents = vc3d::spiral::parseWindingBounds(
        qtArray(generation.bounds), qtArray(generation.windingIds), &error);
    require(artifactComponents && vc3d::spiral::windingMapMatchesBounds(*artifactWindings, *artifactComponents, &error),
            "the artifact's bounds hold: " + error.toStdString());

    Fixture fixture(directory / "fits.vcobj");
    const std::string uuid = fixture.commit(generation, {vc3d::spiral::kSpiralFitTag}, generation.bounds);
    const auto fit = vc3d::spiral::readStoredFit(fixture.session, fixture.attachment, uuid);
    require(fit.uuid == uuid && fit.name == "s1-spiral-g1" && fit.volume == kVolume, "the fit's identity");
    require(cv::countNonZero(fit.windingIds != *artifactWindings) == 0,
            "the stored winding IDs equal the artifact's");
    require(fit.components.size() == artifactComponents->size(), "as many components as the artifact");
    for (std::size_t i = 0; i < fit.components.size(); ++i) {
        const PreviewComponent& a = fit.components[i];
        const PreviewComponent& b = (*artifactComponents)[i];
        require(a.winding == b.winding && a.rowBegin == b.rowBegin && a.rowEnd == b.rowEnd
                    && a.columnBegin == b.columnBegin && a.columnEnd == b.columnEnd,
                "the stored components equal the artifact's");
    }
    const cv::Mat_<cv::Vec3f>& points = *fit.surface->rawPointsPtr();
    require(points.rows == kRows && points.cols == kColumns, "the stored grid's size");
    for (int row = 0; row < kRows; ++row)
        for (int column = 0; column < kColumns; ++column)
            require(points(row, column) == generation.points(row, column), "the stored points are the generation's");
    // Read at that geometry revision too.
    require(vc3d::spiral::readStoredFit(fixture.session, fixture.attachment, uuid, fit.geometryRevision)
                    .geometryRevision == fit.geometryRevision,
            "a pinned revision reads");
    {
        OpenOptions options; options.author = "historical-fit-test";
        auto store = ObjectStore::open(fixture.session->path(fixture.attachment), options);
        auto txn = store.write();
        auto record = *store.objectByUuid(txn, uuid);
        record.name = "new fit";
        store.upsertObject(txn, record);
        store.setMeta(txn, uuid, "source_fit_iteration", "240");
        auto changed = quad_surface_grid(*fit.surface);
        changed.points[1][0] += 10;
        store.putGeometry(txn, uuid, vc::geometry::encode(changed));
        txn.commit();
        auto removal = store.write(); store.deleteObject(removal, uuid); removal.commit();
    }
    const auto historical = vc3d::spiral::readStoredFit(fixture.session, fixture.attachment, uuid, fit.geometryRevision);
    require(historical.name == fit.name && historical.surface->meta.at("source_fit_iteration") == 120 &&
            (*historical.surface->rawPointsPtr())(0, 1) == generation.points(0, 1),
            "historical fits retain their metadata and geometry after replacement and deletion");
}

void onlyCommittedFitsAreListedAndRead(const fs::path& directory)
{
    const Generation generation;
    Fixture fixture(directory / "listing.vcobj");
    const std::string fit = fixture.commit(generation, {vc3d::spiral::kSpiralFitTag}, generation.bounds);
    const std::string other = fixture.commit(generation, {"patch"}, generation.bounds, 1.0, "a patch");
    const auto listed = vc3d::spiral::listStoredFits(*fixture.session, fixture.attachment);
    require(listed.size() == 1 && listed[0].uuid == fit && listed[0].sourceIteration == 120,
            "only the committed fit is listed, with its iteration");
    requireThrows([&] { vc3d::spiral::readStoredFit(fixture.session, fixture.attachment, other); },
                  "is not a committed Spiral fit", "a surface not tagged spiral-fit");
    requireThrows([&] { vc3d::spiral::readStoredFit(fixture.session, fixture.attachment, util::newUuid()); },
                  "is not in the attached object store", "a fit the store does not hold");
    // A changed surface the workspace counts as another input unless it is
    // a committed fit (SpiralWorkspace::noteStoreChanges).
    const auto fits = vc3d::spiral::spiralFitsAmong(*fixture.session, fixture.attachment,
                                                    {fit, other, util::newUuid()});
    require(fits == std::set<std::string>{fit}, "only the committed fit is recognized among changed surfaces");
}

void aFitWhoseMapDisagreesWithItsBoundsIsRefused(const fs::path& directory)
{
    const Generation generation;
    Fixture fixture(directory / "refused.vcobj");
    Json shifted = generation.bounds;
    shifted[1]["column_begin"] = 4;
    const std::string wrongBounds = fixture.commit(generation, {vc3d::spiral::kSpiralFitTag}, shifted);
    requireThrows([&] { vc3d::spiral::readStoredFit(fixture.session, fixture.attachment, wrongBounds); },
                  "do not match the winding-ID map", "bounds the map does not have");
    const std::string scaled = fixture.commit(generation, {vc3d::spiral::kSpiralFitTag}, generation.bounds, 2.0);
    requireThrows([&] { vc3d::spiral::readStoredFit(fixture.session, fixture.attachment, scaled); },
                  "at scale 1", "a fit bound at another scale");
}
}  // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    try {
        QTemporaryDir directory;
        const fs::path root = directory.path().toStdString();
        aStoredFitReadsAsItsArtifact(root);
        onlyCommittedFitsAreListedAndRead(root);
        aFitWhoseMapDisagreesWithItsBoundsIsRefused(root);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "spiral stored fits: ok\n";
    return 0;
}
