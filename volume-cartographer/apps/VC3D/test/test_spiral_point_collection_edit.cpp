#include "SpiralPointCollectionEdit.hpp"
#include "overlays/ScreenSpacePointIndex.hpp"

#include <QJsonArray>
#include <QTest>

class SpiralPointCollectionEditTest : public QObject
{
    Q_OBJECT
private slots:
    void importsNumericOrderAndPreservesCoordinates();
    void appendEraseReverseAndSerialize();
    void transientSurfacePositionsFollowPointEdits();
    void projectionPositionsUsePerDraftRetainedStorage();
    void deletionCanBeUndoneWithoutInventingAnEdit();
    void screenSpaceHitIndexVisitsNearbyCellsOnly();
    void hoverScaleAndIncompleteNewCollection();
    void linkedCollectionsAreReadOnly();
    void relativeWindingsCountFromZeroAndMirrorOnReverse();
    void relativeImportKeepsAnnotationsAndRequiresThemForEditing();
};

namespace {
QJsonObject point(QJsonArray position, qint64 creationTime,
                  QJsonArray links = {})
{
    QJsonObject result{{QStringLiteral("p"), position},
                       {QStringLiteral("creation_time"), creationTime},
                       {QStringLiteral("custom"), QStringLiteral("preserved")}};
    if (!links.isEmpty()) result[QStringLiteral("links")] = links;
    return result;
}

QJsonDocument sourceDocument()
{
    return QJsonDocument(QJsonObject{
        {QStringLiteral("vc_pointcollections_json_version"), QStringLiteral("1")},
        {QStringLiteral("top_custom"), 42},
        {QStringLiteral("collections"), QJsonObject{
            {QStringLiteral("10"), QJsonObject{
                {QStringLiteral("name"), QStringLiteral("ten")},
                {QStringLiteral("color"), QJsonArray{0.1, 0.2, 0.3}},
                {QStringLiteral("collection_custom"), true},
                {QStringLiteral("points"), QJsonObject{
                    {QStringLiteral("9"), point({9.125, 2.25, 3.5}, 90)},
                    {QStringLiteral("2"), point({2.125, 4.25, 6.5}, 20)},
                }},
            }},
            {QStringLiteral("2"), QJsonObject{
                {QStringLiteral("name"), QStringLiteral("two")},
                {QStringLiteral("points"), QJsonObject{
                    {QStringLiteral("1"), point({1.0, 1.0, 1.0}, 1)},
                    {QStringLiteral("0"), point({0.0, 0.0, 0.0}, 0)},
                }},
            }},
        }},
    });
}
}

void SpiralPointCollectionEditTest::importsNumericOrderAndPreservesCoordinates()
{
    auto drafts = vc3d::spiral::importEditablePcls(
        sourceDocument(), 4.0, QString(64, QLatin1Char('a')), true);
    QCOMPARE(drafts.size(), std::size_t(2));
    QCOMPARE(drafts[0].collectionId, QStringLiteral("2"));
    QCOMPARE(drafts[1].collectionId, QStringLiteral("10"));
    QCOMPARE(drafts[1].points[0].previewPosition[0], 8.5f);
    QCOMPARE(drafts[1].points[1].previewPosition[0], 36.5f);

    const QJsonObject serialized = drafts[1].replacementDocument().object();
    QCOMPARE(serialized.value(QStringLiteral("top_custom")).toInt(), 42);
    const QJsonObject collection = serialized.value(QStringLiteral("collections"))
                                       .toObject().value(QStringLiteral("10")).toObject();
    QVERIFY(collection.value(QStringLiteral("collection_custom")).toBool());
    const QJsonArray original = collection.value(QStringLiteral("points")).toObject()
                                    .value(QStringLiteral("0")).toObject()
                                    .value(QStringLiteral("p")).toArray();
    QCOMPARE(original[0].toDouble(), 2.125);
}

void SpiralPointCollectionEditTest::appendEraseReverseAndSerialize()
{
    auto draft = vc3d::spiral::importEditablePcls(
        sourceDocument(), 4.0, QString(64, QLatin1Char('b')), true)[1];
    draft.appendPreviewPoint({80.0f, 40.0f, 20.0f});
    QVERIFY(draft.erase(1));
    draft.reverse();
    const auto once = draft.points;
    draft.reverse();
    draft.reverse();
    QCOMPARE(draft.points.size(), once.size());
    for (std::size_t index = 0; index < once.size(); ++index)
        QCOMPARE(draft.points[index].previewPosition, once[index].previewPosition);

    const QJsonObject points = draft.replacementDocument().object()
                                   .value(QStringLiteral("collections")).toObject()
                                   .value(QStringLiteral("10")).toObject()
                                   .value(QStringLiteral("points")).toObject();
    QCOMPARE(points.keys(), QStringList({QStringLiteral("0"), QStringLiteral("1")}));
    const qint64 first = points.value(QStringLiteral("0")).toObject()
                             .value(QStringLiteral("creation_time")).toInteger();
    const qint64 second = points.value(QStringLiteral("1")).toObject()
                              .value(QStringLiteral("creation_time")).toInteger();
    QVERIFY(first < second);
    const QJsonArray appended = points.value(QStringLiteral("0")).toObject()
                                    .value(QStringLiteral("p")).toArray();
    QCOMPARE(appended[0].toDouble(), 20.0);
    QCOMPARE(appended[1].toDouble(), 10.0);
    QCOMPARE(appended[2].toDouble(), 5.0);
}

void SpiralPointCollectionEditTest::transientSurfacePositionsFollowPointEdits()
{
    vc3d::spiral::EditablePclDraft draft;
    draft.appendPreviewPoint({1.0f, 2.0f, 3.0f}, QPointF(10.0, 20.0));
    draft.appendPreviewPoint({4.0f, 5.0f, 6.0f}, QPointF(30.0, 40.0));

    QVERIFY(draft.points[0].previewSurfacePosition.has_value());
    QCOMPARE(*draft.points[0].previewSurfacePosition, QPointF(10.0, 20.0));
    draft.reverse();
    QCOMPARE(*draft.points[0].previewSurfacePosition, QPointF(30.0, 40.0));
    QVERIFY(draft.erase(1));
    QCOMPARE(draft.points.size(), std::size_t{1});
    QCOMPARE(*draft.points[0].previewSurfacePosition, QPointF(30.0, 40.0));

    const QJsonObject serializedPoint = draft.replacementDocument().object()
        .value(QStringLiteral("collections")).toObject()
        .value(QString()).toObject()
        .value(QStringLiteral("points")).toObject()
        .value(QStringLiteral("0")).toObject();
    QVERIFY(!serializedPoint.contains(QStringLiteral("previewSurfacePosition")));
}

void SpiralPointCollectionEditTest::projectionPositionsUsePerDraftRetainedStorage()
{
    vc3d::spiral::EditablePclDraft first;
    vc3d::spiral::EditablePclDraft second;
    first.appendPreviewPoint({1.0f, 2.0f, 3.0f});
    first.appendPreviewPoint({4.0f, 5.0f, 6.0f});
    second.appendPreviewPoint({7.0f, 8.0f, 9.0f});
    second.appendPreviewPoint({10.0f, 11.0f, 12.0f});

    const auto& firstProjection = first.projectionPositions();
    const auto& secondProjection = second.projectionPositions();
    QCOMPARE(firstProjection.size(), std::size_t{2});
    QCOMPARE(secondProjection.size(), std::size_t{2});
    QVERIFY(firstProjection.data() != secondProjection.data());

    const cv::Vec3f* firstStorage = firstProjection.data();
    QCOMPARE(first.projectionPositions().data(), firstStorage);
    QCOMPARE(first.projectionPositions()[1], cv::Vec3f(4.0f, 5.0f, 6.0f));

    first.reverse();
    QCOMPARE(first.projectionPositions().data(), firstStorage);
    QCOMPARE(first.projectionPositions()[0], cv::Vec3f(4.0f, 5.0f, 6.0f));
}

void SpiralPointCollectionEditTest::linkedCollectionsAreReadOnly()
{
    QJsonObject root = sourceDocument().object();
    QJsonObject collections = root.value(QStringLiteral("collections")).toObject();
    QJsonObject other = collections.value(QStringLiteral("2")).toObject();
    other[QStringLiteral("windings_linked")] = QJsonArray{10};
    collections[QStringLiteral("2")] = other;
    root[QStringLiteral("collections")] = collections;
    const auto drafts = vc3d::spiral::importEditablePcls(
        QJsonDocument(root), 1.0, QString(64, QLatin1Char('c')), true);
    QVERIFY(!drafts[0].editable);
    QVERIFY(!drafts[1].editable);
}

void SpiralPointCollectionEditTest::deletionCanBeUndoneWithoutInventingAnEdit()
{
    auto draft = vc3d::spiral::importEditablePcls(
        sourceDocument(), 1.0, QString(64, QLatin1Char('d')), true)[0];
    QVERIFY(!draft.dirty);
    draft.setDeleted(true);
    QVERIFY(draft.deleted);
    QVERIFY(draft.dirty);
    draft.setDeleted(false);
    QVERIFY(!draft.deleted);
    QVERIFY(!draft.dirty);

    draft.reverse();
    draft.setDeleted(true);
    draft.setDeleted(false);
    QVERIFY(draft.dirty);
}

void SpiralPointCollectionEditTest::screenSpaceHitIndexVisitsNearbyCellsOnly()
{
    ScreenSpacePointIndex index;
    constexpr std::size_t count = 10000;
    index.reserve(count);
    for (std::size_t point = 0; point < count; ++point) {
        index.insert({QPointF(static_cast<qreal>(point * 32 + 4), 4.0),
                      static_cast<std::uint64_t>(point), 0, point});
    }
    QCOMPARE(index.size(), count);

    std::size_t visited = 0;
    const auto hit = index.closest(
        QPointF(5000 * 32 + 5.0, 4.0), 8.0,
        [&visited](std::size_t) {
            ++visited;
            return true;
        });
    QVERIFY(hit.has_value());
    QCOMPARE(*hit, std::size_t{5000});
    QCOMPARE(visited, std::size_t{1});

    ScreenSpacePointIndex ties;
    ties.insert({QPointF(15.0, 8.0), 9, 1, 91});
    ties.insert({QPointF(17.0, 8.0), 2, 4, 24});
    const auto deterministic = ties.closest(QPointF(16.0, 8.0), 8.0);
    QVERIFY(deterministic.has_value());
    QCOMPARE(*deterministic, std::size_t{24});

    const auto filtered = ties.closest(
        QPointF(16.0, 8.0), 8.0,
        [](std::size_t payload) { return payload != 24; });
    QVERIFY(filtered.has_value());
    QCOMPARE(*filtered, std::size_t{91});
}

void SpiralPointCollectionEditTest::hoverScaleAndIncompleteNewCollection()
{
    QCOMPARE(vc3d::spiral::editablePclPointRadius(5.0, false), 5.0);
    QCOMPARE(vc3d::spiral::editablePclPointRadius(5.0, true), 7.0);

    vc3d::spiral::EditablePclDraft draft;
    QVERIFY(draft.isIncompleteNewCollection());
    draft.appendPreviewPoint({1.0f, 2.0f, 3.0f});
    QVERIFY(draft.isIncompleteNewCollection());
    draft.appendPreviewPoint({4.0f, 5.0f, 6.0f});
    QVERIFY(!draft.isIncompleteNewCollection());
    draft.collectionId = QStringLiteral("12");
    QVERIFY(!draft.isIncompleteNewCollection());
}

void SpiralPointCollectionEditTest::relativeWindingsCountFromZeroAndMirrorOnReverse()
{
    using vc3d::spiral::editablePclPointWinding;
    using vc3d::spiral::editablePclPointWindingLabel;

    vc3d::spiral::EditablePclDraft same;
    same.appendPreviewPoint({1.0f, 2.0f, 3.0f});
    same.appendPreviewPoint({4.0f, 5.0f, 6.0f});
    for (const auto& point : same.points) {
        QVERIFY(point.sourcePayload.value(QStringLiteral("wind_a")).isNull());
        QVERIFY(!editablePclPointWinding(point));
        QVERIFY(editablePclPointWindingLabel(point).isEmpty());
    }
    same.reverse();
    QVERIFY(same.points[0].sourcePayload.value(QStringLiteral("wind_a")).isNull());

    vc3d::spiral::EditablePclDraft relative;
    relative.role = vc3d::spiral::PclRole::Relative;
    relative.appendPreviewPoint({1.0f, 0.0f, 0.0f});
    relative.appendPreviewPoint({2.0f, 0.0f, 0.0f});
    relative.appendPreviewPoint({3.0f, 0.0f, 0.0f});
    const auto windings = [](const vc3d::spiral::EditablePclDraft& draft) {
        QList<double> values;
        for (const auto& point : draft.points)
            values.push_back(editablePclPointWinding(point).value_or(-1.0));
        return values;
    };
    QCOMPARE(windings(relative), QList<double>({0.0, 1.0, 2.0}));
    QCOMPARE(editablePclPointWindingLabel(relative.points[2]), QStringLiteral("2"));

    // Flipping reverses the chain and mirrors the annotations, so the
    // winding count still ascends along the new order: the constraint's
    // direction is what changes, not just the point ids.
    relative.reverse();
    QCOMPARE(relative.points[0].previewPosition, cv::Vec3f(3.0f, 0.0f, 0.0f));
    QCOMPARE(windings(relative), QList<double>({0.0, 1.0, 2.0}));
    relative.appendPreviewPoint({4.0f, 0.0f, 0.0f});
    QCOMPARE(windings(relative), QList<double>({0.0, 1.0, 2.0, 3.0}));
    QVERIFY(relative.erase(1));
    QCOMPARE(windings(relative), QList<double>({0.0, 2.0, 3.0}));
    relative.reverse();
    QCOMPARE(windings(relative), QList<double>({0.0, 1.0, 3.0}));

    const QJsonObject points = relative.replacementDocument().object()
        .value(QStringLiteral("collections")).toObject()
        .value(QString()).toObject()
        .value(QStringLiteral("points")).toObject();
    QCOMPARE(points.value(QStringLiteral("0")).toObject()
                 .value(QStringLiteral("wind_a")).toDouble(), 0.0);
    QCOMPARE(points.value(QStringLiteral("2")).toObject()
                 .value(QStringLiteral("wind_a")).toDouble(), 3.0);
}

void SpiralPointCollectionEditTest::relativeImportKeepsAnnotationsAndRequiresThemForEditing()
{
    QJsonObject annotated = point({1.0, 1.0, 1.0}, 1);
    annotated[QStringLiteral("wind_a")] = 9.0;
    QJsonObject annotatedTwo = point({2.0, 2.0, 2.0}, 2);
    annotatedTwo[QStringLiteral("wind_a")] = 10.0;
    QJsonObject annotatedThree = point({3.0, 3.0, 3.0}, 3);
    annotatedThree[QStringLiteral("wind_a")] = 12.0;
    const QJsonDocument document(QJsonObject{
        {QStringLiteral("vc_pointcollections_json_version"), QStringLiteral("1")},
        {QStringLiteral("collections"), QJsonObject{
            {QStringLiteral("4"), QJsonObject{
                {QStringLiteral("name"), QStringLiteral("wraps")},
                {QStringLiteral("points"), QJsonObject{
                    {QStringLiteral("0"), annotated},
                    {QStringLiteral("1"), annotatedTwo},
                    {QStringLiteral("2"), annotatedThree},
                }},
            }},
            {QStringLiteral("5"), QJsonObject{
                {QStringLiteral("name"), QStringLiteral("mixed")},
                {QStringLiteral("points"), QJsonObject{
                    {QStringLiteral("0"), annotated},
                    {QStringLiteral("1"), point({2.0, 2.0, 2.0}, 2)},
                }},
            }},
        }},
    });
    auto drafts = vc3d::spiral::importEditablePcls(
        document, 1.0, QString(64, QLatin1Char('e')), true,
        vc3d::spiral::PclRole::Relative);
    QCOMPARE(drafts.size(), std::size_t(2));
    QCOMPARE(static_cast<int>(drafts[0].role),
             static_cast<int>(vc3d::spiral::PclRole::Relative));
    QVERIFY(drafts[0].editable);
    // An unannotated point would be dropped by the fitter on replacement.
    QVERIFY(!drafts[1].editable);

    auto& wraps = drafts[0];
    wraps.appendPreviewPoint({4.0f, 4.0f, 4.0f});
    QCOMPARE(*vc3d::spiral::editablePclPointWinding(wraps.points[3]), 13.0);
    // Mirroring keeps the range: 9, 10, 12, 13 -> 9, 10, 12, 13 reversed
    // in space, i.e. the former first point now carries 13.
    wraps.reverse();
    QCOMPARE(wraps.points[3].previewPosition, cv::Vec3f(1.0f, 1.0f, 1.0f));
    QCOMPARE(*vc3d::spiral::editablePclPointWinding(wraps.points[3]), 13.0);
    QCOMPARE(*vc3d::spiral::editablePclPointWinding(wraps.points[0]), 9.0);
    QCOMPARE(*vc3d::spiral::editablePclPointWinding(wraps.points[1]), 10.0);
    QCOMPARE(*vc3d::spiral::editablePclPointWinding(wraps.points[2]), 12.0);

    // Same-winding import ignores annotations and stays editable.
    const auto sameWinding = vc3d::spiral::importEditablePcls(
        document, 1.0, QString(64, QLatin1Char('f')), true);
    QVERIFY(sameWinding[1].editable);
}

QTEST_MAIN(SpiralPointCollectionEditTest)
#include "test_spiral_point_collection_edit.moc"
