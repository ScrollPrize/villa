#include "SessionDrafts.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <stdexcept>

#include <QPointer>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <utility>
#include <cstring>

using namespace vc::objects;

SessionDrafts::SessionDrafts(std::shared_ptr<ObjectStoreSession> session, QObject* parent)
    : QObject(parent), _session(std::move(session))
{
}

SessionDrafts::~SessionDrafts()
{
    _resolving.waitForFinished();
    _checkpoint.waitForFinished();
    if (_active) checkpointNow();
}

bool SessionDrafts::start(const std::filesystem::path& directory, int intervalMs)
{
    if (_active) return true;
    try {
        const auto recovery = _session->useDraftDirectory(directory);
        _recovered.insert(recovery.drafts.begin(), recovery.drafts.end());
        for (const auto& pending : recovery.pending) _stalePending.insert(pending.requestUuid);
    } catch (const std::exception& error) {
        _error = QString::fromUtf8(error.what());
        return false;
    }
    _active = true;
    if (intervalMs > 0) {
        auto* timer = new QTimer(this);
        timer->setInterval(intervalMs);
        connect(timer, &QTimer::timeout, this, &SessionDrafts::tick);
        timer->start();
    }
    return true;
}

bool SessionDrafts::recovered(const std::string& objectUuid) const
{
    return _recovered.contains(objectUuid) && _session->hasDraft(objectUuid);
}

std::vector<std::string> SessionDrafts::recoveredDrafts() const
{
    std::vector<std::string> result;
    for (const auto& uuid : _recovered)
        if (_session->hasDraft(uuid)) result.push_back(uuid);
    return result;
}

void SessionDrafts::forgetRecovered(const std::string& objectUuid)
{
    _recovered.erase(objectUuid);
}

bool SessionDrafts::canRestoreSurface(QuadSurface& surface, const vc::geometry::Surface& draft)
{
    const auto* grid = surface.rawPointsPtr();
    if (!grid) return false;
    if (grid->rows == draft.rows && grid->cols == draft.columns) return true;
    // Like Apply, refuse resized grids while component column ranges cannot follow them.
    const auto components = surface.meta.find("components");
    if (components != surface.meta.end() && components->is_array() && !components->empty()) return false;
    const auto recorded = [&](const std::string& name) {
        return std::any_of(draft.channels.begin(), draft.channels.end(),
                           [&](const auto& channel) { return channel.name == name; });
    };
    if (!recorded("growth_exclusions")) return false;
    for (const auto& name : surface.channelNames())
        if (name != "mask" && !recorded(name)) return false;
    return true;
}

namespace {
using vc::geometry::SurfacePoint;
constexpr SurfacePoint missing{-1, -1, -1};

// A registered view can place a real vertex at negative Z or at exactly the
// missing-value sentinel. Keep its presence separate from its display XYZ.
std::vector<SurfacePoint> draftPoints(QuadSurface& surface, const cv::Rect& area,
                                      std::vector<uint8_t>& coordinatePresence)
{
    const auto points = surface.pointsForStorage(area);
    const bool displayed = surface.hasTransformedCoordinates;
    const auto mask = displayed ? surface.vertexMask(area) : cv::Mat_<uint8_t>{};
    const auto hiddenSentinels = displayed ? surface.retainedDisplaySentinels(area) : std::vector<uint8_t>{};
    if (displayed) coordinatePresence.reserve(points.total());
    std::vector<SurfacePoint> result;
    result.reserve(points.total());
    for (int row = 0; row < points.rows; ++row)
        for (int col = 0; col < points.cols; ++col) {
            const auto p = points(row, col);
            const SurfacePoint point{p[0], p[1], p[2]};
            bool present = vc::geometry::validSurfacePoint(point);
            if (displayed) {
                present = std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
                    (point != missing || (!mask.empty() && mask(row, col)) ||
                     (!hiddenSentinels.empty() && hiddenSentinels[result.size()]));
                coordinatePresence.push_back(present);
            }
            result.push_back(present ? point : missing);
        }
    return result;
}
}

void SessionDrafts::restoreSurface(QuadSurface& surface, const vc::geometry::Surface& draft)
{
    draft.validateDraft();
    if (!canRestoreSurface(surface, draft))
        throw std::invalid_argument("Recovered draft grid differs from the open surface");
    auto* grid = surface.rawPointsPtr();
    const cv::Size size(int(draft.columns), int(draft.rows));
    const bool resized = grid->size() != size;
    const auto retainedPoints = surface.pointsForStorage();
    const auto retainedMask = surface.vertexMask();
    cv::Mat_<uint8_t> exclusions = resized ? cv::Mat{} : surface.channel("growth_exclusions", SURF_CHANNEL_NORESIZE);
    if (exclusions.empty()) exclusions = cv::Mat_<uint8_t>(size, uint8_t{0});
    else exclusions = exclusions.clone();
    std::vector<std::pair<std::string, cv::Mat>> channels;
    bool recordedExclusions = false;
    const std::vector<int64_t> shape{draft.rows, draft.columns};
    for (const auto& channel : draft.channels) {
        const auto pixels = quad_channel_mat(channel);
        if (channel.name == "growth_exclusions") {
            if (channel.dtype != "|u1" || channel.shape != shape)
                throw std::invalid_argument("Recovered growth exclusions differ from the surface grid");
            exclusions = pixels;
            recordedExclusions = true;
        } else if (channel.name == "generations") {
            // Surface validation already checks whole-multiple alignment.
            if (channel.dtype != "<u2" || channel.shape.size() != 2)
                throw std::invalid_argument("Recovered generations must be a two-dimensional uint16 channel");
        }
        if (channel.name != "growth_exclusions") channels.emplace_back(channel.name, pixels);
    }
    grid->create(size);
    cv::Mat_<uint8_t> mask(size);
    auto oldHidden = std::move(surface.maskedVertices);
    surface.maskedVertices.clear();
    for (int r = 0; r < grid->rows; ++r)
        for (int c = 0; c < grid->cols; ++c) {
            const auto i = std::size_t(r) * grid->cols + c;
            const auto& p = draft.points[i];
            mask(r, c) = draft.validity.empty() || draft.validity[i] ? 255 : 0;
            const bool hasPoint = draft.coordinatePresence.empty() ? vc::geometry::validSurfacePoint(p)
                                                                   : draft.coordinatePresence[i] != 0;
            const cv::Vec3f displayed{p[0], p[1], p[2]};
            if (!mask(r, c)) {
                if (hasPoint) {
                    surface.maskedVertices.set({c, r}, displayed);
                } else if (!recordedExclusions && !retainedMask.empty() && !retainedMask(r, c)) {
                    if (const auto previous = oldHidden.get({c, r}))
                        surface.maskedVertices.set({c, r}, *previous);
                    else if (vc::geometry::validSurfacePoint(retainedPoints(r, c)))
                        surface.maskedVertices.set({c, r}, retainedPoints(r, c));
                }
            }
            (*grid)(r, c) = mask(r, c) ? displayed : cv::Vec3f(-1, -1, -1);
        }
    surface.setChannel("mask", mask);
    surface.setChannel("growth_exclusions", exclusions);
    for (const auto& [name, pixels] : channels) surface.setChannel(name, pixels);
    surface.invalidateCache();
}

vc::geometry::Surface SessionDrafts::surfaceForDraft(QuadSurface& surface, bool fullChannels)
{
    const cv::Rect bounds(cv::Point{}, surface.rawPointsPtr()->size());
    vc::geometry::Surface draft{bounds.height, bounds.width, {surface.scale()[0], surface.scale()[1]}, {}};
    draft.points = draftPoints(surface, bounds, draft.coordinatePresence);
    if (!surface.channel("mask", SURF_CHANNEL_NORESIZE).empty())
        draft.validity = quad_channels_operation(surface.id, surface, {"mask"}, {}, bounds.height, bounds.width)
                             .channels.front().bytes;
    if (fullChannels) {
        auto names = surface.channelNames();
        std::erase(names, "mask");
        draft.channels = quad_channels_operation(surface.id, surface, names, {}, int(draft.rows), int(draft.columns)).channels;
        if (std::find(names.begin(), names.end(), "growth_exclusions") == names.end())
            draft.channels.push_back({"growth_exclusions", "|u1", {draft.rows, draft.columns},
                                      std::vector<uint8_t>(draft.points.size(), uint8_t{0})});
    } else if (!surface.channel("growth_exclusions", SURF_CHANNEL_NORESIZE).empty()) {
        std::vector<std::string> channels{"growth_exclusions"};
        if (!surface.channel("generations", SURF_CHANNEL_NORESIZE).empty()) channels.push_back("generations");
        draft.channels = quad_channels_operation(surface.id, surface, channels, {}, int(draft.rows), int(draft.columns)).channels;
    }
    return draft;
}

void SessionDrafts::updateSurfaceDraft(ObjectStoreSession& session, QuadSurface& surface, const cv::Rect& rectangle)
{
    const auto area = rectangle & cv::Rect(cv::Point{}, surface.rawPointsPtr()->size());
    if (area.empty()) return;
    std::vector<uint8_t> presence;
    const auto points = draftPoints(surface, area, presence);
    session.updateDraft(surface.id, area.y, area.x, area.height, area.width, points, {}, presence);
}

bool SessionDrafts::checkpointNow()
{
    if (!_active) return false;
    try {
        _session->checkpointDrafts();
        reportCheckpoint({});
        return true;
    } catch (const std::exception& error) {
        reportCheckpoint(QString::fromUtf8(error.what()));
        return false;
    }
}

void SessionDrafts::reportCheckpoint(const QString& error)
{
    if (error == _checkpointError) return;
    const bool wasFailing = !_checkpointError.isEmpty();
    _checkpointError = error;
    if (!error.isEmpty()) emit checkpointFailed(error);
    else if (wasFailing) emit checkpointRestored();
}

void SessionDrafts::tick()
{
    // Pending saves seen at the previous tick are not in flight any more.
    std::set<std::string> now;
    for (const auto& pending : _session->pendingSubmissions()) {
        if (_seenPending.contains(pending.requestUuid)) _stalePending.insert(pending.requestUuid);
        now.insert(pending.requestUuid);
    }
    _seenPending = std::move(now);
    resolvePending();

    // Checkpoints write whole changed grids: off the UI thread.
    if (_checkpoint.isRunning()) return;
    QPointer<SessionDrafts> self(this);
    auto session = _session;
    _checkpoint = QtConcurrent::run([self, session] {
        QString failure;
        try { session->checkpointDrafts(); }
        catch (const std::exception& error) { failure = QString::fromUtf8(error.what()); }
        QMetaObject::invokeMethod(qApp, [self, failure] { if (self) self->reportCheckpoint(failure); }, Qt::QueuedConnection);
    });
}

void SessionDrafts::resolvePending()
{
    if (!_active || _resolving.isRunning()) return;
    std::vector<std::pair<AttachmentIdentity, std::string>> work;
    const auto attachments = _session->attachments();
    for (const auto& pending : _session->pendingSubmissions()) {
        if (!_stalePending.contains(pending.requestUuid)) continue;
        const auto attachment = std::find_if(attachments.begin(), attachments.end(),
                                             [&](const auto& a) { return a.storeUuid == pending.storeUuid; });
        if (attachment != attachments.end()) work.emplace_back(*attachment, pending.requestUuid);
    }
    if (work.empty()) return;
    QPointer<SessionDrafts> self(this);
    auto session = _session;
    _resolving = QtConcurrent::run([self, session, work] {
        for (const auto& [attachment, request] : work) {
            QString message;
            try {
                const auto receipt = session->resolvePending(attachment, request);
                message = receipt.commitSequence
                    ? tr("An earlier save was committed (commit %1).").arg(*receipt.commitSequence)
                    : tr("An earlier save was committed (it changed nothing).");
            } catch (const TransactionConflict& conflict) {
                message = tr("An earlier save conflicts with newer changes and was not committed: %1")
                              .arg(QString::fromUtf8(conflict.what()));
            } catch (const std::invalid_argument&) {
                continue; // resolved meanwhile (by the editor that made it), or invalid: concluded
            } catch (const std::exception&) {
                continue; // still unreachable: stays pending for the next tick
            }
            QMetaObject::invokeMethod(qApp, [self, request, message] {
                if (!self) return;
                self->_stalePending.erase(request);
                emit self->pendingResolved(message);
            }, Qt::QueuedConnection);
        }
    });
}
