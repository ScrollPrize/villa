#include "InkPreviewClient.hpp"

#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaObject>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QNetworkReply>
#include <QNetworkRequest>

#include <algorithm>
#include <utility>

namespace {

int httpStatus(QNetworkReply* reply)
{
    return reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
}

QString seconds(int ms)
{
    return QString::number(ms / 1000.0, 'g', 3);
}

} // namespace

InkPreviewClient::InkPreviewClient(QObject* parent)
    : QObject(parent)
    , _network(new QNetworkAccessManager(this))
    , _url(QStringLiteral("http://127.0.0.1:8765"))
{
    // The server runs on this computer (or behind a local tunnel): never through a proxy,
    // and never to another host by a redirect. A 3xx answer is a failure.
    _network->setProxy(QNetworkProxy::NoProxy);
    _network->setRedirectPolicy(QNetworkRequest::ManualRedirectPolicy);
}

InkPreviewClient::~InkPreviewClient()
{
    cancel();
}

void InkPreviewClient::setServerUrl(const QUrl& url)
{
    cancel();
    _url = url;
}

void InkPreviewClient::setTimeouts(int healthMs, int previewMs)
{
    _healthTimeoutMs = std::max(1, healthMs);
    _previewTimeoutMs = std::max(1, previewMs);
}

QString InkPreviewClient::urlProblem(const QUrl& url)
{
    const QString scheme = url.scheme().toLower();
    if (!url.isValid() || url.host().isEmpty() || (scheme != QLatin1String("http") && scheme != QLatin1String("https"))) {
        return tr("Enter the server address as http://HOST:PORT, for example http://127.0.0.1:8765.");
    }
    return {};
}

QUrl InkPreviewClient::endpoint(const QString& path) const
{
    QUrl url = _url;
    url.setPath(path);
    url.setQuery(QString());
    url.setFragment(QString());
    return url;
}

QString InkPreviewClient::redirectError(int http) const
{
    return tr("The ink preview server at %1 answered with a redirect (HTTP %2); VC3D does not follow redirects.")
        .arg(_url.toString())
        .arg(http);
}

QString InkPreviewClient::transportError(QNetworkReply* reply) const
{
    // cancel() deletes the replies it drops, so a cancelled reply that gets here hit its timeout.
    const int timeoutMs = reply->request().transferTimeout();
    if (reply->error() == QNetworkReply::OperationCanceledError && timeoutMs > 0) {
        return tr("The ink preview server at %1 did not answer within %2 s.").arg(_url.toString(), seconds(timeoutMs));
    }
    return tr("No ink preview server answers at %1: %2").arg(_url.toString(), reply->errorString());
}

void InkPreviewClient::track(QNetworkReply* reply)
{
    _replies.erase(std::remove_if(_replies.begin(), _replies.end(),
                                  [](const QPointer<QNetworkReply>& r) { return r.isNull() || r->isFinished(); }),
                   _replies.end());
    _replies.emplace_back(reply);
}

void InkPreviewClient::cancel()
{
    ++_generation;
    auto replies = std::move(_replies);
    _replies.clear();
    // Deleting a reply aborts its request. abort() would report a POST's cancellation
    // twice on Qt 6.4 (two finished signals and a warning); a deleted reply reports nothing.
    for (const auto& reply : replies) {
        if (reply) {
            QObject::disconnect(reply, nullptr, this, nullptr);
            reply->deleteLater();
        }
    }
}

bool InkPreviewClient::busy() const
{
    return std::any_of(_replies.begin(), _replies.end(),
                       [](const QPointer<QNetworkReply>& r) { return !r.isNull() && !r->isFinished(); });
}

void InkPreviewClient::checkHealth(HealthDone done)
{
    const quint64 generation = _generation;
    if (const QString problem = urlProblem(_url); !problem.isEmpty()) {
        QMetaObject::invokeMethod(this, [this, generation, done, problem]() {
            if (generation == _generation) {
                done(std::nullopt, problem);
            }
        }, Qt::QueuedConnection);
        return;
    }

    QNetworkRequest request(endpoint(QStringLiteral("/health")));
    request.setTransferTimeout(_healthTimeoutMs);
    QNetworkReply* reply = _network->get(request);
    track(reply);
    // Single shot: a reply aborted by its transfer timeout can signal finished twice.
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation, done]() {
        reply->deleteLater();
        if (generation != _generation) {
            return;
        }
        const int http = httpStatus(reply);
        if (http == 0) {
            done(std::nullopt, transportError(reply));
            return;
        }
        if (http >= 300 && http < 400) {
            done(std::nullopt, redirectError(http));
            return;
        }
        const QByteArray body = reply->readAll();
        const QJsonObject object = QJsonDocument::fromJson(body).object();
        if (http != 200 || object.value(QStringLiteral("service")).toString() != QLatin1String("ink-preview")) {
            done(std::nullopt, tr("%1 is not an ink preview server (GET /health answered HTTP %2).")
                                   .arg(_url.toString())
                                   .arg(http));
            return;
        }
        const int api = object.value(QStringLiteral("api")).toInt(-1);
        if (api != kApi) {
            done(std::nullopt, tr("The server at %1 speaks ink preview api %2; VC3D speaks api %3.")
                                   .arg(_url.toString())
                                   .arg(api)
                                   .arg(kApi));
            return;
        }
        Health health;
        health.model = object.value(QStringLiteral("model")).toString();
        health.backend = object.value(QStringLiteral("backend")).toString();
        health.patch = object.value(QStringLiteral("patch")).toInt();
        health.stride = object.value(QStringLiteral("stride")).toInt();
        health.depth = object.value(QStringLiteral("depth")).toInt();
        const QJsonValue voxel = object.value(QStringLiteral("voxel_um"));
        if (voxel.isDouble() && voxel.toDouble() > 0.0) {
            health.voxelUm = voxel.toDouble();
        }
        health.maxViewPx = static_cast<qint64>(object.value(QStringLiteral("max_view_px")).toDouble(16777216.0));
        if (health.depth < 1 || health.patch < 1 || health.maxViewPx < 1) {
            done(std::nullopt, tr("The server at %1 reported no model depth, patch size or view limit.").arg(_url.toString()));
            return;
        }
        done(health, {});
    }, Qt::SingleShotConnection);
}

void InkPreviewClient::requestPreview(const View& view, const Upload* upload, PreviewDone done)
{
    const quint64 generation = _generation;
    if (const QString problem = urlProblem(_url); !problem.isEmpty()) {
        QMetaObject::invokeMethod(this, [this, generation, done, problem]() {
            if (generation == _generation) {
                Reply failed;
                failed.error = problem;
                done(failed);
            }
        }, Qt::QueuedConnection);
        return;
    }

    // The key is the client's text: encode every reserved character, so '+' and '&' survive.
    QByteArray query = "key=" + QUrl::toPercentEncoding(view.key);
    const auto add = [&query](const char* name, qint64 value) {
        query += '&';
        query += name;
        query += '=';
        query += QByteArray::number(value);
    };
    add("cw", view.canvas.width());
    add("ch", view.canvas.height());
    add("x", view.rect.x());
    add("y", view.rect.y());
    add("w", view.rect.width());
    add("h", view.rect.height());
    query += view.reversed ? "&variant=r" : "&variant=f";
    query += "&format=u8";
    if (upload) {
        add("sx", upload->rect.x());
        add("sy", upload->rect.y());
        add("sw", upload->rect.width());
        add("sh", upload->rect.height());
        query += upload->wide ? "&dtype=uint16" : "&dtype=uint8";
    }

    QUrl url = endpoint(QStringLiteral("/preview"));
    url.setQuery(QString::fromLatin1(query), QUrl::TolerantMode);
    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/octet-stream"));
    request.setTransferTimeout(_previewTimeoutMs);
    QNetworkReply* reply = _network->post(request, upload ? upload->voxels : QByteArray());
    track(reply);
    connect(reply, &QNetworkReply::finished, this, [this, reply, generation, view, done]() {
        reply->deleteLater();
        if (generation != _generation) {
            return;
        }
        Reply out;
        out.httpStatus = httpStatus(reply);
        if (out.httpStatus == 0) {
            out.error = transportError(reply);
            done(out);
            return;
        }
        if (out.httpStatus >= 300 && out.httpStatus < 400) {
            out.error = redirectError(out.httpStatus);
            done(out);
            return;
        }
        const QByteArray body = reply->readAll();

        if (out.httpStatus == 200) {
            const auto shape = QString::fromLatin1(reply->rawHeader("X-Shape")).split(QLatin1Char(','));
            const QSize size = shape.size() == 2 ? QSize(shape[1].trimmed().toInt(), shape[0].trimmed().toInt()) : QSize();
            const qint64 expected = qint64(view.rect.width()) * view.rect.height();
            if (size != view.rect.size() || body.size() != expected) {
                out.error = tr("The server sent %1 bytes of shape %2 for a %3 x %4 view; expected %5 bytes.")
                                .arg(body.size())
                                .arg(QString::fromLatin1(reply->rawHeader("X-Shape")))
                                .arg(view.rect.width())
                                .arg(view.rect.height())
                                .arg(expected);
                done(out);
                return;
            }
            out.kind = Reply::Kind::Image;
            out.pixels = body;
            out.size = size;
            out.backend = QString::fromLatin1(reply->rawHeader("X-Backend"));
            const auto tiles = QString::fromLatin1(reply->rawHeader("X-Tiles")).split(QLatin1Char(','));
            if (tiles.size() == 2) {
                out.computed = tiles[0].trimmed().toInt();
                out.cached = tiles[1].trimmed().toInt();
            }
            done(out);
            return;
        }

        const QJsonObject object = QJsonDocument::fromJson(body).object();
        if (out.httpStatus == 409) {
            const QJsonObject need = object.value(QStringLiteral("need")).toObject();
            out.need = QRect(need.value(QStringLiteral("x")).toInt(-1), need.value(QStringLiteral("y")).toInt(-1),
                             need.value(QStringLiteral("w")).toInt(), need.value(QStringLiteral("h")).toInt());
            out.depth = object.value(QStringLiteral("depth")).toInt();
            out.missing = object.value(QStringLiteral("missing")).toInt();
            // A patch that overlaps the view reaches at most patch - 1 pixels past it, so a
            // need beyond that is not one the protocol produces; VC3D would sample and hold
            // all of it in memory.
            const int reach = std::max(0, view.patch - 1);
            const QRect allowed = view.rect.adjusted(-reach, -reach, reach, reach) & QRect(QPoint(0, 0), view.canvas);
            if (out.need.isEmpty() || !allowed.contains(out.need) || out.depth < 1) {
                out.error = tr("The server asked for the layers of canvas rectangle (%1, %2, %3 x %4), which is not "
                               "inside the view (%5, %6, %7 x %8) widened by its patch size %9.")
                                .arg(out.need.x())
                                .arg(out.need.y())
                                .arg(out.need.width())
                                .arg(out.need.height())
                                .arg(view.rect.x())
                                .arg(view.rect.y())
                                .arg(view.rect.width())
                                .arg(view.rect.height())
                                .arg(view.patch);
                done(out);
                return;
            }
            out.kind = Reply::Kind::Need;
            done(out);
            return;
        }

        if (object.contains(QStringLiteral("refused"))) {
            out.error = tr("The server refused the view: %1").arg(object.value(QStringLiteral("refused")).toString());
        } else if (object.contains(QStringLiteral("error"))) {
            out.error = tr("The ink preview server reported an error (HTTP %1): %2")
                            .arg(out.httpStatus)
                            .arg(object.value(QStringLiteral("error")).toString());
        } else {
            out.error = tr("The ink preview server answered HTTP %1: %2").arg(out.httpStatus).arg(reply->errorString());
        }
        done(out);
    }, Qt::SingleShotConnection);
}
