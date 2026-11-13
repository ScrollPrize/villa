#pragma once

#include <QByteArray>
#include <QObject>
#include <QPointer>
#include <QRect>
#include <QSize>
#include <QString>
#include <QUrl>

#include <functional>
#include <optional>
#include <vector>

class QNetworkAccessManager;
class QNetworkReply;

// A client of an ink preview server the user started (ink preview api 1, see
// docs/ink_preview.md): GET /health, and POST /preview in stack mode, where the
// server answers 409 with the canvas rectangle whose layers it needs and the client
// posts again with them. Replies run their callback on the client's thread; a
// request dropped by cancel() or setServerUrl() never runs it.
class InkPreviewClient : public QObject
{
    Q_OBJECT

public:
    static constexpr int kApi = 1;

    struct Health {
        QString model;
        QString backend;
        int patch = 0;
        int stride = 0;
        int depth = 0;                    // layers the model reads
        std::optional<double> voxelUm;    // the model's training voxel size, when the server knows it
        qint64 maxViewPx = 0;             // the largest view it computes
    };

    // A view of a canvas: `key` names everything that decides the canvas's voxels,
    // `rect` is the view in canvas pixels, `reversed` asks for the control view with
    // the layer order reversed.
    struct View {
        QString key;
        QSize canvas;
        QRect rect;
        bool reversed = false;
    };

    // The layers of the canvas rectangle `rect`, in forward order: depth * height *
    // width voxels, layers first, C order, little-endian.
    struct Upload {
        QRect rect;
        bool wide = false;  // uint16 voxels; uint8 otherwise
        QByteArray voxels;
    };

    struct Reply {
        enum class Kind { Image, Need, Failed };
        Kind kind = Kind::Failed;
        // Image: the view, one byte per pixel (floor(255 p)), row-major.
        QByteArray pixels;
        QSize size;
        QString backend;
        int computed = 0;  // patches the server ran for this reply
        int cached = 0;    // patches it took from its cache
        // Need: the canvas rectangle to upload, the layers it wants, the patches missing.
        QRect need;
        int depth = 0;
        int missing = 0;
        // Failed: what went wrong, as a sentence for the user.
        QString error;
        int httpStatus = 0;  // 0 when no HTTP answer arrived
    };

    using HealthDone = std::function<void(const std::optional<Health>& health, const QString& error)>;
    using PreviewDone = std::function<void(const Reply& reply)>;

    explicit InkPreviewClient(QObject* parent = nullptr);
    ~InkPreviewClient() override;

    // Drops the requests in flight.
    void setServerUrl(const QUrl& url);
    [[nodiscard]] QUrl serverUrl() const { return _url; }
    // Per-request limits: a request that receives nothing for this long fails.
    void setTimeouts(int healthMs, int previewMs);

    void checkHealth(HealthDone done);
    // With `upload`, its layers go in the body (sx, sy, sw, sh and dtype in the query).
    void requestPreview(const View& view, const Upload* upload, PreviewDone done);
    // Drops every request in flight: none of their callbacks runs.
    void cancel();
    [[nodiscard]] bool busy() const;

    // Why `url` cannot name a server, or empty when it can.
    [[nodiscard]] static QString urlProblem(const QUrl& url);

private:
    QUrl endpoint(const QString& path) const;
    QString transportError(QNetworkReply* reply) const;
    void track(QNetworkReply* reply);

    QNetworkAccessManager* _network{nullptr};
    QUrl _url;
    quint64 _generation{0};
    int _healthTimeoutMs{5000};
    int _previewTimeoutMs{120000};
    std::vector<QPointer<QNetworkReply>> _replies;
};
