#include "UnifiedBrowserDialog.hpp"

#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <QApplication>
#include <QButtonGroup>
#include <QDir>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QtConcurrent>
#include <QUrl>
#include <QUrlQuery>
#include <QVBoxLayout>
#include <QXmlStreamReader>

#include <algorithm>
#include <exception>
#include <optional>
#include <vector>

namespace {

bool stringStartsWithAny(const QString& s, std::initializer_list<const char*> needles)
{
    for (const auto* n : needles) {
        if (s.startsWith(QLatin1String(n), Qt::CaseInsensitive)) return true;
    }
    return false;
}

QString withTrailingSlash(QString s)
{
    if (!s.endsWith('/')) s += '/';
    return s;
}

QString stripTrailingSlash(QString s)
{
    while (s.size() > 1 && s.endsWith('/')) s.chop(1);
    return s;
}

// Convert local absolute path -> file:// URI.
QString pathToFileUri(const QString& absPath, bool isDir)
{
    QString out = QStringLiteral("file://") + absPath;
    if (isDir) out = withTrailingSlash(out);
    return out;
}

// Convert any incoming URI/path to a normalized form for the given mode.
//   Local mode: returns absolute path (no scheme).
//   Remote mode: returns URL with trailing slash.
QString fileUriToPath(QString uri)
{
    if (uri.startsWith(QLatin1String("file://"), Qt::CaseInsensitive)) {
        return uri.mid(7);
    }
    return uri;
}

struct S3Location {
    QString scheme;
    QString bucket;
    QString prefix;
    QString region;
};

struct RemoteEntry {
    QString name;
    QString uri;
    bool isDir = true;
};

struct RemoteListResult {
    std::vector<RemoteEntry> entries;
    QString nextToken;
    QString error;
};

QString withoutLeadingSlash(QString s)
{
    while (s.startsWith('/')) s.remove(0, 1);
    return s;
}

QString ensurePrefixDirectory(QString prefix)
{
    prefix = withoutLeadingSlash(prefix);
    if (!prefix.isEmpty() && !prefix.endsWith('/')) prefix += '/';
    return prefix;
}

std::optional<S3Location> parseS3Url(const QString& input)
{
    const QString trimmed = input.trimmed();
    const int schemeEnd = trimmed.indexOf(QStringLiteral("://"));
    if (schemeEnd < 0) return std::nullopt;

    const QString scheme = trimmed.left(schemeEnd).toLower();
    if (scheme != QLatin1String("s3") && !scheme.startsWith(QLatin1String("s3+"))) {
        return std::nullopt;
    }

    QString rest = trimmed.mid(schemeEnd + 3);
    const int queryStart = rest.indexOf('?');
    if (queryStart >= 0) rest = rest.left(queryStart);

    const int slash = rest.indexOf('/');
    S3Location loc;
    loc.scheme = scheme;
    loc.bucket = slash >= 0 ? rest.left(slash) : rest;
    loc.prefix = slash >= 0 ? rest.mid(slash + 1) : QString();
    loc.prefix = ensurePrefixDirectory(loc.prefix);
    loc.region = scheme.startsWith(QLatin1String("s3+")) ? scheme.mid(3) : QStringLiteral("us-east-1");
    if (loc.bucket.isEmpty()) return std::nullopt;
    return loc;
}

QString s3UriFor(const S3Location& loc, const QString& prefix)
{
    QString uri = loc.scheme + QStringLiteral("://") + loc.bucket + QStringLiteral("/");
    uri += withoutLeadingSlash(prefix);
    return withTrailingSlash(uri);
}

QString s3ListUrl(const S3Location& loc, const QString& continuationToken = {})
{
    QUrl url;
    url.setScheme(QStringLiteral("https"));
    url.setHost(loc.bucket + QStringLiteral(".s3.") + loc.region + QStringLiteral(".amazonaws.com"));

    QUrlQuery query;
    query.addQueryItem(QStringLiteral("list-type"), QStringLiteral("2"));
    query.addQueryItem(QStringLiteral("delimiter"), QStringLiteral("/"));
    query.addQueryItem(QStringLiteral("encoding-type"), QStringLiteral("url"));
    if (!loc.prefix.isEmpty()) {
        query.addQueryItem(QStringLiteral("prefix"), loc.prefix);
    }
    if (!continuationToken.isEmpty()) {
        query.addQueryItem(QStringLiteral("continuation-token"), continuationToken);
    }
    url.setQuery(query);
    return url.toString(QUrl::FullyEncoded);
}

QString decodedS3Text(const QString& text)
{
    return QUrl::fromPercentEncoding(text.toUtf8());
}

QString displayNameForPrefix(const QString& prefix, const QString& parentPrefix)
{
    QString name = decodedS3Text(prefix);
    QString parent = decodedS3Text(parentPrefix);
    if (!parent.isEmpty() && name.startsWith(parent)) name = name.mid(parent.size());
    while (name.endsWith('/')) name.chop(1);
    const int slash = name.lastIndexOf('/');
    if (slash >= 0) name = name.mid(slash + 1);
    return name + QStringLiteral("/");
}

bool isVesuviusOpenDataRoot(const S3Location& loc)
{
    return loc.bucket.compare(QStringLiteral("vesuvius-challenge-open-data"), Qt::CaseInsensitive) == 0
        && loc.prefix.isEmpty();
}

bool shouldHideVesuviusRootEntry(const S3Location& loc, const QString& decodedKeyOrPrefix)
{
    if (!isVesuviusOpenDataRoot(loc)) return false;

    QString name = decodedKeyOrPrefix;
    while (name.startsWith('/')) name.remove(0, 1);
    if (name.endsWith('/')) name.chop(1);

    return name.compare(QStringLiteral("_thumbnails"), Qt::CaseInsensitive) == 0
        || name.compare(QStringLiteral("index.html"), Qt::CaseInsensitive) == 0
        || name.compare(QStringLiteral("license.txt"), Qt::CaseInsensitive) == 0
        || name.compare(QStringLiteral("metadata.json"), Qt::CaseInsensitive) == 0;
}

RemoteListResult listS3Prefix(const QString& urlPrefix,
                              const vc::HttpAuth& auth,
                              const QString& continuationToken = {})
{
    RemoteListResult result;
    const auto loc = parseS3Url(urlPrefix);
    if (!loc) {
        result.error = QObject::tr("Remote browsing supports s3:// bucket URLs.");
        return result;
    }

    const std::string body = vc::httpGetString(
        s3ListUrl(*loc, continuationToken).toStdString(), auth);
    if (body.empty()) {
        result.error = QObject::tr("No response while listing bucket.");
        return result;
    }

    QXmlStreamReader xml(QString::fromStdString(body));
    while (!xml.atEnd()) {
        xml.readNext();
        if (!xml.isStartElement()) continue;

        if (xml.name() == QLatin1String("CommonPrefixes")) {
            QString prefix;
            while (!(xml.isEndElement() && xml.name() == QLatin1String("CommonPrefixes")) && !xml.atEnd()) {
                xml.readNext();
                if (xml.isStartElement() && xml.name() == QLatin1String("Prefix")) {
                    prefix = xml.readElementText();
                }
            }
            if (!prefix.isEmpty()) {
                const QString decodedPrefix = decodedS3Text(prefix);
                if (shouldHideVesuviusRootEntry(*loc, decodedPrefix)) continue;
                result.entries.push_back({
                    displayNameForPrefix(prefix, loc->prefix),
                    s3UriFor(*loc, decodedPrefix),
                    true
                });
            }
        } else if (xml.name() == QLatin1String("Contents")) {
            QString key;
            while (!(xml.isEndElement() && xml.name() == QLatin1String("Contents")) && !xml.atEnd()) {
                xml.readNext();
                if (xml.isStartElement() && xml.name() == QLatin1String("Key")) {
                    key = xml.readElementText();
                }
            }
            if (!key.isEmpty()) {
                const QString decodedKey = decodedS3Text(key);
                if (shouldHideVesuviusRootEntry(*loc, decodedKey)) continue;
                if (decodedKey == loc->prefix) continue;
                QString name = decodedKey;
                const QString parent = loc->prefix;
                if (!parent.isEmpty() && name.startsWith(parent)) name = name.mid(parent.size());
                if (name.contains('/')) continue;
                const bool isDir = name.endsWith('/');
                result.entries.push_back({
                    isDir ? name : name,
                    isDir ? s3UriFor(*loc, decodedKey) : loc->scheme + QStringLiteral("://") + loc->bucket + QStringLiteral("/") + decodedKey,
                    isDir
                });
            }
        } else if (xml.name() == QLatin1String("NextContinuationToken")) {
            result.nextToken = xml.readElementText();
        }
    }

    if (xml.hasError()) {
        result.entries.clear();
        result.nextToken.clear();
        result.error = QObject::tr("Could not parse S3 listing: %1").arg(xml.errorString());
    }

    std::sort(result.entries.begin(), result.entries.end(), [](const RemoteEntry& a, const RemoteEntry& b) {
        if (a.isDir != b.isDir) return a.isDir > b.isDir;
        return QString::localeAwareCompare(a.name, b.name) < 0;
    });
    return result;
}

}  // namespace

UnifiedBrowserDialog::UnifiedBrowserDialog(QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle(tr("Browse"));
    resize(700, 520);

    auto* layout = new QVBoxLayout(this);

    // Mode toggle
    auto* modeRow = new QHBoxLayout();
    _localRadio = new QRadioButton(tr("Local"));
    _remoteRadio = new QRadioButton(tr("Remote"));
    _localRadio->setChecked(true);
    _modeGroup = new QButtonGroup(this);
    _modeGroup->addButton(_localRadio);
    _modeGroup->addButton(_remoteRadio);
    modeRow->addWidget(_localRadio);
    modeRow->addWidget(_remoteRadio);
    modeRow->addStretch();
    layout->addLayout(modeRow);
    connect(_localRadio, &QRadioButton::toggled,
            this, &UnifiedBrowserDialog::onModeChanged);
    connect(_remoteRadio, &QRadioButton::toggled,
            this, &UnifiedBrowserDialog::onModeChanged);

    // Hint label (hidden until set)
    _hint = new QLabel();
    _hint->setWordWrap(true);
    _hint->hide();
    layout->addWidget(_hint);

    // Path bar + Up button
    auto* navRow = new QHBoxLayout();
    _upButton = new QPushButton(tr("Up"));
    _upButton->setFixedWidth(50);
    connect(_upButton, &QPushButton::clicked,
            this, &UnifiedBrowserDialog::onUpClicked);
    _pathBar = new QLineEdit();
    _pathBar->setPlaceholderText(tr("Path or URL — paste anything"));
    connect(_pathBar, &QLineEdit::returnPressed,
            this, &UnifiedBrowserDialog::onPathBarReturn);
    navRow->addWidget(_upButton);
    navRow->addWidget(_pathBar);
    layout->addLayout(navRow);

    // List
    _list = new QListWidget();
    _list->setSelectionMode(QAbstractItemView::SingleSelection);
    connect(_list, &QListWidget::itemDoubleClicked,
            this, &UnifiedBrowserDialog::onItemDoubleClicked);
    connect(_list, &QListWidget::itemSelectionChanged,
            this, &UnifiedBrowserDialog::onItemSelectionChanged);
    layout->addWidget(_list);

    // Status
    _status = new QLabel();
    layout->addWidget(_status);

    // Buttons
    auto* btnRow = new QHBoxLayout();
    btnRow->addStretch();
    _openButton = new QPushButton(tr("Open"));
    _openButton->setDefault(true);
    connect(_openButton, &QPushButton::clicked,
            this, &UnifiedBrowserDialog::onOpenClicked);
    auto* cancel = new QPushButton(tr("Cancel"));
    connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
    btnRow->addWidget(_openButton);
    btnRow->addWidget(cancel);
    layout->addLayout(btnRow);

    // Default to local home
    _currentLocalDir = QDir::homePath();
    _currentRemoteUrl = QStringLiteral("s3://vesuvius-challenge-open-data/");
    navigateLocal(_currentLocalDir);
}

void UnifiedBrowserDialog::setHint(const QString& text)
{
    if (text.isEmpty()) {
        _hint->hide();
    } else {
        _hint->setText(text);
        _hint->show();
    }
}

void UnifiedBrowserDialog::setStartUri(const QString& uri)
{
    if (uri.isEmpty()) return;
    const QString trimmed = uri.trimmed();
    const Mode m = detectModeFromUri(trimmed);
    if (m == Mode::Remote) {
        _remoteRadio->setChecked(true);
        navigateRemote(withTrailingSlash(trimmed));
    } else {
        _localRadio->setChecked(true);
        QString p = fileUriToPath(trimmed);
        QFileInfo fi(p);
        if (fi.isFile()) p = fi.absolutePath();
        navigateLocal(p);
    }
}

UnifiedBrowserDialog::Mode UnifiedBrowserDialog::detectModeFromUri(const QString& uri)
{
    if (stringStartsWithAny(uri, {"s3://", "http://", "https://"})) return Mode::Remote;
    return Mode::Local;
}

void UnifiedBrowserDialog::onModeChanged()
{
    const Mode previous = _mode;
    _mode = _remoteRadio->isChecked() ? Mode::Remote : Mode::Local;
    if (_mode == previous) return;

    _selectedUri.clear();
    _list->clear();
    if (_mode == Mode::Local) {
        navigateLocal(_currentLocalDir);
    } else {
        navigateRemote(_currentRemoteUrl);
    }
}

void UnifiedBrowserDialog::onPathBarReturn()
{
    const QString text = _pathBar->text().trimmed();
    if (text.isEmpty()) return;
    const Mode m = detectModeFromUri(text);
    if (m == Mode::Remote) {
        _remoteRadio->setChecked(true);
        navigateRemote(withTrailingSlash(text));
    } else {
        _localRadio->setChecked(true);
        QString p = fileUriToPath(text);
        // Expand ~/...
        if (p.startsWith(QLatin1String("~/"))) {
            p.replace(0, 1, QDir::homePath());
        } else if (p == QLatin1String("~")) {
            p = QDir::homePath();
        }
        QFileInfo fi(p);
        if (fi.isFile()) p = fi.absolutePath();
        navigateLocal(p);
    }
}

void UnifiedBrowserDialog::onUpClicked()
{
    if (_mode == Mode::Local) {
        QDir d(_currentLocalDir);
        if (d.cdUp()) navigateLocal(d.absolutePath());
    } else {
        QString u = stripTrailingSlash(_currentRemoteUrl);
        const int slash = u.lastIndexOf('/');
        // Don't strip past "scheme://host"
        const int schemeEnd = u.indexOf(QStringLiteral("://"));
        if (schemeEnd >= 0 && slash <= schemeEnd + 2) {
            return;
        }
        if (slash > 0) {
            navigateRemote(withTrailingSlash(u.left(slash)));
        }
    }
}

void UnifiedBrowserDialog::navigateLocal(const QString& absDir)
{
    _currentLocalDir = absDir;
    _pathBar->setText(absDir);
    _list->clear();

    QDir d(absDir);
    if (!d.exists()) {
        _status->setText(tr("No such directory"));
        return;
    }

    QDir::Filters filters = QDir::AllEntries | QDir::NoDotAndDotDot;
    auto entries = d.entryInfoList(filters, QDir::Name | QDir::DirsFirst);

    QRegularExpression filterRe;
    bool useFilter = false;
    if (!_localFilters.isEmpty()) {
        QStringList parts;
        for (const auto& g : _localFilters) {
            // Convert glob to regex; QRegularExpression::wildcardToRegularExpression
            parts << QRegularExpression::wildcardToRegularExpression(
                g, QRegularExpression::UnanchoredWildcardConversion);
        }
        filterRe.setPattern(QStringLiteral("^(?:") + parts.join('|') + QStringLiteral(")$"));
        filterRe.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        useFilter = filterRe.isValid();
    }

    int shown = 0;
    for (const auto& fi : entries) {
        const bool isDir = fi.isDir();
        if (!isDir) {
            if (!_acceptsFiles) continue;
            if (useFilter && !filterRe.match(fi.fileName()).hasMatch()) continue;
        }
        auto* item = new QListWidgetItem();
        item->setText(isDir ? fi.fileName() + QStringLiteral("/") : fi.fileName());
        item->setData(Qt::UserRole, pathToFileUri(fi.absoluteFilePath(), isDir));
        item->setData(Qt::UserRole + 1, isDir);
        _list->addItem(item);
        ++shown;
    }
    _status->setText(tr("%1 items").arg(shown));
}

bool UnifiedBrowserDialog::ensureRemoteAuth(const QString& probeUrl)
{
    if (_authResolved && probeUrl == _authProbeUrl) return true;
    if (!_authResolver) {
        // No resolver — proceed with empty auth (works for fully public buckets).
        _auth = {};
        _authResolved = true;
        _authProbeUrl = probeUrl;
        return true;
    }
    QString err;
    if (!_authResolver(probeUrl, &_auth, &err)) {
        _status->setText(tr("Auth failed: %1").arg(err));
        return false;
    }
    _authResolved = true;
    _authProbeUrl = probeUrl;
    return true;
}

void UnifiedBrowserDialog::navigateRemote(const QString& urlPrefix)
{
    ++_listSeq;
    const std::uint64_t mySeq = _listSeq;

    _currentRemoteUrl = withTrailingSlash(urlPrefix);
    _pathBar->setText(_currentRemoteUrl);
    _list->clear();
    _status->setText(tr("Loading..."));

    const bool hasScheme =
        _currentRemoteUrl.startsWith(QLatin1String("s3://"), Qt::CaseInsensitive)
        || _currentRemoteUrl.startsWith(QLatin1String("s3+"), Qt::CaseInsensitive)
        || _currentRemoteUrl.startsWith(QLatin1String("http://"), Qt::CaseInsensitive)
        || _currentRemoteUrl.startsWith(QLatin1String("https://"), Qt::CaseInsensitive);
    if (!hasScheme || _currentRemoteUrl == QLatin1String("s3://")) {
        _status->setText(tr("Need a bucket name (e.g. s3://your-bucket/) or full URL"));
        return;
    }

    if (!parseS3Url(_currentRemoteUrl)) {
        _status->setText(tr("Browsing is available for s3:// buckets. Paste a full URL and click Open to attach it directly."));
        return;
    }

    if (!ensureRemoteAuth(_currentRemoteUrl)) {
        return;
    }

    auto* watcher = new QFutureWatcher<RemoteListResult>(this);
    connect(watcher, &QFutureWatcher<RemoteListResult>::finished, this, [this, watcher, mySeq]() {
        watcher->deleteLater();
        if (mySeq != _listSeq) return;

        RemoteListResult result;
        try {
            result = watcher->result();
        } catch (const std::exception& e) {
            _status->setText(tr("Remote listing failed: %1").arg(QString::fromUtf8(e.what())));
            return;
        }

        if (!result.error.isEmpty()) {
            _status->setText(result.error);
            return;
        }

        int shown = 0;
        for (const auto& entry : result.entries) {
            auto* item = new QListWidgetItem();
            item->setText(entry.name);
            item->setData(Qt::UserRole, entry.uri);
            item->setData(Qt::UserRole + 1, entry.isDir);
            _list->addItem(item);
            ++shown;
        }

        if (!result.nextToken.isEmpty()) {
            auto* item = new QListWidgetItem(tr("Load more..."));
            item->setData(Qt::UserRole, _currentRemoteUrl);
            item->setData(Qt::UserRole + 1, true);
            item->setData(Qt::UserRole + 2, result.nextToken);
            _list->addItem(item);
        }

        _status->setText(result.nextToken.isEmpty()
            ? tr("%1 items").arg(shown)
            : tr("%1 items; more available").arg(shown));
    });
    watcher->setFuture(QtConcurrent::run([url = _currentRemoteUrl, auth = _auth]() {
        return listS3Prefix(url, auth);
    }));
}

void UnifiedBrowserDialog::onItemDoubleClicked(QListWidgetItem* item)
{
    if (!item) return;
    const bool isDir = item->data(Qt::UserRole + 1).toBool();
    const QString uri = item->data(Qt::UserRole).toString();
    const QString continuationToken = item->data(Qt::UserRole + 2).toString();
    if (!continuationToken.isEmpty() && _mode == Mode::Remote) {
        _status->setText(tr("Loading more..."));
        if (!ensureRemoteAuth(_currentRemoteUrl)) return;
        auto* loadMoreItem = item;
        auto* watcher = new QFutureWatcher<RemoteListResult>(this);
        const std::uint64_t mySeq = _listSeq;
        connect(watcher, &QFutureWatcher<RemoteListResult>::finished, this,
                [this, watcher, loadMoreItem, mySeq]() {
            watcher->deleteLater();
            if (mySeq != _listSeq) return;
            const int row = _list->row(loadMoreItem);
            if (row >= 0) delete _list->takeItem(row);

            RemoteListResult result;
            try {
                result = watcher->result();
            } catch (const std::exception& e) {
                _status->setText(tr("Remote listing failed: %1").arg(QString::fromUtf8(e.what())));
                return;
            }
            if (!result.error.isEmpty()) {
                _status->setText(result.error);
                return;
            }
            for (const auto& entry : result.entries) {
                auto* newItem = new QListWidgetItem(entry.name);
                newItem->setData(Qt::UserRole, entry.uri);
                newItem->setData(Qt::UserRole + 1, entry.isDir);
                _list->addItem(newItem);
            }
            if (!result.nextToken.isEmpty()) {
                auto* newItem = new QListWidgetItem(tr("Load more..."));
                newItem->setData(Qt::UserRole, _currentRemoteUrl);
                newItem->setData(Qt::UserRole + 1, true);
                newItem->setData(Qt::UserRole + 2, result.nextToken);
                _list->addItem(newItem);
            }
            _status->setText(tr("%1 items").arg(_list->count() - (result.nextToken.isEmpty() ? 0 : 1)));
        });
        watcher->setFuture(QtConcurrent::run([url = _currentRemoteUrl, auth = _auth, continuationToken]() {
            return listS3Prefix(url, auth, continuationToken);
        }));
        return;
    }
    if (isDir) {
        if (_mode == Mode::Local) {
            navigateLocal(fileUriToPath(stripTrailingSlash(uri)));
        } else {
            navigateRemote(withTrailingSlash(uri));
        }
    } else {
        _selectedUri = uri;
        accept();
    }
}

void UnifiedBrowserDialog::onItemSelectionChanged()
{
    auto* item = _list->currentItem();
    if (!item) return;
    _selectedUri = item->data(Qt::UserRole).toString();
}

QString UnifiedBrowserDialog::currentUri() const
{
    return _mode == Mode::Local
        ? pathToFileUri(_currentLocalDir, true)
        : _currentRemoteUrl;
}

QString UnifiedBrowserDialog::itemUri(const QListWidgetItem* item) const
{
    return item ? item->data(Qt::UserRole).toString() : QString();
}

bool UnifiedBrowserDialog::isAcceptableUri(const QString& uri, bool isFile) const
{
    if (uri.isEmpty()) return false;
    if (isFile && !_acceptsFiles) return false;
    if (!isFile && !_acceptsDirs) return false;
    return true;
}

void UnifiedBrowserDialog::onOpenClicked()
{
    auto* selected = _list->currentItem();
    if (selected) {
        if (_mode == Mode::Remote && !selected->data(Qt::UserRole + 2).toString().isEmpty()) {
            onItemDoubleClicked(selected);
            return;
        }
        const bool isDir = selected->data(Qt::UserRole + 1).toBool();
        const QString uri = selected->data(Qt::UserRole).toString();
        if (isAcceptableUri(uri, !isDir)) {
            _selectedUri = uri;
            accept();
            return;
        }
    }
    if (!_acceptsDirs) {
        QMessageBox::information(this, windowTitle(),
            tr("Select an item to open."));
        return;
    }
    // Fall back to the path bar — prefer what's typed there over the
    // last-navigated location, since users often paste a URL and click Open
    // without pressing Enter to navigate first.
    const QString typed = _pathBar->text().trimmed();
    if (typed.isEmpty()) {
        _selectedUri = currentUri();
        accept();
        return;
    }
    const Mode m = detectModeFromUri(typed);
    if (m == Mode::Remote) {
        const int schemeEnd = typed.indexOf(QStringLiteral("://"));
        if (schemeEnd < 0 || typed.size() <= schemeEnd + 3) {
            _status->setText(tr("Need a bucket/host (e.g. s3://your-bucket/)"));
            return;
        }
        _selectedUri = withTrailingSlash(typed);
    } else {
        QString p = fileUriToPath(typed);
        if (p.startsWith(QLatin1String("~/"))) {
            p.replace(0, 1, QDir::homePath());
        } else if (p == QLatin1String("~")) {
            p = QDir::homePath();
        }
        QFileInfo fi(p);
        const bool isDir = fi.isDir();
        if (!fi.exists()) {
            _status->setText(tr("No such path: %1").arg(p));
            return;
        }
        _selectedUri = pathToFileUri(fi.absoluteFilePath(), isDir);
    }
    accept();
}
