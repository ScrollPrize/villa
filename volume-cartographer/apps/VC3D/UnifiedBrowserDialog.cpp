#include "UnifiedBrowserDialog.hpp"
#include "utils/sftp_fetch.hpp"
#include <QCheckBox>
#include <QKeyEvent>

#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <vc_reader/catalog.hpp>
#include <vc_reader/s3_listing.hpp>

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
#include <QVBoxLayout>

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

bool isValidRemoteUri(const QString& uri)
{
    const QUrl parsed(uri, QUrl::StrictMode);
    if (!parsed.isValid() || parsed.host().isEmpty()) return false;

    const QString scheme = parsed.scheme().toLower();
    return scheme == QLatin1String("s3")
        || scheme.startsWith(QLatin1String("s3+"))
        || scheme == QLatin1String("http")
        || scheme == QLatin1String("sftp")
        || scheme == QLatin1String("ssh")
        || scheme == QLatin1String("https");
}

bool remoteUriLooksLikeDirectory(const QString& uri)
{
    const QString path = QUrl(uri).path();
    return path.isEmpty() || path.endsWith('/');
}

QString withTrailingSlash(QString s)
{
    // A directory slash belongs to the path, not a signed query or selector.
    auto end = s.size();
    for (const auto delimiter : {'?', '#'}) {
        const auto pos = s.indexOf(delimiter);
        if (pos >= 0) end = std::min(end, pos);
    }
    if (end == 0 || s[end - 1] != '/') s.insert(end, '/');
    return s;
}

QString stripTrailingSlash(QString s)
{
    while (s.size() > 1 && s.endsWith('/')) s.chop(1);
    return s;
}

// Convert local absolute path -> a standards-compliant file: URI.
// QUrl owns the platform details here: notably, UNC paths keep their server
// authority instead of being flattened into an ordinary slash-prefixed path.
QString pathToFileUri(const QString& absPath, bool isDir)
{
    const QString normalized = QDir::fromNativeSeparators(absPath);
    QString out = QUrl::fromLocalFile(normalized).toString();
    if (isDir) out = withTrailingSlash(out);
    return out;
}

// Convert any incoming URI/path to the normalized local-path form used by
// QFileInfo/QDir.  QUrl::toLocalFile preserves a file://server/share UNC
// authority as //server/share, while fromNativeSeparators accepts paths pasted
// with Windows backslashes (including \\wsl.localhost\...).
QString fileUriToPath(QString uri)
{
    if (uri.startsWith(QLatin1String("file:"), Qt::CaseInsensitive)) {
        const QUrl parsed(uri);
        if (parsed.isLocalFile()) {
            return QDir::fromNativeSeparators(parsed.toLocalFile());
        }
    }
    return QDir::fromNativeSeparators(uri);
}

// A browsable s3:// location: its bucket's URI ("s3://bucket/",
// "s3+REGION://bucket/") and the listing of the key prefix below it, at the
// endpoint the remote transport resolves (vc::resolveRemoteUrl).
struct S3Location {
    QString bucketUri;
    vcr::s3::Prefix listing;  // key is empty at the bucket root, else ends with '/'
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

std::optional<S3Location> parseS3Url(const QString& input)
{
    QString uri = input.trimmed();
    const int schemeEnd = uri.indexOf(QStringLiteral("://"));
    if (schemeEnd < 0) return std::nullopt;
    const QString scheme = uri.left(schemeEnd).toLower();
    if (scheme != QLatin1String("s3") && !scheme.startsWith(QLatin1String("s3+"))) {
        return std::nullopt;
    }
    uri = scheme + uri.mid(schemeEnd);
    if (const int queryStart = uri.indexOf('?'); queryStart >= 0) uri = uri.left(queryStart);

    const int bucketEnd = uri.indexOf('/', schemeEnd + 3);
    const QString bucketUri = bucketEnd >= 0 ? uri.left(bucketEnd) : uri;
    if (bucketUri.size() == schemeEnd + 3) return std::nullopt;
    const auto listing = vcr::s3::splitPrefixUrl(vc::resolveRemoteUrl(uri.toStdString()).httpsUrl);
    if (!listing) return std::nullopt;
    return S3Location{withTrailingSlash(bucketUri), *listing};
}

QString displayNameForPrefix(QString name, const QString& parent)
{
    if (!parent.isEmpty() && name.startsWith(parent)) name = name.mid(parent.size());
    while (name.endsWith('/')) name.chop(1);
    const int slash = name.lastIndexOf('/');
    if (slash >= 0) name = name.mid(slash + 1);
    return name + QStringLiteral("/");
}

// The Open Data catalog's root, whose site files (the manifest, index page,
// licence and thumbnails) are not data to browse.
bool shouldHideCatalogRootEntry(const S3Location& loc, const std::string& keyOrPrefix)
{
    if (vcr::s3::objectUrl(loc.listing.origin, loc.listing.key + "metadata.json") !=
        vcr::catalog::kDefaultManifestUrl) {
        return false;
    }
    QString name = QString::fromStdString(keyOrPrefix).mid(int(loc.listing.key.size()));
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

    vcr::s3::ListPage page;
    try {
        page = vcr::s3::parseListPage(vc::httpGetString(
            vcr::s3::listUrl(loc->listing, "/", continuationToken.toStdString()), auth));
    } catch (const std::exception& e) {
        result.error = QObject::tr("Could not list bucket: %1").arg(QString::fromUtf8(e.what()));
        return result;
    }

    const QString parent = QString::fromStdString(loc->listing.key);
    for (const auto& listedPrefix : page.commonPrefixes) {
        if (shouldHideCatalogRootEntry(*loc, listedPrefix)) continue;
        const QString prefix = QString::fromStdString(listedPrefix);
        result.entries.push_back({displayNameForPrefix(prefix, parent), loc->bucketUri + prefix, true});
    }
    for (const auto& object : page.objects) {
        if (shouldHideCatalogRootEntry(*loc, object.key)) continue;
        const QString key = QString::fromStdString(object.key);
        if (key == parent) continue;
        QString name = key;
        if (!parent.isEmpty() && name.startsWith(parent)) name = name.mid(parent.size());
        if (name.contains('/')) continue;
        result.entries.push_back({name, loc->bucketUri + key, false});
    }
    result.nextToken = QString::fromStdString(page.nextContinuationToken);

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
    _pathBar->installEventFilter(this);
    _pathBar->setPlaceholderText(tr("Path or URL — paste anything"));
    connect(_pathBar, &QLineEdit::textEdited, this, [this]() {
        _pathBarEdited = true;
        _selectedUri.clear();
        _list->clearSelection();
        _list->setCurrentItem(nullptr);
    });
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
    _showHidden = new QCheckBox(tr("Show hidden files"), this);
    layout->addWidget(_showHidden);
    connect(_showHidden, &QCheckBox::toggled, this, [this]() {
        if (_mode == Mode::Local) navigateLocal(_currentLocalDir);
    });

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
    // The bucket the Open Data catalog is published in.
    const std::string_view manifest = vcr::catalog::kDefaultManifestUrl;
    _currentRemoteUrl = QString::fromStdString(
        vcr::catalog::openDataS3Url(std::string(manifest.substr(0, manifest.rfind('/') + 1))).value());
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

void UnifiedBrowserDialog::setStartUri(const QString& uri, bool isFile)
{
    if (uri.isEmpty()) return;
    const QString trimmed = uri.trimmed();
    const Mode m = detectModeFromUri(trimmed);
    if (m == Mode::Remote) {
        _remoteRadio->setChecked(true);
        if (isFile && _acceptsFiles) {
            navigateRemote(trimmed.left(trimmed.lastIndexOf('/') + 1));
            _pathBar->setText(trimmed);
            _pathBarEdited = true;
        } else {
            navigateRemote(withTrailingSlash(trimmed));
        }
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
    if (stringStartsWithAny(uri, {"s3://", "s3+", "http://", "https://", "sftp://", "ssh://"})) return Mode::Remote;
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
    handleTypedPath(text, false);
}

bool UnifiedBrowserDialog::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == _pathBar && event->type() == QEvent::KeyPress) {
        const auto* key = static_cast<QKeyEvent*>(event);
        if (key->key() == Qt::Key_Return || key->key() == Qt::Key_Enter) {
            onPathBarReturn();
            return true; // Do not also activate the dialog's default Open button.
        }
    }
    return QDialog::eventFilter(watched, event);
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
    const QString normalizedDir = QDir::fromNativeSeparators(absDir);
    _currentLocalDir = normalizedDir;
    _pathBar->setText(normalizedDir);
    _pathBarEdited = false;
    _list->clear();

    QDir d(normalizedDir);
    if (!d.exists()) {
        _status->setText(tr("No such directory"));
        return;
    }

    QDir::Filters filters = QDir::AllEntries | QDir::NoDotAndDotDot;
    // Let Qt honor native hidden-file semantics, including Windows attributes;
    // a dot-prefixed filename alone is not hidden on Windows.
    if (_showHidden->isChecked()) filters |= QDir::Hidden;
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
    if (utils::is_sftp_url(probeUrl.toStdString())) return true;
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
    _pathBarEdited = false;
    _list->clear();
    _status->setText(tr("Loading..."));

    const bool hasScheme =
        utils::is_sftp_url(_currentRemoteUrl.toStdString()) ||
        _currentRemoteUrl.startsWith(QLatin1String("s3://"), Qt::CaseInsensitive)
        || _currentRemoteUrl.startsWith(QLatin1String("s3+"), Qt::CaseInsensitive)
        || _currentRemoteUrl.startsWith(QLatin1String("http://"), Qt::CaseInsensitive)
        || _currentRemoteUrl.startsWith(QLatin1String("https://"), Qt::CaseInsensitive);
    if (!hasScheme || _currentRemoteUrl == QLatin1String("s3://")) {
        _status->setText(tr("Need a bucket name (e.g. s3://your-bucket/) or full URL"));
        return;
    }

    if (!parseS3Url(_currentRemoteUrl) && !utils::is_sftp_url(_currentRemoteUrl.toStdString())) {
        _status->setText(tr("Browsing is available for s3:// buckets and sftp:// hosts. Paste a full URL and click Open to attach it directly."));
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
        if (utils::is_sftp_url(url.toStdString())) {
            RemoteListResult result;
            for (const auto& entry : utils::list_sftp_directory(url.toStdString()))
                result.entries.push_back({QString::fromStdString(entry.name),
                    QString::fromStdString(entry.url), entry.directory});
            return result;
        }
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
    _pathBarEdited = false;
}

QString UnifiedBrowserDialog::currentUri() const
{
    return _mode == Mode::Local
        ? pathToFileUri(_currentLocalDir, true)
        : _currentRemoteUrl;
}

bool UnifiedBrowserDialog::isAcceptableUri(const QString& uri, bool isFile) const
{
    if (uri.isEmpty()) return false;
    if (isFile && !_acceptsFiles) return false;
    if (!isFile && !_acceptsDirs) return false;
    return true;
}

void UnifiedBrowserDialog::handleTypedPath(const QString& typed, bool acceptDirectories)
{
    const Mode mode = detectModeFromUri(typed);
    if (mode == Mode::Remote) {
        if (!isValidRemoteUri(typed)) {
            _status->setText(tr("Need a valid remote host or S3 bucket"));
            return;
        }

        _remoteRadio->setChecked(true);
        const bool isDirectory = remoteUriLooksLikeDirectory(typed) || !_acceptsFiles;
        if (isDirectory) {
            if (acceptDirectories && _acceptsDirs) {
                _selectedUri = withTrailingSlash(typed);
                accept();
            } else {
                navigateRemote(withTrailingSlash(typed));
            }
            return;
        }

        if (!_acceptsFiles) {
            _status->setText(tr("Select a directory to open."));
            return;
        }
        _selectedUri = typed;
        accept();
        return;
    }

    _localRadio->setChecked(true);
    QString path = fileUriToPath(typed);
    if (path.startsWith(QLatin1String("~/"))) {
        path.replace(0, 1, QDir::homePath());
    } else if (path == QLatin1String("~")) {
        path = QDir::homePath();
    }

    const QFileInfo info(path);
    if (!info.exists()) {
        _status->setText(tr("No such path: %1").arg(path));
        return;
    }
    if (info.isDir()) {
        if (acceptDirectories && _acceptsDirs) {
            _selectedUri = pathToFileUri(info.absoluteFilePath(), true);
            accept();
        } else {
            navigateLocal(info.absoluteFilePath());
        }
        return;
    }
    if (!_acceptsFiles) {
        _status->setText(tr("Select a directory to open."));
        return;
    }
    _selectedUri = pathToFileUri(info.absoluteFilePath(), false);
    accept();
}

void UnifiedBrowserDialog::onOpenClicked()
{
    auto* selected = _list->currentItem();
    if (selected && selected->isSelected()) {
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
    if (_pathBarEdited) {
        const QString typed = _pathBar->text().trimmed();
        if (!typed.isEmpty()) handleTypedPath(typed, true);
        return;
    }
    if (_acceptsDirs) {
        _selectedUri = currentUri();
        accept();
        return;
    }
    QMessageBox::information(this, windowTitle(),
        tr("Select an item to open."));
}
