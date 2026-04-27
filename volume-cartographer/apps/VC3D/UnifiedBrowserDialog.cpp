#include "UnifiedBrowserDialog.hpp"

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
#include <QUrl>
#include <QVBoxLayout>
#include <QtConcurrent>

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
    _currentRemoteUrl = QStringLiteral("s3://");
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
    _status->setText(tr("Listing..."));

    const bool hasScheme =
        _currentRemoteUrl.startsWith(QLatin1String("s3://"), Qt::CaseInsensitive)
        || _currentRemoteUrl.startsWith(QLatin1String("http://"), Qt::CaseInsensitive)
        || _currentRemoteUrl.startsWith(QLatin1String("https://"), Qt::CaseInsensitive);
    if (!hasScheme || _currentRemoteUrl == QLatin1String("s3://")) {
        _status->setText(tr("Need a bucket name (e.g. s3://your-bucket/) or full URL"));
        return;
    }

    if (!ensureRemoteAuth(_currentRemoteUrl)) return;

    // Convert s3:// to https for listing
    QString httpsUrl = _currentRemoteUrl;
    if (httpsUrl.startsWith(QLatin1String("s3://"), Qt::CaseInsensitive)) {
        auto resolved = vc::resolveRemoteUrl(_currentRemoteUrl.toStdString());
        httpsUrl = QString::fromStdString(resolved.httpsUrl);
        if (!httpsUrl.endsWith('/')) httpsUrl += '/';
    }

    auto auth = _auth;
    auto baseUri = _currentRemoteUrl;
    auto* watcher = new QFutureWatcher<vc::cache::S3ListResult>(this);
    connect(watcher, &QFutureWatcher<vc::cache::S3ListResult>::finished, this,
        [this, watcher, mySeq, baseUri]() {
            watcher->deleteLater();
            if (mySeq != _listSeq) return;
            const auto result = watcher->result();
            if (result.authError) {
                _status->setText(tr("Auth/list error: %1")
                    .arg(QString::fromStdString(result.errorMessage)));
                return;
            }
            int shown = 0;
            for (const auto& pref : result.prefixes) {
                auto* item = new QListWidgetItem();
                const QString name = QString::fromStdString(pref);
                item->setText(name);
                item->setData(Qt::UserRole, baseUri + name);
                item->setData(Qt::UserRole + 1, true);
                _list->addItem(item);
                ++shown;
            }
            for (const auto& obj : result.objects) {
                if (!_acceptsFiles) continue;
                auto* item = new QListWidgetItem();
                const QString name = QString::fromStdString(obj);
                item->setText(name);
                item->setData(Qt::UserRole, baseUri + name);
                item->setData(Qt::UserRole + 1, false);
                _list->addItem(item);
                ++shown;
            }
            _status->setText(tr("%1 items").arg(shown));
        });
    auto fut = QtConcurrent::run([httpsUrl, auth]() {
        return vc::cache::s3ListObjects(httpsUrl.toStdString(), auth);
    });
    watcher->setFuture(fut);
}

void UnifiedBrowserDialog::onItemDoubleClicked(QListWidgetItem* item)
{
    if (!item) return;
    const bool isDir = item->data(Qt::UserRole + 1).toBool();
    const QString uri = item->data(Qt::UserRole).toString();
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
        const bool isDir = selected->data(Qt::UserRole + 1).toBool();
        const QString uri = selected->data(Qt::UserRole).toString();
        if (isAcceptableUri(uri, !isDir)) {
            _selectedUri = uri;
            accept();
            return;
        }
    }
    // Fall back to the path bar (whatever the user typed)
    if (_acceptsDirs) {
        _selectedUri = currentUri();
        accept();
        return;
    }
    QMessageBox::information(this, windowTitle(),
        tr("Select an item to open."));
}
