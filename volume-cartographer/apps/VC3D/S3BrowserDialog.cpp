#include "S3BrowserDialog.hpp"

#include "vc/core/util/RemoteUrl.hpp"

#include <QApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QtConcurrent>
#include <QFutureWatcher>

S3BrowserDialog::S3BrowserDialog(
    const vc::cache::HttpAuth& auth,
    const QString& initialUrl,
    QWidget* parent)
    : QDialog(parent)
    , _auth(auth)
{
    setWindowTitle(tr("Browse S3"));
    resize(600, 450);

    auto* layout = new QVBoxLayout(this);

    // Path bar + navigation buttons
    auto* navLayout = new QHBoxLayout();
    _upButton = new QPushButton(tr("Up"));
    _upButton->setFixedWidth(50);
    connect(_upButton, &QPushButton::clicked, this, &S3BrowserDialog::navigateUp);

    _pathBar = new QLineEdit();
    _pathBar->setPlaceholderText(tr("s3://bucket-name/prefix/"));
    connect(_pathBar, &QLineEdit::returnPressed, this, &S3BrowserDialog::navigateToPathBar);

    navLayout->addWidget(_upButton);
    navLayout->addWidget(_pathBar);
    layout->addLayout(navLayout);

    // List widget
    _listWidget = new QListWidget();
    _listWidget->setSelectionMode(QAbstractItemView::SingleSelection);
    connect(_listWidget, &QListWidget::itemDoubleClicked,
            this, &S3BrowserDialog::onItemDoubleClicked);
    layout->addWidget(_listWidget);

    // Status label
    _statusLabel = new QLabel();
    layout->addWidget(_statusLabel);

    // Buttons
    auto* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();
    _openButton = new QPushButton(tr("Open"));
    _openButton->setDefault(true);
    connect(_openButton, &QPushButton::clicked, this, &S3BrowserDialog::onOpenClicked);
    auto* cancelButton = new QPushButton(tr("Cancel"));
    connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    buttonLayout->addWidget(_openButton);
    buttonLayout->addWidget(cancelButton);
    layout->addLayout(buttonLayout);

    // Navigate to initial URL or show empty
    QString startUrl = initialUrl.isEmpty() ? QStringLiteral("s3://") : initialUrl;
    if (!startUrl.endsWith('/')) startUrl += '/';
    navigateTo(startUrl);
}

QString S3BrowserDialog::selectedUrl() const
{
    return _selectedUrl;
}

QString S3BrowserDialog::s3ToHttps(const QString& s3Url) const
{
    auto resolved = vc::resolveRemoteUrl(s3Url.toStdString());
    QString https = QString::fromStdString(resolved.httpsUrl);
    if (!https.endsWith('/')) https += '/';
    return https;
}

void S3BrowserDialog::navigateTo(const QString& s3Url)
{
    // Bump the sequence FIRST so any in-flight watcher whose
    // textChanged-triggered slot re-enters navigateTo, or which is about
    // to finish during this function body, is invalidated before we mutate
    // _currentUrl / _pathBar. Without this ordering, a late response to an
    // earlier navigation could still pass the seq check if it lands between
    // the assignments above and the ++ below.
    ++_listRequestSeq;

    _currentUrl = s3Url;
    if (!_currentUrl.endsWith('/')) _currentUrl += '/';
    _pathBar->setText(_currentUrl);

    // Need at least s3://bucket/ to list
    // s3Url format: s3://bucket/prefix/
    QString afterScheme = _currentUrl.mid(5); // after "s3://"
    if (afterScheme.isEmpty() || afterScheme == "/") {
        _listWidget->clear();
        _statusLabel->setText(tr("Enter an S3 bucket URL above (e.g. s3://my-bucket/)"));
        return;
    }

    setLoading(true);

    QString httpsUrl = s3ToHttps(_currentUrl);
    vc::cache::HttpAuth auth = _auth;

    // Bump the sequence for every navigation so a late-arriving response
    // to an older request is dropped instead of overwriting the current
    // listing (and spuriously clearing the "loading" indicator). Also
    // capture the requested path so that even if the user navigates back
    // to an already-seen URL, we don't confuse responses whose paths
    // differ from the current _currentUrl at completion time.
    const std::uint64_t seq = _listRequestSeq;
    const QString requestedUrl = _currentUrl;

    auto* watcher = new QFutureWatcher<vc::cache::S3ListResult>(this);
    connect(watcher, &QFutureWatcher<vc::cache::S3ListResult>::finished, this,
        [this, watcher, seq, requestedUrl]() {
            watcher->deleteLater();
            if (seq != _listRequestSeq || requestedUrl != _currentUrl) {
                // A newer navigateTo has already been issued (possibly
                // to a different path); its watcher will toggle the
                // loading state when it lands.
                return;
            }
            setLoading(false);

            vc::cache::S3ListResult result;
            try {
                result = watcher->result();
            } catch (const std::exception& e) {
                _statusLabel->setText(tr("Error: %1").arg(e.what()));
                return;
            }

            if (result.authError) {
                _statusLabel->setText(
                    tr("Auth error: %1").arg(QString::fromStdString(result.errorMessage)));
                return;
            }

            _listWidget->clear();

            // Add folders (prefixes)
            for (const auto& prefix : result.prefixes) {
                auto* item = new QListWidgetItem(
                    QApplication::style()->standardIcon(QStyle::SP_DirIcon),
                    QString::fromStdString(prefix) + "/");
                item->setData(Qt::UserRole, QStringLiteral("folder"));
                _listWidget->addItem(item);
            }

            // Add files (objects)
            for (const auto& obj : result.objects) {
                auto* item = new QListWidgetItem(
                    QApplication::style()->standardIcon(QStyle::SP_FileIcon),
                    QString::fromStdString(obj));
                item->setData(Qt::UserRole, QStringLiteral("file"));
                _listWidget->addItem(item);
            }

            int total = static_cast<int>(result.prefixes.size() + result.objects.size());
            _statusLabel->setText(tr("%1 items").arg(total));
        });

    auto future = QtConcurrent::run([httpsUrl, auth]() {
        return vc::cache::s3ListObjects(httpsUrl.toStdString(), auth);
    });
    watcher->setFuture(future);
}

void S3BrowserDialog::navigateUp()
{
    // s3://bucket/a/b/c/ -> s3://bucket/a/b/
    QString url = _currentUrl;
    if (url.endsWith('/')) url.chop(1);

    int lastSlash = url.lastIndexOf('/');
    int schemeEnd = url.indexOf("://");
    if (schemeEnd < 0) return;

    // Don't go above s3://
    if (lastSlash <= schemeEnd + 2) return;

    url = url.left(lastSlash + 1);
    navigateTo(url);
}

void S3BrowserDialog::navigateToPathBar()
{
    QString url = _pathBar->text().trimmed();
    if (url.isEmpty()) return;
    if (!url.startsWith("s3://") && !url.startsWith("s3+")) {
        url = "s3://" + url;
    }
    navigateTo(url);
}

void S3BrowserDialog::onItemDoubleClicked(QListWidgetItem* item)
{
    if (!item) return;
    QString type = item->data(Qt::UserRole).toString();
    QString name = item->text();

    if (type == "folder") {
        // Navigate into folder
        navigateTo(_currentUrl + name);
    } else {
        // Select and accept
        _selectedUrl = _currentUrl + name;
        accept();
    }
}

void S3BrowserDialog::onOpenClicked()
{
    // If an item is selected, use it; otherwise use the current location
    auto* item = _listWidget->currentItem();
    if (item) {
        QString name = item->text();
        _selectedUrl = _currentUrl + name;
    } else {
        _selectedUrl = _currentUrl;
    }
    // Strip trailing slash for consistency
    while (_selectedUrl.endsWith('/')) _selectedUrl.chop(1);
    accept();
}

void S3BrowserDialog::setLoading(bool loading)
{
    _upButton->setEnabled(!loading);
    _openButton->setEnabled(!loading);
    _listWidget->setEnabled(!loading);
    if (loading) {
        _listWidget->clear();
        _statusLabel->setText(tr("Loading..."));
    }
}
