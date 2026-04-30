#pragma once

#include <QDialog>
#include <QString>
#include <cstdint>
#include <functional>

#include "vc/core/cache/HttpMetadataFetcher.hpp"

class QButtonGroup;
class QLabel;
class QLineEdit;
class QListWidget;
class QListWidgetItem;
class QPushButton;
class QRadioButton;

class UnifiedBrowserDialog : public QDialog
{
    Q_OBJECT

public:
    enum class Mode { Local, Remote };

    explicit UnifiedBrowserDialog(QWidget* parent = nullptr);

    void setHint(const QString& text);
    void setAcceptsFiles(bool v) { _acceptsFiles = v; }
    void setAcceptsDirs(bool v) { _acceptsDirs = v; }
    void setLocalNameFilters(const QStringList& globs) { _localFilters = globs; }
    void setStartUri(const QString& uri);

    using AuthResolver = std::function<bool(const QString& url,
                                             vc::cache::HttpAuth* out,
                                             QString* err)>;
    void setAuthResolver(AuthResolver r) { _authResolver = std::move(r); }

    QString selectedUri() const { return _selectedUri; }

private slots:
    void onModeChanged();
    void onPathBarReturn();
    void onUpClicked();
    void onItemDoubleClicked(QListWidgetItem* item);
    void onItemSelectionChanged();
    void onOpenClicked();

private:
    void navigateLocal(const QString& absDir);
    void navigateRemote(const QString& urlPrefix);
    bool ensureRemoteAuth(const QString& probeUrl);
    static Mode detectModeFromUri(const QString& uri);
    QString currentUri() const;
    QString itemUri(const QListWidgetItem* item) const;
    bool isAcceptableUri(const QString& uri, bool isFile) const;

    Mode _mode = Mode::Local;
    bool _acceptsFiles = true;
    bool _acceptsDirs = true;
    QStringList _localFilters;
    QString _selectedUri;

    QString _currentLocalDir;
    QString _currentRemoteUrl;

    vc::cache::HttpAuth _auth;
    bool _authResolved = false;
    QString _authProbeUrl;
    AuthResolver _authResolver;

    QRadioButton* _localRadio{nullptr};
    QRadioButton* _remoteRadio{nullptr};
    QButtonGroup* _modeGroup{nullptr};
    QLabel* _hint{nullptr};
    QLineEdit* _pathBar{nullptr};
    QPushButton* _upButton{nullptr};
    QListWidget* _list{nullptr};
    QLabel* _status{nullptr};
    QPushButton* _openButton{nullptr};

    std::uint64_t _listSeq{0};
};
