#pragma once

#include <QDialog>
#include <QString>
#include <QStringList>

#include <cstdint>

#include "vc/core/util/RemoteAuth.hpp"

class QLineEdit;
class QListWidget;
class QListWidgetItem;
class QPushButton;
class QLabel;

class S3BrowserDialog : public QDialog
{
    Q_OBJECT

public:
    explicit S3BrowserDialog(const vc::HttpAuth& auth,
                             const QString& initialUrl = QString(),
                             QWidget* parent = nullptr);

    // The s3:// URL the user selected (empty if canceled)
    QString selectedUrl() const;

private slots:
    void navigateTo(const QString& s3Url);
    void navigateUp();
    void navigateToPathBar();
    void onItemDoubleClicked(QListWidgetItem* item);
    void onOpenClicked();

private:
    void setLoading(bool loading);
    // Convert s3://bucket/prefix/ to HTTPS URL for s3ListObjects
    QString s3ToHttps(const QString& s3Url) const;

    vc::HttpAuth _auth;
    QString _currentUrl;   // e.g. "s3://bucket/prefix/"
    QString _selectedUrl;
    // Monotonic counter incremented on every navigateTo() — late-arriving
    // list responses check against this before touching the UI so a slow
    // previous listing can't overwrite the results of a newer one.
    std::uint64_t _listRequestSeq{0};

    QLineEdit* _pathBar{nullptr};
    QListWidget* _listWidget{nullptr};
    QPushButton* _upButton{nullptr};
    QPushButton* _openButton{nullptr};
    QLabel* _statusLabel{nullptr};
};
