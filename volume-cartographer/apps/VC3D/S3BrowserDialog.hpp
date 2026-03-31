#pragma once

#include <QDialog>
#include <QString>
#include <QStringList>

#include "vc/core/cache/HttpMetadataFetcher.hpp"

class QLineEdit;
class QListWidget;
class QListWidgetItem;
class QPushButton;
class QLabel;

class S3BrowserDialog : public QDialog
{
    Q_OBJECT

public:
    explicit S3BrowserDialog(const vc::cache::HttpAuth& auth,
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

    vc::cache::HttpAuth _auth;
    QString _currentUrl;   // e.g. "s3://bucket/prefix/"
    QString _selectedUrl;

    QLineEdit* _pathBar{nullptr};
    QListWidget* _listWidget{nullptr};
    QPushButton* _upButton{nullptr};
    QPushButton* _openButton{nullptr};
    QLabel* _statusLabel{nullptr};
};
