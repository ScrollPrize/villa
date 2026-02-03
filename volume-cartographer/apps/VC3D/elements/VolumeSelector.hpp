#pragma once

#include <QWidget>
#include <QComboBox>
#include <QLabel>
#include <QVector>
#include <QToolButton>

class VolumeSelector : public QWidget {
    Q_OBJECT

public:
    enum class BrowseMode {
        Directory,
        File
    };

    struct VolumeOption {
        QString id;
        QString name;
        QString path;
    };

    explicit VolumeSelector(QWidget* parent = nullptr);

    void setLabelText(const QString& text);
    void setLabelVisible(bool visible);
    void setAllowNone(bool allow, const QString& label = QString());
    void setBrowseEnabled(bool enabled);
    void setBrowseDialogTitle(const QString& title);
    void setBrowseMode(BrowseMode mode);
    void setVolumes(const QVector<VolumeOption>& volumes, const QString& defaultVolumeId = QString());
    QString selectedVolumeId() const;
    QString selectedVolumePath() const;
    bool hasVolumes() const;
    QComboBox* comboBox() const { return _combo; }

private:
    QLabel* _label{nullptr};
    QComboBox* _combo{nullptr};
    QToolButton* _browseButton{nullptr};
    bool _allowNone{false};
    bool _browseEnabled{false};
    QString _noneLabel;
    QString _browseDialogTitle;
    BrowseMode _browseMode{BrowseMode::Directory};
};
