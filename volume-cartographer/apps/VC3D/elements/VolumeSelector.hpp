#pragma once

#include <QWidget>
#include <QComboBox>
#include <QLabel>
#include <QVector>

class VolumeSelector : public QWidget {
    Q_OBJECT

public:
    struct VolumeOption {
        QString id;
        QString name;
        QString path;
    };

    explicit VolumeSelector(QWidget* parent = nullptr);

    void setLabelText(const QString& text);
    void setVolumes(const QVector<VolumeOption>& volumes, const QString& defaultVolumeId = QString());
    QString selectedVolumeId() const;
    QString selectedVolumePath() const;
    bool hasVolumes() const;
    QComboBox* comboBox() const { return _combo; }

private:
    QLabel* _label{nullptr};
    QComboBox* _combo{nullptr};
};
