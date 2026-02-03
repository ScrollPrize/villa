#pragma once

#include "elements/JsonProfileEditor.hpp"
#include "VCSettings.hpp"

#include <QSettings>
#include <QString>
#include <QVector>

namespace vc3d::user_profiles {

struct UserProfile {
    int storageId{0};
    QString name;
    QString jsonText;
};

inline QString profileIdFromStorageId(int storageId)
{
    return QStringLiteral("user_%1").arg(storageId);
}

inline int storageIdFromProfileId(const QString& profileId)
{
    if (!profileId.startsWith(QStringLiteral("user_"))) {
        return -1;
    }
    bool ok = false;
    const int id = profileId.mid(5).toInt(&ok);
    return ok ? id : -1;
}

inline bool isUserProfileId(const QString& profileId)
{
    return storageIdFromProfileId(profileId) > 0;
}

inline QVector<UserProfile> loadAll()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QVector<UserProfile> result;

    settings.beginGroup(QStringLiteral("user_profiles"));
    const int nextId = settings.value(QStringLiteral("next_id"), 1).toInt();

    for (int id = 1; id < nextId; ++id) {
        const QString groupKey = QString::number(id);
        settings.beginGroup(groupKey);
        const QString name = settings.value(QStringLiteral("name")).toString();
        if (!name.isEmpty()) {
            UserProfile profile;
            profile.storageId = id;
            profile.name = name;
            profile.jsonText = settings.value(QStringLiteral("json")).toString();
            result.append(profile);
        }
        settings.endGroup();
    }

    settings.endGroup();
    return result;
}

inline int save(const QString& name, const QString& jsonText)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(QStringLiteral("user_profiles"));
    const int id = settings.value(QStringLiteral("next_id"), 1).toInt();
    settings.setValue(QStringLiteral("next_id"), id + 1);

    settings.beginGroup(QString::number(id));
    settings.setValue(QStringLiteral("name"), name);
    settings.setValue(QStringLiteral("json"), jsonText);
    settings.endGroup();

    settings.endGroup();
    return id;
}

inline void update(int storageId, const QString& name, const QString& jsonText)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(QStringLiteral("user_profiles"));
    settings.beginGroup(QString::number(storageId));
    settings.setValue(QStringLiteral("name"), name);
    settings.setValue(QStringLiteral("json"), jsonText);
    settings.endGroup();
    settings.endGroup();
}

inline void remove(int storageId)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(QStringLiteral("user_profiles"));
    settings.remove(QString::number(storageId));
    settings.endGroup();
}

} // namespace vc3d::user_profiles
