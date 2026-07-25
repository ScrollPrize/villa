#pragma once

#include <QList>
#include <QSettings>
#include <QString>
#include <QUrl>

// One service abstraction: every Spiral service is identified by a base URL
// and a credential, including a service on loopback. The three user-facing
// modes (Localhost, Remote LAN, Remote SSH) map onto this one structure.
struct SpiralServiceProfile
{
    enum class Transport { Direct, SshTunnel };

    QString id;                // settings key; "localhost" is the built-in profile
    QString name;
    Transport transport = Transport::Direct;
    QUrl baseUrl;              // Direct: service URL. Tunnel: derived, not stored.
    QString sshDestination;    // Tunnel: [user@]host, resolved through ~/.ssh/config
    int remoteServicePort = 0; // Tunnel: fixed loopback port of the host service
    QString apiKey;            // held separately from the persisted profile fields
    bool autoLaunch = false;   // launch and own a local service process (Localhost only)
    // Optional service-root -> viewer-root prefix mapping so input overlays can
    // be loaded locally when both machines mount the same dataset.
    QString serviceRootPrefix;
    QString localRootPrefix;

    bool isLocalhost() const { return autoLaunch; }
    bool isRemote() const { return !autoLaunch; }

    static SpiralServiceProfile localhostProfile()
    {
        SpiralServiceProfile profile;
        profile.id = QStringLiteral("localhost");
        profile.name = QObject::tr("Local (this computer)");
        profile.transport = Transport::Direct;
        profile.autoLaunch = true;
        return profile;
    }

    // API keys are deliberately not persisted here: they are session-only or
    // supplied through SPIRAL_API_KEY (SSH profiles read theirs from the host).
    void save(QSettings& settings) const
    {
        settings.beginGroup(QStringLiteral("spiral/profiles/") + id);
        settings.setValue(QStringLiteral("name"), name);
        settings.setValue(QStringLiteral("transport"),
                          transport == Transport::SshTunnel ? QStringLiteral("ssh")
                                                            : QStringLiteral("direct"));
        settings.setValue(QStringLiteral("base_url"), baseUrl.toString());
        settings.setValue(QStringLiteral("ssh_destination"), sshDestination);
        settings.setValue(QStringLiteral("remote_service_port"), remoteServicePort);
        settings.setValue(QStringLiteral("service_root_prefix"), serviceRootPrefix);
        settings.setValue(QStringLiteral("local_root_prefix"), localRootPrefix);
        settings.endGroup();
    }

    static SpiralServiceProfile load(QSettings& settings, const QString& profileId)
    {
        SpiralServiceProfile profile;
        profile.id = profileId;
        settings.beginGroup(QStringLiteral("spiral/profiles/") + profileId);
        profile.name = settings.value(QStringLiteral("name"), profileId).toString();
        profile.transport =
            settings.value(QStringLiteral("transport")).toString() == QStringLiteral("ssh")
                ? Transport::SshTunnel
                : Transport::Direct;
        profile.baseUrl = QUrl(settings.value(QStringLiteral("base_url")).toString());
        profile.sshDestination = settings.value(QStringLiteral("ssh_destination")).toString();
        profile.remoteServicePort = settings.value(QStringLiteral("remote_service_port"), 0).toInt();
        profile.serviceRootPrefix = settings.value(QStringLiteral("service_root_prefix")).toString();
        profile.localRootPrefix = settings.value(QStringLiteral("local_root_prefix")).toString();
        settings.endGroup();
        return profile;
    }
};
