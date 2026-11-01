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
    // Local launch bindings: the owned service is started with
    // --output/--cache/--winding-inference from these values (empty output
    // derives a per-store default under the app data directory; empty cache
    // uses the service's documented user-cache default; empty winding
    // inference launches without winding-model supervision). Remote profiles
    // never carry these: their services are configured on their hosts and
    // advertise their bindings through /bindings.
    QString outputRoot;
    QString cacheRoot;
    QString windingInference;
    // The hub of the service's object store as this machine reaches it (a
    // mount path, or sftp://[user@]host/path), for cloning a local replica
    // when the attached store is not one; empty uses the advertised address
    // (through the SSH destination for an SSH profile).
    QString hubAddress;

    bool isLocalhost() const { return autoLaunch; }
    bool isRemote() const { return !autoLaunch; }

    static SpiralServiceProfile localhostProfile(QSettings* settings = nullptr)
    {
        SpiralServiceProfile profile;
        profile.id = QStringLiteral("localhost");
        profile.name = QObject::tr("Local (this computer)");
        profile.transport = Transport::Direct;
        profile.autoLaunch = true;
        if (settings) {
            settings->beginGroup(QStringLiteral("spiral/profiles/localhost"));
            profile.outputRoot = settings->value(QStringLiteral("output_root")).toString();
            profile.cacheRoot = settings->value(QStringLiteral("cache_root")).toString();
            profile.windingInference =
                settings->value(QStringLiteral("winding_inference")).toString();
            settings->endGroup();
        }
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
        settings.setValue(QStringLiteral("output_root"), outputRoot);
        settings.setValue(QStringLiteral("cache_root"), cacheRoot);
        settings.setValue(QStringLiteral("winding_inference"), windingInference);
        // Retired: services take no dataset root, and nothing maps one.
        settings.remove(QStringLiteral("dataset_root"));
        settings.remove(QStringLiteral("local_root_prefix"));
        settings.setValue(QStringLiteral("hub_address"), hubAddress);
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
        profile.outputRoot = settings.value(QStringLiteral("output_root")).toString();
        profile.cacheRoot = settings.value(QStringLiteral("cache_root")).toString();
        profile.windingInference = settings.value(QStringLiteral("winding_inference")).toString();
        profile.hubAddress = settings.value(QStringLiteral("hub_address")).toString();
        settings.endGroup();
        return profile;
    }
};
