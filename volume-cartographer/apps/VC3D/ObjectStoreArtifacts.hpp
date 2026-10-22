#pragma once
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/Uuid.hpp>
#include <QCryptographicHash>
#include <QSaveFile>
#include <QTemporaryDir>
#include <filesystem>
#include <map>
#include <memory>

namespace vc3d {
// Immutable JSON transport artifacts. Ownership controls lifetime, attachment
// identity prevents reuse across store switches, and content determines reuse.
// File/directory formats such as tifxyz use the native geometry exporter.
inline std::map<std::string, std::filesystem::path> materializeJsonDocuments(
    std::shared_ptr<QTemporaryDir>& temporary, const vc::objects::AttachmentIdentity& attachment,
    const std::map<std::string, std::string>& documents)
{
    QCryptographicHash hash(QCryptographicHash::Sha256);
    for (const auto& [uuid, document] : documents) {
        vc::objects::util::canonicalUuid(uuid);
        hash.addData(QByteArrayView(uuid.data(), qsizetype(uuid.size())));
        hash.addData(QByteArrayView(document.data(), qsizetype(document.size())));
    }
    if (!temporary) temporary = std::make_shared<QTemporaryDir>();
    if (!temporary->isValid()) throw std::runtime_error("Cannot create object transport directory");
    const auto directory = std::filesystem::path(temporary->path().toStdString()) /
        std::to_string(attachment.generation) / attachment.storeUuid / hash.result().toHex().toStdString();
    std::filesystem::create_directories(directory);
    std::map<std::string, std::filesystem::path> paths;
    for (const auto& [uuid, document] : documents) {
        const auto path = directory / (uuid + ".json");
        if (!std::filesystem::exists(path)) {
            QSaveFile output(QString::fromStdString(path.string()));
            if (!output.open(QIODevice::WriteOnly) ||
                output.write(document.data(), document.size()) != qint64(document.size()) || !output.commit())
                throw std::runtime_error("Cannot write object transport artifact " + path.string() + ": " + output.errorString().toStdString());
        }
        paths.emplace(uuid, path);
    }
    return paths;
}
}
