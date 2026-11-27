#pragma once

#include <QByteArray>
#include <QCryptographicHash>
#include <QFile>
#include <QString>

namespace vc3d {

// Lowercase hex SHA-256 of a file's bytes, read in chunks, or an empty string
// if the file cannot be opened or read.
inline QString fileSha256Hex(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
        return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!hash.addData(&file))
        return {};
    return QString::fromLatin1(hash.result().toHex());
}

}  // namespace vc3d
