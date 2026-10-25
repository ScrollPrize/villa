#pragma once
// Asking how an object's stored coordinates relate to the open volume, for
// objects of any kind the store cannot map into it. No scale is assumed.
#include "vc/core/types/Volume.hpp"
#include <nlohmann/json.hpp>
#include <QInputDialog>
#include <QMessageBox>
#include <QObject>
#include <cmath>
#include <optional>

namespace vc3d {
// Stored units per voxel of `volume` for an object of `kind` ("surface",
// "fiber"), shown with its metadata and `details` (further lines, e.g. its
// bounds). nullopt when cancelled or not a positive finite number.
inline std::optional<double> promptStoredUnitsPerVoxel(QWidget* parent, const QString& title, const QString& kind,
                                                       const Volume& volume, const nlohmann::json& metadata,
                                                       const QString& details = {})
{
    const auto shape = volume.shape();
    const auto message = QObject::tr("Bind this %1 to the active volume?\n\nVolume: %2\nDimensions Z/Y/X: %3 / %4 / %5\n"
                                     "Resolution: %6 µm\nMetadata: %7\n%8\nEnter stored %1 units per active-volume voxel. "
                                     "No scale is assumed.")
        .arg(kind, QString::fromStdString(volume.id())).arg(shape[0]).arg(shape[1]).arg(shape[2]).arg(volume.voxelSize())
        .arg(QString::fromStdString(metadata.dump()), details.isEmpty() ? QString() : details + QLatin1Char('\n'));
    bool accepted = false;
    const auto answer = QInputDialog::getText(parent, title, message, QLineEdit::Normal, {}, &accepted);
    if (!accepted) return std::nullopt;
    bool valid = false;
    const double scale = answer.toDouble(&valid);
    if (!valid || !(scale > 0) || !std::isfinite(scale)) {
        QMessageBox::warning(parent, title, QObject::tr("Enter a positive finite coordinate scale."));
        return std::nullopt;
    }
    return scale;
}
}
