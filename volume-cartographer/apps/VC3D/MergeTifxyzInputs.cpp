#include "MergeTifxyzInputs.hpp"

#include <QCoreApplication>
#include <QRegularExpression>
#include <QSet>

#include <algorithm>

namespace vc3d::merge_tifxyz {

namespace {

QString tr(const char* text)
{
    return QCoreApplication::translate("MergeSurfacesDialog", text);
}

QStringList flatten(const Grid& rows)
{
    QStringList out;
    for (const QStringList& row : rows) {
        for (const QString& cell : row) {
            out << cell.trimmed();
        }
    }
    return out;
}

}  // namespace

QStringList distinctSurfaces(const Grid& rows)
{
    QStringList unique;
    for (const auto& n : flatten(rows)) {
        if (n.isEmpty()) continue;
        if (!unique.contains(n)) unique << n;
    }
    return unique;
}

QString gridJsonText(const Grid& rows)
{
    int cols = 0;
    for (const QStringList& row : rows) {
        cols = std::max(cols, static_cast<int>(row.size()));
    }
    auto cell = [&rows](int r, int c) -> QString {
        const QStringList& row = rows[r];
        return c < row.size() ? row[c].trimmed() : QString();
    };

    // Whitespace-delimited string rows can't represent interior empty
    // cells -- the parser splits the row on whitespace and looks each
    // token up as a surface, so a literal "" token would be searched for
    // as the surface `""` and the run would fail. Switch to JSON array rows (with `null` for empty cells)
    // whenever the grid has any blank, which is the form gmResolveGrid
    // already handles. Solid grids keep the more compact string form.
    bool hasEmpty = false;
    for (int r = 0; r < rows.size() && !hasEmpty; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (cell(r, c).isEmpty()) {
                hasEmpty = true;
                break;
            }
        }
    }

    QString out;
    out += QStringLiteral("{\n  \"rows\": [\n");

    if (hasEmpty) {
        for (int r = 0; r < rows.size(); ++r) {
            QStringList cells;
            cells.reserve(cols);
            for (int c = 0; c < cols; ++c) {
                const QString name = cell(r, c);
                if (name.isEmpty()) {
                    cells << QStringLiteral("null");
                } else {
                    cells << QStringLiteral("\"") + name + QLatin1Char('"');
                }
            }
            out += QStringLiteral("    [")
                 + cells.join(QStringLiteral(", "))
                 + QLatin1Char(']');
            if (r + 1 < rows.size()) out += QLatin1Char(',');
            out += QLatin1Char('\n');
        }
    } else {
        // Pad each cell so visual columns line up in the preview.
        QVector<int> widths(cols, 0);
        for (int r = 0; r < rows.size(); ++r) {
            for (int c = 0; c < cols; ++c) {
                const int w = static_cast<int>(cell(r, c).size());
                if (w > widths[c]) widths[c] = w;
            }
        }
        for (int r = 0; r < rows.size(); ++r) {
            QStringList tokens;
            tokens.reserve(cols);
            for (int c = 0; c < cols; ++c) {
                tokens << cell(r, c).leftJustified(std::max(widths[c], 1), ' ');
            }
            out += QStringLiteral("    \"")
                 + tokens.join(' ').trimmed()
                 + QLatin1Char('"');
            if (r + 1 < rows.size()) out += QLatin1Char(',');
            out += QLatin1Char('\n');
        }
    }

    out += QStringLiteral("  ]\n}\n");
    return out;
}

QString validate(const Grid& rows,
                 const QStringList& availableSurfaces,
                 const QString& refSurface,
                 double ransacMinThresh,
                 double ransacMaxThresh)
{
    const QStringList unique = distinctSurfaces(rows);
    if (unique.size() < 2) {
        return tr("Merge requires at least 2 distinct surfaces in the grid; "
                  "currently %1.").arg(unique.size());
    }
    const QSet<QString> avail(availableSurfaces.begin(), availableSurfaces.end());
    QStringList missing;
    for (const auto& n : unique) {
        if (!avail.contains(n)) missing << n;
    }
    if (!missing.isEmpty()) {
        return tr("These are not surfaces of the current volume:\n  %1")
            .arg(missing.join(QLatin1String(", ")));
    }
    for (const auto& n : unique) {
        if (n.contains(QRegularExpression("\\s"))) {
            return tr("Surface '%1' contains whitespace; the merge.json "
                      "format does not support that.")
                .arg(n);
        }
    }
    if (ransacMinThresh >= ransacMaxThresh) {
        return tr("RANSAC min threshold must be strictly less than max.");
    }
    if (!refSurface.isEmpty() && !unique.contains(refSurface)) {
        return tr("Reference '%1' is not in the grid.").arg(refSurface);
    }
    return {};
}

}  // namespace vc3d::merge_tifxyz
