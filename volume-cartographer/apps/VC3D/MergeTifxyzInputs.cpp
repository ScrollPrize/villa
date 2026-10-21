#include "MergeTifxyzInputs.hpp"

#include <QCoreApplication>
#include <QRegularExpression>
#include <QSet>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace vc3d::merge_tifxyz {

namespace {

QString tr(const char* text)
{
    return QCoreApplication::translate("MergeTifxyzDialog", text);
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

QString alphaFirst(const QStringList& names)
{
    QString out;
    for (const auto& n : names) {
        if (n.isEmpty()) continue;
        if (out.isEmpty() || n < out) out = n;
    }
    return out;
}

QString resolveOutputName(const QString& volpkgDir, const QString& base)
{
    if (volpkgDir.isEmpty() || base.isEmpty()) return base;
    namespace fs = std::filesystem;
    const fs::path paths = fs::path(volpkgDir.toStdString()) / "paths";
    auto exists = [&](const std::string& name) {
        return fs::exists(paths / name);
    };
    if (!exists(base.toStdString())) return base;
    for (int v = 2; v < 1000; ++v) {
        QString candidate = base + QStringLiteral("_v%1").arg(v);
        if (!exists(candidate.toStdString())) return candidate;
    }
    return base + QStringLiteral("_v?");
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
    // token up as a directory under <paths_dir>, so a literal "" token
    // would be searched for as the directory `""` and the run would
    // fail. Switch to JSON array rows (with `null` for empty cells)
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
                 const QStringList& availableSegments,
                 const QString& pathsDir,
                 const QString& refSurface,
                 double ransacMinThresh,
                 double ransacMaxThresh)
{
    const QStringList names = flatten(rows);
    QStringList unique;
    for (const auto& n : names) {
        if (n.isEmpty()) continue;
        if (!unique.contains(n)) unique << n;
    }
    if (unique.size() < 2) {
        return tr("Merge requires at least 2 distinct surfaces in the grid; "
                  "currently %1.").arg(unique.size());
    }
    const QSet<QString> avail(availableSegments.begin(), availableSegments.end());
    QStringList missing;
    for (const auto& n : unique) {
        if (!avail.contains(n)) missing << n;
    }
    if (!missing.isEmpty()) {
        return tr("These names are not present under %1:\n  %2")
            .arg(pathsDir, missing.join(QLatin1String(", ")));
    }
    for (const auto& n : unique) {
        if (n.contains(QRegularExpression("\\s"))) {
            return tr("Surface name '%1' contains whitespace; the merge.json "
                      "format does not support that. Rename the surface first.")
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

QString writeMergeJson(const QString& volpkgDir, const Grid& rows, QString* error)
{
    auto fail = [error](const QString& message) {
        if (error) *error = message;
        return QString();
    };
    if (volpkgDir.isEmpty()) {
        return fail(tr("No volpkg directory available; cannot write merge.json."));
    }
    namespace fs = std::filesystem;
    const QString stem = alphaFirst(flatten(rows)) + QStringLiteral("_merged");
    const QString outName = resolveOutputName(volpkgDir, stem);
    const fs::path mj = fs::path(volpkgDir.toStdString()) /
                        (outName.toStdString() + ".merge.json");
    try {
        std::ofstream f(mj);
        if (!f) throw std::runtime_error("ofstream failed to open " + mj.string());
        const std::string s = gridJsonText(rows).toStdString();
        f.write(s.data(), static_cast<std::streamsize>(s.size()));
        if (!f) throw std::runtime_error("ofstream write failed for " + mj.string());
    } catch (const std::exception& e) {
        return fail(tr("Failed to write merge.json: %1").arg(QString::fromUtf8(e.what())));
    }
    return QString::fromStdString(mj.string());
}

}  // namespace vc3d::merge_tifxyz
