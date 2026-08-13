#include "FiberNameDisplay.hpp"

#include <algorithm>
#include <optional>

namespace vc3d {
namespace {

bool allDigits(const QString& text)
{
    if (text.isEmpty()) {
        return false;
    }
    for (const QChar ch : text) {
        if (!ch.isDigit()) {
            return false;
        }
    }
    return true;
}

bool looksLikeFiberTimestamp(const QString& text)
{
    if (text.size() < 10 || text[8] != QLatin1Char('T')) {
        return false;
    }
    for (int i = 0; i < text.size(); ++i) {
        if (i == 8) {
            continue;
        }
        if (!text[i].isDigit()) {
            return false;
        }
    }
    return true;
}

struct FiberNameParts {
    QString prefix;
    QString timestamp;
    QString sequence;
};

std::optional<FiberNameParts> splitFiberName(QString name)
{
    const int lastSeparator = name.lastIndexOf(QLatin1Char('_'));
    const int timestampSeparator = lastSeparator > 0
        ? name.lastIndexOf(QLatin1Char('_'), lastSeparator - 1)
        : -1;
    if (timestampSeparator >= 0 && lastSeparator > timestampSeparator) {
        const QString prefix = name.left(timestampSeparator);
        const QString timestamp =
            name.mid(timestampSeparator + 1, lastSeparator - timestampSeparator - 1);
        const QString sequence = name.mid(lastSeparator + 1);
        if (!prefix.isEmpty() &&
            looksLikeFiberTimestamp(timestamp) &&
            sequence.size() == 6 &&
            allDigits(sequence)) {
            return FiberNameParts{prefix, timestamp, sequence};
        }
    }

    return std::nullopt;
}

QString rightElideAscii(const QString& text, const QFontMetrics& metrics, int width)
{
    if (width <= 0) {
        return QString();
    }
    if (metrics.horizontalAdvance(text) <= width) {
        return text;
    }

    const QString ellipsis = QStringLiteral("...");
    if (metrics.horizontalAdvance(ellipsis) > width) {
        return QString();
    }

    for (int kept = text.size() - 1; kept >= 0; --kept) {
        const QString candidate = text.left(kept) + ellipsis;
        if (metrics.horizontalAdvance(candidate) <= width) {
            return candidate;
        }
    }

    return ellipsis;
}

QString elidePrefixBeforeSuffix(const QString& prefix,
                                const QString& suffix,
                                const QFontMetrics& metrics,
                                int width)
{
    if (width <= 0) {
        return QString();
    }

    const QString full = prefix + suffix;
    if (metrics.horizontalAdvance(full) <= width) {
        return full;
    }

    if (metrics.horizontalAdvance(suffix) >= width) {
        return metrics.elidedText(suffix, Qt::ElideLeft, width);
    }

    const int prefixWidth = width - metrics.horizontalAdvance(suffix);
    QString shortenedPrefix = rightElideAscii(prefix, metrics, prefixWidth);
    QString candidate = shortenedPrefix + suffix;
    if (!shortenedPrefix.isEmpty() && metrics.horizontalAdvance(candidate) <= width) {
        return candidate;
    }

    return suffix;
}

} // namespace

QString displayStemForFiberFile(QString fileName)
{
    fileName = fileName.trimmed();
    const int slash = std::max(fileName.lastIndexOf(QLatin1Char('/')),
                               fileName.lastIndexOf(QLatin1Char('\\')));
    if (slash >= 0) {
        fileName = fileName.mid(slash + 1);
    }
    if (fileName.endsWith(QStringLiteral(".json"), Qt::CaseInsensitive)) {
        fileName.chop(5);
    }
    if (fileName.isEmpty()) {
        return QString();
    }

    return fileName;
}

QString adaptFiberNameToWidth(const QString& name, const QFontMetrics& metrics, int width)
{
    if (width <= 0 || metrics.horizontalAdvance(name) <= width) {
        return name;
    }

    const std::optional<FiberNameParts> parts = splitFiberName(name);
    if (!parts) {
        return metrics.elidedText(name, Qt::ElideRight, width);
    }

    const QString fixedText = parts->prefix + QStringLiteral("__") + parts->sequence;
    const int timestampWidth = width - metrics.horizontalAdvance(fixedText);
    QString timestamp = rightElideAscii(parts->timestamp, metrics, timestampWidth);
    if (timestamp.isEmpty()) {
        timestamp = QStringLiteral("...");
    }

    const QString candidate = parts->prefix + QLatin1Char('_') + timestamp +
                              QLatin1Char('_') + parts->sequence;
    if (metrics.horizontalAdvance(candidate) <= width) {
        return candidate;
    }

    return elidePrefixBeforeSuffix(parts->prefix,
                                   QStringLiteral("_..._") + parts->sequence,
                                   metrics,
                                   width);
}

} // namespace vc3d
