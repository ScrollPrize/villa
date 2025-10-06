#include "elements/ProgressUtil.hpp"

#include <QProgressBar>

#include <algorithm>
#include <cmath>


namespace {

constexpr int kDefaultProgressMinimum = 0;
constexpr int kDefaultProgressMaximum = 100;

QString replaceToken(QString text, const QString& token, const QString& value)
{
    if (text.contains(token)) {
        text.replace(token, value);
    }
    return text;
}

} // namespace


ProgressUtil::ProgressUtil(QStatusBar* statusBar, QObject* parent)
    : QObject(parent)
    , _statusBar(statusBar)
    , _animTimer(nullptr)
    , _animFrame(0)
    , _progressBar(nullptr)
    , _progressTotal(0)
    , _progressValue(0)
    , _progressActive(false)
    , _hasStoredBarFont(false)
    , _hasStoredBarFormat(false)
    , _storedBarMinimum(kDefaultProgressMinimum)
    , _storedBarMaximum(kDefaultProgressMaximum)
{
}

ProgressUtil::~ProgressUtil()
{
    stopProgress();

    if (_animTimer) {
        if (_animTimer->isActive()) {
            _animTimer->stop();
        }
        delete _animTimer;
    }
}

void ProgressUtil::startAnimation(const QString& message)
{
    _animFrame = 0;
    _message = message;
    if (!_animTimer) {
        _animTimer = new QTimer(this);
        connect(_animTimer, &QTimer::timeout, this, &ProgressUtil::updateAnimation);
    }

    if (_statusBar) {
        _statusBar->showMessage(message + QStringLiteral(" |"), 0); // 0 timeout keeps it visible
    }
    _animTimer->start(300);
}

void ProgressUtil::stopAnimation(const QString& message, int timeout)
{
    if (_animTimer && _animTimer->isActive()) {
        _animTimer->stop();
    }

    if (_statusBar) {
        _statusBar->showMessage(message, timeout);
    }
}

void ProgressUtil::setProgressBar(QProgressBar* progressBar)
{
    if (_progressBar == progressBar) {
        return;
    }

    if (_progressBar && _progressBar != progressBar) {
        restoreProgressBarFont();
        restoreProgressBarFormat();
        _progressBar->setVisible(false);
    }

    _progressBar = progressBar;

    if (_progressBar) {
        _storedBarFormat = _progressBar->format();
        _storedBarFont = _progressBar->font();
        _storedBarMinimum = _progressBar->minimum();
        _storedBarMaximum = _progressBar->maximum();
        _hasStoredBarFont = true;
        _hasStoredBarFormat = true;
        _progressBar->setVisible(false);
    }
}

void ProgressUtil::configureProgressBar(const ProgressBarOptions& options)
{
    _defaultBarOptions = options;
    if (!_progressActive) {
        applyProgressBarFont();
        if (_progressBar) {
            if (_defaultBarOptions.textMode == ProgressTextMode::None) {
                _progressBar->setTextVisible(false);
            } else if (_hasStoredBarFormat) {
                _progressBar->setTextVisible(true);
                _progressBar->setFormat(_storedBarFormat);
            }
        }
    }
}

void ProgressUtil::startProgress(int totalSteps, const ProgressBarOptions* options)
{
    if (!_progressBar) {
        return;
    }

    _progressActive = true;
    _progressTotal = std::max(0, totalSteps);
    _progressValue = 0;
    _activeBarOptions = options ? *options : _defaultBarOptions;

    applyProgressBarFont();

    if (_progressTotal <= 0) {
        _progressBar->setRange(0, 0); // busy indicator
    } else {
        _progressBar->setRange(0, _progressTotal);
        _progressBar->setValue(0);
    }

    if (_activeBarOptions.textMode == ProgressTextMode::None) {
        _progressBar->setTextVisible(false);
    } else {
        _progressBar->setTextVisible(true);
    }

    updateProgressFormat();
    _progressBar->setVisible(true);
}

void ProgressUtil::updateProgress(int completedSteps)
{
    if (!_progressBar || !_progressActive) {
        return;
    }

    if (_progressTotal > 0) {
        _progressValue = std::clamp(completedSteps, 0, _progressTotal);
        _progressBar->setValue(_progressValue);
    } else {
        _progressValue = std::max(0, completedSteps);
        _progressBar->setValue(_progressValue);
    }

    updateProgressFormat();
}

void ProgressUtil::advanceProgress(int stepDelta)
{
    if (!_progressActive) {
        return;
    }

    int newValue = _progressValue + stepDelta;
    if (_progressTotal > 0) {
        newValue = std::clamp(newValue, 0, _progressTotal);
    } else {
        newValue = std::max(0, newValue);
    }
    updateProgress(newValue);
}

void ProgressUtil::stopProgress(bool resetValue)
{
    _progressActive = false;
    _progressTotal = 0;
    _progressValue = 0;

    if (_progressBar) {
        if (resetValue) {
            _progressBar->setRange(_storedBarMinimum, _storedBarMaximum);
            _progressBar->setValue(_storedBarMinimum);
        }
        restoreProgressBarFont();
        restoreProgressBarFormat();
        _progressBar->setVisible(false);
    }
}

void ProgressUtil::updateAnimation()
{
    static const QChar animChars[] = {'|', '/', '-', '\\'};
    _animFrame = (_animFrame + 1) % 4;
    if (_statusBar) {
        _statusBar->showMessage(_message + QStringLiteral(" ") + animChars[_animFrame], 0);
    }
}

void ProgressUtil::applyProgressBarFont()
{
    if (!_progressBar) {
        return;
    }

    const ProgressBarOptions& options = _progressActive ? _activeBarOptions : _defaultBarOptions;
    if (options.fontPointSize > 0) {
        QFont font = _progressBar->font();
        if (font.pointSize() != options.fontPointSize) {
            font.setPointSize(options.fontPointSize);
            _progressBar->setFont(font);
        }
    } else if (_hasStoredBarFont) {
        _progressBar->setFont(_storedBarFont);
    }
}

void ProgressUtil::restoreProgressBarFont()
{
    if (_progressBar && _hasStoredBarFont) {
        _progressBar->setFont(_storedBarFont);
    }
}

void ProgressUtil::restoreProgressBarFormat()
{
    if (_progressBar) {
        if (_hasStoredBarFormat) {
            _progressBar->setFormat(_storedBarFormat);
        } else {
            _progressBar->setFormat(QStringLiteral("%p%"));
        }
        _progressBar->setTextVisible(true);
    }
}

void ProgressUtil::updateProgressFormat()
{
    if (!_progressBar) {
        return;
    }

    const ProgressBarOptions& options = _progressActive ? _activeBarOptions : _defaultBarOptions;

    if (options.textMode == ProgressTextMode::None) {
        _progressBar->setTextVisible(false);
        return;
    }

    _progressBar->setTextVisible(true);

    if (!_progressActive) {
        if (_hasStoredBarFormat) {
            _progressBar->setFormat(_storedBarFormat);
        }
        return;
    }

    const int total = _progressTotal;
    const int maxValue = (total > 0) ? total : _progressValue;
    const int value = std::clamp(_progressValue, 0, maxValue);
    const int remaining = (total > 0) ? std::max(0, total - value) : 0;
    const int percentPrecision = std::max(0, options.percentPrecision);
    const double percent = (total > 0) ? (static_cast<double>(value) * 100.0) / static_cast<double>(total)
                                       : 0.0;

    QString formatted;

    switch (options.textMode) {
    case ProgressTextMode::Percent: {
        if (total <= 0) {
            formatted = QStringLiteral("--");
        } else if (percentPrecision == 0) {
            formatted = QString::number(static_cast<int>(std::round(percent)));
        } else {
            formatted = QString::number(percent, 'f', percentPrecision);
        }
        formatted.append(QStringLiteral("%"));
        break;
    }
    case ProgressTextMode::Fraction: {
        if (total > 0) {
            formatted = QStringLiteral("%1 / %2").arg(value).arg(total);
        } else {
            formatted = QString::number(value);
        }
        break;
    }
    case ProgressTextMode::DoneRemaining: {
        if (total > 0) {
            formatted = QStringLiteral("%1 done : %2 left").arg(value).arg(remaining);
        } else {
            formatted = QStringLiteral("%1 done").arg(value);
        }
        break;
    }
    case ProgressTextMode::Custom: {
        formatted = options.customFormat;
        formatted = replaceToken(formatted, QStringLiteral("{value}"), QString::number(value));
        formatted = replaceToken(formatted, QStringLiteral("{total}"), total > 0 ? QString::number(total) : QStringLiteral("-"));
        formatted = replaceToken(formatted, QStringLiteral("{remaining}"), QString::number(remaining));
        if (percentPrecision == 0) {
            formatted = replaceToken(formatted, QStringLiteral("{percent}"),
                                     QString::number(static_cast<int>(std::round(percent))));
        } else {
            formatted = replaceToken(formatted, QStringLiteral("{percent}"),
                                     QString::number(percent, 'f', percentPrecision));
        }
        break;
    }
    case ProgressTextMode::None:
        // Already handled above
        break;
    }

    if (!options.prefix.isEmpty()) {
        formatted.prepend(options.prefix);
    }
    if (!options.suffix.isEmpty()) {
        formatted.append(options.suffix);
    }

    if (!formatted.isEmpty()) {
        _progressBar->setFormat(formatted);
    }
}
