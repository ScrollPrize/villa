#pragma once

#include <QElapsedTimer>
#include <QLabel>
#include <QProgressBar>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

// Independent of fit progress: editing access is prepared before a fit exists.
class SpiralActivityWidget : public QWidget
{
public:
    explicit SpiralActivityWidget(QWidget* parent = nullptr) : QWidget(parent)
    {
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        _label = new QLabel(this);
        _label->setObjectName(QStringLiteral("spiralInputActivityText"));
        _label->setWordWrap(true);
        auto* progress = new QProgressBar(this);
        progress->setRange(0, 0);
        layout->addWidget(_label);
        layout->addWidget(progress);
        _timer.setInterval(1000);
        connect(&_timer, &QTimer::timeout, this, [this]() { refresh(); });
        hide();
    }

    void setPreparation(const QString& message) { _preparation = message; refresh(); }
    void setCopy(int active, const QString& message) { _copy = active > 0 ? message : QString(); refresh(); }
    void reset() { _preparation.clear(); _copy.clear(); refresh(); }

private:
    void refresh()
    {
        const bool active = !_preparation.isEmpty() || !_copy.isEmpty();
        if (active && !_timer.isActive()) { _elapsed.start(); _timer.start(); }
        if (!active) { _timer.stop(); _label->clear(); hide(); return; }
        QString text = _preparation;
        if (!_copy.isEmpty()) {
            if (!text.isEmpty()) text += QLatin1Char('\n');
            text += _copy;
        }
        const auto seconds = _elapsed.elapsed() / 1000;
        _label->setText(tr("%1\nElapsed: %2m %3s").arg(text)
                            .arg(seconds / 60).arg(seconds % 60));
        show();
    }

    QLabel* _label = nullptr;
    QTimer _timer;
    QElapsedTimer _elapsed;
    QString _preparation;
    QString _copy;
};
