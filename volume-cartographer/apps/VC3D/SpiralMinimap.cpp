#include "SpiralMinimap.hpp"

#include <QHelpEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QToolTip>

#include <algorithm>
#include <cmath>

namespace
{
// Golden-angle hue walk: neighboring windings get clearly different, stable
// colors, and the same winding keeps its color across previews.
QColor windingColor(int winding)
{
    const double hue = std::fmod(static_cast<double>(winding) * 0.381966, 1.0);
    return QColor::fromHsvF(hue, 0.55, 0.80);
}
} // namespace

SpiralMinimap::SpiralMinimap(QWidget* parent)
    : QWidget(parent)
{
    setFixedHeight(20);
    setCursor(Qt::PointingHandCursor);
    hide();
}

void SpiralMinimap::setBands(std::vector<Band> bands,
                                    float columnBegin, float columnEnd)
{
    _bands = std::move(bands);
    std::sort(_bands.begin(), _bands.end(),
              [](const Band& a, const Band& b) {
                  return a.columnBegin < b.columnBegin;
              });
    _columnBegin = columnBegin;
    _columnEnd = columnEnd;
    setVisible(!_bands.empty() && hasSpan());
    update();
}

void SpiralMinimap::clearBands()
{
    _bands.clear();
    _columnBegin = 0.0f;
    _columnEnd = 0.0f;
    hide();
}

void SpiralMinimap::setViewIndicator(float columnBegin, float columnEnd)
{
    if (_viewBegin == columnBegin && _viewEnd == columnEnd) return;
    _viewBegin = columnBegin;
    _viewEnd = columnEnd;
    update();
}

float SpiralMinimap::columnAtX(qreal x) const
{
    if (!hasSpan() || width() <= 0) return _columnBegin;
    const float fraction = std::clamp(
        static_cast<float>(x / width()), 0.0f, 1.0f);
    return _columnBegin + fraction * (_columnEnd - _columnBegin);
}

qreal SpiralMinimap::xAtColumn(float column) const
{
    return (column - _columnBegin) / (_columnEnd - _columnBegin) * width();
}

const SpiralMinimap::Band* SpiralMinimap::bandAtX(qreal x) const
{
    const float column = columnAtX(x);
    for (const Band& band : _bands)
        if (column >= band.columnBegin && column < band.columnEnd)
            return &band;
    return nullptr;
}

void SpiralMinimap::paintEvent(QPaintEvent*)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor(30, 30, 30));
    if (!hasSpan()) return;

    QFont font = painter.font();
    font.setPixelSize(10);
    painter.setFont(font);
    const QFontMetricsF metrics(font);
    for (const Band& band : _bands) {
        const qreal left = xAtColumn(band.columnBegin);
        const qreal right = xAtColumn(band.columnEnd);
        const QRectF bandRect(left, 0.0, right - left, height());
        painter.fillRect(bandRect, windingColor(band.winding));
        const QString label = QString::number(band.winding);
        if (bandRect.width() >= metrics.horizontalAdvance(label) + 6.0) {
            painter.setPen(QColor(0, 0, 0, 180));
            painter.drawText(bandRect, Qt::AlignCenter, label);
        }
    }

    if (_viewEnd > _viewBegin) {
        const qreal left = std::max(0.0, xAtColumn(_viewBegin));
        const qreal right = std::min(
            static_cast<qreal>(width()), xAtColumn(_viewEnd));
        if (right > left) {
            const QRectF viewRect(left, 0.0, right - left, height());
            painter.fillRect(viewRect, QColor(255, 255, 255, 60));
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1.0));
            painter.drawRect(viewRect.adjusted(0.5, 0.5, -0.5, -0.5));
        }
    }
}

void SpiralMinimap::mousePressEvent(QMouseEvent* event)
{
    if (event->button() != Qt::LeftButton || !hasSpan()) {
        QWidget::mousePressEvent(event);
        return;
    }
    emit columnClicked(columnAtX(event->position().x()));
    event->accept();
}

void SpiralMinimap::mouseMoveEvent(QMouseEvent* event)
{
    if ((event->buttons() & Qt::LeftButton) && hasSpan()) {
        emit columnClicked(columnAtX(event->position().x()));
        event->accept();
        return;
    }
    QWidget::mouseMoveEvent(event);
}

bool SpiralMinimap::event(QEvent* event)
{
    if (event->type() == QEvent::ToolTip) {
        auto* helpEvent = static_cast<QHelpEvent*>(event);
        if (const Band* band = bandAtX(helpEvent->pos().x())) {
            QToolTip::showText(helpEvent->globalPos(),
                               tr("Winding %1").arg(band->winding), this);
        } else {
            QToolTip::hideText();
        }
        return true;
    }
    return QWidget::event(event);
}
