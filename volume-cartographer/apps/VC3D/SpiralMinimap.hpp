#pragma once

#include <QWidget>

#include <vector>

// Fixed-height strip docked below the flattened Spiral preview viewer. Each
// displayed winding is one colored band; the full widget width maps to the
// full column span of the displayed preview (the whole spiral length).
// Clicking or dragging requests a horizontal pan to the matching column, and
// a translucent marker mirrors the viewer's currently visible column range.
class SpiralMinimap : public QWidget
{
    Q_OBJECT
public:
    struct Band {
        int winding = 0;
        // Grid columns of the displayed preview crop.
        float columnBegin = 0.0f;
        float columnEnd = 0.0f;
    };

    explicit SpiralMinimap(QWidget* parent = nullptr);

    void setBands(std::vector<Band> bands, float columnBegin, float columnEnd);
    void clearBands();
    // Visible column range of the viewer; pass an empty/inverted range to
    // hide the marker.
    void setViewIndicator(float columnBegin, float columnEnd);

signals:
    void columnClicked(float column);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    bool event(QEvent* event) override;

private:
    bool hasSpan() const { return _columnEnd > _columnBegin; }
    float columnAtX(qreal x) const;
    qreal xAtColumn(float column) const;
    const Band* bandAtX(qreal x) const;

    std::vector<Band> _bands;
    float _columnBegin = 0.0f;
    float _columnEnd = 0.0f;
    float _viewBegin = 0.0f;
    float _viewEnd = -1.0f;
};
