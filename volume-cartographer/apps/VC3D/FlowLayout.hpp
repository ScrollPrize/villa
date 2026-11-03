#pragma once

#include <QLayout>
#include <QList>

// Left-to-right layout that wraps onto another row when the width runs out,
// instead of squeezing its widgets below their size hint.
class FlowLayout final : public QLayout
{
public:
    explicit FlowLayout(QWidget* parent = nullptr, int hSpacing = 4, int vSpacing = 3);
    ~FlowLayout() override;

    void addItem(QLayoutItem* item) override;
    void insertWidget(int index, QWidget* widget);
    int count() const override;
    QLayoutItem* itemAt(int index) const override;
    QLayoutItem* takeAt(int index) override;

    Qt::Orientations expandingDirections() const override;
    bool hasHeightForWidth() const override;
    int heightForWidth(int width) const override;
    QSize minimumSize() const override;
    QSize sizeHint() const override;
    void setGeometry(const QRect& rect) override;

private:
    int doLayout(const QRect& rect, bool testOnly) const;

    QList<QLayoutItem*> _items;
    int _hSpace;
    int _vSpace;
};
