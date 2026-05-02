#pragma once

#include <QLabel>
#include <QStringList>

class ViewerStatsBar : public QLabel
{
    Q_OBJECT

public:
    explicit ViewerStatsBar(QWidget* parent = nullptr);

    void setItems(const QStringList& items);
};
