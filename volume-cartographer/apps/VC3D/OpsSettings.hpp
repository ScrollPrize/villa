#pragma once

#include <QWidget>
#include <ui_OpsSettings.h>

#include "OpChain.hpp"
#include "vc/core/util/Surface.hpp"


class OpsSettings : public QWidget
{
    Q_OBJECT

public:
    explicit OpsSettings(QWidget* parent = nullptr);
    ~OpsSettings();

public slots:
    void onOpSelected(Surface *op, OpChain *chain);
    void onEnabledChanged();

signals:
    void sendOpChainChanged(OpChain *chain);

private:
    Ui::OpsSettings* ui;
    QGroupBox *_box = nullptr;
    QCheckBox *_enable = nullptr;

    Surface *_op = nullptr;
    OpChain *_chain = nullptr;
    
    QWidget *_form = nullptr;
};
