#pragma once

#include <QWidget>
#include <ui_formsetsrc.h>

#include "OpChain.hpp"


class FormSetSrc : public QWidget
{
    Q_OBJECT

public:
    explicit FormSetSrc(Surface *op, QWidget* parent = nullptr);
    ~FormSetSrc();
    
private slots:
    void onAlgoIdxChanged(int index);
    
signals:
    void sendOpChainChanged(OpChain *chain);

private:
    Ui::FormSetSrc* ui;
    
    OpChain *_chain;
    QComboBox *_combo;
};

