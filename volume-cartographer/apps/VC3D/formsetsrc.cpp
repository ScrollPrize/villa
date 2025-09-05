#include "formsetsrc.hpp"
#include "ui_formsetsrc.h"

FormSetSrc::FormSetSrc(Surface *op, QWidget* parent)
    : QWidget(parent), ui(new Ui::FormSetSrc)
{
    ui->setupUi(this);
    
    _chain = dynamic_cast<OpChain*>(op);
    assert(_chain);
    
    _combo = this->findChild<QComboBox*>("comboBox");
    
    _combo->setCurrentIndex(int(_chain->_src_mode));
    
    connect(_combo, &QComboBox::currentIndexChanged, this, &FormSetSrc::onAlgoIdxChanged);
}

FormSetSrc::~FormSetSrc() { delete ui; }

void FormSetSrc::onAlgoIdxChanged(int index)
{
    if (!_chain)
        return;
    
    _chain->_src_mode = OpChainSourceMode(index);
    
    sendOpChainChanged(_chain);
}