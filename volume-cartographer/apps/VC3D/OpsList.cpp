#include "OpsList.hpp"
#include "ui_OpsList.h"

#include "OpChain.hpp"

#include <iostream>


OpsList::OpsList(QWidget* parent) : QWidget(parent), ui(new Ui::OpsList)
{
    ui->setupUi(this);

    _tree = ui->treeWidget;
    _add_sel = ui->comboOp;
    connect(_tree, &QTreeWidget::currentItemChanged, this, &OpsList::onSelChanged);
    connect(ui->pushAppendOp, &QPushButton::clicked, this, &OpsList::onAppendOpClicked);
}

OpsList::~OpsList() { delete ui; }

void OpsList::onOpChainSelected(OpChain *ops)
{
    _op_chain = ops;
    std::cout << "OpChain/Layer selected " << ops << std::endl;

    _tree->clear();

    if (_op_chain) {
        QTreeWidgetItem *item = new QTreeWidgetItem(_tree);
        item->setText(0, QString(op_name(_op_chain)));
        item->setData(0, Qt::UserRole, QVariant::fromValue((void*)_op_chain));

        for (auto& op : _op_chain->ops()) {
            QTreeWidgetItem *item = new QTreeWidgetItem(_tree);
            item->setText(0, QString(op_name(op)));
            item->setData(0, Qt::UserRole, QVariant::fromValue((void*)op));
        }
    }
}

void OpsList::onSelChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous)
{
    if (!current)
        sendOpSelected(nullptr, _op_chain);
    else
        sendOpSelected((Surface*)qvariant_cast<void*>(current->data(0, Qt::UserRole)), _op_chain);
}

void OpsList::onAppendOpClicked()
{
    if (!_op_chain)
        return;

    QString sel = _add_sel->currentText();

    if (sel == "refineAlphaComp") {
        assert(_cache);
        assert(_ds);
        _op_chain->append(new RefineCompSurface(_ds, _cache));
    }

    onOpChainSelected(_op_chain);
    sendOpChainChanged(_op_chain);
}

void OpsList::setDataset(z5::Dataset *ds, ChunkCache *cache, float scale)
{
    _ds = ds;
    _cache = cache;
    _scale = scale;
}
