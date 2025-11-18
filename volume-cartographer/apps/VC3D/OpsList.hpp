#pragma once

#include <QWidget>
#include <ui_OpsList.h>
#include <z5/dataset.hxx>

#include "OpChain.hpp"
#include "SurfaceTreeWidget.hpp"
#include "vc/core/util/Slicing.hpp"


class OpsList : public QWidget
{
    Q_OBJECT

public:
    explicit OpsList(QWidget* parent = nullptr);
    ~OpsList();

    void setDataset(z5::Dataset *ds, ChunkCache<uint8_t> *cache, float scale);


private slots:
    void onSelChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous);

public slots:
    void onOpChainSelected(OpChain *ops);
    void onAppendOpClicked();

signals:
    void sendOpSelected(Surface *surf, OpChain *chain);
    void sendOpChainChanged(OpChain *chain);

private:
    Ui::OpsList* ui;
    QTreeWidget *_tree;
    QComboBox *_add_sel;
    OpChain *_op_chain = nullptr;

    //FIXME currently stored for refinement layer - make this somehow generic ...
    z5::Dataset *_ds = nullptr;
    ChunkCache<uint8_t> *_cache = nullptr;
    float _scale = 0.0;
};

