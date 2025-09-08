#include "OpsSettings.hpp"
#include "ui_OpsSettings.h"

#include "OpChain.hpp"
#include "formsetsrc.hpp"

OpsSettings::OpsSettings(QWidget* parent)
    : QWidget(parent), ui(new Ui::OpsSettings)
{
    ui->setupUi(this);
    // Prefer direct UI pointers over findChild and guard for nulls
    _box = ui->groupBox;
    _enable = ui->chkEnableLayer;

    if (_box)
        _box->setVisible(false); // start invisible until a layer is selected

    if (_enable)
        connect(_enable, &QCheckBox::stateChanged, this, &OpsSettings::onEnabledChanged);
}

OpsSettings::~OpsSettings() { delete ui; }

void OpsSettings::onEnabledChanged()
{
    if (_chain) {
        _chain->setEnabled((DeltaSurface*)_op, _enable->isChecked());
        sendOpChainChanged(_chain);
    }
}

QWidget *op_form_widget(Surface *op, OpsSettings *parent)
{
    if (!op)
        return nullptr;
    
    if (dynamic_cast<OpChain*>(op)) {
        auto w = new FormSetSrc(op, parent);
        //TODO inherit all settings form widgets from common base, unify the connect
        QWidget::connect(w, &FormSetSrc::sendOpChainChanged, parent, &OpsSettings::sendOpChainChanged);
        return w;
    }

    return nullptr;
}


void OpsSettings::onOpSelected(Surface *op, OpChain *chain)
{
    _op = op;
    _chain = chain;

    // If we have no layer selected (e.g. because a new surface was selected to
    // display which resets the layer selection), hide the box until a
    // layer actually is selected.
    if(!_op) {
        if (_box)
            _box->setVisible(false);
        return;
    }

    if (_box)
        _box->setTitle(tr("Selected Layer: %1").arg(QString(op_name(op))));

    if (_enable) {
        if (!dynamic_cast<DeltaSurface*>(_op)) {
            _enable->setEnabled(false);
        } else if (_chain) {
            _enable->setEnabled(true);
            QSignalBlocker blocker(_enable);
            _enable->setChecked(_chain->enabled((DeltaSurface*)_op));
        } else {
            _enable->setEnabled(false);
        }
    }
    
    if (_form)
        delete _form;
    
    _form = op_form_widget(op, this);
    if (_form && _box && _box->layout()) {
        _box->layout()->addWidget(_form);
        _box->setVisible(true);
    }
}
