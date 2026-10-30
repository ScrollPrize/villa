#pragma once
// The Transforms dock: the volumes the object store relates to the open one
// and the route it uses for each. Registering a new affine and choosing among
// a pair's routes write store objects; the store's views do the rest.
#include "VolumeRegistrations.hpp"

#include <QWidget>

#include <vector>

class CState;
class QLabel;
class QPushButton;
class QTreeWidget;

class VolumeRegistrationsPanel : public QWidget
{
    Q_OBJECT

public:
    explicit VolumeRegistrationsPanel(CState* state, QWidget* parent = nullptr);

    // Read the open volume's relations from the store again.
    void refresh();

private:
    void registerTransform();
    void chooseRoute();
    void updateButtons();

    CState* _state{nullptr};
    QLabel* _heading{nullptr};
    QTreeWidget* _list{nullptr};
    QPushButton* _choose{nullptr};
    QPushButton* _register{nullptr};
    QLabel* _status{nullptr};
    vcr::catalog::RegistrationGraph _graph;
    std::vector<vc::objects::VolumeTransforms::Related> _related;
};
