#pragma once
#include <QWidget>
#include <QLabel>
#include <QMouseEvent>
#include <algorithm>

// A viewport panel that follows the selection until the user places it.
class ConstraintGraphSelectionBar : public QWidget {
public:
    explicit ConstraintGraphSelectionBar(QWidget* viewport):QWidget(viewport) {
        viewport->installEventFilter(this);
    }
    QLabel* makeGrip() {
        auto* grip=new QLabel(tr("⠿  Drag to move"),this);
        grip->setObjectName("constraintGraphSelectionGrip");
        grip->setCursor(Qt::SizeAllCursor);grip->installEventFilter(this);_grip=grip;
        return grip;
    }
    void placeNear(QPoint anchor) {
        adjustSize();move(clamped(_placed?pos():anchor));raise();
    }
protected:
    bool eventFilter(QObject* object,QEvent* event) override {
        if(object==parentWidget()&&event->type()==QEvent::Resize)move(clamped(pos()));
        if(object==_grip) {
            if(event->type()==QEvent::MouseButtonPress) {
                auto* mouse=static_cast<QMouseEvent*>(event);
                if(mouse->button()==Qt::LeftButton){_dragging=true;_start=mouse->globalPosition().toPoint();_origin=pos();return true;}
            }
            if(event->type()==QEvent::MouseMove&&_dragging) {
                auto* mouse=static_cast<QMouseEvent*>(event);
                _placed=true;move(clamped(_origin+mouse->globalPosition().toPoint()-_start));return true;
            }
            if(event->type()==QEvent::MouseButtonRelease&&_dragging) {_dragging=false;return true;}
        }
        return QWidget::eventFilter(object,event);
    }
private:
    QPoint clamped(QPoint p) const {
        return {std::clamp(p.x(),0,std::max(0,parentWidget()->width()-width())),
                std::clamp(p.y(),0,std::max(0,parentWidget()->height()-height()))};
    }
    QLabel* _grip=nullptr;
    bool _dragging=false,_placed=false;
    QPoint _start,_origin;
};
