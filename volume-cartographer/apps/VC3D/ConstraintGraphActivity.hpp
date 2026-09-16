#pragma once

#include <QElapsedTimer>
#include <QLabel>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <array>

// Independent rows prevent background discovery from hiding a foreground fit.
class ConstraintGraphActivity : public QWidget {
public:
    enum Channel { Operation, Discovery, Comparison };
    explicit ConstraintGraphActivity(QWidget* parent=nullptr):QWidget(parent) {
        auto* layout=new QVBoxLayout(this);layout->setContentsMargins(8,4,8,4);
        const std::array<QString,3> names{tr("Operation"),tr("Attachment search"),tr("Source comparison")};
        for(int i=0;i<3;++i) {
            auto& r=rows_[i];auto* line=new QHBoxLayout;
            r.text=new QLabel(names[i]+tr(": Idle"),this);r.text->setWordWrap(true);
            r.text->setTextFormat(Qt::PlainText);r.text->setTextInteractionFlags(Qt::TextSelectableByMouse);
            r.text->setObjectName(QString("graphActivity%1").arg(i));
            r.elapsed=new QLabel(this);r.bar=new QProgressBar(this);
            r.bar->setFixedWidth(100);r.bar->setFixedHeight(12);r.bar->setTextVisible(false);r.bar->hide();
            line->addWidget(r.text,1);line->addWidget(r.elapsed);line->addWidget(r.bar);
            if(i==0){stop_=new QPushButton(tr("Stop operation"),this);stop_->setEnabled(false);line->addWidget(stop_);}
            layout->addLayout(line);
        }
        result_=new QLabel(this);result_->setWordWrap(true);result_->setTextFormat(Qt::PlainText);
        result_->setTextInteractionFlags(Qt::TextSelectableByMouse);layout->addWidget(result_);
        auto* details=new QPushButton(tr("Show activity log"),this);details->setCheckable(true);
        layout->addWidget(details,0,Qt::AlignLeft);
        log_=new QPlainTextEdit(this);log_->setReadOnly(true);log_->setMaximumBlockCount(200);
        log_->setMaximumHeight(150);log_->hide();layout->addWidget(log_);
        connect(details,&QPushButton::toggled,this,[this,details](bool shown){
            log_->setVisible(shown);details->setText(shown?tr("Hide activity log"):tr("Show activity log"));
        });
        auto* timer=new QTimer(this);timer->setInterval(250);
        connect(timer,&QTimer::timeout,this,[this]{for(auto& r:rows_)if(r.active)updateClock(r);});timer->start();
    }
    QPushButton* stopButton()const{return stop_;}
    QLabel* resultLabel()const{return result_;}
    void note(const QString& message){result_->setText(message);log_->appendPlainText(tr("Result: ")+message);}
    void begin(Channel channel,const QString& title,bool stoppable=false) {
        auto& r=rows_[channel];r.title=title;r.active=true;r.clock.start();r.stage=tr("Starting…");
        r.text->setText(title+": "+r.stage);r.bar->setRange(0,0);r.bar->show();updateClock(r);
        if(channel==Operation){stop_->setEnabled(stoppable);result_->clear();}
        append(r,title);
    }
    void update(Channel channel,const QString& stage) {
        auto& r=rows_[channel];if(!r.active||stage==r.stage)return;
        r.stage=stage;r.text->setText(r.title+": "+stage);append(r,stage);
    }
    void finish(Channel channel,const QString& outcome) {
        auto& r=rows_[channel];if(!r.active)return;
        updateClock(r);r.active=false;r.bar->hide();r.text->setText(r.title+": "+outcome);append(r,outcome);
        if(channel==Operation)stop_->setEnabled(false);
    }
private:
    struct Row {QLabel *text{},*elapsed{};QProgressBar* bar{};QElapsedTimer clock;QString title,stage;bool active=false;};
    static QString duration(qint64 ms){return QString("%1:%2").arg(ms/60000).arg(ms/1000%60,2,10,QChar('0'));}
    static void updateClock(Row& r){r.elapsed->setText(tr("Elapsed %1").arg(duration(r.clock.elapsed())));}
    void append(const Row& r,const QString& text){log_->appendPlainText(QString("[%1] %2 — %3").arg(duration(r.clock.elapsed()),r.title,text));}
    std::array<Row,3> rows_;
    QPushButton* stop_{};QLabel* result_{};QPlainTextEdit* log_{};
};
