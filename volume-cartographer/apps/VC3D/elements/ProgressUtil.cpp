#include "elements/ProgressUtil.hpp"



ProgressUtil::ProgressUtil(QStatusBar* statusBar, QObject* parent)
    : QObject(parent)
    , _statusBar(statusBar)
    , _animTimer(nullptr)
    , _animFrame(0)
{
}

ProgressUtil::~ProgressUtil()
{
    if (_animTimer) {
        if (_animTimer->isActive()) {
            _animTimer->stop();
        }
        delete _animTimer;
    }
}

void ProgressUtil::startAnimation(const QString& message)
{

    _animFrame = 0;
    _message = message;
    if (!_animTimer) {
        _animTimer = new QTimer(this);
        connect(_animTimer, &QTimer::timeout, this, &ProgressUtil::updateAnimation);
    }
    
    if (_statusBar) _statusBar->showMessage(message + " |", 0); // 0 timeout means it stays until changed
    _animTimer->start(300); // updates every 300 ms 
}

void ProgressUtil::stopAnimation(const QString& message, int timeout)
{
    if (_animTimer && _animTimer->isActive()) {
        _animTimer->stop();
    }
    
    if (_statusBar) _statusBar->showMessage(message, timeout);
}

void ProgressUtil::updateAnimation()
{
    static const QChar animChars[] = {'|', '/', '-', '\\'};
    _animFrame = (_animFrame + 1) % 4;
    if (_statusBar) _statusBar->showMessage(_message + " " + animChars[_animFrame], 0);
}

