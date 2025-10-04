#pragma once

#include <QObject>
#include <QStatusBar>
#include <QTimer>
#include <QString>



class ProgressUtil : public QObject
{
    Q_OBJECT

public:
    explicit ProgressUtil(QStatusBar* statusBar, QObject* parent = nullptr);
    
    ~ProgressUtil();

    /**
     * @brief Start a progress animation in the status bar
     * @param message The message to display alongside the animation
     */
    void startAnimation(const QString& message);
    
    /**
     * @brief Stop the progress animation and display a final message
     * @param message The final message to display
     * @param timeout How long to display the message (in ms, 0 for indefinite)
     */
    void stopAnimation(const QString& message, int timeout = 15000);

private slots:
    void updateAnimation();

private:
    QStatusBar* _statusBar;
    QTimer* _animTimer;
    int _animFrame;
    QString _message;
};


