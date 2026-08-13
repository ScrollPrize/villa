#pragma once

#include <QObject>
#include <QPointer>
#include <QStatusBar>
#include <QTimer>
#include <QString>
#include <QFont>


class QProgressBar;



class ProgressUtil : public QObject
{
    Q_OBJECT

public:
    enum class ProgressTextMode {
        Percent,
        Fraction,
        DoneRemaining,
        Custom,
        None
    };

    struct ProgressBarOptions {
        ProgressTextMode textMode = ProgressTextMode::Percent;
        QString prefix;
        QString suffix;
        QString customFormat; // Supports {value}, {total}, {remaining}, {percent}
        int percentPrecision = 0;
        int fontPointSize = -1;
    };

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

    /**
     * @brief Associate a progress bar widget that this util can manage.
     */
    void setProgressBar(QProgressBar* progressBar);

    /**
     * @brief Configure default options used for progress bar rendering.
     */
    void configureProgressBar(const ProgressBarOptions& options);

    /**
     * @brief Start tracking progress using the managed progress bar.
     * @param totalSteps Number of steps expected (<=0 enables busy indicator).
     * @param options Optional override of the default progress bar options.
     */
    void startProgress(int totalSteps, const ProgressBarOptions* options = nullptr);

    /**
     * @brief Update the current progress to a specific step count.
     */
    void updateProgress(int completedSteps);

    /**
     * @brief Increment progress by a delta (default 1).
     */
    void advanceProgress(int stepDelta = 1);

    /**
     * @brief Stop progress tracking and optionally reset progress bar state.
     */
    void stopProgress(bool resetValue = true);

    bool isProgressActive() const { return _progressActive; }
    int progressTotal() const { return _progressTotal; }
    int progressValue() const { return _progressValue; }

private slots:
    void updateAnimation();

private:
    void applyProgressBarFont();
    void restoreProgressBarFont();
    void restoreProgressBarFormat();
    void updateProgressFormat();

    QStatusBar* _statusBar;
    QTimer* _animTimer;
    int _animFrame;
    QString _message;

    QPointer<QProgressBar> _progressBar;
    ProgressBarOptions _defaultBarOptions;
    ProgressBarOptions _activeBarOptions;
    int _progressTotal;
    int _progressValue;
    bool _progressActive;
    bool _hasStoredBarFont;
    bool _hasStoredBarFormat;
    int _storedBarMinimum;
    int _storedBarMaximum;
    QString _storedBarFormat;
    QFont _storedBarFont;
};
