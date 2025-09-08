#pragma once

#include <QWidget>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QLabel>


class ConsoleOutputWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ConsoleOutputWidget(QWidget* parent = nullptr);
    ~ConsoleOutputWidget();
    void appendOutput(const QString& text);

    void clear();
    void setTitle(const QString& title);

public slots:
    void copyToClipboard();

private:
    QPlainTextEdit* _textEdit;
    QPushButton* _clearButton;
    QPushButton* _copyButton;
    QLabel* _titleLabel;
};


