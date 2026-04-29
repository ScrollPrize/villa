#pragma once

#include <QDialog>
#include <QString>

class JsonProfileEditor;
class QDialogButtonBox;

class JsonFileEditorDialog : public QDialog {
    Q_OBJECT

public:
    explicit JsonFileEditorDialog(const QString& filePath,
                                  const QString& title,
                                  QWidget* parent = nullptr);

private slots:
    void accept() override;

private:
    void updateOkButton();

    QString _filePath;
    JsonProfileEditor* _editor{nullptr};
    QDialogButtonBox* _buttons{nullptr};
};
