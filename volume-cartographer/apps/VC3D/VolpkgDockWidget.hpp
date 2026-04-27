#pragma once

#include <QWidget>
#include <memory>

class QLineEdit;
class QTreeWidget;
class QTreeWidgetItem;
class QCheckBox;
class QPoint;

namespace vc { class Volpkg; }

// Dedicated "Project" panel that shows every DataSource in the active
// Project with type / location / tags / state columns. Lets the user
// reload, remove, rename, enable/disable, and edit tags per source via a
// right-click context menu, without needing to walk through the Data
// menu each time.
class VolpkgDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit VolpkgDockWidget(QWidget* parent = nullptr);
    ~VolpkgDockWidget() override;

public slots:
    // Rebuild the tree from the given project (null → clear).
    void setProject(std::shared_ptr<vc::Volpkg> project);

signals:
    void reloadSourceRequested(const QString& sourceId);
    void removeSourceRequested(const QString& sourceId);
    void renameSourceRequested(const QString& sourceId);
    void toggleEnabledRequested(const QString& sourceId, bool enabled);
    void editTagsRequested(const QString& sourceId);
    void openSourceLocationRequested(const QString& sourceId);

private slots:
    void applyFilter();
    void handleContextMenu(const QPoint& pos);
    void toggleGroupedView(bool grouped);

private:
    void rebuildTree();
    QString selectedSourceId() const;

    std::shared_ptr<vc::Volpkg> _project;

    QLineEdit* _filterEdit{nullptr};
    QCheckBox* _groupedCheck{nullptr};
    QTreeWidget* _tree{nullptr};
    bool _grouped{true};
};
