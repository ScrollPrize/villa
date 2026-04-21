#pragma once

#include <QWidget>
#include <memory>

class QLineEdit;
class QTreeWidget;
class QTreeWidgetItem;
class QCheckBox;
class QPoint;

namespace vc { class Project; }

// Dedicated "Project" panel that shows every DataSource in the active
// Project with type / location / tags / state columns. Lets the user
// reload, remove, rename, enable/disable, and edit tags per source via a
// right-click context menu, without needing to walk through the Data
// menu each time.
class ProjectDockWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ProjectDockWidget(QWidget* parent = nullptr);
    ~ProjectDockWidget() override;

public slots:
    // Rebuild the tree from the given project (null → clear).
    void setProject(std::shared_ptr<vc::Project> project);

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

    std::shared_ptr<vc::Project> _project;

    QLineEdit* _filterEdit{nullptr};
    QCheckBox* _groupedCheck{nullptr};
    QTreeWidget* _tree{nullptr};
    bool _grouped{true};
};
