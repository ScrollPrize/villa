#include "ProjectDockWidget.hpp"

#include <QAction>
#include <QCheckBox>
#include <QDesktopServices>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLineEdit>
#include <QMenu>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QUrl>
#include <QVBoxLayout>

#include "vc/core/types/Project.hpp"

namespace {
constexpr int kColName = 0;
constexpr int kColType = 1;
constexpr int kColWhere = 2;
constexpr int kColTags = 3;
constexpr int kColState = 4;

QString sourceTypeLabel(vc::DataSourceType t)
{
    return QString::fromStdString(vc::data_source_type_to_string(t));
}

QString locationDisplay(const vc::DataSource& ds)
{
    QString prefix;
    if (ds.location_kind == vc::LocationKind::Remote) {
        prefix = QObject::tr("[remote] ");
    }
    return prefix + QString::fromStdString(ds.location);
}

QString tagsDisplay(const vc::DataSource& ds)
{
    QStringList parts;
    for (const auto& t : ds.tags) parts.push_back(QString::fromStdString(t));
    return parts.join(QStringLiteral(", "));
}

QString stateDisplay(const vc::DataSource& ds)
{
    QStringList parts;
    if (!ds.enabled) parts.push_back(QObject::tr("disabled"));
    if (ds.imported) parts.push_back(QObject::tr("linked"));
    return parts.join(QStringLiteral(", "));
}
} // namespace

ProjectDockWidget::ProjectDockWidget(QWidget* parent)
    : QWidget(parent)
{
    auto* vbox = new QVBoxLayout(this);
    vbox->setContentsMargins(4, 4, 4, 4);

    auto* topRow = new QHBoxLayout();
    _filterEdit = new QLineEdit(this);
    _filterEdit->setPlaceholderText(tr("Filter by name, type, tag, location..."));
    connect(_filterEdit, &QLineEdit::textChanged,
            this, &ProjectDockWidget::applyFilter);
    topRow->addWidget(_filterEdit, 1);

    _groupedCheck = new QCheckBox(tr("Grouped"), this);
    _groupedCheck->setChecked(true);
    connect(_groupedCheck, &QCheckBox::toggled,
            this, &ProjectDockWidget::toggleGroupedView);
    topRow->addWidget(_groupedCheck);

    vbox->addLayout(topRow);

    _tree = new QTreeWidget(this);
    _tree->setColumnCount(5);
    _tree->setHeaderLabels({tr("Source"), tr("Type"), tr("Location"),
                            tr("Tags"), tr("State")});
    _tree->setContextMenuPolicy(Qt::CustomContextMenu);
    _tree->setRootIsDecorated(true);
    _tree->setUniformRowHeights(true);
    _tree->header()->setSectionResizeMode(kColName, QHeaderView::Interactive);
    _tree->header()->setSectionResizeMode(kColWhere, QHeaderView::Stretch);
    connect(_tree, &QTreeWidget::customContextMenuRequested,
            this, &ProjectDockWidget::handleContextMenu);
    vbox->addWidget(_tree, 1);
}

ProjectDockWidget::~ProjectDockWidget() = default;

void ProjectDockWidget::setProject(std::shared_ptr<vc::Project> project)
{
    _project = std::move(project);
    rebuildTree();
}

void ProjectDockWidget::toggleGroupedView(bool grouped)
{
    _grouped = grouped;
    rebuildTree();
}

void ProjectDockWidget::rebuildTree()
{
    if (!_tree) return;
    _tree->clear();
    if (!_project) return;

    auto makeItem = [&](const vc::DataSource& ds) {
        auto* item = new QTreeWidgetItem();
        item->setText(kColName, QString::fromStdString(ds.id));
        item->setData(kColName, Qt::UserRole, QString::fromStdString(ds.id));
        item->setText(kColType, sourceTypeLabel(ds.type));
        item->setText(kColWhere, locationDisplay(ds));
        item->setText(kColTags, tagsDisplay(ds));
        item->setText(kColState, stateDisplay(ds));
        if (!ds.enabled) {
            for (int c = 0; c < _tree->columnCount(); ++c) {
                item->setForeground(c, QBrush(Qt::gray));
            }
        }
        return item;
    };

    if (_grouped && !_project->groups.empty()) {
        // Show each group as a top-level section. Ungrouped sources go
        // under a final "Other" group.
        std::vector<bool> used(_project->data_sources.size(), false);

        for (const auto& g : _project->groups) {
            auto* groupItem = new QTreeWidgetItem(_tree);
            groupItem->setText(kColName, QString::fromStdString(
                g.name.empty() ? g.id : g.name));
            groupItem->setFirstColumnSpanned(true);

            for (const auto& sid : g.source_ids) {
                for (std::size_t i = 0; i < _project->data_sources.size(); ++i) {
                    if (_project->data_sources[i].id == sid) {
                        groupItem->addChild(makeItem(_project->data_sources[i]));
                        used[i] = true;
                        break;
                    }
                }
            }
            groupItem->setExpanded(true);
        }

        auto* other = new QTreeWidgetItem(_tree);
        other->setText(kColName, tr("(ungrouped)"));
        other->setFirstColumnSpanned(true);
        for (std::size_t i = 0; i < _project->data_sources.size(); ++i) {
            if (!used[i]) other->addChild(makeItem(_project->data_sources[i]));
        }
        if (other->childCount() == 0) {
            delete _tree->takeTopLevelItem(_tree->indexOfTopLevelItem(other));
        } else {
            other->setExpanded(true);
        }
    } else {
        for (const auto& ds : _project->data_sources) {
            _tree->addTopLevelItem(makeItem(ds));
        }
    }

    // Linked projects at the bottom (read-only informational).
    if (!_project->linked_projects.empty()) {
        auto* linksHeader = new QTreeWidgetItem(_tree);
        linksHeader->setText(kColName, tr("Linked projects"));
        linksHeader->setFirstColumnSpanned(true);
        for (const auto& lp : _project->linked_projects) {
            auto* row = new QTreeWidgetItem(linksHeader);
            row->setText(kColName, QString::fromStdString(lp.path));
            row->setText(kColType, tr("link"));
            row->setText(kColState,
                         lp.read_only ? tr("read-only") : tr("writable"));
        }
        linksHeader->setExpanded(true);
    }

    _tree->resizeColumnToContents(kColType);
    _tree->resizeColumnToContents(kColTags);
    _tree->resizeColumnToContents(kColState);
    applyFilter();
}

void ProjectDockWidget::applyFilter()
{
    if (!_tree) return;
    const QString filter = _filterEdit ? _filterEdit->text().trimmed() : QString();
    auto matches = [&](QTreeWidgetItem* item) {
        if (filter.isEmpty()) return true;
        for (int c = 0; c < _tree->columnCount(); ++c) {
            if (item->text(c).contains(filter, Qt::CaseInsensitive)) return true;
        }
        return false;
    };

    for (int i = 0; i < _tree->topLevelItemCount(); ++i) {
        auto* top = _tree->topLevelItem(i);
        bool anyVisible = false;
        if (top->childCount() > 0) {
            for (int j = 0; j < top->childCount(); ++j) {
                auto* child = top->child(j);
                const bool vis = matches(child);
                child->setHidden(!vis);
                anyVisible = anyVisible || vis;
            }
            top->setHidden(!anyVisible);
        } else {
            top->setHidden(!matches(top));
        }
    }
}

QString ProjectDockWidget::selectedSourceId() const
{
    if (!_tree) return {};
    auto items = _tree->selectedItems();
    if (items.isEmpty()) return {};
    return items.first()->data(kColName, Qt::UserRole).toString();
}

void ProjectDockWidget::handleContextMenu(const QPoint& pos)
{
    if (!_tree) return;
    auto* item = _tree->itemAt(pos);
    if (!item) return;
    const QString id = item->data(kColName, Qt::UserRole).toString();
    if (id.isEmpty()) return;

    bool imported = false;
    bool enabled = true;
    if (_project) {
        if (const auto* ds = _project->find_source(id.toStdString())) {
            imported = ds->imported;
            enabled = ds->enabled;
        }
    }

    QMenu menu(_tree);
    QAction* reloadAct = menu.addAction(tr("Reload"));
    QAction* toggleAct = menu.addAction(
        enabled ? tr("Disable") : tr("Enable"));
    QAction* tagsAct = menu.addAction(tr("Edit tags..."));
    menu.addSeparator();
    QAction* openAct = menu.addAction(tr("Reveal location"));
    menu.addSeparator();
    QAction* renameAct = menu.addAction(tr("Rename..."));
    QAction* removeAct = menu.addAction(tr("Remove"));
    if (imported) {
        renameAct->setEnabled(false);
        removeAct->setEnabled(false);
    }

    auto* chosen = menu.exec(_tree->viewport()->mapToGlobal(pos));
    if (!chosen) return;
    if (chosen == reloadAct)  emit reloadSourceRequested(id);
    else if (chosen == removeAct)  emit removeSourceRequested(id);
    else if (chosen == renameAct)  emit renameSourceRequested(id);
    else if (chosen == toggleAct)  emit toggleEnabledRequested(id, !enabled);
    else if (chosen == tagsAct)    emit editTagsRequested(id);
    else if (chosen == openAct)    emit openSourceLocationRequested(id);
}
