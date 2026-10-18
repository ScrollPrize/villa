#include "agent_bridge/AgentBridgeAccessible.hpp"

#include <QAccessible>
#include <QComboBox>
#include <QDockWidget>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QMetaEnum>
#include <QPlainTextEdit>
#include <QTextDocumentFragment>
#include <QTextEdit>
#include <QVariant>
#include <QWidget>

#include "agent_bridge/AgentBridgeUi.hpp"

namespace {

constexpr int kMaxTextChars = 500;
constexpr int kMaxItems = 200;

QString plainText(const QString& text)
{
    return Qt::mightBeRichText(text) ? QTextDocumentFragment::fromHtml(text).toPlainText()
                                     : text;
}

// Drops mnemonic markers and a trailing colon: "&Steps:" -> "Steps".
QString cleanLabel(const QString& text)
{
    QString label = plainText(text);
    label.replace(QStringLiteral("&&"), QStringLiteral("\x01"));
    label.remove(QLatin1Char('&'));
    label.replace(QStringLiteral("\x01"), QStringLiteral("&"));
    label = label.trimmed();
    if (label.endsWith(QLatin1Char(':')))
        label.chop(1);
    return label.trimmed();
}

QString roleName(QAccessible::Role role)
{
    const char* key = QMetaEnum::fromType<QAccessible::Role>().valueToKey(role);
    return key ? QString::fromLatin1(key) : QString::number(int(role));
}

bool isEntry(QWidget* w)
{
    if (qobject_cast<QLineEdit*>(w) || qobject_cast<QTextEdit*>(w) ||
        qobject_cast<QPlainTextEdit*>(w))
        return true;
    auto* combo = qobject_cast<QComboBox*>(w);
    return combo && combo->isEditable();
}

// The name a person reads for a widget: its accessible name; else the text
// of the label whose buddy it is (Qt names item views without their buddy);
// else the name of the plain container it sits in (a form row that holds the
// field and its buttons). It does not depend on where a tree walk started.
QString widgetName(QWidget* w, QAccessibleInterface* iface)
{
    QString name = iface ? iface->text(QAccessible::Name) : QString();
    QWidget* parent = w && !w->isWindow() ? w->parentWidget() : nullptr;
    if (name.isEmpty() && parent) {
        for (QObject* child : parent->children()) {
            auto* label = qobject_cast<QLabel*>(child);
            if (label && label->buddy() == w) {
                name = label->text();
                break;
            }
        }
    }
    if (name.isEmpty() && parent) {
        QAccessibleInterface* container = QAccessible::queryAccessibleInterface(parent);
        if (container && container->role() == QAccessible::Client)
            name = container->text(QAccessible::Name);
    }
    return cleanLabel(name);
}

bool credentialEntry(QWidget* w, QAccessibleInterface* iface, const QString& name)
{
    if (iface && iface->state().passwordEdit)
        return true;
    return agentBridgeCredentialLabel(name) ||
           agentBridgeCredentialLabel(iface ? iface->text(QAccessible::Description) : QString()) ||
           agentBridgeCredentialLabel(w->objectName());
}

bool isSkipped(const QWidget* w)
{
    const QString name = w->objectName();
    if (name == QLatin1String("qt_scrollarea_hcontainer") ||
        name == QLatin1String("qt_scrollarea_vcontainer") ||
        name == QLatin1String("qt_toolbar_ext_button") ||
        name == QLatin1String("ScrollLeftButton") ||
        name == QLatin1String("ScrollRightButton") ||
        name.startsWith(QLatin1String("qt_dockwidget_")))
        return true;
    const QByteArray cls = w->metaObject()->className();
    return cls == "QSplitterHandle" || cls == "QSizeGrip" || cls == "QRubberBand";
}

bool isLeafRole(QAccessible::Role role)
{
    switch (role) {
    case QAccessible::EditableText:
    case QAccessible::StaticText:
    case QAccessible::SpinBox:
    case QAccessible::Slider:
    case QAccessible::Dial:
    case QAccessible::ScrollBar:
    case QAccessible::ProgressBar:
    case QAccessible::ComboBox:
    case QAccessible::List:
    case QAccessible::Tree:
    case QAccessible::Table:
    case QAccessible::Button:
    case QAccessible::CheckBox:
    case QAccessible::RadioButton:
    case QAccessible::ButtonMenu:
    case QAccessible::ButtonDropDown:
    case QAccessible::Graphic:
    case QAccessible::Separator:
    case QAccessible::Indicator:
        return true;
    default:
        return false;
    }
}

QString pathSegment(QAccessibleInterface* iface, int index)
{
    if (QObject* object = iface->object()) {
        const QString name = object->objectName();
        if (!name.isEmpty() && !name.startsWith(QLatin1String("qt_")))
            return name;
        return QString::fromLatin1(object->metaObject()->className());
    }
    return QStringLiteral("%1[%2]").arg(roleName(iface->role())).arg(index);
}

class Walker
{
public:
    Walker(QWidget* root, const AgentBridgeInspectOptions& options)
        : _root(root), _options(options)
    {
    }

    int nodes = 0;
    bool truncated = false;

    QJsonObject node(QAccessibleInterface* iface, int depth, const QString& parentPath,
                     int index)
    {
        ++nodes;
        QJsonObject out;
        QObject* object = iface->object();
        auto* widget = qobject_cast<QWidget*>(object);
        const QAccessible::State state = iface->state();

        out["role"] = roleName(iface->role());
        if (object) {
            out["class"] = QString::fromLatin1(object->metaObject()->className());
            if (!object->objectName().isEmpty())
                out["objectName"] = object->objectName();
        }
        const QString segment = pathSegment(iface, index);
        const QString path = parentPath.isEmpty() ? segment : parentPath + QLatin1Char('/') + segment;
        out["path"] = path;

        const QString name = widget ? widgetName(widget, iface)
                                    : cleanLabel(iface->text(QAccessible::Name));
        const bool secretEntry = widget && isEntry(widget) && credentialEntry(widget, iface, name);
        bool redacted = secretEntry;
        auto report = [&](const char* key, const QString& text) {
            if (text.isEmpty())
                return;
            if (agentBridgeCredentialContent(text)) {
                redacted = true;
                return;
            }
            out[QLatin1String(key)] =
                text.size() > kMaxTextChars ? text.left(kMaxTextChars) : text;
            if (text.size() > kMaxTextChars)
                out["textTruncated"] = true;
        };
        report("name", name);
        report("description", plainText(iface->text(QAccessible::Description)));

        if (QAccessibleValueInterface* value = iface->valueInterface()) {
            out["value"] = QJsonValue::fromVariant(value->currentValue());
            out["minimum"] = QJsonValue::fromVariant(value->minimumValue());
            out["maximum"] = QJsonValue::fromVariant(value->maximumValue());
        } else if (!secretEntry) {
            report("value", plainText(iface->text(QAccessible::Value)));
        }
        if (iface->role() == QAccessible::ComboBox && !secretEntry) {
            for (int i = 0; i < iface->childCount(); ++i) {
                QAccessibleInterface* list = iface->child(i);
                if (list && list->role() == QAccessible::List) {
                    describeItems(list, out, redacted);
                    break;
                }
            }
        } else if (iface->role() == QAccessible::List || iface->role() == QAccessible::Tree ||
                   iface->role() == QAccessible::Table) {
            describeItems(iface, out, redacted);
        }

        out["enabled"] = !state.disabled;
        out["visible"] = widget ? widget->isVisible() : !state.invisible;
        if (state.checkable)
            out["checked"] = bool(state.checked);
        if (state.selected)
            out["selected"] = true;
        if (state.focused)
            out["focused"] = true;
        if (state.readOnly && widget && isEntry(widget))
            out["readOnly"] = true;
        if (redacted)
            out["redacted"] = true;

        const auto viewer = object ? _options.viewerIds.constFind(object)
                                   : _options.viewerIds.cend();
        if (viewer != _options.viewerIds.cend()) {
            out["viewerId"] = viewer.value();
            return out;
        }
        if (isLeafRole(iface->role()))
            return out;
        if (depth >= _options.maxDepth) {
            if (hasListedChild(iface))
                out["childrenOmitted"] = true;
            return out;
        }
        QJsonArray children;
        appendChildren(iface, depth, path, children);
        if (!children.isEmpty())
            out["children"] = children;
        return out;
    }

private:
    bool listed(QAccessibleInterface* child) const
    {
        if (auto* w = qobject_cast<QWidget*>(child->object())) {
            if (w->isWindow() && !qobject_cast<QDockWidget*>(w))
                return false;
            if (isSkipped(w))
                return false;
            return _options.includeHidden || w->isVisibleTo(_root);
        }
        return _options.includeHidden || !child->state().invisible;
    }

    bool hasListedChild(QAccessibleInterface* iface) const
    {
        for (int i = 0; i < iface->childCount(); ++i) {
            QAccessibleInterface* child = iface->child(i);
            if (child && listed(child))
                return true;
        }
        return false;
    }

    // Unnamed layout containers add nothing to read; their children are
    // listed in their place and do not count toward depth.
    bool anonymous(QAccessibleInterface* iface) const
    {
        const QAccessible::Role role = iface->role();
        if (role != QAccessible::Client && role != QAccessible::Pane &&
            role != QAccessible::LayeredPane && role != QAccessible::Splitter)
            return false;
        QObject* object = iface->object();
        if (!object || _options.viewerIds.contains(object))
            return false;
        const QString objectName = object->objectName();
        if (!objectName.isEmpty() && !objectName.startsWith(QLatin1String("qt_")))
            return false;
        return iface->text(QAccessible::Name).isEmpty();
    }

    void appendChildren(QAccessibleInterface* parent, int depth, const QString& path,
                        QJsonArray& out)
    {
        for (int i = 0; i < parent->childCount(); ++i) {
            QAccessibleInterface* child = parent->child(i);
            if (!child || !listed(child))
                continue;
            if (nodes >= _options.maxNodes) {
                truncated = true;
                return;
            }
            if (anonymous(child))
                appendChildren(child, depth, path, out);
            else
                out.append(node(child, depth + 1, path, i));
        }
    }

    // Rows of a list, tree or table: the first column's text of the first
    // kMaxItems rows, the row count, and the selected rows.
    void describeItems(QAccessibleInterface* view, QJsonObject& out, bool& redacted) const
    {
        QAccessibleTableInterface* table = view->tableInterface();
        if (!table)
            return;
        const int rows = table->rowCount();
        out["rows"] = rows;
        if (table->columnCount() > 1)
            out["columns"] = table->columnCount();
        QJsonArray items;
        for (int row = 0; row < rows && row < kMaxItems; ++row) {
            QAccessibleInterface* cell = table->cellAt(row, 0);
            const QString text = cell ? cleanLabel(cell->text(QAccessible::Name)) : QString();
            if (agentBridgeCredentialContent(text)) {
                redacted = true;
                items.append(QJsonValue(QJsonValue::Null));
            } else {
                items.append(text.left(kMaxTextChars));
            }
        }
        out["items"] = items;
        QJsonArray selected;
        for (int row : table->selectedRows()) {
            if (selected.size() >= kMaxItems)
                break;
            selected.append(row);
        }
        if (!selected.isEmpty())
            out["selectedRows"] = selected;
    }

    QWidget* _root;
    const AgentBridgeInspectOptions& _options;
};

}  // namespace


AgentBridgeInspection agentBridgeInspect(QWidget* root, const AgentBridgeInspectOptions& options)
{
    AgentBridgeInspection result;
    QAccessibleInterface* iface = root ? QAccessible::queryAccessibleInterface(root) : nullptr;
    if (!iface)
        return result;
    Walker walker(root, options);
    result.tree = walker.node(iface, 0, QString(), 0);
    result.nodeCount = walker.nodes;
    result.truncated = walker.truncated;
    return result;
}


bool agentBridgeWidgetShowsSecret(QWidget* widget)
{
    if (!widget)
        return false;
    QAccessibleInterface* iface = QAccessible::queryAccessibleInterface(widget);
    const QString name = widgetName(widget, iface);
    if (isEntry(widget) && credentialEntry(widget, iface, name))
        return true;
    if (!iface)
        return false;
    return agentBridgeCredentialContent(name) ||
           agentBridgeCredentialContent(iface->text(QAccessible::Value)) ||
           agentBridgeCredentialContent(iface->text(QAccessible::Description));
}
