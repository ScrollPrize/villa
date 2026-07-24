#include "agent_bridge/AgentBridgeMethod.hpp"

#include <cmath>

#include <QJsonArray>
#include <QSet>

#include "agent_bridge/AgentBridgeError.hpp"

void AgentBridgeParam::validate(
    const QJsonValue& value, const QString& path) const
{
    const QString param = path.isEmpty() ? name : path;
    if (value.isUndefined()) {
        if (required)
            fail(param, QStringLiteral("is required"));
        return;
    }
    if (value.isNull()) {
        if (!nullable)
            fail(param, QStringLiteral("must not be null"));
        return;
    }

    if (!matchesType(value, type)) {
        for (AgentBridgeParamType alternateType : alternateTypes) {
            if (!matchesType(value, alternateType))
                continue;
            AgentBridgeParam alternate = *this;
            alternate.type = alternateType;
            alternate.alternateTypes.clear();
            alternate.validate(value, path);
            return;
        }
    }

    switch (type) {
    case AgentBridgeParamType::String:
        if (!value.isString())
            fail(param, QStringLiteral("must be a string"));
        if (!values.isEmpty() && !values.contains(value.toString()))
            fail(param, QStringLiteral("has an invalid value"));
        break;
    case AgentBridgeParamType::Number:
        validateNumber(value, param, false);
        break;
    case AgentBridgeParamType::Integer:
        validateNumber(value, param, true);
        break;
    case AgentBridgeParamType::Boolean:
        if (!value.isBool())
            fail(param, QStringLiteral("must be a boolean"));
        break;
    case AgentBridgeParamType::Object: {
        if (!value.isObject())
            fail(param, QStringLiteral("must be an object"));
        const QJsonObject object = value.toObject();
        for (const AgentBridgeParam& property : properties) {
            if (property.required && !object.contains(property.name)) {
                fail(
                    param,
                    QStringLiteral("requires %1").arg(property.name));
            }
            property.validate(
                object.value(property.name),
                QStringLiteral("%1.%2").arg(param, property.name));
        }
        break;
    }
    case AgentBridgeParamType::Array:
        if (!value.isArray())
            fail(param, QStringLiteral("must be an array"));
        if (items) {
            const QJsonArray array = value.toArray();
            for (const QJsonValue& item : array)
                items->validate(item, param);
        }
        break;
    }
}

QJsonObject AgentBridgeParam::describe() const
{
    QJsonObject schema;
    if (nullable || !alternateTypes.empty()) {
        QJsonArray types{typeName(type)};
        for (AgentBridgeParamType alternateType : alternateTypes)
            types.append(typeName(alternateType));
        if (nullable)
            types.append(QStringLiteral("null"));
        schema["type"] = types;
    } else {
        schema["type"] = typeName(type);
    }
    if (!defaultValue.isUndefined())
        schema["default"] = defaultValue;
    if (!values.isEmpty())
        schema["enum"] = QJsonArray::fromStringList(values);
    if (minimum)
        schema[exclusiveMinimum ? "exclusiveMinimum" : "minimum"] = *minimum;
    if (maximum)
        schema[exclusiveMaximum ? "exclusiveMaximum" : "maximum"] = *maximum;
    if (type == AgentBridgeParamType::Object) {
        QJsonObject childSchemas;
        QJsonArray requiredProperties;
        for (const AgentBridgeParam& property : properties) {
            childSchemas[property.name] = property.describe();
            if (property.required)
                requiredProperties.append(property.name);
        }
        schema["properties"] = childSchemas;
        if (!requiredProperties.isEmpty())
            schema["required"] = requiredProperties;
    } else if (type == AgentBridgeParamType::Array && items) {
        schema["items"] = items->describe();
    }
    return schema;
}

QString AgentBridgeParam::definitionError(const QString& path) const
{
    const QString param = path.isEmpty() ? name : path;
    if (!values.isEmpty() && type != AgentBridgeParamType::String)
        return QStringLiteral("%1 has enum values but is not a string").arg(param);
    QSet<AgentBridgeParamType> types{type};
    for (AgentBridgeParamType alternateType : alternateTypes) {
        if (types.contains(alternateType))
            return QStringLiteral("%1 has duplicate parameter types").arg(param);
        types.insert(alternateType);
    }
    if (QSet<QString>(values.begin(), values.end()).size() != values.size())
        return QStringLiteral("%1 has duplicate enum values").arg(param);
    if (minimum && maximum && *minimum > *maximum)
        return QStringLiteral("%1 has an inverted numeric range").arg(param);
    if (exclusiveMinimum && !minimum)
        return QStringLiteral("%1 has an exclusive minimum without a bound").arg(param);
    if (exclusiveMaximum && !maximum)
        return QStringLiteral("%1 has an exclusive maximum without a bound").arg(param);
    if ((minimum || maximum) &&
        type != AgentBridgeParamType::Number &&
        type != AgentBridgeParamType::Integer) {
        return QStringLiteral("%1 has bounds but is not numeric").arg(param);
    }

    if (type == AgentBridgeParamType::Object) {
        QSet<QString> names;
        for (const AgentBridgeParam& property : properties) {
            if (property.name.isEmpty())
                return QStringLiteral("%1 has an unnamed property").arg(param);
            if (names.contains(property.name)) {
                return QStringLiteral("%1 has duplicate property %2")
                    .arg(param, property.name);
            }
            names.insert(property.name);
            const QString error = property.definitionError(
                QStringLiteral("%1.%2").arg(param, property.name));
            if (!error.isEmpty())
                return error;
        }
    } else if (!properties.empty()) {
        return QStringLiteral("%1 has properties but is not an object").arg(param);
    }

    if (type == AgentBridgeParamType::Array) {
        if (!items)
            return QStringLiteral("%1 has no item schema").arg(param);
        return items->definitionError(QStringLiteral("%1[]").arg(param));
    }
    if (items)
        return QStringLiteral("%1 has items but is not an array").arg(param);
    return {};
}

void AgentBridgeParam::fail(const QString& param, const QString& detail)
{
    QJsonObject data;
    data["param"] = param;
    throw AgentBridgeError{
        -32602,
        QStringLiteral("%1 %2").arg(param, detail),
        data,
    };
}

void AgentBridgeParam::validateNumber(
    const QJsonValue& value, const QString& param, bool integer) const
{
    if (!value.isDouble()) {
        fail(param, integer ? QStringLiteral("must be an integer")
                            : QStringLiteral("must be a number"));
    }
    const double number = value.toDouble();
    if ((finite && !std::isfinite(number)) ||
        (integer && (!std::isfinite(number) || std::floor(number) != number))) {
        fail(param, integer ? QStringLiteral("must be an integer")
                            : QStringLiteral("must be a finite number"));
    }
    if (minimum &&
        (exclusiveMinimum ? number <= *minimum : number < *minimum)) {
        fail(param, QStringLiteral("is below the allowed minimum"));
    }
    if (maximum &&
        (exclusiveMaximum ? number >= *maximum : number > *maximum)) {
        fail(param, QStringLiteral("is above the allowed maximum"));
    }
}

QString AgentBridgeParam::typeName(AgentBridgeParamType type)
{
    switch (type) {
    case AgentBridgeParamType::String:  return QStringLiteral("string");
    case AgentBridgeParamType::Number:  return QStringLiteral("number");
    case AgentBridgeParamType::Integer: return QStringLiteral("integer");
    case AgentBridgeParamType::Boolean: return QStringLiteral("boolean");
    case AgentBridgeParamType::Object:  return QStringLiteral("object");
    case AgentBridgeParamType::Array:   return QStringLiteral("array");
    }
    return {};
}

bool AgentBridgeParam::matchesType(
    const QJsonValue& value, AgentBridgeParamType type)
{
    switch (type) {
    case AgentBridgeParamType::String:  return value.isString();
    case AgentBridgeParamType::Number:
    case AgentBridgeParamType::Integer: return value.isDouble();
    case AgentBridgeParamType::Boolean: return value.isBool();
    case AgentBridgeParamType::Object:  return value.isObject();
    case AgentBridgeParamType::Array:   return value.isArray();
    }
    return false;
}

QString AgentBridgeMcp::mappedParamName(const QString& wireName) const
{
    const auto rename = paramRenames.constFind(wireName);
    if (rename != paramRenames.cend())
        return *rename;
    if (!snakeCaseParams)
        return wireName;

    QString result;
    for (const QChar character : wireName) {
        if (character.isUpper()) {
            if (!result.isEmpty())
                result.append(QLatin1Char('_'));
            result.append(character.toLower());
        } else {
            result.append(character);
        }
    }
    return result;
}

QJsonObject AgentBridgeMcp::describe(
    const std::vector<AgentBridgeParam>& params) const
{
    QJsonObject result;
    result["tool"] = tool;
    QJsonObject renames;
    for (const AgentBridgeParam& param : params) {
        const QString mappedName = mappedParamName(param.name);
        if (mappedName != param.name)
            renames[param.name] = mappedName;
    }
    if (!renames.isEmpty())
        result["paramRenames"] = renames;
    if (!extraParams.isEmpty())
        result["extraParams"] = QJsonArray::fromStringList(extraParams);
    return result;
}

void AgentBridgeMethod::validate(const QJsonValue& value) const
{
    if (!value.isUndefined() && !value.isNull() && !value.isObject()) {
        QJsonObject data;
        data["param"] = QStringLiteral("params");
        throw AgentBridgeError{
            -32602,
            QStringLiteral("params must be an object"),
            data,
        };
    }

    const QJsonObject object = value.toObject();
    for (const AgentBridgeParam& param : params)
        param.validate(object.value(param.name));
}

QString AgentBridgeMethod::definitionError() const
{
    if (name.isEmpty())
        return QStringLiteral("method name is empty");

    QSet<QString> names;
    for (const AgentBridgeParam& param : params) {
        if (param.name.isEmpty())
            return QStringLiteral("%1 has an unnamed parameter").arg(name);
        if (names.contains(param.name)) {
            return QStringLiteral("%1 has duplicate parameter %2")
                .arg(name, param.name);
        }
        names.insert(param.name);
        const QString error = param.definitionError(param.name);
        if (!error.isEmpty())
            return error;
    }

    QSet<int> errorCodes;
    for (int error : errors) {
        if (error >= 0)
            return QStringLiteral("%1 has a non-negative error code").arg(name);
        if (errorCodes.contains(error)) {
            return QStringLiteral("%1 has duplicate error code %2")
                .arg(name)
                .arg(error);
        }
        errorCodes.insert(error);
    }

    if (mcp.tool.isEmpty()) {
        if (mcp.snakeCaseParams ||
            !mcp.paramRenames.isEmpty() ||
            !mcp.extraParams.isEmpty()) {
            return QStringLiteral("%1 has MCP parameters without an MCP tool")
                .arg(name);
        }
        return {};
    }

    QSet<QString> toolParams;
    for (const AgentBridgeParam& param : params) {
        const QString toolParam = mcp.mappedParamName(param.name);
        if (toolParam.isEmpty())
            return QStringLiteral("%1 maps %2 to an empty MCP parameter")
                .arg(name, param.name);
        if (toolParams.contains(toolParam)) {
            return QStringLiteral("%1 has duplicate MCP parameter %2")
                .arg(name, toolParam);
        }
        toolParams.insert(toolParam);
    }
    for (auto it = mcp.paramRenames.cbegin();
         it != mcp.paramRenames.cend();
         ++it) {
        if (!names.contains(it.key())) {
            return QStringLiteral("%1 renames unknown parameter %2")
                .arg(name, it.key());
        }
    }
    for (const QString& extra : mcp.extraParams) {
        if (extra.isEmpty())
            return QStringLiteral("%1 has an empty extra MCP parameter").arg(name);
        if (toolParams.contains(extra)) {
            return QStringLiteral("%1 has duplicate MCP parameter %2")
                .arg(name, extra);
        }
        toolParams.insert(extra);
    }
    return {};
}

QJsonObject AgentBridgeMethod::describe() const
{
    QJsonObject properties;
    QJsonArray required;
    for (const AgentBridgeParam& param : params) {
        properties[param.name] = param.describe();
        if (param.required)
            required.append(param.name);
    }

    QJsonObject paramsSchema;
    paramsSchema["type"] = QStringLiteral("object");
    paramsSchema["properties"] = properties;
    if (!required.isEmpty())
        paramsSchema["required"] = required;

    QJsonArray errorCodes;
    for (int error : errors)
        errorCodes.append(error);

    QJsonObject result;
    result["params"] = paramsSchema;
    result["errors"] = errorCodes;
    if (!mcp.tool.isEmpty())
        result["mcp"] = mcp.describe(params);
    return result;
}
