#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QMap>
#include <QSet>
#include <QString>
#include <QStringList>

#include "agent_bridge/AgentBridgeError.hpp"

enum class AgentBridgeParamType {
    String,
    Number,
    Integer,
    Boolean,
    Object,
    Array,
};

struct AgentBridgeParam
{
    QString name;
    AgentBridgeParamType type;
    std::vector<AgentBridgeParamType> alternateTypes;
    bool required{false};
    bool nullable{false};
    bool finite{false};
    std::optional<double> minimum;
    bool exclusiveMinimum{false};
    std::optional<double> maximum;
    bool exclusiveMaximum{false};
    QJsonValue defaultValue{QJsonValue::Undefined};
    QStringList values;
    std::vector<AgentBridgeParam> properties;
    std::shared_ptr<AgentBridgeParam> items;

    void validate(const QJsonValue& value, const QString& path = {}) const
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

    QJsonObject describe() const
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
        if (minimum) {
            schema[exclusiveMinimum ? "exclusiveMinimum" : "minimum"] = *minimum;
        }
        if (maximum) {
            schema[exclusiveMaximum ? "exclusiveMaximum" : "maximum"] = *maximum;
        }
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

    QString definitionError(const QString& path = {}) const
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
            return items->definitionError(
                QStringLiteral("%1[]").arg(param));
        }
        if (items)
            return QStringLiteral("%1 has items but is not an array").arg(param);
        return {};
    }

private:
    [[noreturn]] static void fail(const QString& param, const QString& detail)
    {
        QJsonObject data;
        data["param"] = param;
        throw AgentBridgeError{
            -32602,
            QStringLiteral("%1 %2").arg(param, detail),
            data,
        };
    }

    void validateNumber(
        const QJsonValue& value,
        const QString& param,
        bool integer) const
    {
        if (!value.isDouble())
            fail(param, integer ? QStringLiteral("must be an integer")
                                : QStringLiteral("must be a number"));
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

    static QString typeName(AgentBridgeParamType type)
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

    static bool matchesType(
        const QJsonValue& value,
        AgentBridgeParamType type)
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
};

namespace AgentBridgeParams {

inline AgentBridgeParam withDefault(
    AgentBridgeParam param,
    const QJsonValue& defaultValue)
{
    param.defaultValue = defaultValue;
    return param;
}

inline AgentBridgeParam nullable(AgentBridgeParam param)
{
    param.nullable = true;
    return param;
}

inline AgentBridgeParam required(AgentBridgeParam param)
{
    param.required = true;
    return param;
}

inline AgentBridgeParam optionalString(
    const QString& name,
    const QJsonValue& defaultValue = QJsonValue(QJsonValue::Undefined))
{
    return {
        .name = name,
        .type = AgentBridgeParamType::String,
        .defaultValue = defaultValue,
    };
}

inline AgentBridgeParam requiredString(const QString& name)
{
    AgentBridgeParam param = optionalString(name);
    param.required = true;
    return param;
}

inline AgentBridgeParam optionalStringEnum(
    const QString& name,
    QStringList values,
    const QJsonValue& defaultValue = QJsonValue(QJsonValue::Undefined))
{
    AgentBridgeParam param = optionalString(name, defaultValue);
    param.values = std::move(values);
    return param;
}

inline AgentBridgeParam requiredStringEnum(const QString& name, QStringList values)
{
    return required(optionalStringEnum(name, std::move(values)));
}

inline AgentBridgeParam optionalNumber(
    const QString& name,
    const QJsonValue& defaultValue = QJsonValue(QJsonValue::Undefined))
{
    return {
        .name = name,
        .type = AgentBridgeParamType::Number,
        .finite = true,
        .defaultValue = defaultValue,
    };
}

inline AgentBridgeParam requiredNumber(const QString& name)
{
    AgentBridgeParam param = optionalNumber(name);
    param.required = true;
    return param;
}

inline AgentBridgeParam requiredPositiveNumber(const QString& name)
{
    AgentBridgeParam param = requiredNumber(name);
    param.minimum = 0.0;
    param.exclusiveMinimum = true;
    return param;
}

inline AgentBridgeParam optionalInteger(
    const QString& name,
    const QJsonValue& defaultValue = QJsonValue(QJsonValue::Undefined))
{
    return {
        .name = name,
        .type = AgentBridgeParamType::Integer,
        .finite = true,
        .minimum = std::numeric_limits<int>::lowest(),
        .maximum = std::numeric_limits<int>::max(),
        .defaultValue = defaultValue,
    };
}

inline AgentBridgeParam requiredInteger(const QString& name)
{
    return required(optionalInteger(name));
}

inline AgentBridgeParam optionalSafeId(const QString& name)
{
    AgentBridgeParam param{
        .name = name,
        .type = AgentBridgeParamType::Integer,
        .finite = true,
        .minimum = 1.0,
        .maximum = 9007199254740991.0,
    };
    return param;
}

inline AgentBridgeParam requiredSafeId(const QString& name)
{
    return required(optionalSafeId(name));
}

inline AgentBridgeParam optionalBoolean(
    const QString& name,
    const QJsonValue& defaultValue = QJsonValue(QJsonValue::Undefined))
{
    return {
        .name = name,
        .type = AgentBridgeParamType::Boolean,
        .defaultValue = defaultValue,
    };
}

inline AgentBridgeParam requiredBoolean(const QString& name)
{
    AgentBridgeParam param = optionalBoolean(name);
    param.required = true;
    return param;
}

inline AgentBridgeParam optionalObject(
    const QString& name,
    std::vector<AgentBridgeParam> properties)
{
    return {
        .name = name,
        .type = AgentBridgeParamType::Object,
        .properties = std::move(properties),
    };
}

inline AgentBridgeParam requiredObject(
    const QString& name,
    std::vector<AgentBridgeParam> properties)
{
    return required(optionalObject(name, std::move(properties)));
}

inline AgentBridgeParam optionalArray(
    const QString& name,
    AgentBridgeParam itemSchema)
{
    return {
        .name = name,
        .type = AgentBridgeParamType::Array,
        .items = std::make_shared<AgentBridgeParam>(std::move(itemSchema)),
    };
}

inline AgentBridgeParam optionalArray(
    const QString& name,
    AgentBridgeParamType itemType)
{
    return optionalArray(name, AgentBridgeParam{.type = itemType});
}

inline AgentBridgeParam requiredArray(
    const QString& name,
    AgentBridgeParam itemSchema)
{
    AgentBridgeParam param = optionalArray(name, std::move(itemSchema));
    param.required = true;
    return param;
}

inline AgentBridgeParam requiredArray(
    const QString& name,
    AgentBridgeParamType itemType)
{
    AgentBridgeParam param = optionalArray(name, itemType);
    param.required = true;
    return param;
}

}  // namespace AgentBridgeParams

struct AgentBridgeMcp
{
    QString tool;
    bool snakeCaseParams{false};
    QMap<QString, QString> paramRenames;
    QStringList extraParams;

    QString mappedParamName(const QString& wireName) const
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

    QJsonObject describe(const std::vector<AgentBridgeParam>& params) const
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
};

namespace AgentBridgeMcpTools {

inline AgentBridgeMcp exact(QString tool, QStringList extraParams = {})
{
    return {
        .tool = std::move(tool),
        .extraParams = std::move(extraParams),
    };
}

inline AgentBridgeMcp snakeCase(QString tool, QStringList extraParams = {})
{
    return {
        .tool = std::move(tool),
        .snakeCaseParams = true,
        .extraParams = std::move(extraParams),
    };
}

}  // namespace AgentBridgeMcpTools

struct AgentBridgeMethod
{
    QString name;
    std::vector<AgentBridgeParam> params;
    std::vector<int> errors;
    AgentBridgeMcp mcp;

    void validate(const QJsonValue& value) const
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

    QString definitionError() const
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
            if (errorCodes.contains(error))
                return QStringLiteral("%1 has duplicate error code %2").arg(name).arg(error);
            errorCodes.insert(error);
        }

        if (mcp.tool.isEmpty()) {
            if (mcp.snakeCaseParams ||
                !mcp.paramRenames.isEmpty() ||
                !mcp.extraParams.isEmpty()) {
                return QStringLiteral("%1 has MCP parameters without an MCP tool").arg(name);
            }
            return {};
        }

        QSet<QString> toolParams;
        for (const AgentBridgeParam& param : params) {
            const QString toolParam = mcp.mappedParamName(param.name);
            if (toolParam.isEmpty())
                return QStringLiteral("%1 maps %2 to an empty MCP parameter").arg(name, param.name);
            if (toolParams.contains(toolParam))
                return QStringLiteral("%1 has duplicate MCP parameter %2").arg(name, toolParam);
            toolParams.insert(toolParam);
        }
        for (auto it = mcp.paramRenames.cbegin(); it != mcp.paramRenames.cend(); ++it) {
            if (!names.contains(it.key()))
                return QStringLiteral("%1 renames unknown parameter %2").arg(name, it.key());
        }
        for (const QString& extra : mcp.extraParams) {
            if (extra.isEmpty())
                return QStringLiteral("%1 has an empty extra MCP parameter").arg(name);
            if (toolParams.contains(extra))
                return QStringLiteral("%1 has duplicate MCP parameter %2").arg(name, extra);
            toolParams.insert(extra);
        }
        return {};
    }

    QJsonObject describe() const
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
};
