#pragma once

#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
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
        if (nullable) {
            schema["type"] = QJsonArray{typeName(type), QStringLiteral("null")};
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
};

namespace AgentBridgeParams {

inline AgentBridgeParam optionalString(const QString& name)
{
    return {
        .name = name,
        .type = AgentBridgeParamType::String,
    };
}

inline AgentBridgeParam requiredString(const QString& name)
{
    AgentBridgeParam param = optionalString(name);
    param.required = true;
    return param;
}

inline AgentBridgeParam optionalStringEnum(const QString& name, QStringList values)
{
    AgentBridgeParam param = optionalString(name);
    param.values = std::move(values);
    return param;
}

inline AgentBridgeParam optionalNumber(const QString& name)
{
    return {
        .name = name,
        .type = AgentBridgeParamType::Number,
        .finite = true,
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

inline AgentBridgeParam optionalInteger(const QString& name)
{
    return {
        .name = name,
        .type = AgentBridgeParamType::Integer,
        .finite = true,
    };
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

struct AgentBridgeMethod
{
    QString name;
    std::vector<AgentBridgeParam> params;
    std::vector<int> errors;

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
        return result;
    }
};
