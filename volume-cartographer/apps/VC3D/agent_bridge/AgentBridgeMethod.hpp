#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <QJsonObject>
#include <QJsonValue>
#include <QMap>
#include <QString>
#include <QStringList>

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

    void validate(const QJsonValue& value, const QString& path = {}) const;
    QJsonObject describe() const;
    QString definitionError(const QString& path = {}) const;

private:
    [[noreturn]] static void fail(const QString& param, const QString& detail);
    void validateNumber(
        const QJsonValue& value, const QString& param, bool integer) const;
    static QString typeName(AgentBridgeParamType type);
    static bool matchesType(
        const QJsonValue& value, AgentBridgeParamType type);
};

namespace AgentBridgeParams {

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

    QString mappedParamName(const QString& wireName) const;
    QJsonObject describe(const std::vector<AgentBridgeParam>& params) const;
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

    void validate(const QJsonValue& value) const;
    QString definitionError() const;
    QJsonObject describe() const;
};
