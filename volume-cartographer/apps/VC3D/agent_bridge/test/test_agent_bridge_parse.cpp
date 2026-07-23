// Unit tests for the Agent Bridge strict wire-parameter helpers (SPEC §5).
// Pure functions over QJsonValue/QJsonObject, so a tiny assert harness suffices.
//
// QJsonValue::toDouble()/toInt()/toBool() silently coerce a wrong-typed value to
// 0/false; the jsonRequire*/jsonTo* helpers must instead reject a PRESENT-but-
// malformed value with AgentBridgeError{-32602} carrying data["param"], while
// still ACCEPTING well-formed values.

#include <cstdio>
#include <limits>
#include <string>

#include <QJsonObject>
#include <QJsonValue>
#include <QString>

#include <opencv2/core.hpp>

#include "agent_bridge/AgentBridgeCommandError.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void reportFail(const char* file, int line, const std::string& what)
{
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what.c_str());
    ++g_failures;
}

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond))                                                           \
            reportFail(__FILE__, __LINE__, std::string("expected: ") + #cond); \
    } while (0)

// Runs `fn`, asserts it threw AgentBridgeError{-32602} with data["param"] equal
// to `expectedParam`.
template <typename Fn>
void expectParamError(const char* label, const char* expectedParam, Fn&& fn)
{
    ++g_checks;
    try {
        fn();
        reportFail(__FILE__, __LINE__,
                   std::string(label) + ": expected AgentBridgeError, none thrown");
    } catch (const AgentBridgeError& e) {
        if (e.code != -32602)
            reportFail(__FILE__, __LINE__,
                       std::string(label) + ": code=" + std::to_string(e.code) +
                           " (expected -32602)");
        const QString param = e.data.value(QStringLiteral("param")).toString();
        if (param != QLatin1String(expectedParam))
            reportFail(__FILE__, __LINE__,
                       std::string(label) + ": data.param=\"" +
                           param.toStdString() + "\" (expected \"" +
                           expectedParam + "\")");
    } catch (...) {
        reportFail(__FILE__, __LINE__,
                   std::string(label) + ": threw a non-AgentBridgeError exception");
    }
}

template <typename Fn>
void expectNoThrow(const char* label, Fn&& fn)
{
    ++g_checks;
    try {
        fn();
    } catch (const AgentBridgeError& e) {
        reportFail(__FILE__, __LINE__,
                   std::string(label) + ": unexpected AgentBridgeError code=" +
                       std::to_string(e.code) + " msg=" + e.message.toStdString());
    } catch (...) {
        reportFail(__FILE__, __LINE__,
                   std::string(label) + ": unexpected non-AgentBridgeError exception");
    }
}

}  // namespace

int main()
{
    const double kInf = std::numeric_limits<double>::infinity();
    const double kNaN = std::numeric_limits<double>::quiet_NaN();

    expectNoThrow("params<-absent", [&] {
        CHECK(paramsObject(QJsonValue(QJsonValue::Undefined)).isEmpty());
    });
    expectNoThrow("params<-null", [&] {
        CHECK(paramsObject(QJsonValue(QJsonValue::Null)).isEmpty());
    });
    expectNoThrow("params<-object", [&] {
        CHECK(paramsObject(QJsonObject{{"value", 1}}).value("value").toInt() == 1);
    });
    expectParamError("params<-array", "params", [&] {
        paramsObject(QJsonArray{});
    });
    expectParamError("params<-number", "params", [&] {
        paramsObject(QJsonValue(1));
    });
    expectParamError("string<-number", "name", [&] {
        jsonRequireString(QJsonValue(1), "name");
    });

    // --- jsonRequireNumber: reject non-numbers, accept numbers ---
    expectParamError("number<-string", "amount",
        [&] { jsonRequireNumber(QJsonValue(QStringLiteral("abc")), "amount"); });
    expectParamError("number<-bool", "amount",
        [&] { jsonRequireNumber(QJsonValue(true), "amount"); });
    expectParamError("number<-null", "amount",
        [&] { jsonRequireNumber(QJsonValue(QJsonValue::Null), "amount"); });
    expectNoThrow("number<-double", [&] {
        CHECK(jsonRequireNumber(QJsonValue(2.5), "amount") == 2.5);
    });
    expectNoThrow("number<-integer", [&] {
        CHECK(jsonRequireNumber(QJsonValue(7), "amount") == 7.0);
    });

    // --- jsonRequireFinite: reject NaN/Inf ---
    expectParamError("finite<-inf", "gain",
        [&] { jsonRequireFinite(QJsonValue(kInf), "gain"); });
    expectParamError("finite<-nan", "gain",
        [&] { jsonRequireFinite(QJsonValue(kNaN), "gain"); });
    expectNoThrow("finite<-double", [&] {
        CHECK(jsonRequireFinite(QJsonValue(-3.25), "gain") == -3.25);
    });

    // --- jsonRequireFiniteFloat: reject a finite double beyond float range ---
    // (1e300 is finite as double but overflows to +inf when narrowed to float).
    expectParamError("finiteFloat<-overflow", "coord",
        [&] { jsonRequireFiniteFloat(QJsonValue(1.0e300), "coord"); });
    expectParamError("finiteFloat<-neg-overflow", "coord",
        [&] { jsonRequireFiniteFloat(QJsonValue(-1.0e300), "coord"); });
    expectParamError("finiteFloat<-inf", "coord",
        [&] { jsonRequireFiniteFloat(QJsonValue(kInf), "coord"); });
    expectNoThrow("finiteFloat<-inrange", [&] {
        CHECK(jsonRequireFiniteFloat(QJsonValue(123.5), "coord") == 123.5);
    });

    // --- jsonRequireInt: reject wrong type, fractional, non-finite, overflow ---
    expectParamError("int<-string", "steps",
        [&] { jsonRequireInt(QJsonValue(QStringLiteral("3")), "steps"); });
    expectParamError("int<-fractional", "steps",
        [&] { jsonRequireInt(QJsonValue(1.5), "steps"); });
    expectParamError("int<-inf", "steps",
        [&] { jsonRequireInt(QJsonValue(kInf), "steps"); });
    expectParamError("int<-overflow", "steps",
        [&] { jsonRequireInt(QJsonValue(1.0e300), "steps"); });
    expectNoThrow("int<-integer", [&] {
        CHECK(jsonRequireInt(QJsonValue(42), "steps") == 42);
    });
    expectNoThrow("int<-integral-double", [&] {
        CHECK(jsonRequireInt(QJsonValue(-8.0), "steps") == -8);
    });

    // --- jsonRequireBool: reject non-bool, accept bool ---
    expectParamError("bool<-number", "flag",
        [&] { jsonRequireBool(QJsonValue(1), "flag"); });
    expectParamError("bool<-string", "flag",
        [&] { jsonRequireBool(QJsonValue(QStringLiteral("true")), "flag"); });
    expectNoThrow("bool<-bool", [&] {
        CHECK(jsonRequireBool(QJsonValue(true), "flag") == true);
    });

    // --- jsonRequireString / optionals: absent vs present-but-wrong-type ---
    {
        QJsonObject o;
        o["name"] = QStringLiteral("hi");
        o["num"] = 5;
        expectNoThrow("reqString<-string", [&] {
            CHECK(jsonRequireString(o, "name") == QLatin1String("hi"));
        });
        expectParamError("reqString<-number", "num",
            [&] { jsonRequireString(o, "num"); });
        // absent required string -> also -32602 (not a string).
        expectParamError("reqString<-absent", "missing",
            [&] { jsonRequireString(o, "missing"); });
        // Optional: absent returns default, present-but-wrong-type rejects.
        expectNoThrow("optString<-absent", [&] {
            CHECK(jsonOptionalString(o, "missing", QStringLiteral("def")) ==
                  QLatin1String("def"));
        });
        expectParamError("optInt<-fractional", "num2", [&] {
            QJsonObject bad;
            bad["num2"] = 2.5;
            jsonOptionalInt(bad, "num2", 0);
        });
        expectNoThrow("optBool<-absent", [&] {
            CHECK(jsonOptionalBool(o, "missing", true) == true);
        });
    }

    // --- jsonToVec3 {x, y, z} ---
    expectParamError("vec3<-not-object", "point",
        [&] { jsonToVec3(QJsonValue(42), "point"); });
    expectParamError("vec3<-missing-z", "point", [&] {
        QJsonObject o;
        o["x"] = 1.0;
        o["y"] = 2.0;
        jsonToVec3(QJsonValue(o), "point");
    });
    expectParamError("vec3<-bool-coord", "z", [&] {
        QJsonObject o;
        o["x"] = 1.0;
        o["y"] = 2.0;
        o["z"] = true;
        jsonToVec3(QJsonValue(o), "point");
    });
    expectParamError("vec3<-float-overflow-z", "z", [&] {
        QJsonObject o;
        o["x"] = 1.0;
        o["y"] = 2.0;
        o["z"] = 1.0e300;  // finite double, +inf as float
        jsonToVec3(QJsonValue(o), "point");
    });
    expectNoThrow("vec3<-valid", [&] {
        QJsonObject o;
        o["x"] = 1.0;
        o["y"] = 2.0;
        o["z"] = 3.0;
        const cv::Vec3f v = jsonToVec3(QJsonValue(o), "point");
        CHECK(v[0] == 1.0f);
        CHECK(v[1] == 2.0f);
        CHECK(v[2] == 3.0f);
    });

    CHECK(volumePointInBounds(cv::Vec3f{0.0f, 1.0f, 2.0f}, {3, 4, 5}));
    CHECK(volumePointInBounds(cv::Vec3f{2.9f, 3.9f, 4.9f}, {3, 4, 5}));
    CHECK(!volumePointInBounds(cv::Vec3f{-0.1f, 1.0f, 2.0f}, {3, 4, 5}));
    CHECK(!volumePointInBounds(cv::Vec3f{3.0f, 1.0f, 2.0f}, {3, 4, 5}));

    const auto mapLaunchError = [](CommandLaunchError::Kind kind) {
        return commandLaunchErrorToBridgeError(
            {kind, QStringLiteral("detail")},
            QStringLiteral("fallback"),
            QStringLiteral("segment-1"),
            QStringLiteral("tool"));
    };
    CHECK(mapLaunchError(CommandLaunchError::Other).code == -32005);
    CHECK(mapLaunchError(CommandLaunchError::InvalidState).code == -32005);
    const auto segmentError = mapLaunchError(CommandLaunchError::SegmentNotFound);
    CHECK(segmentError.code == -32007);
    CHECK(segmentError.data.value("kind").toString() == QLatin1String("segment"));
    CHECK(segmentError.data.value("id").toString() == QLatin1String("segment-1"));
    const auto volumeError = mapLaunchError(CommandLaunchError::VolumeNotFound);
    CHECK(volumeError.code == -32007);
    CHECK(volumeError.data.value("kind").toString() == QLatin1String("volume"));
    const auto inputError = mapLaunchError(CommandLaunchError::InputNotFound);
    CHECK(inputError.code == -32007);
    CHECK(inputError.data.value("kind").toString() == QLatin1String("file"));
    CHECK(mapLaunchError(CommandLaunchError::RemoteVolume).code == -32009);
    CHECK(mapLaunchError(CommandLaunchError::ToolUnavailable).code == -32006);
    const auto busyError = mapLaunchError(CommandLaunchError::Busy);
    CHECK(busyError.code == -32004);
    CHECK(busyError.data.value("source").toString() == QLatin1String("tool"));

    if (g_failures == 0) {
        std::printf("PASS: all %d checks in test_agent_bridge_parse\n", g_checks);
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d of %d checks failed\n", g_failures, g_checks);
    return 1;
}
