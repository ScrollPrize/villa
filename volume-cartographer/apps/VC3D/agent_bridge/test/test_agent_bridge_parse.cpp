// Unit tests for the Agent Bridge strict wire-parameter helpers (SPEC §5).
//
// These are pure functions over QJsonValue/QJsonObject — no server, no event
// loop — so a tiny dependency-free assert harness suffices (matches the
// "plain main() returns nonzero on failure" style noted for this test).
//
// Covers the C4 hardening: QJsonValue::toDouble()/toInt()/toBool() silently
// coerce a wrong-typed value to 0/false; the jsonRequire*/jsonTo* helpers must
// instead reject a PRESENT-but-malformed value with AgentBridgeError{-32602}
// carrying data["param"], while still ACCEPTING well-formed values.

#include <cstdio>
#include <limits>
#include <string>

#include <QJsonObject>
#include <QJsonValue>
#include <QPointF>
#include <QString>

#include <opencv2/core.hpp>

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

// Runs `fn`, asserts it does NOT throw.
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

    // --- jsonBoundedInt: reject out of range, accept in range ---
    expectParamError("bounded<-below", "level",
        [&] { jsonBoundedInt(QJsonValue(-1), "level", 0, 5); });
    expectParamError("bounded<-above", "level",
        [&] { jsonBoundedInt(QJsonValue(6), "level", 0, 5); });
    expectNoThrow("bounded<-inrange", [&] {
        CHECK(jsonBoundedInt(QJsonValue(3), "level", 0, 5) == 3);
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

    // --- jsonToScenePoint {x, y} ---
    expectParamError("scene<-not-object", "pt",
        [&] { jsonToScenePoint(QJsonValue(QStringLiteral("x")), "pt"); });
    expectParamError("scene<-missing-y", "pt", [&] {
        QJsonObject o;
        o["x"] = 1.0;
        jsonToScenePoint(QJsonValue(o), "pt");
    });
    expectParamError("scene<-string-coord", "x", [&] {
        QJsonObject o;
        o["x"] = QStringLiteral("nope");
        o["y"] = 2.0;
        jsonToScenePoint(QJsonValue(o), "pt");
    });
    expectNoThrow("scene<-valid", [&] {
        QJsonObject o;
        o["x"] = 10.5;
        o["y"] = -4.0;
        const QPointF p = jsonToScenePoint(QJsonValue(o), "pt");
        CHECK(p.x() == 10.5);
        CHECK(p.y() == -4.0);
    });

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

    if (g_failures == 0) {
        std::printf("PASS: all %d checks in test_agent_bridge_parse\n", g_checks);
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d of %d checks failed\n", g_failures, g_checks);
    return 1;
}
