#pragma once
// QtTest across the Qt versions the project builds with: Qt 6.3 renamed
// QVERIFY_EXCEPTION_THROWN(expression, type) to
// QVERIFY_THROWS_EXCEPTION(type, expression); Ubuntu 22.04's Qt 6.2 has
// only the former.
#include <QtTest/QtTest>
#ifndef QVERIFY_THROWS_EXCEPTION
#define QVERIFY_THROWS_EXCEPTION(exceptiontype, ...) QVERIFY_EXCEPTION_THROWN((__VA_ARGS__), exceptiontype)
#endif
